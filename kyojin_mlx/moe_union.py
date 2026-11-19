# SPDX-License-Identifier: Apache-2.0
"""Flag moe_union (KYOJIN_MOE_UNION, default off): routed-expert block of a 2..8-row pass (MTP verify rows) with each DISTINCT expert read once for all the rows that chose it.

The per-slot gather (ForeignSwitchGLU, one threadgroup per (token, expert) slot) reads an expert's bytes once per row that picked it; verify rows overlap a lot, so the
union kernels run one threadgroup per distinct expert: the weight chunk is decoded once and multiplied by every row of the group (the kernel text is the stored
format's own gather GEMV with R = rows, so the k-quant decoders, the saturation to +-65504 and the sum order per row are the same as the gather's).

  tables  inds [T, K] -> eids [S] (the distinct experts, first-seen order, padded with 0), oslot [S, T] (group g, row r: the slot r * K + k where row r chose expert g, else NONE)
          one threadgroup, no host read: static shapes, so the compiled layer program keeps it.
  gate_up one launch over the stored gate and up matrices (grid y = 2 S groups; groups past the distinct count return at once): y [2 S, n], rows [0, S) gate, [S, 2 S) up, slot order.
  down    the down projection with the gated activation silu(gate) * up applied while the input is staged (glu), input row of (g, r) = its slot.
Generic by stored type: every k-quant of foreign_metal.FORMATSK that the gather GEMV already runs (q3_k, q4_k, q5_k, q6_k, iq*, ...), Metal only."""
from __future__ import annotations

import os

import mlx.core as mx

from . import foreign_metal as FM
from . import kernels as KN

UNION = os.environ.get("KYOJIN_MOE_UNION", "0") not in ("0", "")       # flag moe_union (flags.py rewrites it at runtime)
NONE_U = "0xFFFFFFFFu"
MAX_ROWS = 8
PATHS = {"union": 0}

_TABLES = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint NT = 128u;
    threadgroup uint first_s[S_], cnt_s[S_], base_s[S_], ge_s[S_], gc_s[S_];
    for (uint s = tid; s < uint(S_); s += NT) {
      const uint e = inds[s]; uint f = 1u, c = 0u;
      for (uint p = 0; p < s; ++p) if (inds[p] == e) { f = 0u; break; }
      if (f != 0u) for (uint p = s; p < uint(S_); ++p) c += (inds[p] == e) ? 1u : 0u;
      first_s[s] = f; cnt_s[s] = c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tid; s < uint(S_); s += NT) {
      uint b = 0u;
      for (uint p = 0; p < s; ++p) b += first_s[p] != 0u ? (cnt_s[p] + uint(R) - 1u) / uint(R) : 0u;
      base_s[s] = b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tid; s < uint(S_); s += NT) if (first_s[s] != 0u) {
      const uint nch = (cnt_s[s] + uint(R) - 1u) / uint(R);
      for (uint ch = 0; ch < nch; ++ch) { ge_s[base_s[s] + ch] = inds[s]; gc_s[base_s[s] + ch] = ch; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint U = 0u;
    for (uint p = 0; p < uint(S_); ++p) U += first_s[p] != 0u ? (cnt_s[p] + uint(R) - 1u) / uint(R) : 0u;
    for (uint g = tid; g < uint(S_); g += NT) eids[g] = g < U ? ge_s[g] : 0u;
    for (uint i = tid; i < uint(S_) * uint(R); i += NT) {
      const uint g = i / uint(R), j = i % uint(R);
      uint res = 0xFFFFFFFFu;
      if (g < U) {
        const uint e = ge_s[g], want = gc_s[g] * uint(R) + j;
        uint seen = 0u;
        for (uint r = 0; r < uint(T_) && res == 0xFFFFFFFFu; ++r)
          for (uint k = 0; k < uint(K); ++k) if (inds[r * uint(K) + k] == e) { if (seen == want) res = r * uint(K) + k; ++seen; break; }
      }
      oslot[i] = res;
    }
"""


def rows_per_group() -> int:
    """R of the union kernels: the most rows one threadgroup handles for one expert (an expert chosen by more rows takes several groups, its bytes read once per group). 2 by default
    (the mean expert of a verify pass is chosen by ~1.5 rows); KYOJIN_MOE_UNION_R. R = rows = the whole union."""
    return max(1, int(os.environ.get("KYOJIN_MOE_UNION_R", "2")))


def tables(inds: mx.array, T: int, K: int, R: int):
    """inds [T, K] (any int dtype) -> (eids uint32 [S], oslot uint32 [S * R]) with S = T * K: group g = (an expert, a chunk of <= R of its rows): eids[g] its expert (0 past the last group),
    oslot[g * R + j] the slot (row * K + k) of its j-th row, or 0xFFFFFFFF."""
    S = T * K
    kern = KN._kern(("moe_union_tables2",), name="kyojin_moe_union_tables2", input_names=["inds"], output_names=["eids", "oslot"], source=_TABLES)
    return kern(inputs=[inds.reshape(-1).astype(mx.uint32)], template=[("S_", S), ("R", R), ("T_", T), ("K", K)], grid=(128, 1, 1), threadgroup=(128, 1, 1),
                output_shapes=[(S,), (S * R,)], output_dtypes=[mx.uint32, mx.uint32])


def family(d):
    """"k" (the K / IQ gather GEMV, gemvk_source), "n" (the legacy-format gather GEMV, gemv_source) or None (nvfp4 / 8-bit / dense: not covered)."""
    if d.fmt == "dense" or not d.gemv_ok() or not d.E or d.g2a_ok(True):
        return None
    if d.fmt in FM.FORMATSK:
        return "k"
    return None if d.fmt in FM.FORMATS8 else "n"


def source(fam: str, fmt: str, nbr: int, glu: bool, pair: bool, partial: bool) -> tuple[str, list[str]]:
    """The union text of the stored format's gather GEMV (+ the input names after x [x2] oslot eids). Template ints: R (rows per group), KIN, N, NBR, SP (groups), SO (slots), SLOTIN, KK (experts per row: the x row of a slot is slot / KK)."""
    if fam == "k":
        src = FM.gemvk_source(fmt, nbr, True, glu, partial)
        base = FM.gemvk_inputs(fmt, True, False)[3:]                   # qs, mt, [hi]
    else:
        assert not glu
        src = FM.gemv_source(fmt, nbr, True, False, False, partial)
        base = FM.gemv_inputs(True)[3:]                                # qs, mt, gs
    head = "const uint slot = threadgroup_position_in_grid.y;"
    assert src.count(head) == 1
    for old in (" const uint ex = eids[slot];", "\n    const uint ex = eids[slot];"):
        src = src.replace(old, "")
    setup = ("\n    const uint ex = eids[slot];\n    threadgroup uint osl_s[R];\n    if (tid < uint(R)) osl_s[tid] = oslot[slot * uint(R) + tid];\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
             f"    {{ bool any_ = false; for (uint r = 0; r < uint(R); ++r) any_ = any_ || osl_s[r] != {NONE_U}; if (!any_) return; }}\n")
    if pair:
        sel = "".join(f" auto {n} = sec_ ? {n}B : {n}A;" for n in base)
        pre = "const uint s2_ = threadgroup_position_in_grid.y; const bool sec_ = s2_ >= uint(SP); const uint slot = sec_ ? s2_ - uint(SP) : s2_;" + sel + "\n    const uint OFFS = sec_ ? uint(SO) : 0u;"
    else:
        pre = "const uint slot = threadgroup_position_in_grid.y;\n    const uint OFFS = 0u;"
    src = src.replace(head, pre + setup)
    ri = f"(osl_s[r] == {NONE_U} ? 0u : (SLOTIN != 0 ? osl_s[r] : osl_s[r] / uint(KK)))"          # gate / up read the token row of the slot (slot / K)
    for old in ("XP + r * uint(KIN)", "XP2 + r * uint(KIN)"):
        if old in src:
            xn = "x" if old.startswith("XP ") else "x2"
            src = src.replace(old, f"(device const half*)({xn} + ulong({ri}) * uint(KIN))")
    for old in ("    device const half* XP = (device const half*)(x + ulong(xrows[slot]) * uint(KIN));\n", "    device const half* XP2 = (device const half*)(x2 + ulong(xrows[slot]) * uint(KIN));\n",
                "    device half* YP = y + ulong(slot) * uint(N);\n"):
        src = src.replace(old, "")
    a = "if (ul == 0u) YP[r * uint(N) + rbase + j] ="
    assert src.count(a) == 1
    src = src.replace(a, f"if (ul == 0u && osl_s[r] != {NONE_U}) y[(ulong(osl_s[r]) + OFFS) * uint(N) + rbase + j] =")
    c_ = "uint r = i >> 6, j = i & 63u;"
    assert src.count(c_) == 1
    src = src.replace(c_, c_ + f" if (osl_s[r] == {NONE_U}) continue;")                 # an inactive row stages nothing
    b = "threadgroup const half* xr = xs + r * 640u;"
    assert src.count(b) == 1
    src = src.replace(b, f"if (osl_s[r] == {NONE_U}) continue;\n        " + b)
    assert "XP" not in src and "xrows" not in src and "YP" not in src, "moe_union: unexpected gather text"
    return src, ([n + "A" for n in base] + [n + "B" for n in base]) if pair else base


def _kern(fam, fmt, nbr, glu, pair, partial):
    from . import foreign_linear as FL
    src, names = source(fam, fmt, nbr, glu, pair, partial)
    ins = ["x"] + (["x2"] if glu else []) + ["oslot", "eids"] + names
    cf = FL.cfmt() if fam == "n" else False
    hdr = FM.hdrk(fmt) + (FM.glu_helper(None) if glu else "") if fam == "k" else FM.header(fmt, "gemv", cf)
    return KN._kern(("moe_union", fam, fmt, nbr, glu, pair, partial, cf), name=f"kyojin_moe_union_{fam}_{fmt}_n{nbr}" + ("u" if glu else "") + ("p" if pair else "") + ("_p" if partial else ""),
                    input_names=ins, output_names=["y"], header=hdr, source=src)


def _arrs(d, fam):
    return [d.qs, d.meta] + ([d.hi] if d.hi is not None else []) if fam == "k" else [d.qs, d.meta, d.gs]


def plan(g, u, d, T: int, K: int, nbr: int):
    """None, or (family of gate / up, family of down, fused glu down?) when the union path can run this block."""
    if not (UNION and 2 <= T <= MAX_ROWS and mx.default_device() == mx.gpu and g.tl is None and d.tl is None and g.n == d.k and g.pair_ok(u)):
        return None
    fg, fd = family(g), family(d)
    if fg is None or fd is None or g.n % (8 * nbr) or d.n % (8 * nbr) or g.k % 32 or d.k % 32:
        return None
    return fg, fd, bool(fd == "k" and d.glu_ok())


def gate_up(g, u, fam, x, eids, oslot, R: int, S: int, nbr: int, K: int):
    """x half [T, k] -> y half [2 S, n]: rows [0, S) = gate, [S, 2 S) = up, slot order (slot = row * K + k)."""
    kern = _kern(fam, g.fmt, nbr, False, True, g._part())
    return kern(inputs=[x, oslot, eids] + _arrs(g, fam) + _arrs(u, fam), template=[("R", R), ("KIN", g.k), ("N", g.n), ("NBR", nbr), ("SP", S), ("SO", S), ("SLOTIN", 0), ("KK", K)],
                grid=(g.n // (8 * nbr) * 128, 2 * S, 1), threadgroup=(128, 1, 1), output_shapes=[(2 * S, g.n)], output_dtypes=[mx.float16])[0]


def down(d, fam, glu, a, b, eids, oslot, R: int, S: int, nbr: int):
    """glu: a, b half [S, k] (gate / up of every slot), the input of slot s is silu(a) * b (K family only); else a = h half [S, k] (b unused). -> y half [S, n]."""
    kern = _kern(fam, d.fmt, nbr, glu, False, d._part())
    return kern(inputs=([a, b] if glu else [a]) + [oslot, eids] + _arrs(d, fam), template=[("R", R), ("KIN", d.k), ("N", d.n), ("NBR", nbr), ("SP", S), ("SO", S), ("SLOTIN", 1), ("KK", 1)],
                grid=(d.n // (8 * nbr) * 128, S, 1), threadgroup=(128, 1, 1), output_shapes=[(S, d.n)], output_dtypes=[mx.float16])[0]
