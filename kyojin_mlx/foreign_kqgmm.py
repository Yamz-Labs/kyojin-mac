# SPDX-License-Identifier: Apache-2.0
"""Expert-grouped GEMM on the STORED rows of a routed-expert stack (prefill of a GGUF / foreign MoE with K-quant or 4-bit experts), one resident copy.

The prefill path used to decode the whole expert stack of a projection to fp16 once per chunk and run mx.gather_mm over it (a fixed ~280 ms per 2048-token chunk on the
35B-A3B UD-Q4_K_XL: 148 GB of fp16 traffic for 18 GB of stored bytes). Here the slots are sorted by expert and every threadgroup owns a tile of at most MT sorted rows of ONE
expert: foreign_wide.wide_source_reg (register-fed NAX product, the half2 decoders write straight into the cooperative tensor, fp32 accumulation) with the weight rows offset by
the tile's expert. The stored bytes are read once per tile, nothing is decoded to memory. The tile list is built on the device from the sorted expert ids (no host readback):
tb[e] = first tile of expert e (tb[E] = tiles in use), off[e] = first sorted row of expert e. Threadgroups past tb[E] return at once.
Output: y half [S, n] in sorted order, saturated to +-65504 (the kernel clamps). Needs k % 128 == 0 and a format with a few-row MMA decoder (q4_k, q5_k, q6_k, iq4_xs, q4_0, ...)."""
from __future__ import annotations

import os

import mlx.core as mx

from . import foreign_metal as FM, foreign_wide as FW
from . import kernels as KN

_HEAD = """    const uint mb = threadgroup_position_in_grid.y * uint(MT);
    const uint R = min(uint(MT), uint(x_shape[0]) - mb);
"""
_TILE = """    const uint ti = threadgroup_position_in_grid.y;
    if (ti >= tb[uint(E)]) return;
    uint e_lo = 0u, e_hi = uint(E);
    while (e_hi - e_lo > 1u) { const uint e_mid = (e_lo + e_hi) >> 1; if (tb[e_mid] <= ti) e_lo = e_mid; else e_hi = e_mid; }
    const uint ex = e_lo, tin = ti - tb[ex];
    const uint mb = off[ex] + tin * uint(MT);
    const uint R = min(uint(MT), off[ex + 1u] - mb);
"""
_ROWT = "rowt[j] = min(n0 + nb + 8u * j, uint(N) - 1u);"
_ROWT_E = "rowt[j] = uint(ex) * uint(N) + min(n0 + nb + 8u * j, uint(N) - 1u);"


def on() -> bool:
    """Flag kqgmm (default on; KYOJIN_KQ_GMM=0 turns it off). Read per call."""
    return os.environ.get("KYOJIN_KQ_GMM", "1") not in ("0", "")


KINDS = ("q4_k", "q5_k", "q6_k", "iq4_xs")                      # the formats checked against the fp16 path (tools/kqgmm_micro.py)


def ok(dp, S=None) -> bool:
    """A stacked expert pack the grouped kernel can read as stored."""
    return bool(dp.E) and dp.fmt in KINDS and dp.k % 128 == 0 and not dp.rowgs


_YST = "y[ulong(mb + m) * uint(N) + n0 + nl]"
_YST_S = "y[ulong(orow[mb + m]) * uint(N) + n0 + nl]"


def source(fmt: str, scatter: bool = False, dep: bool = False) -> str:
    src = FW.wide_source_reg(fmt)
    if scatter:                                                 # output row of sorted row r = orow[r] (slot order: no unsort pass)
        assert src.count(_YST) == 1
        src = src.replace(_YST, _YST_S)
    for a, b in ((_HEAD, _TILE), (_ROWT, _ROWT_E)):
        assert src.count(a) == 1, f"wide_source_reg changed: {a[:40]!r} found {src.count(a)} times"
        src = src.replace(a, b)
    if dep:                                                     # a never-taken read of `dep`: the launch waits for the array that produced it (two NAX products running together at 4K rows are 2x slower than one after the other)
        assert src.count("if (ti >= tb[uint(E)]) return;") == 1
        src = src.replace("if (ti >= tb[uint(E)]) return;", "if (ti >= tb[uint(E)]) { if (ti > 0x7FFFFFFFu) y[0] = half(dep[0]); return; }")
    if fmt not in FM._FMMA_K:                                   # per-expert global scale (nvfp4-class gs[E]; 1.0 for Q4_K)
        assert src.count("gs[0]") == 1
        src = src.replace("gs[0]", "gs[ex]")
    return src


def names(fmt: str, scatter: bool = False, dep: bool = False):
    return FW.wide_inputs(fmt) + ["tb", "off"] + (["orow"] if scatter else []) + (["dep"] if dep else [])


def csort_on() -> bool:
    """Flag kqcsort (default on; KYOJIN_KQ_CSORT=0 turns it off). Read per call."""
    return os.environ.get("KYOJIN_KQ_CSORT", "1") not in ("0", "")


_CSORT = r"""
    threadgroup atomic_uint cnt[E];
    threadgroup atomic_uint cur[E];
    const uint t = thread_position_in_threadgroup.x;
    for (uint i = t; i < uint(E); i += 1024u) atomic_store_explicit(&cnt[i], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = t; s < uint(S); s += 1024u) atomic_fetch_add_explicit(&cnt[e[s]], 1u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0u) {
      uint o = 0u, b = 0u;
      for (uint i = 0u; i < uint(E); ++i) {
        const uint c = atomic_load_explicit(&cnt[i], memory_order_relaxed);
        off[i] = o; tb[i] = b; atomic_store_explicit(&cur[i], o, memory_order_relaxed);
        o += c; b += (c + uint(MT) - 1u) / uint(MT);
      }
      off[E] = o; tb[E] = b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = t; s < uint(S); s += 1024u) order[atomic_fetch_add_explicit(&cur[e[s]], 1u, memory_order_relaxed)] = s;
"""


def csort(e32, E: int, MT: int):
    """One launch: counting sort of the slots by expert. Returns (order uint32 [S] (slots of one expert in arbitrary order), tb [E + 1], off [E + 1]) like argsort + plan."""
    S = int(e32.shape[0])
    kern = KN._kern(("kq_csort",), name="kyojin_kq_csort", input_names=["e"], output_names=["order", "tb", "off"], header="", source=_CSORT)
    return kern(inputs=[e32], template=[("S", S), ("E", E), ("MT", MT)], grid=(1024, 1, 1), threadgroup=(1024, 1, 1),
                output_shapes=[(S,), (E + 1,), (E + 1,)], output_dtypes=[mx.uint32] * 3)


_IDX: dict = {}


def xperm(fmt: str, k: int):
    key = (FW.wide_layout(fmt), k)
    if key not in _IDX:
        _IDX[key] = mx.array(FW.wide_xperm_index_reg(fmt, k))
    return _IDX[key]


def plan(es, E: int, MT: int):
    """(tb uint32 [E + 1], off uint32 [E + 1]) from the sorted expert ids es (uint32 [S])."""
    cnt = mx.zeros((E,), mx.int32).at[es.astype(mx.int32)].add(1)
    z = mx.zeros((1,), mx.int32)
    off = mx.concatenate([z, mx.cumsum(cnt)])
    tb = mx.concatenate([z, mx.cumsum((cnt + (MT - 1)) // MT)])
    return tb.astype(mx.uint32), off.astype(mx.uint32)


def gmm(dp, xp, tb, off, S: int, MT: int = 32, orow=None, dep=None):
    """xp half [S, k] = sorted activations already permuted with xperm(dp.fmt, dp.k); returns y half [S, n] in sorted order."""
    E = dp.E
    ws = int(os.environ.get("KYOJIN_KQ_WS", "1") or 1) if MT <= 32 else (3 if MT >= 48 else 4)   # one simdgroup per 32 columns walks all of k: no cross-simdgroup reduction, more resident groups (-5 % chunk vs 4 k-splits at 8K; roofline 04/11)
    rc = min(MT, 7168 // (ws * 32))
    nm = names(dp.fmt, orow is not None, dep is not None)
    arrs = {"x": xp, "qs": dp.qs, "mt": dp.meta, "gs": dp.gs, "hi": getattr(dp, "hi", None), "tb": tb, "off": off, "orow": orow, "dep": dep}
    kern = KN._kern(("kqgmm", dp.fmt, orow is not None, dep is not None), name=f"kyojin_fg_kqgmm_{dp.fmt}" + ("_s" if orow is not None else "") + ("_d" if dep is not None else ""), input_names=nm, output_names=["y"],
                    header=FW.wide_header(dp.fmt), source=source(dp.fmt, orow is not None, dep is not None))
    maxt = S // MT + E
    return kern(inputs=[arrs[a] for a in nm], template=[("MT", MT), ("KIN", dp.k), ("N", dp.n), ("WS", ws), ("RC", rc), ("E", E)],
                grid=(-(-dp.n // 32) * 32 * ws, maxt, 1), threadgroup=(32 * ws, 1, 1), output_shapes=[(S, dp.n)], output_dtypes=[mx.float16])[0]


_GATHER = r"""
    const uint g = thread_position_in_grid.x;                 // 4 outputs per thread
    const uint s = g / (uint(KIN) / 4u), p = (g - s * (uint(KIN) / 4u)) * 4u;
    const device half* xr = x + ulong(tok[s]) * uint(KIN);
    const int4 ix = *(device const int4*)(idx + p);
    *(device half4*)(o + ulong(s) * uint(KIN) + p) = half4(xr[ix.x], xr[ix.y], xr[ix.z], xr[ix.w]);
"""
_GATHER_ORD = _GATHER.replace("tok[s]", "(order[s] / uint(K))")
_SILU_PERM = r"""
    const uint g = thread_position_in_grid.x;
    const uint s = g / (uint(KIN) / 4u), p = (g - s * (uint(KIN) / 4u)) * 4u;
    const int4 ix = *(device const int4*)(idx + p);
    const device half* ar = a + ulong(s) * uint(KIN);
    const device half* br = b + ulong(s) * uint(KIN);
    float4 av = float4(float(ar[ix.x]), float(ar[ix.y]), float(ar[ix.z]), float(ar[ix.w]));
    float4 bv = float4(float(br[ix.x]), float(br[ix.y]), float(br[ix.z]), float(br[ix.w]));
    float4 hv = av * (1.0f / (1.0f + metal::precise::exp(-av))) * bv;
    *(device half4*)(o + ulong(s) * uint(KIN) + p) = half4(hv);
"""


def gather_perm(x2, order, K: int, idx):
    """x2 half [T, k] -> half [S, k]: row s = x2[order[s] // K] with its columns permuted by idx (one pass instead of two takes)."""
    S, k = int(order.shape[0]), int(x2.shape[1])
    kern = KN._kern(("kq_gather",), name="kyojin_kq_gather_perm", input_names=["x", "order", "idx"], output_names=["o"], header="", source=_GATHER_ORD)
    return kern(inputs=[x2, order, idx], template=[("KIN", k), ("K", K)], grid=(S * k // 4, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(S, k)], output_dtypes=[mx.float16])[0]


def silu_perm(a, b, idx):
    """silu(a) * b in f32 (rounded to fp16 like the stock path) with the columns permuted by idx: the down projection's staged input."""
    S, k = int(a.shape[0]), int(a.shape[1])
    kern = KN._kern(("kq_silu",), name="kyojin_kq_silu_perm", input_names=["a", "b", "idx"], output_names=["o"], header="", source=_SILU_PERM)
    return kern(inputs=[a, b, idx], template=[("KIN", k)], grid=(S * k // 4, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(S, k)], output_dtypes=[mx.float16])[0]


def mlp(glu, x2, e32, K: int, MT: int = 32):
    """ForeignSwitchGLU prefill on stored rows: x2 half [T, H], e32 uint32 [T * K] (slot s = token s // K). Returns y half [S, H] in slot order, or None if a projection is not eligible."""
    if not all(ok(p) for p in (glu.g, glu.u, glu.d)):
        return None
    S = int(e32.shape[0])
    E = glu.g.E
    if csort_on() and E <= 1024:
        order, tb, off = csort(e32, E, MT)
    else:
        order = mx.argsort(e32).astype(mx.uint32)
        tb, off = plan(e32[order], E, MT)
    xg = gather_perm(x2, order, K, xperm(glu.g.fmt, glu.g.k))
    a = gmm(glu.g, xg, tb, off, S, MT)
    xu = xg if (glu.u.fmt == glu.g.fmt and glu.u.k == glu.g.k) else gather_perm(x2, order, K, xperm(glu.u.fmt, glu.u.k))
    b = gmm(glu.u, xu, tb, off, S, MT, dep=a)
    return gmm(glu.d, silu_perm(a, b, xperm(glu.d.fmt, glu.d.k)), tb, off, S, MT, orow=order)
