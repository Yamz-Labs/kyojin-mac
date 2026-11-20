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
import re

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


def glm_on() -> bool:
    """Flag kqgmm for NVFP4 experts and the clamped SwiGLU (GLM-5.3 prefill). KYOJIN_KQ_GMM_GLM=1 turns it on (default off until the 5-layer cut check). Read per call."""
    return os.environ.get("KYOJIN_KQ_GMM_GLM", "0") not in ("0", "")


def sched_on() -> bool:
    """KYOJIN_KQ_GMM_SCHED=omlx: the tile-decode schedule (foreign_kqnax, NVFP4 only, from NAX_MIN_PER_EXPERT mean rows per expert). Default off. Read per call."""
    return os.environ.get("KYOJIN_KQ_GMM_SCHED", "") == "omlx"


NAX_MIN_PER_EXPERT = int(os.environ.get("KYOJIN_KQ_NAX_MIN", "48"))                   # below this the register-fed kqgmm is as fast or faster (57 rows per expert: 1.03x)

def iq_on() -> bool:
    """Flag kqgmm-iq: the grouped kernel also reads iq2_s / iq3_s / iq3_xxs experts as stored (few-row MMA decoders of foreign_metal, grid tables from the decode kernels). KYOJIN_KQGMM_IQ=1 turns it on (default off). Read per call."""
    return os.environ.get("KYOJIN_KQGMM_IQ", "0") not in ("0", "")


def moe_pk_on() -> bool:
    """Flag moe_partialk (KYOJIN_MOE_PARTIALK, default off, read per call): the grouped kernel also reads q4_1 / q5_0 / q5_1 (kdec float decoders) and iq4_nl (table pair) experts as stored, any k % 128 == 0 (640 included). Read per call."""
    return os.environ.get("KYOJIN_MOE_PARTIALK", "0") not in ("0", "")


def klow_on() -> bool:
    """Flag kqgmm-klow: the grouped kernel also reads q3_k / q2_k experts as stored (half2 decoders dp_q3k / dp_q2k). KYOJIN_KQGMM_KLOW=1 turns it on (default off). Read per call."""
    return os.environ.get("KYOJIN_KQGMM_KLOW", "0") not in ("0", "")


def iq2x_on() -> bool:
    """Flag kqgmm-iq2x: the grouped kernel also reads iq2_xs / iq2_xxs experts as stored (half2 decoders dp_iq2xs / dp_iq2xxs). KYOJIN_KQGMM_IQ2X=1 turns it on (default off). Read per call."""
    return os.environ.get("KYOJIN_KQGMM_IQ2X", "0") not in ("0", "")


IQ2X_KINDS = ("iq2_xs", "iq2_xxs")
KLOW_KINDS = ("q3_k", "q2_k")
IQ_KINDS = ("iq2_s", "iq3_s", "iq3_xxs")
PK_KINDS = ("q4_1", "q5_0", "q5_1", "iq4_nl")
KINDS = ("q4_k", "q5_k", "q6_k", "iq4_xs", "nvfp4")                      # the formats checked against the fp16 path (tools/kqgmm_micro.py)


def ok(dp, S=None) -> bool:
    """A stacked expert pack the grouped kernel can read as stored."""
    return bool(dp.E) and (dp.fmt in KINDS or (dp.fmt in IQ_KINDS and iq_on()) or (dp.fmt in KLOW_KINDS and klow_on()) or (dp.fmt in IQ2X_KINDS and iq2x_on()) or (dp.fmt in PK_KINDS and moe_pk_on())) and dp.k % 128 == 0 and not dp.rowgs and (dp.fmt != "nvfp4" or glm_on())


def why_not(glu) -> str:
    """Flag audit: why `mlp(glu, ...)` was not used for a prompt chunk (only called under fired.ON)."""
    if not on():
        return "flag_off"
    if glu.limit is not None and not glm_on():
        return "glm_flag_off"
    for nm, p in (("gate", glu.g), ("up", glu.u), ("down", glu.d)):
        if not ok(p):
            return f"{nm}:" + ("not_experts" if not p.E else "fmt=" + str(p.fmt) if p.fmt not in KINDS and not (p.fmt in IQ_KINDS and iq_on()) and not (p.fmt in KLOW_KINDS and klow_on()) and not (p.fmt in IQ2X_KINDS and iq2x_on()) and not (p.fmt in PK_KINDS and moe_pk_on()) else "k%128" if p.k % 128 else "rowgs" if p.rowgs else "glm_flag_off")
    return "?"


_YST = "y[ulong(mb + m) * uint(N) + n0 + nl]"
_YST_S = "y[ulong(orow[mb + m]) * uint(N) + n0 + nl]"


def _hdr(fmt: str) -> str:
    with FM.kqp(klow_on()), FM.kqnl(fmt == "iq4_nl"):
        return FW.wide_header(fmt)


def source(fmt: str, scatter: bool = False, dep: bool = False, rt: int = 1) -> str:
    with FM.kqp(klow_on()), FM.kqnl(fmt == "iq4_nl"):
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
        if fmt in FM.FORMATS8 and "gs[0]" not in src:           # dense 8-bit linear (one group, ex = 0): per-row scales gs[n0 + nl] stay as they are (dense() only, never an expert stack)
            assert "gs[n0 + nl]" in src
        else:
            assert src.count("gs[0]") == 1
            src = src.replace("gs[0]", "gs[ex]")
    return rtn_source(src, rt) if rt > 1 else src


_RT_TA = "tensor<device half, dextents<int32_t, 2>, tensor_inline> tA((device half*)x + ulong(mb) * uint(KIN), dextents<int32_t, 2>(int(KIN), int(R)));"
_RT_MB = "const uint mb = off[ex] + tin * uint(MT);"
_RT_R = "const uint R = min(uint(MT), off[ex + 1u] - mb);"
_RT_CT = ("    auto cT = op.template get_destination_cooperative_tensor<RA<decltype(tA)>, RA<decltype(rB0)>, float>();\n"
          "    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;")
_RT_RUN = "if (step == 0) op.run(sA, rB0, cT); else op.run(sA, rB1, cT);"
_RT_EPI = "    threadgroup float red[WS][RC * 32];\n"


def rtn_source(src: str, n: int) -> str:
    """Decode-sharing variant (template RTT = n, 2 <= n <= 4): a threadgroup owns n consecutive row tiles of MT rows of one expert (a group of n * MT rows of tb / off) and decodes every
    weight k-block ONCE for all of them: n accumulators (cT, cT1, ...), n activation tensors, n epilogues. Row r of tile i is computed by the same instructions in the same order as in the
    one-tile kernel, so the output is bit-identical. Tiles past the end of the group have 0 rows (R_i = 0): their run / epilogue are skipped."""
    for a in (_RT_TA, _RT_MB, _RT_R, _RT_CT, _RT_RUN, _RT_EPI):
        assert src.count(a) == 1, f"kqgmm source changed: {a[:50]!r} found {src.count(a)} times"
    assert 2 <= n <= 4
    head, epi = src.split(_RT_EPI)
    src = head.replace(_RT_MB, "const uint mb = off[ex] + tin * uint(MT * RTT);")
    rows = "const uint RALL = min(uint(MT * RTT), off[ex + 1u] - mb);\n    const uint R = min(uint(MT), RALL);"
    for i in range(1, n):
        rows += f"\n    const uint mb{i} = mb + {i}u * uint(MT);\n    const uint R{i} = RALL > {i}u * uint(MT) ? min(uint(MT), RALL - {i}u * uint(MT)) : 0u;"
    src = src.replace(_RT_R, rows)
    src = src.replace(_RT_TA, _RT_TA + "".join(f"\n    tensor<device half, dextents<int32_t, 2>, tensor_inline> tA{i}((device half*)x + ulong(mb{i}) * uint(KIN), dextents<int32_t, 2>(int(KIN), int(R{i})));" for i in range(1, n)))
    src = src.replace(_RT_CT, _RT_CT + "".join(f"\n    auto cT{i} = op.template get_destination_cooperative_tensor<RA<decltype(tA)>, RA<decltype(rB0)>, float>();\n"
                                                f"    for (uint16_t i = 0; i < cT{i}.get_capacity(); ++i) if (cT{i}.is_valid_element(i)) cT{i}[i] = 0.0f;" for i in range(1, n)))
    src = src.replace(_RT_RUN, _RT_RUN + "".join(f"\n        if (R{i} > 0u) {{ auto sA{i} = tA{i}.slice(int(kb * 128u + step * 64u), 0); if (step == 0) op.run(sA{i}, rB0, cT{i}); else op.run(sA{i}, rB1, cT{i}); }}" for i in range(1, n)))
    out = src + _RT_EPI + epi
    for i in range(1, n):
        ei = re.sub(r"\bcT\b", f"cT{i}", re.sub(r"\bR\b", f"R{i}", epi.replace("mb + m", f"mb{i} + m")))      # tile i's epilogue: cT_i, R_i, rows from mb_i
        assert f"mb{i} + m" in ei and f"cT{i}" in ei and "mb + m" not in ei
        out += f"    if (R{i} > 0u) {{\n" + ei + "    }\n"
    return out


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
    with FM.kqp(klow_on()):
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


def rt_of() -> int:
    """KYOJIN_KQ_RT (read per call, default 1): row tiles of one expert that share each decoded weight k-block (n = decode once for n * MT rows). 1..4."""
    v = os.environ.get("KYOJIN_KQ_RT", "1")
    return int(v) if v in ("2", "3", "4") else 1


def gmm(dp, xp, tb, off, S: int, MT: int = 32, orow=None, dep=None, RT: int = 1, E=None):
    """xp half [S, k] = sorted activations already permuted with xperm(dp.fmt, dp.k); returns y half [S, n] in sorted order. E: the group count (default dp.E; 1 for a dense pack, see dense())."""
    E = dp.E if E is None else E
    if dp.fmt in PK_KINDS:
        from . import fired as FD
        FD.fire("moe_partialk_kqgmm_" + dp.fmt)
    ws = int(os.environ.get("KYOJIN_KQ_WS", "1") or 1) if MT <= 32 else (3 if MT >= 48 else 4)   # one simdgroup per 32 columns walks all of k: no cross-simdgroup reduction, more resident groups (-5 % chunk vs 4 k-splits at 8K; roofline 04/11)
    rc = min(MT, 7168 // (ws * 32))
    nm = names(dp.fmt, orow is not None, dep is not None)
    arrs = {"x": xp, "qs": dp.qs, "mt": dp.meta, "gs": dp.gs, "hi": getattr(dp, "hi", None), "tb": tb, "off": off, "orow": orow, "dep": dep}
    kern = KN._kern(("kqgmm", dp.fmt, orow is not None, dep is not None) + ((f"rt{RT}",) if RT > 1 else ()), name=f"kyojin_fg_kqgmm_{dp.fmt}" + ("_s" if orow is not None else "") + ("_d" if dep is not None else "") + (f"_rt{RT}" if RT > 1 else ""), input_names=nm, output_names=["y"],
                    header=_hdr(dp.fmt), source=source(dp.fmt, orow is not None, dep is not None, RT))
    maxt = S // (MT * RT) + E
    return kern(inputs=[arrs[a] for a in nm], template=[("MT", MT), ("KIN", dp.k), ("N", dp.n), ("WS", ws), ("RC", rc), ("E", E)] + ([("RTT", RT)] if RT > 1 else []),
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
    LIMIT_CLAMP
    float4 hv = av * (1.0f / (1.0f + metal::precise::exp(-av))) * bv;
    *(device half4*)(o + ulong(s) * uint(KIN) + p) = half4(hv);
"""


def gather_perm(x2, order, K: int, idx):
    """x2 half [T, k] -> half [S, k]: row s = x2[order[s] // K] with its columns permuted by idx (one pass instead of two takes)."""
    S, k = int(order.shape[0]), int(x2.shape[1])
    kern = KN._kern(("kq_gather",), name="kyojin_kq_gather_perm", input_names=["x", "order", "idx"], output_names=["o"], header="", source=_GATHER_ORD)
    return kern(inputs=[x2, order, idx], template=[("KIN", k), ("K", K)], grid=(S * k // 4, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(S, k)], output_dtypes=[mx.float16])[0]


_SILU_OLD = "float4 hv = av * (1.0f / (1.0f + metal::precise::exp(-av))) * bv;"


def act_perm_source(src: str, act):
    """Activation kind of the silu_perm source: act None = the silu-gated form (unchanged); ("situ", scale_gate, scale_lin) = ((sg tanh(a / sg)) * sigmoid(a)) * (sl tanh(b / sl)), no clamp."""
    if act is None:
        return src
    kind, sg, sl = act
    assert kind == "situ" and sg > 0.0 and sl > 0.0 and src.count(_SILU_OLD) == 1, act
    return src.replace(_SILU_OLD, f"float4 hv = (({float(sg)!r}f * metal::precise::tanh(av / {float(sg)!r}f)) * (1.0f / (1.0f + metal::precise::exp(-av)))) * ({float(sl)!r}f * metal::precise::tanh(bv / {float(sl)!r}f));")


def silu_perm(a, b, idx, limit=None, act=None):
    """silu(a) * b in f32 (rounded to fp16 like the stock path) with the columns permuted by idx: the down projection's staged input.
    limit (GLM-5.3 / DS4.1 swiglu_limit): a = min(a, limit), b = clip(b, -limit, limit) first, as ForeignSwitchGLU._glu."""
    S, k = int(a.shape[0]), int(a.shape[1])
    lim = "" if (limit is None or act) else f"av = min(av, {float(limit)!r}f); bv = clamp(bv, -{float(limit)!r}f, {float(limit)!r}f);"
    kern = KN._kern(("kq_silu", limit) + ((act,) if act else ()), name="kyojin_kq_silu_perm" + ("_situ" if act else ""), input_names=["a", "b", "idx"], output_names=["o"], header="",
                    source=act_perm_source(_SILU_PERM.replace("LIMIT_CLAMP", lim), act))
    return kern(inputs=[a, b, idx], template=[("KIN", k)], grid=(S * k // 4, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(S, k)], output_dtypes=[mx.float16])[0]


def mlp(glu, x2, e32, K: int, MT: int = 32):
    """ForeignSwitchGLU prefill on stored rows: x2 half [T, H], e32 uint32 [T * K] (slot s = token s // K). Returns y half [S, H] in slot order, or None if a projection is not eligible."""
    if not all(ok(p) for p in (glu.g, glu.u, glu.d)):
        return None
    S = int(e32.shape[0])
    E = glu.g.E
    if sched_on() and glu.g.fmt == "nvfp4" and S >= NAX_MIN_PER_EXPERT * E:           # flag KYOJIN_KQ_GMM_SCHED=omlx: weights decoded once per tile into threadgroup memory (foreign_kqnax)
        from . import foreign_kqnax as NX
        y = NX.mlp(glu, x2, e32, K)
        if y is not None:
            return y
    RT = rt_of() if MT * 4 <= 128 else 1                                                     # decode-sharing groups of RT row tiles: groups of RT * MT rows per expert
    if csort_on() and E <= 1024:
        order, tb, off = csort(e32, E, MT * RT)
    else:
        order = mx.argsort(e32).astype(mx.uint32)
        tb, off = plan(e32[order], E, MT * RT)
    xg = gather_perm(x2, order, K, xperm(glu.g.fmt, glu.g.k))
    a = gmm(glu.g, xg, tb, off, S, MT, RT=RT)
    xu = xg if (glu.u.fmt == glu.g.fmt and glu.u.k == glu.g.k) else gather_perm(x2, order, K, xperm(glu.u.fmt, glu.u.k))
    b = gmm(glu.u, xu, tb, off, S, MT, dep=a, RT=RT)
    return gmm(glu.d, silu_perm(a, b, xperm(glu.d.fmt, glu.d.k), glu.limit, getattr(glu, "act_kind", None)), tb, off, S, MT, orow=order, RT=RT)


def dense_on() -> bool:
    """Flag kqdense (KYOJIN_KQ_DENSE, default off, read per call): big-row dense 8-bit linears (Q8_0) through the grouped kernel as ONE group of rows."""
    return os.environ.get("KYOJIN_KQ_DENSE", "0") not in ("0", "")


DENSE_MIN_ROWS = int(os.environ.get("KYOJIN_KQ_DENSE_ROWS", "128"))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, "") or default)          # read per call: the harness sets the arm env after import


# KYOJIN_KQ_DENSE_MINSIZE: n * k floor, default 16 M: below it (q_a, shared) the decode + matmul / wide product is as fast (lev1 micro, small weights)
# KYOJIN_KQ_DENSE_MT: its own row tile (default 32; KQ_MT 16 is the expert tile) · KYOJIN_KQ_DENSE_RT: 0 = by shape (dense_rt), 1 / 2 / 4 force


def dense_rt(n: int, k: int, R: int) -> int:
    """RT 2 (one decoded weight k-block serves two row tiles) pays when there are many tiles and a long decode to share: lev2 micro R512 gate 3072->16384 RT1 1.50 / RT2 1.01 ms,
    gate 3072->8192 RT1 0.94 / RT2 1.24, o 16384->3072 0.949 / 0.967, q_b 1024->16384 0.435 / 0.484; R2048 gate RT2 2.61 vs RT1 2.69, o RT1 3.03 vs RT2 3.16."""
    return 2 if (k >= 2048 and n * R >= (1 << 23)) else 1


def dense_ok(dp, R: int) -> bool:
    return dense_on() and not dp.E and dp.fmt == "q8_0" and R >= DENSE_MIN_ROWS and dp.k % 128 == 0 and dp.n * dp.k >= _env_int("KYOJIN_KQ_DENSE_MINSIZE", 16 << 20)       # q8_0 packs carry per-row scales (rowgs): the kernel reads gs[n0 + nl]


def dense(dp, x2, MT=None, RT=None):
    """x2 [R, k] -> half [R, n]: a dense stored linear as ONE group of the grouped kernel (tb = [0, tiles], off = [0, R]); with RT >= 2 each decoded weight k-block serves RT row tiles
    (KYOJIN_KQ_DENSE_RT, default by shape), so the stored bytes are decoded and read once per RT * MT rows instead of once per 32."""
    R = int(x2.shape[0])
    if x2.dtype != mx.float16:
        x2 = mx.clip(x2, -65504.0, 65504.0).astype(mx.float16)
    MT = _env_int("KYOJIN_KQ_DENSE_MT", 32) if MT is None else MT
    rt = (_env_int("KYOJIN_KQ_DENSE_RT", 0) or dense_rt(int(dp.n), int(dp.k), R)) if RT is None else RT           # its own switch: the expert kernel's KYOJIN_KQ_RT does not pay on Q3_K
    xp = mx.take(x2, xperm(dp.fmt, dp.k), axis=1)
    tb = mx.array([0, -(-R // (MT * rt))], dtype=mx.uint32)
    off = mx.array([0, R], dtype=mx.uint32)
    return gmm(dp, xp, tb, off, R, MT, RT=rt, E=1)
