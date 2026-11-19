# SPDX-License-Identifier: Apache-2.0
"""G2c rows path: routed-expert projections of a decode / verify block (T rows, T * K slots), one threadgroup column per DISTINCT expert.

Design after oMLX's moe_verify_gather (https://github.com/jundot/omlx, Apache-2.0, omlx/patches/moe_verify_gather.py: the pairs of a short verify block are
processed in expert order so rows that pick the same expert read its weights once) and moe_gate_up_swiglu (the clamped SwiGLU is computed in the epilogue of
the gate | up launch). The kernels here are our own: NVFP4 expert rows in the G2a register-fed layout (foreign_metal.gemv_g2_source: same decode, same order
of the products and sums per output, so the outputs are bit-identical to ForeignSwitchGLU's pair / glu / down launches).

    group     one launch (T > 1): slots -> table [S, 2 + T] of (expert, count, slot_0 .. slot_{count - 1}), count = 0 for the unused positions
    pair_glu  grid.y = position in the table: the expert's gate AND up rows of an output tile are read once into registers, then for each row that
              picked the expert x is staged and h = clamped silu(gate) * up is written (no separate glu launch, no gate | up intermediate)
    down      the same for the down projection (y[slot] = down_e(h[slot]))

T = 1: every position holds one slot, no table, no group launch. The weight bytes of a distinct expert are read once per block instead of once per row.
"""
from __future__ import annotations

import functools

import mlx.core as mx

from . import kernels as KN
from . import moe_scale_compact as MC

T_MAX = 8                                              # rows per block of the table path (an expert is picked at most once per row)

_HDR = r"""
#define UNR _Pragma("clang loop unroll(full)")
inline float unit_acc(uint4 qv, ushort m, thread const float4* xf, float acc) {
  const uint mm = m;
  const half2 sh = as_type<half2>(((mm & 0x7Fu) << 7) | ((mm & 0x7F00u) << 15));
  const float2 sc = float2(sh) * 4194304.0f;                       // 2^22 = 2^8 (e4m3 -> half shift) * 2^14 (e2m1 -> half)
  float pb[2] = {0.0f, 0.0f};
  UNR for (uint wi = 0; wi < 4; ++wi) {
    const uint wd = qv[wi];
    float p = pb[wi >> 1];
    UNR for (uint t = 0; t < 4; ++t) {
      const uint mg = (t < 3u ? (wd << (9u - 4u * t)) : (wd >> 3)) & 0x0E000E00u;
      const uint sn = (wd << (12u - 4u * t)) & 0x80008000u;
      const float2 f = float2(as_type<half2>(mg | sn));
      p = fma(f.x, xf[2u * wi][t], p);
      p = fma(f.y, xf[2u * wi + 1u][t], p);
    }
    pb[wi >> 1] = p;
  }
  return fma(pb[0], sc.x, fma(pb[1], sc.y, acc));
}
"""

_ROWS = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tgi = threadgroup_position_in_grid.x;
    const uint p = threadgroup_position_in_grid.y;
    uint cnt = 1u, ex = eids[p];
    if (!SINGLE) {
      cnt = tab[p * uint(W) + 1u];
      if (cnt == 0u) return;                                           // unused position: the whole threadgroup leaves
      ex = tab[p * uint(W)];
    }
    constexpr uint UPR = uint(KIN) / 32u, NU = uint(KIN) / 1024u, NT = 32u * uint(SGN);
    constexpr uint NBU = PG ? NU : 1u;
    const uint rbase = tgi * uint(SGN) + sg;
    device const uint4* qa = (device const uint4*)(qsA + ulong(ex) * (ulong(N) * uint(KIN) / 8u));
    device const ushort* ma = (device const ushort*)(mtA + ulong(ex) * (ulong(N) * (uint(KIN) / 32u)));
    device const uint4* qb = (device const uint4*)(qsB + ulong(ex) * (ulong(N) * uint(KIN) / 8u));
    device const ushort* mb = (device const ushort*)(mtB + ulong(ex) * (ulong(N) * (uint(KIN) / 32u)));
    const float gsa = gsA[ex];
    const float gsb = PG ? gsB[ex] : 1.0f;
    uint4 qva[NU]; ushort mva[NU]; uint4 qvb[NBU]; ushort mvb[NBU];
    UNR for (uint i = 0; i < NU; ++i) {
      const ulong o = ulong(rbase) * UPR + lane + 32u * i;
      qva[i] = qa[o]; mva[i] = ma[o];
    }
    if (PG) {
      UNR for (uint i = 0; i < NU; ++i) {
        const ulong o = ulong(rbase) * UPR + lane + 32u * i;
        qvb[i] = qb[o]; mvb[i] = mb[o];
      }
    }
    threadgroup half xs[(uint(KIN) / 32u) * 40u];
    for (uint r = 0; r < cnt; ++r) {
      const uint slot = SINGLE ? p : tab[p * uint(W) + 2u + r];
      device const half* XP = (device const half*)(x + ulong(slot / uint(XDIV)) * uint(KIN));
      if (r > 0u) threadgroup_barrier(mem_flags::mem_threadgroup);   // the previous row's reads of xs are done
      for (uint j = tid; j < uint(KIN) / 8u; j += NT) *(threadgroup uint4*)(xs + j * 8u + (j >> 2) * 8u) = *(device const uint4*)(XP + j * 8u);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float acca = 0.0f, accb = 0.0f;
      UNR for (uint i = 0; i < NU; ++i) {
        threadgroup const half4* xp = (threadgroup const half4*)(xs + (lane + 32u * i) * 40u);
        float4 xf[8];
        UNR for (uint t = 0; t < 8; ++t) xf[t] = float4(xp[t]);
        acca = unit_acc(qva[i], mva[i], xf, acca);
        if (PG) accb = unit_acc(qvb[i], mvb[i], xf, accb);
      }
      const float va = simd_sum(acca);
      const float vb = simd_sum(accb);
      if (lane == 0u) {
        if (PG) {
          float a = float(half(va * gsa)), b = float(half(vb * gsb));  // the gate and up rows as the pair launch stores them (fp16)
          LIMIT_CLAMP
          const float e = exp(-abs(a));                                // mx.sigmoid: 1 / (1 + exp(-|x|)), 1 - that for x < 0
          const float y_ = 1.0f / (1.0f + e);
          const float sg_ = a < 0.0f ? 1.0f - y_ : y_;
          y[ulong(slot) * uint(N) + rbase] = half((a * sg_) * b);
        } else {
          y[ulong(slot) * uint(N) + rbase] = half(va * gsa);
        }
      }
    }
"""

_GROUP = r"""
    const uint s = thread_position_in_threadgroup.x;
    threadgroup uint lead[256];
    const bool valid = s < uint(S);
    const uint ex = valid ? inds[s] : 0xFFFFFFFFu;
    bool isl = valid;
    for (uint j = 0; j < s && j < uint(S); ++j) if (inds[j] == ex) isl = false;
    lead[s] = isl ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint rk = 0u, tot = 0u;
    for (uint j = 0; j < uint(S); ++j) { if (j < s) rk += lead[j]; tot += lead[j]; }
    if (valid && isl) {
      uint c = 0u;
      for (uint j = s; j < uint(S); ++j) if (inds[j] == ex) { tab[rk * uint(W) + 2u + c] = j; ++c; }
      tab[rk * uint(W)] = ex;
      tab[rk * uint(W) + 1u] = c;
    }
    if (valid && s >= tot) tab[s * uint(W) + 1u] = 0u;
"""


_ACT_OLD = "half((a * sg_) * b)"


def act_source(src: str, act):
    """Activation kind of a pair_glu kernel source. act None = the clamped silu-gated form already in the source; act = ("situ", scale_gate, scale_lin):
    half(((scale_gate * tanh(a / scale_gate)) * sigmoid(a)) * (scale_lin * tanh(b / scale_lin))), no clamp (the tanh bounds both); the sigmoid is the one the source computes (sg_)."""
    if act is None:
        return src
    kind, sg, sl = act
    assert kind == "situ" and sg > 0.0 and sl > 0.0 and src.count(_ACT_OLD) == 1, act
    return src.replace(_ACT_OLD, f"half((({float(sg)!r}f * metal::precise::tanh(a / {float(sg)!r}f)) * sg_) * ({float(sl)!r}f * metal::precise::tanh(b / {float(sl)!r}f)))")


@functools.lru_cache(maxsize=None)
def _rows_kern(pg: bool, limit, act=None, scm: int = 0):
    clamp = "" if (limit is None or act) else f"a = min(a, {float(limit)!r}f); b = clamp(b, -{float(limit)!r}f, {float(limit)!r}f);"
    kw = dict(name="kyojin_glm_rows_" + ("pair_glu" if pg else "down") + ("_situ" if act else ""), input_names=["x", "eids", "tab", "qsA", "mtA", "gsA", "qsB", "mtB", "gsB"], output_names=["y"], header=_HDR,
              source=act_source(_ROWS.replace("LIMIT_CLAMP", clamp), act))
    key = ("glm_rows", pg, limit) + ((act,) if act else ())
    if scm:                                              # flag KYOJIN_MOE_SCALE_COMPACT: the scale reads go through the compact helper (flag off: the source above, untouched)
        key, kw["name"] = key + ("scm",), kw["name"] + "_scm"
        kw["input_names"], kw["header"], kw["source"] = MC.scm_names(kw["input_names"]), _HDR + MC.CSC_HDR, MC.scm_source(kw["source"], "inl")
    return KN._kern(key, **kw)


@functools.lru_cache(maxsize=None)
def _group_kern():
    return KN._kern(("glm_group",), name="kyojin_glm_group", input_names=["inds"], output_names=["tab"], header="", source=_GROUP)


def group(inds, T, K):
    """inds uint32 [S = T * K] -> table uint32 [S * (2 + T)] (see the module doc)."""
    S = T * K
    nt = -(-S // 32) * 32
    return _group_kern()(inputs=[inds], template=[("S", S), ("W", 2 + T)], grid=(nt, 1, 1), threadgroup=(nt, 1, 1), output_shapes=[(S * (2 + T),)], output_dtypes=[mx.uint32])[0]


def ok(sm_g, sm_u, sm_d, T, K) -> bool:
    return (T <= T_MAX and T * K <= 256 and sm_g.g2a_ok() and sm_d.g2a_ok() and sm_g.pair_ok(sm_u) and sm_g.n % 4 == 0 and sm_d.k == sm_g.n)


def pair_glu(g, u, x2, inds, tab, T, K, limit, act=None):
    """h half [S, g.n] = clamped silu(gate_e(x[slot // K])) * up_e(x[slot // K]) for slot s of expert inds[s]."""
    from . import foreign_linear as FL
    S = T * K
    SGN = FL.G2A_SGN
    scm = MC.usable(g, u)
    kern = _rows_kern(True, limit, act, scm)
    return kern(inputs=[x2, inds, tab if tab is not None else inds, g.qs, *MC.mt(g, scm), g.gs, u.qs, *MC.mt(u, scm), u.gs],
                template=[("KIN", g.k), ("N", g.n), ("SGN", SGN), ("W", 2 + T), ("SINGLE", int(tab is None)), ("PG", 1), ("XDIV", K)],
                grid=(g.n // SGN * 32 * SGN, S, 1), threadgroup=(32 * SGN, 1, 1), output_shapes=[(S, g.n)], output_dtypes=[mx.float16])[0]


def down(d, h, inds, tab, T, K):
    """y half [S, d.n] = down_e(h[slot])."""
    from . import foreign_linear as FL
    S = T * K
    SGN = FL.G2A_SGN
    scm = MC.usable(d)
    kern = _rows_kern(False, None, None, scm)
    return kern(inputs=[h, inds, tab if tab is not None else inds, d.qs, *MC.mt(d, scm), d.gs, d.qs, *MC.mt(d, scm), d.gs],
                template=[("KIN", d.k), ("N", d.n), ("SGN", SGN), ("W", 2 + T), ("SINGLE", int(tab is None)), ("PG", 0), ("XDIV", 1)],
                grid=(d.n // SGN * 32 * SGN, S, 1), threadgroup=(32 * SGN, 1, 1), output_shapes=[(S, d.n)], output_dtypes=[mx.float16])[0]
