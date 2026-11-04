# SPDX-License-Identifier: Apache-2.0
"""Decode-size MoE block of a Qwen3-Next-style model (router -> routed experts -> weighted combine + shared expert) in a few launches.

mlx-lm's block spends ~30 dispatches per layer on the router (gemv, softmax, argpartition, take, sum, divide, the shared gate and its sigmoid), on the
index arrays, on the silu*up of the routed experts and on the combine (scores multiply, sum over slots, sigmoid-gated shared add). At one row the GPU
time of those small kernels is their launch floor, not their bytes. Here:

    router_logits  one launch   x . gate rows (+ the shared-expert gate row), 8 rows per threadgroup
    router_select  one launch   softmax (f32), top-k by value (lowest index on ties), renormalise, sigmoid of the shared gate
    gate / up      ForeignSwitchGLU's gather GEMV (experts read in their stored format)
    silu_mul       one launch   silu(gate) * up, f32 math, fp16 out (same rounding as the stock path)
    down           the gather GEMV
    combine        one launch   sum_k score_k * down_k + sigmoid(shared gate) * shared, f32 out (the stock block's dtype)

Exactness: the products and the fp16 roundings are the stock ones; the logit dot product and the order of the f32 sum over slots differ (f32 rounding only).
Rows above the gather limit (prefill, big verify) and blocks that do not match (no f32 router, sharded) run the stock `__call__`.
"""
from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn

from . import kernels as KN

def STOCK_F16() -> bool:                                                    # flag moef16 (KYOJIN_MOE_STOCK_F16, default on, read per call):
    return os.environ.get("KYOJIN_MOE_STOCK_F16", "1") not in ("0", "")
# the stock (prefill) MoE block hands an fp16 stream on as fp16
MODE = os.environ.get("KYOJIN_FOREIGN_MOE", "2")       # 0 stock block, 1 fused (f32 out, the stock dtype), 2 fused with fp16 out (saturated) on an fp16 stream: the stream stays fp16 and the casts of the f32 detour go
ON = MODE not in ("0", "")
GLU = os.environ.get("KYOJIN_FOREIGN_MOE_GLU", "1") not in ("0", "")
E_MAX = 1024
LTB = int(os.environ.get("KYOJIN_MOE_LOGITS_TB", "4"))     # tokens per threadgroup row in the prefill router logits

_LOGITS = r"""
    const uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    const uint r = threadgroup_position_in_grid.x * 8u + sg;
    const uint t = threadgroup_position_in_grid.y;
    if (r >= uint(E) + uint(HAS_S)) return;
    device const float4* wp = (device const float4*)((r < uint(E)) ? (gw + r * uint(H)) : sgw);
    device const vec<XT, 4>* xp = (device const vec<XT, 4>*)(x + t * uint(H));
    float4 s0 = 0.0f, s1 = 0.0f;
    for (uint i = lane; i < uint(H) / 4u; i += 64u) {
      s0 += wp[i] * float4(xp[i]);
      s1 += wp[i + 32u] * float4(xp[i + 32u]);
    }
    float4 s4 = s0 + s1;
    float s = simd_sum(s4[0] + s4[1] + s4[2] + s4[3]);
    if (lane == 0u) lg[t * (uint(E) + 1u) + r] = s;
"""

_LOGITS_TB = r"""
    const uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    const uint r = threadgroup_position_in_grid.x * 8u + sg;
    const uint t0 = threadgroup_position_in_grid.y * uint(TB);
    if (r >= uint(E) + uint(HAS_S)) return;
    device const float4* wp = (device const float4*)((r < uint(E)) ? (gw + r * uint(H)) : sgw);
    float4 s0[TB], s1[TB];
    for (uint j = 0; j < uint(TB); ++j) { s0[j] = 0.0f; s1[j] = 0.0f; }
    for (uint i = lane; i < uint(H) / 4u; i += 64u) {
      const float4 w0 = wp[i], w1 = wp[i + 32u];
      for (uint j = 0; j < uint(TB); ++j) {
        const uint t = min(t0 + j, uint(T) - 1u);
        device const vec<XT, 4>* xp = (device const vec<XT, 4>*)(x + t * uint(H));
        s0[j] += w0 * float4(xp[i]);
        s1[j] += w1 * float4(xp[i + 32u]);
      }
    }
    for (uint j = 0; j < uint(TB); ++j) {
      float4 s4 = s0[j] + s1[j];
      float s = simd_sum(s4[0] + s4[1] + s4[2] + s4[3]);
      if (lane == 0u && t0 + j < uint(T)) lg[(t0 + j) * (uint(E) + 1u) + r] = s;
    }
"""

_SELECT = r"""
    const uint t = threadgroup_position_in_grid.x, lane = thread_index_in_simdgroup;
    device const float* L = lg + t * (uint(E) + 1u);
    float p[EL];
    float m = -INFINITY;
    for (uint j = 0; j < EL; ++j) { const uint i = lane + 32u * j; p[j] = i < uint(E) ? L[i] : -INFINITY; m = max(m, p[j]); }
    m = simd_max(m);
    float sum = 0.0f;
    for (uint j = 0; j < EL; ++j) { p[j] = metal::precise::exp(p[j] - m); sum += p[j]; }
    const float inv = 1.0f / simd_sum(sum);
    for (uint j = 0; j < EL; ++j) p[j] *= inv;
    float top[K];
    for (uint k = 0; k < uint(K); ++k) {
      float best = -1.0f; uint bi = 0xFFFFFFFFu;
      for (uint j = 0; j < EL; ++j) if (p[j] > best) { best = p[j]; bi = lane + 32u * j; }
      const float gv = simd_max(best);
      const uint gi = simd_min(best == gv ? bi : 0xFFFFFFFFu);
      top[k] = gv;
      for (uint j = 0; j < EL; ++j) if (lane + 32u * j == gi) p[j] = -1.0f;
      if (lane == 0u) inds[t * uint(K) + k] = gi;
    }
    if (lane == 0u) {
      float ts = 0.0f;
      for (uint k = 0; k < uint(K); ++k) ts += top[k];
      for (uint k = 0; k < uint(K); ++k) scs[t * uint(K) + k] = NORM ? top[k] / ts : top[k];
      sgs[t] = 1.0f / (1.0f + metal::precise::exp(-L[E]));
    }
"""

_SILU = r"""
    const uint i = thread_position_in_grid.x;
    float av = float(a[i]);
    h[i] = half(av * (1.0f / (1.0f + metal::precise::exp(-av))) * float(b[i]));
"""

_COMBINE = r"""
    const uint g = thread_position_in_grid.x;
    const uint t = g / uint(H), c = g - t * uint(H);
    float acc = 0.0f;
    for (uint k = 0; k < uint(K); ++k) acc += float(y[(t * uint(K) + k) * uint(H) + c]) * scs[t * uint(K) + k];
    if (HAS_S) acc += sgs[t] * float(sh[t * uint(H) + c]);
    o[g] = OT(OT_SAT ? clamp(acc, -65504.0f, 65504.0f) : acc);
"""


def _logits(x2, gw, sgw, T, E, H, has_s):
    if T >= 64 and LTB > 1:                                  # prefill rows: LTB tokens share each weight-row read
        k = KN._kern(("mf_logits_tb",), name="kyojin_moe_logits_tb", input_names=["x", "gw", "sgw"], output_names=["lg"], header="", source=_LOGITS_TB)
        return k(inputs=[x2, gw, sgw], template=[("H", H), ("E", E), ("HAS_S", int(has_s)), ("XT", x2.dtype), ("TB", LTB), ("T", T)], grid=(((E + 1 + 7) // 8) * 256, (T + LTB - 1) // LTB, 1),
                 threadgroup=(256, 1, 1), output_shapes=[(T, E + 1)], output_dtypes=[mx.float32])[0]
    k = KN._kern(("mf_logits",), name="kyojin_moe_logits", input_names=["x", "gw", "sgw"], output_names=["lg"], header="", source=_LOGITS)
    return k(inputs=[x2, gw, sgw], template=[("H", H), ("E", E), ("HAS_S", int(has_s)), ("XT", x2.dtype)], grid=(((E + 1 + 7) // 8) * 256, T, 1), threadgroup=(256, 1, 1),
             output_shapes=[(T, E + 1)], output_dtypes=[mx.float32])[0]


def _select(lg, T, E, K, norm):
    k = KN._kern(("mf_select",), name="kyojin_moe_select", input_names=["lg"], output_names=["inds", "scs", "sgs"], header="", source=_SELECT)
    return k(inputs=[lg], template=[("E", E), ("EL", (E + 31) // 32), ("K", K), ("NORM", bool(norm))], grid=(T * 32, 1, 1), threadgroup=(32, 1, 1),
             output_shapes=[(T * K,), (T * K,), (T,)], output_dtypes=[mx.uint32, mx.float32, mx.float32])


def _silu_mul(a, b):
    k = KN._kern(("mf_silu",), name="kyojin_moe_silu_mul", input_names=["a", "b"], output_names=["h"], header="", source=_SILU)
    return k(inputs=[a, b], grid=(a.size, 1, 1), threadgroup=(256, 1, 1), output_shapes=[a.shape], output_dtypes=[mx.float16])[0]


def _combine(y, scs, sgs, sh, T, K, H, has_s, ot=mx.float32):
    k = KN._kern(("mf_combine",), name="kyojin_moe_combine", input_names=["y", "scs", "sgs", "sh"], output_names=["o"], header="", source=_COMBINE)
    return k(inputs=[y, scs, sgs, sh], template=[("H", H), ("K", K), ("HAS_S", bool(has_s)), ("OT", ot), ("OT_SAT", ot == mx.float16)], grid=(T * H, 1, 1), threadgroup=(256, 1, 1),
             output_shapes=[(T, H)], output_dtypes=[ot])[0]


_IDX: dict = {}


def _slot_rows(S, K):
    """Constant device arrays for S slots of K per token: (row of x of each slot, slot ids)."""
    v = _IDX.get((S, K))
    if v is None:
        import numpy as np
        v = _IDX[(S, K)] = (mx.array((np.arange(S) // K).astype(np.uint32)), mx.array(np.arange(S).astype(np.uint32)))
    return v


def _eligible(blk, x) -> bool:
    from . import foreign_linear as FL
    sm = blk.switch_mlp
    if getattr(blk, "sharding_group", None) is not None:
        return False
    T = 1
    for d in x.shape[:-1]:
        T *= d
    K, E, H = blk.top_k, blk.num_experts, x.shape[-1]
    w = blk.gate.weight
    return (T * K <= FL.GATHER_MAX and w.dtype == mx.float32 and E <= E_MAX and K <= 16 and H % 256 == 0 and x.dtype in (mx.float16, mx.float32) and w.shape == (E, H)
            and all(p.fmt != "dense" and p.gemv_ok() for p in (sm.g, sm.u, sm.d)))


class Pending:
    """A fused block's combine that has not run: decfuse.norm_of folds it into the add + RMSNorm that follows; value() runs it alone."""

    def __init__(self, y, scs, sgs, sh, T, K, H, lead):
        self.y, self.scs, self.sgs, self.sh, self.T, self.K, self.H, self.lead = y, scs, sgs, sh, T, K, H, lead

    def value(self):
        return _combine(self.y, self.scs, self.sgs, self.sh, self.T, self.K, self.H, True, mx.float16).reshape(*self.lead, self.H)


def _deferred_call(self, x):
    """_fused_call, but the combine of a block with a shared expert on an fp16 stream is returned as a Pending (a plain array in every other case)."""
    return _fused_call(self, x, defer=True)


def PF_ROUTER() -> bool:                                                    # flag moepf3 (KYOJIN_MOE_PF_ROUTER, default on, read per call)
    return os.environ.get("KYOJIN_MOE_PF_ROUTER", "1") not in ("0", "")


def _pf_eligible(blk, x) -> bool:
    """Prefill rows of a block the decode kernels can route (f32 gate, shared expert, fp16 stream) whose experts run through the grouped GEMM."""
    from . import foreign_linear as FL, foreign_kqgmm as KQ
    sm = blk.switch_mlp
    if MODE != "2" or x.dtype != mx.float16 or getattr(blk, "sharding_group", None) is not None or getattr(blk, "shared_expert", None) is None:
        return False
    T = 1
    for d in x.shape[:-1]:
        T *= d
    K, E, H = blk.top_k, blk.num_experts, x.shape[-1]
    w = blk.gate.weight
    return (T * K > FL.GATHER_MAX and KQ.on() and w.dtype == mx.float32 and E <= E_MAX and K <= 16 and H % 256 == 0 and w.shape == (E, H)
            and all(KQ.ok(p) for p in (sm.g, sm.u, sm.d)))


def _pf_call(self, x):
    """Prefill block: router logits + softmax/top-k/renormalise in two launches, grouped-GEMM experts, one combine (scores, sigmoid-gated shared expert, saturated fp16 out)."""
    lead, H = x.shape[:-1], x.shape[-1]
    K, E = self.top_k, self.num_experts
    x2 = x.reshape(-1, H)
    T = x2.shape[0]
    lg = _logits(x2, self.gate.weight, self.shared_expert_gate.weight.reshape(-1), T, E, H, True)
    inds, scs, sgs = _select(lg, T, E, K, self.norm_topk_prob)
    y = self.switch_mlp(x, inds.reshape(*lead, K))
    sh = self.shared_expert(x).reshape(T, H)
    return _combine(y.reshape(T * K, H), scs, sgs, sh, T, K, H, True, mx.float16).reshape(*lead, H)


def _fused_call(self, x, defer=False):
    if PF_ROUTER() and _pf_eligible(self, x):
        return _pf_call(self, x)
    if not _eligible(self, x):
        r = self._stock_call(x)
        if MODE == "2" and x.dtype == mx.float16 and r.dtype == mx.float32 and STOCK_F16():       # prefill rows: the stock block returns f32 (f32 router scores, f32 shared gate); an f32 stream there skips the fused GDN prefill block and runs every norm / attention / projection input in f32
            r = mx.clip(r, -65504.0, 65504.0).astype(mx.float16)                                   # same saturating cast as the fused combine
        return r
    sm = self.switch_mlp
    lead, H = x.shape[:-1], x.shape[-1]
    K, E = self.top_k, self.num_experts
    x2 = x.reshape(-1, H)
    T = x2.shape[0]
    has_s = getattr(self, "shared_expert", None) is not None
    gw = self.gate.weight
    sgw = self.shared_expert_gate.weight.reshape(-1) if has_s else gw
    lg = _logits(x2, gw, sgw, T, E, H, has_s)
    inds, scs, sgs = _select(lg, T, E, K, self.norm_topk_prob)
    x16 = x2 if x2.dtype == mx.float16 else x2.astype(mx.float16)
    tok, ar = _slot_rows(T * K, K)
    a, b = sm.g.gather_pair(sm.u, x16, tok, inds) if sm.g.pair_ok(sm.u) else (sm.g.gather(x16, tok, inds), sm.u.gather(x16, tok, inds))
    if GLU and sm.d.glu_ok():
        y = sm.d.gather_glu(a, b, ar, inds)                 # silu(a) * b folded into the down projection's input staging
    else:
        y = sm.d.gather(_silu_mul(a, b), ar, inds)
    sh = self.shared_expert(x).reshape(T, H) if has_s else y
    ot = mx.float16 if MODE == "2" and x.dtype == mx.float16 else mx.float32
    if defer and has_s and ot == mx.float16 and sh.dtype == mx.float16 and y.dtype == mx.float16:
        return Pending(y, scs, sgs, sh, T, K, H, lead)
    return _combine(y, scs, sgs, sh, T, K, H, has_s, ot).reshape(*lead, H)


def install(model, log=print) -> int:
    """Swap the call of every MoE block that has a ForeignSwitchGLU. Returns the number of blocks."""
    if not ON:
        return 0
    from . import foreign_linear as FL
    n = 0
    for _, mod in model.named_modules():
        if isinstance(getattr(mod, "switch_mlp", None), FL.ForeignSwitchGLU) and hasattr(mod, "gate") and hasattr(mod, "top_k"):
            cls = mod.__class__
            if getattr(cls, "_foreign_moe", False):
                continue
            mod.__class__ = type("Fused" + cls.__name__, (cls,), {"__call__": _fused_call, "_deferred_call": _deferred_call, "_stock_call": cls.__call__, "_foreign_moe": True})
            n += 1
    if n:
        log(f"[foreign] foreign_moe: {n} MoE blocks run router / experts / combine in fused launches")
    return n
