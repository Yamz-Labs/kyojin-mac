# SPDX-License-Identifier: Apache-2.0
"""Generic MoE router selection in ONE launch (flag KYOJIN_MOE_ROUTE_SEL, default off, read per call): logits -> gating -> selection bias -> optional group limit -> top-k ->
renormalise -> scale, for 1 .. SEL_MAX_ROWS rows.

The stock path (foreign_moe_generic.GenericMoE.route) spends ~6 launches after the logits: gating (sigmoid / softmax / sqrt-softplus), + selection bias, argsort (a block sort),
take_along_axis, sum, divide, scale. The reference design is llama.cpp's `kernel_topk_moe_f32` (ggml-metal argsort.metal: one simdgroup per token row, expert scores in
registers, k rounds of simd arg-max, norm + scale in the same launch); this kernel keeps that layout and adds what the Kyojin MoeSpec describes per op class:

    gating        softmax (precise) | sigmoid | sqrt_softplus | none | softmax_weight (logits as scores, softmax over the K chosen weights)
    router bias   added to the logits before the gating (f32 [E]), optional temperature (logits / temperature)
    select bias   added to the score for the choice only; the weights use the plain score
    groups        n_group > 1: the experts of a group are E / n_group wide, a group scores the sum of its 2 best selection scores, the `group_used` best groups stay
    norm          off | sum + eps (eps > 0) | max(sum, F16_MIN_NORM) (eps = 0, llama.cpp's clamp)
    scale         weights * scale

The logits are an INPUT (f32 [T, E], the stock matmul / rowdot result): they are bit-identical to the stock path's, so indices can be compared exactly with it.
Tie rule: value descending, LOWEST expert index first (= the stable argsort(-score) of the stock path, = llama.cpp's `wt > best || (wt == best && expert < best_expert)`).
Group ties: lowest group index first. A NaN score counts as -inf (never wins before a finite score; the stock argsort order of NaN is unspecified). When fewer than K scores are
finite (heavy group masking) the remaining slots take the -inf entries in index order, like the stable argsort. The weights follow the stock arithmetic in f32; the reductions
(softmax sum, weight sum) use simd / slot order, so weights agree with the stock path to a few fp32 ulps, not bit for bit.

Eligible: Metal, 1 <= T <= SEL_MAX_ROWS, E <= 1024, K <= 32, K <= E, n_group <= 32 with E % n_group == 0 and group_used <= n_group. Anything else runs the stock composition.
`route_ref` is that composition (the reference of the unit test)."""
from __future__ import annotations

import os

import mlx.core as mx

SEL_MAX_ROWS = 16
GATING_ID = {"softmax": 0, "sigmoid": 1, "sqrt_softplus": 2, "none": 3, "softmax_weight": 4}
F16_MIN_NORM = 6.103515625e-5

_SRC = r"""
    const uint t = threadgroup_position_in_grid.x;
    const uint lane = thread_index_in_simdgroup;
    device const float* L = lg + t * uint(E);
    const float inf = INFINITY;
    float zs[EL], pr[EL], sl[EL];
    threadgroup float tsel[(NG > 1) ? E : 1];
    for (uint j = 0; j < EL; ++j) {
      const uint i = lane + 32u * j;
      float z = -inf;
      if (i < uint(E)) {
        z = L[i];
        if (HAS_RB) { z = z + rb[i]; }
        if (HAS_TEMP) { z = z / @TEMP@; }
      }
      zs[j] = z;
    }
    if (GATING == 0) {                                                             // softmax (precise)
      float m = -inf;
      for (uint j = 0; j < EL; ++j) { m = metal::max(m, zs[j]); }
      m = simd_max(m);
      float s = 0.0f;
      for (uint j = 0; j < EL; ++j) {
        const uint i = lane + 32u * j;
        const float e = (i < uint(E)) ? metal::exp(zs[j] - m) : 0.0f;
        pr[j] = e; s += e;
      }
      s = simd_sum(s);
      for (uint j = 0; j < EL; ++j) { pr[j] = pr[j] / s; }
    } else {
      for (uint j = 0; j < EL; ++j) {
        const float z = zs[j];
        float p = z;
        if (GATING == 1) {                                                         // mx.sigmoid
          const float y = 1.0f / (1.0f + metal::exp(-metal::abs(z)));
          p = z < 0.0f ? 1.0f - y : y;
        } else if (GATING == 2) {                                                  // sqrt(softplus(z)), softplus = logaddexp(z, 0)
          const float mx_ = metal::max(z, 0.0f), mn_ = metal::min(z, 0.0f);
          const float x = metal::exp(mn_ - mx_);
          const float u = 1.0f + x;
          const float lp = (u == 1.0f) ? x : metal::log(u) * (x / (u - 1.0f));      // log1p(x)
          p = metal::sqrt((mn_ == -inf || mx_ == inf) ? mx_ : (mx_ + lp));
        }
        pr[j] = p;
      }
    }
    for (uint j = 0; j < EL; ++j) {
      const uint i = lane + 32u * j;
      float c = (i < uint(E)) ? pr[j] : -inf;
      if (HAS_SB && i < uint(E)) { c = c + sb[i]; }
      sl[j] = (c == c) ? c : -inf;                                                 // NaN counts as -inf
    }
    if (NG > 1) {                                                                  // group limit: group score = sum of the 2 best, keep the GU best groups
      for (uint j = 0; j < EL; ++j) { const uint i = lane + 32u * j; if (i < uint(E)) { tsel[i] = sl[j]; } }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      const uint gs = uint(E) / uint(NG);
      float gsc = -inf;
      if (lane < uint(NG)) {
        float a = -inf, b = -inf;
        for (uint e = 0; e < gs; ++e) {
          const float v = tsel[lane * gs + e];
          if (v > a) { b = a; a = v; } else if (v > b) { b = v; }
        }
        gsc = a + b;
        gsc = (gsc == gsc) ? gsc : -inf;
      }
      float keepf = 0.0f;
      uint rank = 0u;
      for (uint h = 0; h < uint(NG); ++h) {
        const float o = simd_shuffle(gsc, ushort(h));
        rank += (o > gsc || (o == gsc && h < lane)) ? 1u : 0u;
      }
      keepf = (lane < uint(NG) && rank < uint(GU)) ? 1.0f : 0.0f;
      for (uint j = 0; j < EL; ++j) {
        const uint i = lane + 32u * j;
        const float kg = simd_shuffle(keepf, ushort(i < uint(E) ? i / gs : 0u));
        if (kg == 0.0f) { sl[j] = -inf; }
      }
    }
    float top_p[K];
    uint top_i[K];
    uint taken = 0u;
    for (uint k = 0; k < uint(K); ++k) {
      float bv = -inf;
      uint bi = 0xFFFFFFFFu;
      for (uint j = 0; j < EL; ++j) {
        const uint i = lane + 32u * j;
        if (i < uint(E) && ((taken >> j) & 1u) == 0u && (bi == 0xFFFFFFFFu || sl[j] > bv)) { bv = sl[j]; bi = i; }
      }
      const float gv = simd_max(bv);
      const uint gi = simd_min(bv == gv ? bi : 0xFFFFFFFFu);                       // lowest index among equal values
      float p = 0.0f;
      for (uint j = 0; j < EL; ++j) {
        if (lane + 32u * j == gi) { p = pr[j]; taken |= (1u << j); }
      }
      top_p[k] = simd_sum(p);
      top_i[k] = gi;
    }
    if (lane == 0u) {
      if (GATING == 4) {                                                           // softmax over the chosen weights
        float m = -inf;
        for (uint k = 0; k < uint(K); ++k) { m = metal::max(m, top_p[k]); }
        float s = 0.0f;
        for (uint k = 0; k < uint(K); ++k) { top_p[k] = metal::exp(top_p[k] - m); s += top_p[k]; }
        for (uint k = 0; k < uint(K); ++k) { top_p[k] = top_p[k] / s; }
      }
      float tot = 0.0f;
      for (uint k = 0; k < uint(K); ++k) { tot += top_p[k]; }                      // slot order
      if (NORM) { tot = (@EPS@ > 0.0f) ? (tot + @EPS@) : metal::max(tot, @FMIN@); }
      for (uint k = 0; k < uint(K); ++k) {
        float w = top_p[k];
        if (NORM) { w = w / tot; }
        if (HAS_SCALE) { w = w * @SCALE@; }
        inds[t * uint(K) + k] = top_i[k];
        wts[t * uint(K) + k] = w;
      }
    }
"""

_KERN: dict = {}


def _f(v: float) -> str:
    return repr(float(v)) + "f"


def _tag(v: float) -> str:
    return repr(float(v)).replace(".", "p").replace("-", "m").replace("+", "")


def eligible(T: int, E: int, K: int, n_group: int = 1, group_used: int = 1, gating: str = "sigmoid") -> bool:
    if gating not in GATING_ID or not (1 <= T <= SEL_MAX_ROWS) or not (1 <= K <= min(E, 32)) or E > 1024 or E < 1:
        return False
    if n_group > 1 and (n_group > 32 or E % n_group != 0 or not (1 <= group_used <= n_group) or K > (E // n_group) * group_used):
        return False
    return mx.default_device() == mx.gpu


def _kern(E, K, gating, norm, eps, scale, n_group, group_used, has_rb, has_sb, temp):
    scalek = scale not in (0.0, 1.0)
    key = (E, K, gating, bool(norm), float(eps), float(scale) if scalek else 1.0, n_group if n_group > 1 else 1, group_used if n_group > 1 else 1, bool(has_rb), bool(has_sb), float(temp))
    k = _KERN.get(key)
    if k is None:
        src = (_SRC.replace("@FMIN@", _f(F16_MIN_NORM)).replace("@EPS@", _f(eps)).replace("@SCALE@", _f(scale if scalek else 1.0)).replace("@TEMP@", _f(temp)))
        name = "kyojin_moe_sel_" + "_".join(str(x) for x in (E, K, GATING_ID[gating], int(bool(norm)), _tag(eps), _tag(scale if scalek else 1.0), key[6], key[7], int(has_rb), int(has_sb), _tag(temp)))
        k = _KERN[key] = mx.fast.metal_kernel(name=name, input_names=["lg", "rb", "sb"], output_names=["inds", "wts"], header="", source=src)
    return k, scalek


_ZB: dict = {}


def _zeros(E: int):
    z = _ZB.get(E)
    if z is None:
        z = _ZB[E] = mx.zeros((E,), dtype=mx.float32)
        mx.eval(z)
    return z


def select(lg, router_bias, sel_bias, K, gating="sigmoid", norm=True, eps=0.0, scale=1.0, n_group=1, group_used=1, temperature=1.0):
    """lg f32 [T, E] router logits (before the router bias / temperature), router_bias / sel_bias f32 [E] or None -> (inds uint32 [T, K], weights f32 [T, K])."""
    T, E = lg.shape
    norm = bool(norm)
    k, scalek = _kern(E, K, gating, norm, eps, scale, n_group, group_used, router_bias is not None, sel_bias is not None, temperature)
    rb = router_bias if router_bias is not None else _zeros(E)
    sb = sel_bias if sel_bias is not None else _zeros(E)
    inds, wts = k(inputs=[lg, rb, sb], template=[("E", E), ("K", K), ("EL", -(-E // 32)), ("GATING", GATING_ID[gating]), ("NORM", norm), ("HAS_RB", router_bias is not None),
                                                 ("HAS_SB", sel_bias is not None), ("NG", n_group if n_group > 1 else 1), ("GU", group_used if n_group > 1 else 1),
                                                 ("HAS_TEMP", temperature != 1.0), ("HAS_SCALE", scalek)],
                  grid=(T * 32, 1, 1), threadgroup=(32, 1, 1), output_shapes=[(T, K), (T, K)], output_dtypes=[mx.uint32, mx.float32])
    return inds, wts


def route_ref(lg, router_bias, sel_bias, K, gating="sigmoid", norm=True, eps=0.0, scale=1.0, n_group=1, group_used=1, temperature=1.0):
    """The stock composition (what GenericMoE.route runs after the logits), same arguments as `select`."""
    import mlx.nn as nn
    if router_bias is not None:
        lg = lg + router_bias
    if temperature != 1.0:
        lg = lg / temperature
    if gating == "softmax":
        probs = mx.softmax(lg, axis=-1, precise=True)
    elif gating == "sigmoid":
        probs = mx.sigmoid(lg)
    elif gating in ("softmax_weight", "none"):
        probs = lg
    else:
        probs = mx.sqrt(nn.softplus(lg))
    sel = probs + sel_bias if sel_bias is not None else probs
    T, E = sel.shape
    if n_group > 1:
        grp = sel.reshape(T, n_group, E // n_group)
        gscore = mx.sort(grp, axis=-1)[..., -2:].sum(-1)
        keep = mx.argsort(-gscore, axis=-1)[:, :group_used]
        mask = mx.put_along_axis(mx.zeros((T, n_group), dtype=mx.bool_), keep, mx.ones(keep.shape, dtype=mx.bool_), axis=-1)
        sel = mx.where(mask[..., None], grp, -mx.inf).reshape(T, E)
    inds = mx.argsort(-sel, axis=-1)[:, :K]
    w = mx.take_along_axis(probs, inds, axis=-1)
    if gating == "softmax_weight":
        w = mx.softmax(w, axis=-1, precise=True)
    if norm:
        tot = w.sum(-1, keepdims=True)
        tot = tot + eps if eps > 0 else mx.maximum(tot, F16_MIN_NORM)
        w = w / tot
    if scale not in (0.0, 1.0):
        w = w * scale
    return inds.astype(mx.uint32), w
