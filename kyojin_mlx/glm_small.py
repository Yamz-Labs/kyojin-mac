# SPDX-License-Identifier: Apache-2.0
# Adapted from oMLX (https://github.com/jundot/omlx), Apache-2.0,
# omlx/patches/mlx_vlm_glm5_next_compat/decode_kernels.py (`kda_decode_step`, `_KDA_SOURCE`): the one-dispatch KDA layer body (one 1024-thread
# threadgroup per head: short conv + SiLU, q / k norms, gate projections, vector-gated delta rule, gated output norm). Re-written here for this
# repository's layout (separate fp8 qkv GEMV and bf16 fbg_a / f_b / g_b, stock rms_norm q / k, B >= 1) and its stock rounding sites.
"""GLM-5.3 linear-attention (KDA) decode body in ONE launch (flag KYOJIN_GLM_FUSE_SMALL, G10b).

Stock decode of a Glm5NextLinearAttention layer (L <= 8 tokens) around its two projection GEMVs is: concat of the conv state, depthwise conv, silu, a
cache slice copy, split, two rms_norm + two scalar multiplies, the fbg_a split, the f_b matmul, sigmoid(b), compute_g_safe, the gated-delta kernel,
the g_b matmul, o_norm, sigmoid, a multiply (plus dtype casts): ~20 launches. Here, after `qkv = qkv_proj(x)` and `fbg = fbg_a_proj(x)` (the stock
modules, untouched), ONE kernel per layer does

    conv (4 taps) + silu + new conv state | f_b / g_b projections of the head's 128 rows | q / k rms norms | g = exp(lb * sigmoid(exp(A_log) * (a + dt_bias))),
    beta = sigmoid(b) | the vector-gated delta rule (stock arithmetic, fp32 state in registers) | o_norm * sigmoid(gate)

and its output feeds `o_proj`. Rounding sites mirror the stock graph (conv -> T, silu -> T, rms_norm -> T, scalar multiply -> T, projection -> T,
o_norm -> norm weight dtype, sigmoid -> T, product -> T). The two small projections (K = 128) are summed per quad in a different order than MLX's gemv
(fp32 accumulate, different association): allclose, not bit equal; every token is computed independently (row invariance).
Not covered (stock path): prefill rows (> MAX_TOKENS), masks, speculative-recording caches, missing conv / recurrent state, non-plain linears."""
from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn

from . import glm_rowfold as RF

ENABLED = os.environ.get("KYOJIN_GLM_FUSE_SMALL", "0") not in ("0", "")
SPEC = os.environ.get("KYOJIN_GLM_FUSE_SMALL_SPEC", "1") not in ("0", "")      # G8d: also serve the verify rows of a speculating cache (under FUSE_SMALL; 0 = the stock path there, G10b as shipped)
MAX_TOKENS = 8
STATS = {"fused": 0, "stock": 0}
_HEADER = "#pragma METAL fp contract(off)\n"

_SRC = r"""
  constexpr int CK = 4;
  constexpr int NROW = CK - 1 + TOK;
  constexpr int NP = 3 * QKV;
  const uint tid = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const int h = int(threadgroup_position_in_grid.x);
  const int bi = int(threadgroup_position_in_grid.y);
  const device T* qkv_b = qkv + size_t(bi) * TOK * NP;
  const device T* fbg_b = fbg + size_t(bi) * TOK * FBG_W;
  const device T* cs_b = conv_state + size_t(bi) * 3 * NP;
  const device float* st_b = state_in + size_t(bi) * HEADS * DK * DK;
  device T* cso_b = conv_state_out + size_t(bi) * 3 * NP;
  device float* sto_b = state_out + size_t(bi) * HEADS * DK * DK;
  device OT* y_b = y + size_t(bi) * TOK * QKV;
  const float eps_qk = consts[0];
  const float norm_eps = consts[1];
  const float lower = consts[2];

  threadgroup T qs[TOK][DK];
  threadgroup T ks[TOK][DK];
  threadgroup T vs[TOK][DK];
  threadgroup T as_[TOK][DK];
  threadgroup T gates[TOK][DK];
  threadgroup T ys[TOK][DK];
  threadgroup float gs[TOK][DK];
  threadgroup T betas[TOK];

  // ---- 1. short conv + silu (stock: conv -> T, silu -> T) and the new conv state ----
  if (tid < uint(3 * DK)) {
    const int part = int(tid) / DK;
    const int i = int(tid) % DK;
    const int gc = part * QKV + h * DK + i;
    T win[NROW];
    for (int r = 0; r < CK - 1; r++) win[r] = cs_b[r * NP + gc];
    for (int t = 0; t < TOK; t++) win[CK - 1 + t] = qkv_b[t * NP + gc];
    const device T* w = conv_w + gc * CK;
    for (int t = 0; t < TOK; t++) {
      float acc = static_cast<float>(w[0]) * static_cast<float>(win[t]);
      acc += static_cast<float>(w[1]) * static_cast<float>(win[t + 1]);
      acc += static_cast<float>(w[2]) * static_cast<float>(win[t + 2]);
      acc += static_cast<float>(w[3]) * static_cast<float>(win[t + 3]);
      const float cv = static_cast<float>(static_cast<T>(acc));
      const float se = 1.0f / (1.0f + metal::precise::exp(metal::abs(cv)));
      const float sgm = static_cast<float>(static_cast<T>((cv < 0.0f) ? se : 1.0f - se));
      const T sv = static_cast<T>(cv * sgm);
      if (part == 0) qs[t][i] = sv; else if (part == 1) ks[t][i] = sv; else vs[t][i] = sv;
    }
    for (int r = 0; r < CK - 1; r++) cso_b[r * NP + gc] = win[TOK + r];
  }

  // ---- 3a. f_b / g_b rows of this head (K = 128), one quad per row: 256 rows x 4 lanes ----
  {
    const int quad = int(tid) / 4;
    const int ql = int(tid) % 4;
    const int which = quad / DK;              // 0: f_b (a), 1: g_b (gate)
    const int i = quad % DK;
    const int row = h * DK + i;
    const device T* wl = (which == 0 ? fb_w : gb_w) + size_t(row) * 128 + ql * 32;
    float wv[32];
    for (int j = 0; j < 32; j++) wv[j] = static_cast<float>(wl[j]);
    for (int t = 0; t < TOK; t++) {
      const device T* xin = fbg_b + t * FBG_W + (which == 0 ? OFF_FA : OFF_GA) + ql * 32;
      float acc = 0.0f;
      for (int j = 0; j < 32; j++) acc += wv[j] * static_cast<float>(xin[j]);
      acc += simd_shuffle_xor(acc, 1);
      acc += simd_shuffle_xor(acc, 2);
      if (ql == 0) {
        if (which == 0) as_[t][i] = static_cast<T>(acc); else gates[t][i] = static_cast<T>(acc);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- 2. rms_norm(q, k) (eps_qk) then the scalar scale: both rounded to T like the stock graph ----
  if (sg < uint(2 * TOK)) {
    const int t = int(sg) / 2;
    const bool is_q = (sg % 2) == 0;
    threadgroup T* row = is_q ? qs[t] : ks[t];
    float x[4];
    float acc = 0.0f;
    for (int e = 0; e < 4; e++) { x[e] = static_cast<float>(row[4 * lane + e]); acc += x[e] * x[e]; }
    acc = simd_sum(acc);
    const float r = metal::precise::rsqrt(acc / 128.0f + eps_qk);
    const float sc = static_cast<float>(static_cast<T>(is_q ? 0.0078125f : 0.08838834764831845f));
    for (int e = 0; e < 4; e++) {
      const float xn = static_cast<float>(static_cast<T>(x[e] * r));
      row[4 * lane + e] = static_cast<T>(sc * xn);
    }
  }

  // ---- 3b. g (compute_g_safe), beta = sigmoid(b) ----
  if (tid < uint(TOK * DK)) {
    const int t = int(tid) / DK;
    const int i = int(tid) % DK;
    const float ea = metal::precise::exp(a_log[h]);
    const float s1 = static_cast<float>(as_[t][i]) + dt_bias[h * DK + i];
    const float s2 = ea * s1;
    const float e = 1.0f / (1.0f + metal::precise::exp(metal::abs(s2)));
    const float s3 = (s2 < 0.0f) ? e : 1.0f - e;
    gs[t][i] = metal::precise::exp(lower * s3);
  }
  if (tid < uint(TOK)) {
    const float bb = static_cast<float>(fbg_b[tid * FBG_W + OFF_B + h]);
    const float be = 1.0f / (1.0f + metal::precise::exp(metal::abs(bb)));
    betas[tid] = static_cast<T>((bb < 0.0f) ? be : 1.0f - be);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- 4. vector-gated delta rule (stock gated_delta_step_vec statements) ----
  for (int j = 0; j < DK / 32; j++) {
    const int dv_idx = int(sg) + 32 * j;
    constexpr int n_per_t = DK / 32;
    const int dk_idx = int(lane);
    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) state[i] = st_b[(size_t(h) * DK + dv_idx) * DK + n_per_t * dk_idx + i];
    for (int t = 0; t < TOK; ++t) {
      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        const int s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] * gs[t][s_idx];
        kv_mem += state[i] * static_cast<float>(ks[t][s_idx]);
      }
      kv_mem = simd_sum(kv_mem);
      const float delta = (static_cast<float>(vs[t][dv_idx]) - kv_mem) * static_cast<float>(betas[t]);
      float out = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        const int s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] + static_cast<float>(ks[t][s_idx]) * delta;
        out += state[i] * static_cast<float>(qs[t][s_idx]);
      }
      out = simd_sum(out);
      if (lane == 0) ys[t][dv_idx] = static_cast<T>(out);
      if (SPEC != 0 && t + 1 < TOK) {      // speculative verify: the post-token state of every row but the last (the cache keeps them for a partial accept)
        for (int i = 0; i < n_per_t; ++i) inter[((((size_t(bi) * (TOK - 1) + size_t(t)) * HEADS + size_t(h)) * DK + size_t(dv_idx)) * DK) + n_per_t * dk_idx + i] = state[i];
      }
    }
    for (int i = 0; i < n_per_t; ++i) sto_b[(size_t(h) * DK + dv_idx) * DK + n_per_t * dk_idx + i] = state[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- 5. o_norm (rms_norm with weight) * sigmoid(gate) ----
  if (sg < uint(TOK)) {
    const int t = int(sg);
    float x[4];
    float acc = 0.0f;
    for (int e = 0; e < 4; e++) { x[e] = static_cast<float>(ys[t][4 * lane + e]); acc += x[e] * x[e]; }
    acc = simd_sum(acc);
    const float r = metal::precise::rsqrt(acc / 128.0f + norm_eps);
    for (int e = 0; e < 4; e++) {
      const int c = 4 * lane + e;
      const float wx = static_cast<float>(static_cast<WT>(static_cast<float>(norm_w[c]) * (x[e] * r)));
      const float gf = static_cast<float>(gates[t][c]);
      const float ge = 1.0f / (1.0f + metal::precise::exp(metal::abs(gf)));
      const float gsg = static_cast<float>(static_cast<T>((gf < 0.0f) ? ge : 1.0f - ge));
      y_b[t * QKV + h * DK + c] = static_cast<OT>(wx * gsg);
    }
  }
"""

_kern = None


def _kernel():
    global _kern
    if _kern is None:
        _kern = mx.fast.metal_kernel(
            name="kyojin_glm_kda_dec",
            input_names=["qkv", "fbg", "conv_w", "conv_state", "state_in", "a_log", "dt_bias", "norm_w", "fb_w", "gb_w", "consts"],
            output_names=["y", "conv_state_out", "state_out", "inter"], source=_SRC, header=_HEADER)
    return _kern


def kda_decode(qkv, fbg, conv_w, conv_state, state, a_log, dt_bias, norm_w, fb_w, gb_w, consts, *, heads, out_dtype, steps=False):
    """qkv [B, T, 3 * H * 128], fbg [B, T, 320] -> y [B, T, H * 128] (out_dtype), new conv state [B, 3, 3HD], new recurrent state [B, H, 128, 128] fp32.
    steps=True (speculative verify, T > 1): a fourth output, the post-token states of rows 0 .. T - 2, [B, T - 1, H, 128, 128] fp32 (what the stock kernel's state_steps returns)."""
    B, T, NP = qkv.shape
    P = heads * 128
    spec = bool(steps) and T > 1
    return _kernel()(
        inputs=[qkv, fbg, conv_w, conv_state, state, a_log, dt_bias, norm_w, fb_w, gb_w, consts],
        template=[("T", qkv.dtype), ("WT", norm_w.dtype), ("OT", out_dtype), ("TOK", T), ("DK", 128), ("QKV", P), ("HEADS", heads),
                  ("FBG_W", fbg.shape[-1]), ("OFF_FA", 0), ("OFF_B", 128), ("OFF_GA", 128 + heads), ("SPEC", 1 if spec else 0)],
        grid=(1024 * heads, B, 1), threadgroup=(1024, 1, 1),
        output_shapes=[(B, T, P), (B, 3, NP), (B, heads, 128, 128), (B, T - 1, heads, 128, 128) if spec else (1,)],
        output_dtypes=[out_dtype, qkv.dtype, mx.float32, mx.float32])[: 4 if spec else 3]


def _plain(m, rows, cols, dtype):
    """A plain dense (rows, cols) weight in the model dtype: nn.Linear, or the row-invariant RowLinear / RowLinearS the served load path swaps in for every nn.Linear (multiuser / qspec)."""
    return (type(m) is nn.Linear or type(m).__name__ in ("RowLinear", "RowLinearS")) and "bias" not in m and m.weight.shape == (rows, cols) and m.weight.dtype == dtype


def _eligible(self, x, mask, cache):
    if not ENABLED or cache is None or mask is not None:
        return False
    if x.ndim != 3 or not 1 <= x.shape[1] <= MAX_TOKENS or x.dtype not in (mx.bfloat16, mx.float16):
        return False
    if self.head_dim != 128 or cache.lengths is not None or cache.left_padding is not None:
        return False
    if getattr(cache, "is_speculating", False):
        if not SPEC or not _spec_room(cache, x.shape[1]):
            return False
    cs, st = cache[0], cache[1]
    P = self.projection_dim
    if cs is None or st is None or cs.shape != (x.shape[0], 3, 3 * P) or cs.dtype != x.dtype or st.shape != (x.shape[0], self.num_heads, 128, 128) or st.dtype != mx.float32:
        return False
    if not (_plain(self.f_b_proj, P, 128, x.dtype) and _plain(self.g_b_proj, P, 128, x.dtype)):
        return False
    cw = self.qkv_conv.conv.weight
    return cw.shape == (3 * P, 4, 1) and cw.dtype == x.dtype and self.o_norm.weight.dtype in (mx.bfloat16, mx.float16, mx.float32)


def _spec_room(cache, T) -> bool:
    """The speculative block can take T more rows in slots 0 (conv window) and 1 (recurrent state), through the cache's own record API."""
    try:
        cap = cache.history_capacity
        return all(cache._recorded_length(i) + T <= cap for i in (0, 1))
    except Exception:      # noqa: a cache class without the record API: stock path
        return False


def _consts(self):
    c = getattr(self, "_g10b_consts", None)
    if c is None:
        c = self._g10b_consts = (mx.array([1e-6 / self.head_dim, self.o_norm.eps, self.lower_bound], dtype=mx.float32),
                                 self.A_log.reshape(-1), self.dt_bias.reshape(-1))
    return c


def _fused_call(self, x, mask=None, cache=None):
    from mlx_vlm.models.linear import linear
    if not _eligible(self, x, mask, cache):
        STATS["stock"] += 1
        return _STOCK(self, x, mask, cache)
    STATS["fused"] += 1
    fold = RF.qkv_fbg(self.qkv_proj, self.fbg_a_proj, x) if RF.ENABLED else None      # N2-ROWDOT (flag KYOJIN_GLM_ROWFOLD): fbg_a row-dot as extra threadgroups of the qkv GEMV launch
    if fold is None:
        if RF.ENABLED:
            RF.STATS["stock"] += 1
        qkv = linear(self.qkv_proj, x)
        fbg = linear(self.fbg_a_proj, x)
    else:
        RF.STATS["fold"] += 1
        qkv, fbg = fold
    consts, a_log, dtb = _consts(self)
    nw = self.o_norm.weight
    # stock: o_norm(output) has the dtype of (output, weight) -> multiplied by the model-dtype sigmoid
    odt = mx.float32 if nw.dtype == mx.float32 else x.dtype
    spec = getattr(cache, "is_speculating", False)
    T = x.shape[1]
    if spec:
        win = mx.concatenate([cache[0], qkv], axis=1)           # the stock conv input: the window after t kept rows is win[:, t:t + 3]
    res = kda_decode(qkv, fbg, self.qkv_conv.conv.weight, cache[0], cache[1], a_log, dtb, nw, self.f_b_proj.weight, self.g_b_proj.weight, consts,
                     heads=self.num_heads, out_dtype=odt, steps=spec)
    y, cs, st = res[:3]
    if spec:
        cache.record_speculative_window(0, win, 3)
        cache.record_speculative_states(1, res[3] if T > 1 else None, st)
    cache[0] = cs
    cache[1] = st
    cache.advance(x.shape[1])
    return linear(self.o_proj, y)


_STOCK = None


def install(model=None, log=print) -> int:
    """Patch Glm5NextLinearAttention.__call__ once per process (the flag is read at call time)."""
    global _STOCK
    try:
        import mlx_vlm.models.glm5_next.language as G
    except Exception:
        return 0
    cls = G.Glm5NextLinearAttention
    if getattr(cls, "_g10b_patched", False):
        return 0
    _STOCK = cls.__call__
    cls.__call__ = _fused_call
    cls._g10b_patched = True
    log("[glm_small] Glm5NextLinearAttention decode body: " + ("ON" if ENABLED else "off (KYOJIN_GLM_FUSE_SMALL)"))
    return 1
