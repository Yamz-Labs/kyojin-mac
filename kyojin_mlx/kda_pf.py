# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 KDA (Glm5NextLinearAttention) prompt path, flag glm_kda_pf (KYOJIN_GLM_KDA_PF, default 0): prompt chunks (> 64 rows, batch 1, plain cache) only.

Stock (G0, 8K rows, 34 layers): qkv_conv 0.35 s (concatenate with the conv state, 24576-channel conv1d, silu, cache slice: 4-5 passes over a 400 MB tensor), q / k rms_norm + scale, the fp32
[T, H, 128] gate tensor of compute_g_safe, then the sequential gated-delta kernel, then o_norm * sigmoid(gate). Here
  prework    ONE kernel: causal 4-tap conv + silu + L2 norm of q / k (+ the q scale), q / k / v written as [1, T, H, 128]; the next conv state is a 3-row slice.
  recurrence kyojin_mlx.kda_nax (chunked NAX scan, gate evaluated in the kernel) - KYOJIN_GLM_KDA_PF_REC=nax (default) | stock (mlx_vlm's kernel, fp32 gate tensor).
  norm-gate  ONE kernel: o_norm (rms_norm, rounded like the stock op) * sigmoid(gate).
The prework and norm-gate kernels follow oMLX (Apache-2.0, https://github.com/jundot/omlx), omlx/patches/glm53_kda_prework.py (one simdgroup per row and head, four channels per lane, the stock rounding
sites mirrored); written again here with a row-blocked sliding window over the conv taps. Anything unexpected (mask, speculation, left padding, odd shapes) runs the stock path."""
from __future__ import annotations

import os

import mlx.core as mx

from . import fired as FD

ENABLED = os.environ.get("KYOJIN_GLM_KDA_PF", "0") != "0"
REC = os.environ.get("KYOJIN_GLM_KDA_PF_REC", "nax")                      # nax | stock
PRE = os.environ.get("KYOJIN_GLM_KDA_PF_PRE", "1") != "0"                 # prework + norm-gate kernels (0 = stock conv chain)
RB = int(os.environ.get("KYOJIN_GLM_KDA_PF_RB", "16"))                    # rows per simdgroup of the prework kernel (conv window reused)
MIN_ROWS = 64
STATS = {"calls": 0, "fallback": 0}

_PRE_SRC = r"""
    const uint lane = thread_position_in_threadgroup.x;
    const uint rb = threadgroup_position_in_grid.y;
    const uint lh = threadgroup_position_in_grid.z;                        // 0..H-1 q heads, H..2H-1 k heads, 2H..3H-1 v heads
    const uint S = uint(s_len);
    const bool is_q = lh < uint(H);
    const bool is_k = (lh >= uint(H)) && (lh < 2u * uint(H));
    const uint head = is_q ? lh : (is_k ? lh - uint(H) : lh - 2u * uint(H));
    const uint ch = lh * uint(D) + lane * 4;                               // first of this lane's 4 channels, in [0, 3 H D)
    float w[4][4];
    for (uint i = 0; i < 4; ++i) { for (uint t = 0; t < 4; ++t) { w[i][t] = float(conv_w[(ch + i) * 4 + t]); } }
    const uint r0 = rb * uint(RB);
    const uint r1 = min(r0 + uint(RB), S);
    T win[3][4];                                                           // inputs of rows r-3, r-2, r-1
    for (uint j = 0; j < 3; ++j) {
        const int rr = int(r0) - 3 + int(j);
        for (uint i = 0; i < 4; ++i) {
            win[j][i] = (rr < 0) ? conv_state[uint(3 + rr) * uint(C) + ch + i] : qkv[uint(rr) * uint(RS) + ch + i];
        }
    }
    for (uint r = r0; r < r1; ++r) {
        T cur[4];
        for (uint i = 0; i < 4; ++i) { cur[i] = qkv[r * uint(RS) + ch + i]; }
        T act[4];
        float sumsq = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            const float acc = float(win[0][i]) * w[i][0] + float(win[1][i]) * w[i][1] + float(win[2][i]) * w[i][2] + float(cur[i]) * w[i][3];
            const T conv = T(acc);
            auto sy = 1 / (1 + metal::precise::exp(metal::abs(conv)));      // MLX Sigmoid, in the array type T (every intermediate rounded like the compiled stock silu)
            const T sg = (conv < T(0)) ? sy : 1 - sy;
            act[i] = conv * sg;
            const float f = float(act[i]);
            sumsq += f * f;
        }
        for (uint i = 0; i < 4; ++i) { win[0][i] = win[1][i]; win[1][i] = win[2][i]; win[2][i] = cur[i]; }
        const uint ob = (r * uint(H) + head) * uint(D) + lane * 4;
        if (is_q || is_k) {
            const float tv = simd_sum(sumsq);                                    // as the stock rms_norm kernel
            const float inv = metal::precise::rsqrt(tv / float(D) + consts[2]);   // the stock rms_norm (eps = 1e-6 / D): T(x * inv), then * scale in T
            for (uint i = 0; i < 4; ++i) {
                const T r = T(float(act[i]) * inv);
                if (is_q) { q_out[ob + i] = T(float(r) * consts[0]); } else { k_out[ob + i] = T(float(r) * consts[1]); }
            }
        } else {
            for (uint i = 0; i < 4; ++i) { v_out[ob + i] = act[i]; }
        }
    }
"""

_NG_SRC = r"""
    const uint lane = thread_position_in_threadgroup.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint head = threadgroup_position_in_grid.z;
    const uint base = (row * uint(H) + head) * uint(D) + lane * 4;
    float xs[4];
    float sumsq = 0.0f;
    for (uint i = 0; i < 4; ++i) { xs[i] = float(y[base + i]); sumsq += xs[i] * xs[i]; }
    const float tv = simd_sum(sumsq);
    const float inv = metal::precise::rsqrt(tv / float(D) + epsv[0]);
    for (uint i = 0; i < 4; ++i) {
        const T r = T(xs[i] * inv);                                            // the stock o_norm: w * T(x * inv), both rounded
        const T nrm = T(float(norm_w[lane * 4 + i]) * float(r));
        const T zv = z[base + i];
        auto sy = 1 / (1 + metal::precise::exp(metal::abs(zv)));              // mx.sigmoid in T
        const T sg = (zv < T(0)) ? sy : 1 - sy;
        out[base + i] = nrm * sg;
    }
"""

_K = {}


def _kernels():
    if not _K:
        _K["pre"] = mx.fast.metal_kernel(name="kyojin_glm_kda_pre", input_names=["qkv", "conv_state", "conv_w", "consts", "s_len"],
                                         output_names=["q_out", "k_out", "v_out"], source=_PRE_SRC)
        _K["ng"] = mx.fast.metal_kernel(name="kyojin_glm_kda_normgate", input_names=["y", "z", "norm_w", "epsv"], output_names=["out"], source=_NG_SRC)
    return _K


def _consts(dtype, dim):
    """[scale^2, scale, eps] as the stock path sees them: python float scalars times a T array are rounded to T first; eps stays a float32 kernel argument"""
    sc = dim ** -0.5
    return mx.concatenate([mx.array([sc ** 2, sc], dtype=dtype).astype(mx.float32), mx.array([1e-6 / dim], dtype=mx.float32)])


def prework(qkv, conv_state, conv_w, heads, dim):
    """qkv [1, S, 3 H D] (the qkv_proj output), conv_state [1, 3, 3 H D], conv_w [3 H D, 4, 1] -> q, k, v [1, S, H, D] and the next conv state [1, 3, 3 H D]."""
    S, C = qkv.shape[1], qkv.shape[2]
    q, k, v = _kernels()["pre"](
        inputs=[qkv, conv_state, conv_w, _consts(qkv.dtype, dim), mx.array(S, dtype=mx.int32)],
        template=[("T", qkv.dtype), ("H", heads), ("D", dim), ("C", C), ("RS", C), ("RB", RB)],
        grid=(32, (S + RB - 1) // RB, 3 * heads), threadgroup=(32, 1, 1),
        output_shapes=[(1, S, heads, dim)] * 3, output_dtypes=[qkv.dtype] * 3)
    return q, k, v, mx.contiguous(qkv[:, S - 3:, :])


def norm_gate(y, gate, norm_w, eps, heads, dim):
    """y, gate [1, S, H, D] -> o_norm(y) * sigmoid(gate) as [1, S, H D]."""
    return _kernels()["ng"](inputs=[y, gate, norm_w, mx.array([eps], dtype=mx.float32)], template=[("T", y.dtype), ("H", heads), ("D", dim)],
                            grid=(32, y.shape[1], heads), threadgroup=(32, 1, 1), output_shapes=[(1, y.shape[1], heads * dim)], output_dtypes=[y.dtype])[0]


def _why(self, x, mask, cache):
    """None when the prompt path serves this call, else the reason (flag audit key; the guards of `_eligible`, same order)."""
    if mask is not None:
        return "mask"
    if cache is None:
        return "no_cache"
    if not isinstance(x, mx.array) or x.ndim != 3:
        return "x_shape"
    if x.shape[1] < MIN_ROWS:
        return f"rows<{MIN_ROWS}"                          # decode / verify call: checked first so the batch reasons below are prompt chunks only
    if x.shape[0] != 1:
        return f"batch={x.shape[0]}"
    if x.dtype not in (mx.bfloat16, mx.float16):
        return "x_dtype=" + str(x.dtype).split(".")[-1]
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return "no_gpu"
    if self.head_dim != 128 or self.conv_kernel != 4 or self.lower_bound is None or not self.lower_bound < 0:
        return "kda_dims"
    if len(getattr(cache, "cache", ())) != 2:
        return "cache=" + type(cache).__name__
    if getattr(cache, "is_speculating", True):
        return "speculating"
    if getattr(cache, "history_capacity", 1):
        return "history_capacity"
    if getattr(cache, "lengths", 1) is not None or getattr(cache, "left_padding", 1) is not None:
        return "batch_cache_padding"
    if self.A_log.dtype != mx.float32 or self.dt_bias.dtype != mx.float32:
        return "param_dtype"
    if self.o_norm.weight.dtype != x.dtype or self.qkv_conv.conv.weight.dtype != x.dtype:
        return "norm_conv_dtype"
    return None


def _eligible(self, x, mask, cache) -> bool:
    return _why(self, x, mask, cache) is None


def _make_call(orig, lang):
    from .kda_nax import kda_nax

    def __call__(self, x, mask=None, cache=None):
        why = _why(self, x, mask, cache)
        if why is not None:
            STATS["fallback"] += 1
            FD.fb("kda_pf", why)                          # "rows<64" = a decode / verify call, not a prompt chunk: the audit table reads it as not applicable
            return orig(self, x, mask, cache)
        STATS["calls"] += 1
        FD.fire("kda_pf")
        T = x.shape[1]
        H, D = self.num_heads, self.head_dim
        shape = (1, T, H, D)
        qkv = lang.linear(self.qkv_proj, x)
        conv_state = cache.cache[0]
        if conv_state is None:
            conv_state = mx.zeros((1, 3, 3 * H * D), dtype=qkv.dtype)
        if PRE:
            q, k, v, cache.cache[0] = prework(qkv, conv_state, self.qkv_conv.conv.weight, H, D)
        else:
            qkv = self.qkv_conv(qkv, mask=None, cache=cache)
            q, k, v = (t.reshape(shape) for t in mx.split(qkv, 3, axis=-1))
            eps = 1e-6 / D
            q = self.scale ** 2 * mx.fast.rms_norm(q, None, eps)
            k = self.scale * mx.fast.rms_norm(k, None, eps)
        f_a, b, g_a = mx.split(lang.linear(self.fbg_a_proj, x), (D, D + H), axis=-1)
        a = lang.linear(self.f_b_proj, f_a).reshape(shape)
        state = cache.cache[1]
        if state is None:
            state = mx.zeros((1, H, D, D), dtype=mx.float32)
        if REC == "nax":
            out, state = kda_nax(q, k, v, a, mx.sigmoid(b), self.A_log, self.dt_bias, self.lower_bound, state)
        else:
            out, state = lang.gated_delta_update(q, k, v, a, b.reshape(1, T, H), self.A_log.reshape(H, 1), self.dt_bias.reshape(H, D), state=state,
                                                 use_kernel=True, lower_bound=self.lower_bound)
        cache.cache[1] = state
        cache.advance(T)
        gate = lang.linear(self.g_b_proj, g_a).reshape(shape)
        if PRE:
            flat = norm_gate(out, gate, self.o_norm.weight, self.o_norm.eps, H, D)
        else:
            flat = (self.o_norm(out) * mx.sigmoid(gate)).reshape(1, T, -1)
        return lang.linear(self.o_proj, flat)

    return __call__


def install(model) -> int:
    """Patch Glm5NextLinearAttention.__call__ (process wide, once). Returns 1 when installed."""
    if not ENABLED:
        return 0
    lm = getattr(model, "language_model", None)
    if not type(lm).__module__.startswith("mlx_vlm.models.glm5_next"):
        return 0
    from mlx_vlm.models.glm5_next import language as lang
    cls = lang.Glm5NextLinearAttention
    if getattr(cls, "_kyojin_kda_pf", False):
        return 1
    cls.__call__ = _make_call(cls.__call__, lang)
    cls._kyojin_kda_pf = True
    return 1
