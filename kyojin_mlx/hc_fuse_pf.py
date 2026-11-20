# SPDX-License-Identifier: Apache-2.0
"""Hyper-connection glue of a prompt chunk (identity form: fn has 2 * hc rows, no Sinkhorn) in two launches instead of ~20 per sub-layer.

Stock chain (foreign_skeleton_hc.HcStream, eager at prefill): fp32 copy, rms_norm, a skinny fp32 matmul against fn [2 hc, hc * D], the sigmoid / scale / base chain on [rows, 8]
(about ten tiny launches), the collapse (a broadcast multiply and a sum over hc: two full passes over the [rows, hc, D] stream) and, after the sub-layer, the expand (h + post * y).
Here:
  pre    one threadgroup (256 threads) per 4 rows: sum of squares and the 2 hc dot products against fn in ONE pass over the stream (fn read once per 4 rows), then pre / post from the
         sums and the collapse x = sum_i pre_i h_i (second read of the same rows, cache hot) -> x [rows, D] in the stream dtype and post [rows, hc] fp32;
  expand h' = h + post * y, vector loads, one pass.
Same formulas as the stock path (mixes = rms(flatten(h)) @ fn.T; pre = sigmoid(m[:hc] * scale[0] + base[:hc]) + eps; post = magnitude * sigmoid(m[hc:] * scale[1] + base[hc:]) (+ eps with
post_eps)); the fp32 sums run in a different order (1e-6 relative). Flag KYOJIN_HY4_HC_FUSE (read per call, default ON since 19/11: a16 cell, NLL inside the null band), prompt chunks only (rows >= KYOJIN_HY4_HC_FUSE_ROWS, 64)."""
from __future__ import annotations

import os

import mlx.core as mx

RB = 4
_PRE = r"""
    constexpr int R = 2 * HC;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint row0 = threadgroup_position_in_grid.x * uint(RB);
    const uint Lr = uint(h_shape[0]);
    float acc[RB][R + 1];
    for (int rb = 0; rb < RB; ++rb) for (int c = 0; c <= R; ++c) acc[rb][c] = 0.0f;
    const device vec<T, 4>* h4 = (const device vec<T, 4>*)h;
    const device float4* f4 = (const device float4*)fn;
    for (uint k = tid; k < uint(KD) / 4u; k += 256u) {
        float4 fv[R];
        for (int r = 0; r < R; ++r) fv[r] = f4[uint(r) * (uint(KD) / 4u) + k];
        for (int rb = 0; rb < RB; ++rb) {
            if (row0 + uint(rb) < Lr) {
                const float4 hv = float4(h4[(row0 + uint(rb)) * (uint(KD) / 4u) + k]);
                acc[rb][R] += dot(hv, hv);
                for (int r = 0; r < R; ++r) acc[rb][r] += dot(hv, fv[r]);
            }
        }
    }
    threadgroup float red[8][RB * (R + 1)];
    for (int rb = 0; rb < RB; ++rb) for (int c = 0; c <= R; ++c) {
        const float v = simd_sum(acc[rb][c]);
        if (lane == 0) red[sg][rb * (R + 1) + c] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float pre_s[RB][HC];
    if (tid < uint(RB)) {
        const int rb = int(tid);
        float t[R + 1];
        for (int c = 0; c <= R; ++c) { float s = 0.0f; for (int g = 0; g < 8; ++g) s += red[g][rb * (R + 1) + c]; t[c] = s; }
        const float inv = metal::precise::rsqrt(t[R] / float(KD) + prm[2]);
        for (int i = 0; i < HC; ++i) {
            pre_s[rb][i] = 1.0f / (1.0f + metal::precise::exp(-(t[i] * inv * sc[0] + base[i]))) + prm[0];
            const float po = prm[1] / (1.0f + metal::precise::exp(-(t[HC + i] * inv * sc[1] + base[HC + i]))) + (prm[3] != 0.0f ? prm[0] : 0.0f);
            if (row0 + uint(rb) < Lr) post[(row0 + uint(rb)) * uint(HC) + uint(i)] = po;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int rb = 0; rb < RB; ++rb) {
        if (row0 + uint(rb) >= Lr) break;
        const device T* hr = h + ulong(row0 + uint(rb)) * uint(KD);
        for (uint d = tid; d < uint(DD); d += 256u) {
            float s = 0.0f;
            for (int i = 0; i < HC; ++i) s += pre_s[rb][i] * float(hr[uint(i) * uint(DD) + d]);
            x[ulong(row0 + uint(rb)) * uint(DD) + d] = T(s);
        }
    }
"""
_EXP = r"""
    const uint g = thread_position_in_grid.x;
    const uint k4 = g % (uint(KD) / 4u), row = g / (uint(KD) / 4u);
    const uint i = (k4 * 4u) / uint(DD), d = k4 * 4u - i * uint(DD);
    const float pv = post[row * uint(HC) + i];
    const float4 hv = float4(*(const device vec<T, 4>*)(h + ulong(row) * uint(KD) + k4 * 4u));
    const float4 yv = float4(*(const device vec<TY, 4>*)(y + ulong(row) * uint(DD) + d));
    *(device vec<T, 4>*)(ho + ulong(row) * uint(KD) + k4 * 4u) = vec<T, 4>(hv + pv * yv);
"""
_k = {}
ROWS_MIN = int(os.environ.get("KYOJIN_HY4_HC_FUSE_ROWS", "64"))
_DT = (mx.float32, mx.float16, mx.bfloat16)


def on() -> bool:
    return os.environ.get("KYOJIN_HY4_HC_FUSE", "1") not in ("0", "")


def ok(hc, h) -> bool:
    """h [B, L, hc, D] stream, identity-form weights (fn [2 hc, hc D] fp32, no Sinkhorn rows), prompt rows."""
    if h.ndim != 4 or h.dtype not in _DT:
        return False
    B, L, n, D = h.shape
    return n == hc.hc and n <= 4 and hc.fn.shape == (2 * n, n * D) and hc.fn.dtype == mx.float32 and D % 4 == 0 and B * L >= ROWS_MIN


def pre(hc, h):
    """-> x [B, L, D] (h dtype) = collapse(h, pre), post [B, L, hc] fp32."""
    B, L, n, D = h.shape
    if "pre" not in _k:
        _k["pre"] = mx.fast.metal_kernel(name="kyojin_hy4_hc_pre", input_names=["h", "fn", "sc", "base", "prm"], output_names=["x", "post"], source=_PRE)
    Lr = B * L
    prm = mx.array([hc.eps, hc.magnitude, hc.norm_eps, 1.0 if hc.post_eps else 0.0], dtype=mx.float32)
    x, post = _k["pre"](inputs=[h.reshape(Lr, n * D), hc.fn, hc.scale.astype(mx.float32), hc.base.astype(mx.float32), prm],
                        template=[("T", h.dtype), ("KD", n * D), ("DD", D), ("HC", n), ("RB", RB)], grid=(-(-Lr // RB) * 256, 1, 1), threadgroup=(256, 1, 1),
                        output_shapes=[(Lr, D), (Lr, n)], output_dtypes=[h.dtype, mx.float32])
    return x.reshape(B, L, D), post.reshape(B, L, n)


def expand(y, h, post):
    """h' = h + post * y (identity form), h [B, L, hc, D], y [B, L, D], post [B, L, hc] -> h dtype."""
    B, L, n, D = h.shape
    if "exp" not in _k:
        _k["exp"] = mx.fast.metal_kernel(name="kyojin_hy4_hc_expand", input_names=["h", "y", "post"], output_names=["ho"], source=_EXP)
    Lr = B * L
    if y.dtype not in _DT:
        y = y.astype(mx.float32)
    ho = _k["exp"](inputs=[h.reshape(Lr, n * D), y.reshape(Lr, D), post.reshape(Lr, n)], template=[("T", h.dtype), ("TY", y.dtype), ("KD", n * D), ("DD", D), ("HC", n)],
                   grid=(Lr * n * D // 4, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(Lr, n * D)], output_dtypes=[h.dtype])[0]
    return ho.reshape(B, L, n, D)
