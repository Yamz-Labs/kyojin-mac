# SPDX-License-Identifier: Apache-2.0
"""Row-invariant dense products for multi-user decode: the result of a row does not depend on how many other rows share the launch.

mx.matmul picks another kernel for another number of rows (about 2 % of rows differ in the last bit between 1 and 4 rows, fp16 and fp32).
`rowdot` computes y[r, o] = sum_k x[r, k] * w[o, k] with a summation order that is a function of (K, N) only: lane l of simdgroup s takes
k = 8 (32 s + l) + 256 KS i + j (j = 0..7 in order, fp32 fma), then simd_sum, then (KS > 1) the KS simdgroup sums in index order.
x and w may be fp16, bf16 or fp32 (accumulation is always fp32); the output has the promoted dtype of the two (the dtype of mixed fp16 / bf16 is fp32, as mx.matmul)."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

_ROWDOT = r"""
    // one threadgroup per (output o, group of MR rows): KS simdgroups split k, the MR rows at once (the weight row is read once per group).
    const uint o = threadgroup_position_in_grid.x;
    const uint rb = threadgroup_position_in_grid.y * MR;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint K = params[0];
    const uint N = params[1];
    const uint ROWS = params[2];
    threadgroup float part[MR * KS];
    float acc[MR];
    for (uint r = 0; r < MR; ++r) acc[r] = 0.0f;
    auto wp = w + ulong(o) * K;
    for (uint kk = (sg * 32 + lane) * 8; kk < K; kk += 256 * KS) {
        float wv[8];
        if (V8) {                                       // K % 8 == 0: no tail, 8 values in vector loads (same values, same fma order)
            ld8(wp + kk, wv);
            for (uint r = 0; r < MR; ++r) {
                float xv[8];
                ld8(x + ulong(min(rb + r, ROWS - 1)) * K + kk, xv);
                for (uint j = 0; j < 8; ++j) acc[r] = fma(wv[j], xv[j], acc[r]);
            }
            continue;
        }
        for (uint j = 0; j < 8; ++j) wv[j] = (kk + j < K) ? float(wp[kk + j]) : 0.0f;
        for (uint r = 0; r < MR; ++r) {
            auto xp = x + ulong(min(rb + r, ROWS - 1)) * K + kk;
            for (uint j = 0; j < 8; ++j) acc[r] = fma(wv[j], (kk + j < K) ? float(xp[j]) : 0.0f, acc[r]);
        }
    }
    for (uint r = 0; r < MR; ++r) {
        const float t = simd_sum(acc[r]);
        if (lane == 0) part[r * KS + sg] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        for (uint r = 0; r < MR; ++r) {
            float s = part[r * KS];
            for (uint i = 1; i < KS; ++i) s += part[r * KS + i];
            if (rb + r < ROWS) y[ulong(rb + r) * N + o] = static_cast<YT>(s);
        }
    }
"""
# 8 values of a row as float: float4 / half4 vector loads for fp32 / fp16 (bf16: element loads). Apple GPUs load unaligned vectors correctly.
_LD8 = r"""
#pragma METAL fp contract(off)
template <typename T> inline void ld8(const device T* p, thread float* v) { for (uint j = 0; j < 8; ++j) v[j] = float(p[j]); }
inline void ld8(const device float* p, thread float* v) {
    const float4 a = *(const device float4*)p, b = *(const device float4*)(p + 4);
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}
inline void ld8(const device half* p, thread float* v) {
    const float4 a = float4(*(const device half4*)p), b = float4(*(const device half4*)(p + 4));
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}
"""
_kern = None
ROWDOT_MAX = 8
_ks_cache: dict = {}
_DT = (mx.float16, mx.bfloat16, mx.float32)


def pick_ks(K: int, N: int) -> int:
    """Simdgroups per output: more k-splits when there are few outputs (hyper-connection mixes: N = 24, K = 20480). A function of (K, N) only."""
    key = (K, N)
    ks = _ks_cache.get(key)
    if ks is None:
        ks = 1
        while N * ks < 192 and K // (256 * ks * 2) >= 2 and ks < 16:
            ks *= 2
        _ks_cache[key] = ks
    return ks


_params: dict = {}


def rowdot(x: mx.array, w: mx.array, out_dtype=None) -> mx.array:
    """x [B, K], w [N, K] (fp16, bf16 or fp32 each) -> [B, N], row-invariant. B > ROWDOT_MAX: groups of ROWDOT_MAX rows on the grid, still one launch (a row keeps its bits).
    out_dtype: the output type when the promoted one is not wanted (an fp32 gate read as a bf16 copy still outputs fp32)."""
    global _kern
    if _kern is None:
        _kern = mx.fast.metal_kernel(name="kyojin_rowdot2", input_names=["x", "w", "params"], output_names=["y"], source=_ROWDOT,
                                     header=_LD8)
    B, K = x.shape
    N = w.shape[0]
    mr = min(B, ROWDOT_MAX)
    assert x.dtype in _DT and w.dtype in _DT, (x.dtype, w.dtype)
    out = out_dtype or (x.dtype if x.dtype == w.dtype else mx.float32)
    ks = pick_ks(K, N)
    p = _params.get((K, N, B))
    if p is None:
        p = _params[(K, N, B)] = mx.array([K, N, B], dtype=mx.uint32)
    return _kern(inputs=[x, w, p], template=[("MR", mr), ("KS", ks), ("YT", out), ("V8", int(K % 8 == 0))], grid=(N * 32 * ks, (B + mr - 1) // mr, 1), threadgroup=(32 * ks, 1, 1),
                 output_shapes=[(B, N)], output_dtypes=[out])[0]


def decode_rows(x: mx.array) -> bool:
    """True for a decode activation [B, 1, K] (one token per user) the row-invariant products apply to."""
    return x.ndim == 3 and x.shape[1] == 1 and x.dtype in _DT


def rowmm(x: mx.array, w: mx.array) -> mx.array:
    """x [B, 1, K] @ w[N, K].T for a decode activation, row-invariant; anything else (prompt chunks, odd dtypes) takes the stock matmul."""
    if decode_rows(x) and w.dtype in _DT:
        return rowdot(x[:, 0], w)[:, None]
    return x @ w.T


class RowLinear(nn.Module):
    """Plain linear (router, tiny projections) whose decode output does not depend on the number of users. Decode steps (one token per
    user, fp16 or fp32 activation) use `rowdot`; prompts (S > 1) use the stock matmul."""

    def __init__(self, lin: nn.Linear):
        super().__init__()
        self.weight = lin.weight
        if "bias" in lin:
            self.bias = lin.bias

    def __call__(self, x: mx.array) -> mx.array:
        y = rowmm(x, self.weight)
        return y + self.bias if "bias" in self else y
