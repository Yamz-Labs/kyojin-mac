# SPDX-License-Identifier: Apache-2.0
"""Row-invariant dense products for multi-user decode: the result of a row does not depend on how many other rows share the launch.

mx.matmul picks another kernel for another number of rows (about 2 % of rows differ in the last bit between 1 and 4 rows, fp16 and fp32).
`rowdot` computes y[r, o] = sum_k x[r, k] * w[o, k] with a summation order that is a function of (K, N) only: lane l of simdgroup s takes
k = 8 (32 s + l) + 256 KS i + j (j = 0..7 in order, fp32 fma), then simd_sum, then (KS > 1) the KS simdgroup sums in index order.
x and w may be fp16, bf16 or fp32 (accumulation is always fp32); the output has the promoted dtype of the two (the dtype of mixed fp16 / bf16 is fp32, as mx.matmul)."""
from __future__ import annotations

import os

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


# flag skinny_rows (KYOJIN_SKINNY_ROWS = largest row count, 0 = off): skinny fp32 / fp16 products of a few-row pass (chained verify rows, hyper-connection mixes, routers) run as ONE rowdot
# launch instead of mx.matmul. mx.matmul picks the gemv kernel for ONE row and the steel GEMM (split-K accumulate pair when K is large and N small) from the second row on: on Hy4 the
# 2-row verify pass paid ~300 extra dispatches (steel_gemm_splitk + accum, steel_gemm_fused_nax) for the hyper-connection mixes and routers alone.
SKINNY_ROWS = int(os.environ.get("KYOJIN_SKINNY_ROWS", "0"))


def skinny_mm(x: mx.array, w: mx.array) -> mx.array:
    """x [..., K] @ w[N, K].T. With the flag on and 2 <= rows <= SKINNY_ROWS (rows = product of the leading dims) the product is one `rowdot` launch (row-invariant, fp32 accumulate,
    the weight row read once for all rows); one row and many rows keep the stock matmul (one row = the gemv kernel the single-token step uses, many rows = prompt chunks)."""
    if SKINNY_ROWS >= 2 and w.ndim == 2 and x.ndim >= 2 and x.shape[-1] == w.shape[1] and x.dtype in _DT and w.dtype in _DT:
        rows = 1
        for d in x.shape[:-1]:
            rows *= d
        if 2 <= rows <= SKINNY_ROWS:
            return rowdot(x.reshape(rows, x.shape[-1]), w).reshape(*x.shape[:-1], w.shape[0])
    return x @ w.T


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


# ---- batched rowdot: x [H, B, K], w [H, N, K] -> [H, B, N] (one weight matrix per head, grid z = head). The MLA absorb products (q_nope @ k_b^T, o @ v_b^T) of a 2..8-row pass.
def _batched_source() -> str:
    src = _ROWDOT
    for a, b in (("    const uint o = threadgroup_position_in_grid.x;\n", "    const uint o = threadgroup_position_in_grid.x;\n    const ulong hz = threadgroup_position_in_grid.z;\n"),
                 ("auto wp = w + ulong(o) * K;", "auto wp = w + (hz * N + o) * K;"),
                 ("x + ulong(min(rb + r, ROWS - 1)) * K + kk", "x + (hz * ROWS + min(rb + r, ROWS - 1)) * K + kk"),
                 ("y[ulong(rb + r) * N + o]", "y[(hz * ROWS + rb + r) * N + o]")):
        assert a in src, a
        src = src.replace(a, b)
    return src


_kern_b = None
SKINNY_ABSORB = int(os.environ.get("KYOJIN_SKINNY_ABSORB", "0"))


def rowdot_batched(x: mx.array, w: mx.array, out_dtype=None) -> mx.array:
    """x [H, B, K], w [H, N, K] (fp16 / bf16 / fp32) -> [H, B, N]; fp32 accumulate, B <= ROWDOT_MAX rows share one weight read. Same arithmetic as rowdot per head."""
    global _kern_b
    if _kern_b is None:
        _kern_b = mx.fast.metal_kernel(name="kyojin_rowdot_b", input_names=["x", "w", "params"], output_names=["y"], source=_batched_source(), header=_LD8)
    H, B, K = x.shape
    N = w.shape[1]
    assert w.shape[0] == H and w.shape[2] == K and 1 <= B <= ROWDOT_MAX and x.dtype in _DT and w.dtype in _DT
    out = out_dtype or (x.dtype if x.dtype == w.dtype else mx.float32)
    ks = pick_ks(K, N)
    p = _params.get((K, N, B))
    if p is None:
        p = _params[(K, N, B)] = mx.array([K, N, B], dtype=mx.uint32)
    return _kern_b(inputs=[x, w, p], template=[("MR", B), ("KS", ks), ("YT", out), ("V8", int(K % 8 == 0))], grid=(N * 32 * ks, 1, H), threadgroup=(32 * ks, 1, 1),
                   output_shapes=[(H, B, N)], output_dtypes=[out])[0]


def absorb_mm(x: mx.array, w: mx.array):
    """x [1, H, L, K] @ w[H, N, K]^T -> [1, H, L, N] for a 2..SKINNY_ABSORB-row pass (flag skinny_absorb), else None (the caller keeps the stock matmul)."""
    if SKINNY_ABSORB < 2 or x.ndim != 4 or x.shape[0] != 1 or not (2 <= x.shape[2] <= min(SKINNY_ABSORB, ROWDOT_MAX)) or w.ndim != 3 or w.shape[0] != x.shape[1] or w.shape[2] != x.shape[3]:
        return None
    if x.dtype not in _DT or w.dtype not in _DT:
        return None
    return rowdot_batched(x[0], w)[None]


# ---- router weight normalisation in one launch: w [T, K] f32 -> w / (sum(w) + eps | max(sum(w), floor)) * scale  (replaces row_reduce + Add / Max + Divide + Multiply)
_ROUTE_NORM = r"""
    const uint t = thread_position_in_grid.x;
    if (t >= T) return;
    float s = 0.0f;
    for (uint k = 0; k < K; ++k) s += w[t * K + k];
    const float tot = c[0] > 0.0f ? s + c[0] : fmax(s, c[1]);
    for (uint k = 0; k < K; ++k) y[t * K + k] = (w[t * K + k] / tot) * c[2];
"""
_kern_rn = None
ROUTE_NORM_FUSE = os.environ.get("KYOJIN_ROUTE_NORM_FUSE", "0") not in ("0", "")


def route_norm(w: mx.array, eps: float, floor: float, scale: float) -> mx.array:
    global _kern_rn
    if _kern_rn is None:
        _kern_rn = mx.fast.metal_kernel(name="kyojin_route_norm", input_names=["w", "c"], output_names=["y"], source=_ROUTE_NORM)
    T, K = w.shape
    c = mx.array([float(eps), float(floor), float(scale) if scale not in (0.0, 1.0) else 1.0], dtype=mx.float32)
    return _kern_rn(inputs=[w.astype(mx.float32), c], template=[("T", T), ("K", K)],
                    grid=(T, 1, 1), threadgroup=(min(T, 32), 1, 1), output_shapes=[(T, K)], output_dtypes=[mx.float32])[0]
