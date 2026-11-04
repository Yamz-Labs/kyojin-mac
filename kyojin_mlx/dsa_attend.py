# SPDX-License-Identifier: Apache-2.0
"""DS4.1 / GLM decode attention over the sliding window + top-k pooled rows, multi-query form (one KV head shared by every query head).

The scalar kernel in `fuse_attn._ATTEND` gives every head its own threadgroup and every head re-reads the same 640 key rows (window 128 + top-k 512, 1 KB each):
the time grows linearly with the number of heads (5.5 us per head on the M5 Air, 64 heads: 0.3 ms per layer) while the ALUs idle. Here a threadgroup takes 8 heads
and 64 keys (a chunk), so the rows are read once per 8 heads, and both products run on simdgroup 8x8 matrices:
  pass 1 (HG = H / 8 x NC = ceil((W + K) / 64) threadgroups): S = Q' K^T (Q' = rope, rounded to fp16, scaled, and truncated like the stock fp32 matmul operand: exactly
          representable in fp16), chunk max m and sum l of exp(S - m), partial O = exp(S - m) V (fp32), V staged 64 dims at a time through threadgroup memory
          (the gather of the top-k rows happens in that copy);
  pass 2 (H threadgroups): merge the NC chunks (sdpa_vector_2pass style), the sink joins the denominator, inverse rope on the tail, round to fp16.
fp16 only (the probabilities are rounded to fp16 for the PV product); same inputs and output as `fuse_attn.attend`."""
from __future__ import annotations

import mlx.core as mx

CK = 64

_HEADER = """
#include <metal_simdgroup_matrix>
using namespace metal;
inline float trunc10(float x) { return as_type<float>(as_type<uint>(x) & 0xFFFFE000u); }
"""

_P1 = r"""
    constexpr uint NTH = 128;
    const uint HG = uint(H) / 8;
    const uint tg = threadgroup_position_in_grid.x;
    const uint hg = tg % HG, c = tg / HG;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint K = idx_shape[0];
    const uint N = pool_shape[0];
    const uint NK = uint(W) + K;
    const int posv = pos[0];
    const float inf = metal::numeric_limits<float>::infinity();
    threadgroup half Qs[8 * D];
    threadgroup uint4 stg[CK * 8];
    threadgroup half* stage = (threadgroup half*)stg;
    threadgroup float Sg[8 * CK];
    threadgroup half Pg[8 * CK];
    threadgroup int rowi[CK];
    threadgroup uchar isr[CK];
    if (tid < CK) {
        const uint j = c * CK + tid;
        int code = -1; uchar ring_row = 0;
        if (j < NK) {
            if (j < uint(W)) { ring_row = 1; code = ((posv >= int(W) - 1) || (int(j) <= posv)) ? int(j) : -1; }
            else { const int ii = idx[j - uint(W)]; code = (ii >= 0 && uint(ii) < N) ? ii : -1; }
        }
        rowi[tid] = code; isr[tid] = ring_row;
    }
    for (uint e = tid; e < 8 * (D / 2); e += NTH) {
        const uint r = e / (D / 2), dp = e % (D / 2);
        const uint h = hg * 8 + r;
        float x1 = float(q[ulong(h) * D + 2 * dp]), x2 = float(q[ulong(h) * D + 2 * dp + 1]);
        if (2 * dp >= uint(D - RD)) {                              // q rope (mx.fast.rope traditional=True), rounded to T like the stock op
            const uint p0 = dp - uint(D - RD) / 2;
            const float cs = rt[2 * p0], sn = rt[2 * p0 + 1];
            const float y1 = float(static_cast<T>(x1 * cs - x2 * sn));
            const float y2 = float(static_cast<T>(x1 * sn + x2 * cs));
            x1 = y1; x2 = y2;
        }
        x1 = x1 * SCALE; x2 = x2 * SCALE;
        if (TRUNC) { x1 = trunc10(x1); x2 = trunc10(x2); }
        Qs[r * D + 2 * dp] = static_cast<half>(x1); Qs[r * D + 2 * dp + 1] = static_cast<half>(x2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // S = Q' K^T over the 64-dim slices of the key rows
    simdgroup_matrix<float, 8, 8> S0(0.0f), S1(0.0f);
    for (uint s = 0; s < uint(D) / 64; ++s) {
        for (uint i = tid; i < CK * 8; i += NTH) {
            const uint r = i / 8, part = i % 8;
            const int code = rowi[r];
            uint4 v = uint4(0);
            if (code >= 0) { v = *(const device uint4*)((isr[r] ? (ring + ulong(code) * D) : (pool + ulong(code) * D)) + s * 64 + part * 8); }
            stg[i] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ks = 0; ks < 8; ++ks) {
            simdgroup_matrix<half, 8, 8> aq, b0, b1;
            simdgroup_load(aq, &Qs[s * 64 + ks * 8], D);
            simdgroup_load(b0, &stage[(sg * 2) * 8 * 64 + ks * 8], 64, ulong2(0, 0), true);
            simdgroup_load(b1, &stage[(sg * 2 + 1) * 8 * 64 + ks * 8], 64, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(S0, aq, b0, S0);
            simdgroup_multiply_accumulate(S1, aq, b1, S1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(S0, &Sg[(sg * 2) * 8], CK);
    simdgroup_store(S1, &Sg[(sg * 2 + 1) * 8], CK);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // chunk softmax numerators: head r = tid / 16, keys 4 * (tid % 16) .. +3
    {
        const uint r = tid / 16, q16 = tid % 16;
        float sv[4];
        float m = -inf;
        for (uint i = 0; i < 4; ++i) {
            const uint k = q16 * 4 + i;
            sv[i] = (rowi[k] >= 0) ? Sg[r * CK + k] : -inf;
            m = metal::max(m, sv[i]);
        }
        m = metal::max(m, simd_shuffle_xor(m, 1u)); m = metal::max(m, simd_shuffle_xor(m, 2u));
        m = metal::max(m, simd_shuffle_xor(m, 4u)); m = metal::max(m, simd_shuffle_xor(m, 8u));
        float l = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            const float p = (m == -inf) ? 0.0f : metal::exp(sv[i] - m);
            l += p;
            Pg[r * CK + q16 * 4 + i] = static_cast<half>(p);
        }
        l += simd_shuffle_xor(l, 1u); l += simd_shuffle_xor(l, 2u); l += simd_shuffle_xor(l, 4u); l += simd_shuffle_xor(l, 8u);
        if (q16 == 0) { mpart[(c * HG + hg) * 8 + r] = m; lpart[(c * HG + hg) * 8 + r] = l; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // O = P V, 64 output dims at a time
    device float* ob = opart + ulong((c * HG + hg) * 8) * D;
    for (uint s = 0; s < uint(D) / 64; ++s) {
        for (uint i = tid; i < CK * 8; i += NTH) {
            const uint r = i / 8, part = i % 8;
            const int code = rowi[r];
            uint4 v = uint4(0);
            if (code >= 0) { v = *(const device uint4*)((isr[r] ? (ring + ulong(code) * D) : (pool + ulong(code) * D)) + s * 64 + part * 8); }
            stg[i] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_matrix<float, 8, 8> O0(0.0f), O1(0.0f);
        for (uint kt = 0; kt < 8; ++kt) {
            simdgroup_matrix<half, 8, 8> a, b0, b1;
            simdgroup_load(a, &Pg[kt * 8], CK);
            simdgroup_load(b0, &stage[kt * 8 * 64 + (sg * 2) * 8], 64);
            simdgroup_load(b1, &stage[kt * 8 * 64 + (sg * 2 + 1) * 8], 64);
            simdgroup_multiply_accumulate(O0, a, b0, O0);
            simdgroup_multiply_accumulate(O1, a, b1, O1);
        }
        simdgroup_store(O0, ob + s * 64 + (sg * 2) * 8, D);
        simdgroup_store(O1, ob + s * 64 + (sg * 2 + 1) * 8, D);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""


_HEADER_P = _HEADER + """
constant float FP4_LUT[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
inline float e4m3_to_float(uchar b) {
    const int e = (b >> 3) & 15, m = b & 7;
    const float mag = e == 0 ? float(m) * 0.001953125f : (1.0f + float(m) * 0.125f) * metal::exp2(float(e - 7));
    return (b & 128) ? -mag : mag;
}
// 8 dims [d0, d0 + 8) of a pooled row stored as fp4 codes (2 per byte, vendor block 16 / e4m3 scale) -> 8 halves, bit-identical to the fake-quantized fp16 row
inline uint4 fp4_row8(const device uchar* codes, const device uchar* scales, ulong row, uint d0, uint D) {
    const uint w = *(const device uint*)(codes + row * (D / 2) + d0 / 2);
    const float sc = e4m3_to_float(scales[row * (D / 16) + d0 / 16]);
    half h[8];
    for (uint i = 0; i < 8; ++i) { const uint c = (w >> (4 * i)) & 15u; const float mag = FP4_LUT[c & 7u]; h[i] = static_cast<half>(((c & 8u) ? -mag : mag) * sc); }
    return *reinterpret_cast<thread uint4*>(h);
}
"""
_OLD_LOAD = 'if (code >= 0) { v = *(const device uint4*)((isr[r] ? (ring + ulong(code) * D) : (pool + ulong(code) * D)) + s * 64 + part * 8); }'
_P1P = _P1.replace(_OLD_LOAD, "if (code >= 0) { v = isr[r] ? *(const device uint4*)(ring + ulong(code) * D + s * 64 + part * 8) : fp4_row8(poolc, pools, ulong(code), s * 64 + part * 8, uint(D)); }").replace("const uint N = pool_shape[0];", "const uint N = poolc_shape[0];")

_P2 = r"""
    const uint h = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint HG = uint(H) / 8;
    const uint hg = h / 8, r = h % 8;
    const float inf = metal::numeric_limits<float>::infinity();
    float m = -inf;
    for (uint c = 0; c < uint(NC); ++c) { m = metal::max(m, mpart[(c * HG + hg) * 8 + r]); }
    float l = 0.0f, a0 = 0.0f, a1 = 0.0f;
    if (m != -inf) {
        for (uint c = 0; c < uint(NC); ++c) {
            const uint ix = (c * HG + hg) * 8 + r;
            const float mc = mpart[ix];
            if (mc == -inf) { continue; }
            const float f = metal::exp(mc - m);
            l += f * lpart[ix];
            a0 += f * opart[ulong(ix) * D + 2 * tid];
            a1 += f * opart[ulong(ix) * D + 2 * tid + 1];
        }
        const float s = sink[h];
        const float den = l + metal::exp(s - m);
        a0 = a0 / den; a1 = a1 / den;
    }
    float x1 = float(static_cast<T>(a0)), x2 = float(static_cast<T>(a1));
    if (tid >= uint(D - RD) / 2) {                                 // inverse rope on the tail (the table holds cos / sin of the negated angles)
        const uint p = tid - uint(D - RD) / 2;
        const float cs = it[2 * p], sn = it[2 * p + 1];
        const float y1 = float(static_cast<T>(x1 * cs - x2 * sn));
        const float y2 = float(static_cast<T>(x1 * sn + x2 * cs));
        x1 = y1; x2 = y2;
    }
    out[ulong(h) * D + 2 * tid] = static_cast<T>(x1);
    out[ulong(h) * D + 2 * tid + 1] = static_cast<T>(x2);
"""

_k: dict = {}


def usable(q: mx.array, ring: mx.array, heads: int) -> bool:
    return q.dtype == mx.float16 and heads % 8 == 0 and ring.shape[-1] % 64 == 0 and ring.shape[-1] >= 128


def attend(q, ring, pool, idx, sink, rt, it, pos, heads: int, rd: int, scale: float, trunc: bool) -> mx.array:
    """Same contract as `fuse_attn.attend` (q [1, 1, H * D] fp16, ring [1, W, D], pool [1, N, D], idx int32 [K], sink fp32 [H]) -> [1, H, 1, D]."""
    import numpy as np
    _, W, D = ring.shape
    K = int(idx.size)
    NC = -(-(W + K) // CK)
    HG = heads // 8
    sc = float(np.float32(scale))
    key = ("p1", repr(sc))
    k1 = _k.get(key)
    if k1 is None:
        k1 = _k[key] = mx.fast.metal_kernel(name="kyojin_attn_mma_p1", input_names=["q", "ring", "pool", "idx", "rt", "pos"], output_names=["opart", "mpart", "lpart"],
                                            source=_P1.replace("SCALE", f"{sc!r}f"), header=_HEADER)
        _k["p2"] = mx.fast.metal_kernel(name="kyojin_attn_mma_p2", input_names=["opart", "mpart", "lpart", "sink", "it"], output_names=["out"], source=_P2, header=_HEADER)
    opart, mpart, lpart = k1(inputs=[q, ring, pool.reshape(-1, D), idx.reshape(-1), rt, pos],
                             template=[("T", q.dtype), ("W", W), ("D", D), ("RD", rd), ("H", heads), ("CK", CK), ("TRUNC", trunc)],
                             grid=(HG * NC * 128, 1, 1), threadgroup=(128, 1, 1),
                             output_shapes=[(NC * heads * D,), (NC * heads,), (NC * heads,)], output_dtypes=[mx.float32] * 3)
    return _k["p2"](inputs=[opart, mpart, lpart, sink, it], template=[("T", q.dtype), ("D", D), ("RD", rd), ("H", heads), ("NC", NC)],
                    grid=(heads * (D // 2), 1, 1), threadgroup=(D // 2, 1, 1), output_shapes=[(1, heads, 1, D)], output_dtypes=[q.dtype])[0]


def attend_packed(q, ring, poolc, pools, idx, sink, rt, it, pos, heads: int, rd: int, scale: float, trunc: bool) -> mx.array:
    """`attend` with the pool stored as fp4 codes (`fp4pack.pack(pool, 16, True)`: poolc uint8 [N, D / 2], pools uint8 [N, D / 16]); dequantized inside the key / value
    staging copy, bit-identical to attending over the fake-quantized fp16 pool (the vendor already rounded the rows to this grid)."""
    import numpy as np
    _, W, D = ring.shape
    K = int(idx.size)
    NC = -(-(W + K) // CK)
    HG = heads // 8
    sc = float(np.float32(scale))
    key = ("p1p", repr(sc))
    k1 = _k.get(key)
    if k1 is None:
        k1 = _k[key] = mx.fast.metal_kernel(name="kyojin_attn_mma_p1_fp4", input_names=["q", "ring", "poolc", "pools", "idx", "rt", "pos"], output_names=["opart", "mpart", "lpart"],
                                            source=_P1P.replace("SCALE", f"{sc!r}f"), header=_HEADER_P)
        if "p2" not in _k:
            _k["p2"] = mx.fast.metal_kernel(name="kyojin_attn_mma_p2", input_names=["opart", "mpart", "lpart", "sink", "it"], output_names=["out"], source=_P2, header=_HEADER)
    opart, mpart, lpart = k1(inputs=[q, ring, poolc.reshape(-1, D // 2), pools.reshape(-1, D // 16), idx.reshape(-1), rt, pos],
                             template=[("T", q.dtype), ("W", W), ("D", D), ("RD", rd), ("H", heads), ("CK", CK), ("TRUNC", trunc)],
                             grid=(HG * NC * 128, 1, 1), threadgroup=(128, 1, 1),
                             output_shapes=[(NC * heads * D,), (NC * heads,), (NC * heads,)], output_dtypes=[mx.float32] * 3)
    return _k["p2"](inputs=[opart, mpart, lpart, sink, it], template=[("T", q.dtype), ("D", D), ("RD", rd), ("H", heads), ("NC", NC)],
                    grid=(heads * (D // 2), 1, 1), threadgroup=(D // 2, 1, 1), output_shapes=[(1, heads, 1, D)], output_dtypes=[q.dtype])[0]
