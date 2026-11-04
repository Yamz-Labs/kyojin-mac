# SPDX-License-Identifier: Apache-2.0
"""Few-row (1..8) GEMV for MLX affine 4-bit, group 64, on simdgroup_matrix: the weights are decoded ONCE per tile (half2 magic-number decode,
fma with scale and bias) and multiply all rows in one 8x8x8 MMA (C^T = W x^T, weights = A operand straight from registers), so 4..8 rows cost
about one row of decode ALU instead of MLX qmv_wide's R fp32 FMAs per weight. Row results do not depend on the number of rows (fixed k order,
padded rows are zero), 1 row included.

    xt = prep_x(x)                       # x [R, K] bf16/fp16 (R <= 8) -> half [K, 8] (transposed, zero padded); shared by every linear of the same input
    y  = q4mma(xt, w, scales, biases, R) # [R, N] in the dtype of the scales
Fragment layout of simdgroup_matrix 8x8: element 0/1 at (fm, fn)/(fm, fn+1), fm = 4 bit4(lane) + ((lane >> 1) & 3), fn = 4 bit3(lane) + 2 bit0(lane).
Lane (fm, c = fn / 2) decodes row n = fm of each of NT n-tiles: its uint4 holds k = 32c .. 32c+31 of a 128-k block; tile (q, j) pairs nibbles
j and j + 4 of word q, i.e. columns fn, fn+1 <-> k = 32c + 8q + j, +4. The B operand (x^T) row l = fm uses the same map with c = fm >> 1, e = fm & 1."""
from __future__ import annotations

import mlx.core as mx

_SRC = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint warp = simdgroup_index_in_threadgroup;
    const uint tid = thread_position_in_threadgroup.x;
    const uint blk = threadgroup_position_in_grid.x;
    const uint fm = ((lane >> 4) & 1u) * 4u + ((lane >> 1) & 3u);
    const uint c = ((lane >> 3) & 1u) * 2u + (lane & 1u);
    const uint fn = c * 2u;
    const uint n0 = blk * (8u * NT);
    constexpr uint KB = K / 128, KW = K / 8, KG = K / 64;

    simdgroup_matrix<float, 8, 8> acc[NT];
    uint rowt[NT];
    _Pragma("clang loop unroll(full)") for (uint t = 0; t < NT; ++t) {
        acc[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        rowt[t] = min(n0 + 8u * t + fm, uint(N) - 1u);
    }
    for (uint kb = warp; kb < KB; kb += WS) {
        uint4 wv[NT]; half2 s2[NT], b2[NT];
        const uint g = kb * 2u + (c >> 1);
        _Pragma("clang loop unroll(full)") for (uint t = 0; t < NT; ++t) {
            wv[t] = ((const device uint4*)(w + ulong(rowt[t]) * KW + kb * 16u))[c];
            s2[t] = half2(half(float(sc[ulong(rowt[t]) * KG + g])));
            b2[t] = half2(half(float(bi[ulong(rowt[t]) * KG + g])));
        }
        const device half* xb = xt + ulong(kb * 128u + 32u * (fm >> 1) + 4u * (fm & 1u)) * 8u + fn;
        _Pragma("clang loop unroll(full)") for (uint q = 0; q < 4; ++q) {
            _Pragma("clang loop unroll(full)") for (uint j = 0; j < 4; ++j) {
                const half2 bv = *((const device half2*)(xb + (8u * q + j) * 8u));
                simdgroup_matrix<half, 8, 8> B;
                B.thread_elements()[0] = bv.x; B.thread_elements()[1] = bv.y;
                _Pragma("clang loop unroll(full)") for (uint t = 0; t < NT; ++t) {
                    const uint u = wv[t][q];
                    const half2 h = as_type<half2>(((u >> (4u * j)) & 0x000F000Fu) | 0x64006400u) - half2(1024.0h);
                    const half2 w2 = fma(s2[t], h, b2[t]);
                    simdgroup_matrix<half, 8, 8> A;
                    A.thread_elements()[0] = w2.x; A.thread_elements()[1] = w2.y;
                    simdgroup_multiply_accumulate(acc[t], A, B, acc[t]);
                }
            }
        }
    }
    threadgroup float red[WS][8 * NT][8];
    _Pragma("clang loop unroll(full)") for (uint t = 0; t < NT; ++t) {
        red[warp][8u * t + fm][fn] = acc[t].thread_elements()[0];
        red[warp][8u * t + fm][fn + 1u] = acc[t].thread_elements()[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < 64u * NT; idx += 32u * WS) {
        const uint nl = idx >> 3, r = idx & 7u;
        if (r < MR && n0 + nl < uint(N)) {
            float v = 0.0f;
            _Pragma("clang loop unroll(full)") for (uint s = 0; s < WS; ++s) v += red[s][nl][r];
            y[ulong(r) * N + n0 + nl] = static_cast<OT>(v);
        }
    }
"""
_PREP = r"""
    const uint k = thread_position_in_grid.x;
    if (k >= uint(K)) return;
    device half4* o = (device half4*)(xt + ulong(k) * XW);
    for (uint r4 = 0; r4 < XW / 4; ++r4) {
        half v[4];
        for (uint i = 0; i < 4; ++i) { const uint r = 4 * r4 + i; v[i] = r < MR ? half(x[ulong(r) * K + k]) : half(0.0f); }
        o[r4] = half4(v[0], v[1], v[2], v[3]);
    }
"""
_k: dict = {}


def prep_x(x: mx.array) -> mx.array:
    """x [..., K] with at most 16 rows in total -> half [K, 8] (<= 8 rows) or [K, 16] (9..16 rows), zero padded rows."""
    K = x.shape[-1]
    R = x.size // K
    if "prep" not in _k:
        _k["prep"] = mx.fast.metal_kernel(name="kyojin_q4mma_prep", input_names=["x"], output_names=["xt"], source=_PREP)
    XW = 8 if R <= 8 else 16
    return _k["prep"](inputs=[x.reshape(R, K)], template=[("K", K), ("MR", R), ("XW", XW)], grid=(K, 1, 1), threadgroup=(256, 1, 1),
                      output_shapes=[(K, XW)], output_dtypes=[mx.float16])[0]


def q4mma(xt: mx.array, w: mx.array, scales: mx.array, biases: mx.array, rows: int, nt: int = 4, ws: int = 4) -> mx.array:
    """y [rows, N] = x @ dequant(w).T for affine 4-bit group 64; xt = prep_x(x); rows <= 8; K % 128 == 0."""
    N, K = w.shape[0], w.shape[1] * 8
    assert K % 128 == 0 and rows <= 8 and scales.shape[1] == K // 64
    if "mma" not in _k:
        _k["mma"] = mx.fast.metal_kernel(name="kyojin_q4mma", input_names=["w", "sc", "bi", "xt"], output_names=["y"], source=_SRC,
                                         header="#include <metal_simdgroup_matrix>\n")
    ntg = (N + 8 * nt - 1) // (8 * nt)
    return _k["mma"](inputs=[w, scales, biases, xt], template=[("K", K), ("N", N), ("NT", nt), ("WS", ws), ("MR", rows), ("OT", scales.dtype)],
                     grid=(ntg * 32 * ws, 1, 1), threadgroup=(32 * ws, 1, 1), output_shapes=[(rows, N)], output_dtypes=[scales.dtype])[0]



_SRC2 = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint warp = simdgroup_index_in_threadgroup;
    const uint tid = thread_position_in_threadgroup.x;
    const uint blk = threadgroup_position_in_grid.x;
    const uint fm = ((lane >> 4) & 1u) * 4u + ((lane >> 1) & 3u);
    const uint c = ((lane >> 3) & 1u) * 2u + (lane & 1u);
    const uint fn = c * 2u;
    constexpr uint NBLK = 4 / KS, KW = K / 8, NG = K / 64;
    const uint kslice = warp % KS, nb = warp / KS;
    const uint n0 = (blk * NBLK + nb) * (8u * NT);

    simdgroup_matrix<float, 8, 8> acc[NT];
    uint rowt[NT];
    _Pragma("clang loop unroll(full)") for (uint t = 0; t < NT; ++t) {
        acc[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        rowt[t] = min(n0 + 8u * t + fm, uint(N) - 1u);
    }
    for (uint g = kslice; g < NG; g += KS) {
        uint2 wv[NT]; float sc_[NT], bi_[NT];
        _Pragma("clang loop unroll(full)") for (uint t = 0; t < NT; ++t) {
            wv[t] = (DBG & 8) ? uint2(g * 77u + t, g * 31u + c) : ((const device uint2*)(w + ulong(rowt[t]) * KW + g * 8u))[c];
            sc_[t] = float(sc[ulong(rowt[t]) * NG + g]);
            bi_[t] = float(bi[ulong(rowt[t]) * NG + g]);
        }
        const float2 sx = *((const device float2*)(xs + g * 8u + fn));
        simdgroup_matrix<float, 8, 8> tmp[NT];
        _Pragma("clang loop unroll(full)") for (uint t = 0; t < NT; ++t) tmp[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        const device half* xb = xt + ulong(g * 64u + 16u * (fm >> 1) + 4u * (fm & 1u)) * 8u + fn;
        if (DBG & 4) { _Pragma("clang loop unroll(full)") for (uint t = 0; t < NT; ++t) { tmp[t].thread_elements()[0] = float(wv[t].x ^ wv[t].y); } }
        else
        _Pragma("clang loop unroll(full)") for (uint q = 0; q < 2; ++q) {
            _Pragma("clang loop unroll(full)") for (uint j = 0; j < 4; ++j) {
                const half2 bv = *((const device half2*)(xb + (8u * q + j) * 8u));
                simdgroup_matrix<half, 8, 8> B;
                B.thread_elements()[0] = bv.x; B.thread_elements()[1] = bv.y;
                _Pragma("clang loop unroll(full)") for (uint t = 0; t < NT; ++t) {
                    const uint u = q ? wv[t].y : wv[t].x;
                    const half2 h = (DBG & 1) ? as_type<half2>(u ^ (j << 12)) : as_type<half2>(((u >> (4u * j)) & 0x000F000Fu) | 0x64006400u);
                    simdgroup_matrix<half, 8, 8> A;
                    A.thread_elements()[0] = h.x; A.thread_elements()[1] = h.y;
                    if (DBG & 2) { tmp[t].thread_elements()[0] += float(h.x) * float(bv.x); tmp[t].thread_elements()[1] += float(h.y) * float(bv.y); }
                    else simdgroup_multiply_accumulate(tmp[t], A, B, tmp[t]);
                }
            }
        }
        _Pragma("clang loop unroll(full)") for (uint t = 0; t < NT; ++t) {
            const float cb = bi_[t] - 1024.0f * sc_[t];
            acc[t].thread_elements()[0] += fma(sc_[t], tmp[t].thread_elements()[0], cb * sx.x);
            acc[t].thread_elements()[1] += fma(sc_[t], tmp[t].thread_elements()[1], cb * sx.y);
        }
    }
    if (KS == 1) {
        _Pragma("clang loop unroll(full)") for (uint t = 0; t < NT; ++t) {
            const uint n = n0 + 8u * t + fm;
            if (n < uint(N)) {
                if (fn < MR) y[ulong(fn) * N + n] = static_cast<OT>(acc[t].thread_elements()[0]);
                if (fn + 1u < MR) y[ulong(fn + 1u) * N + n] = static_cast<OT>(acc[t].thread_elements()[1]);
            }
        }
    } else {
        threadgroup float red[4][8 * NT][8];
        _Pragma("clang loop unroll(full)") for (uint t = 0; t < NT; ++t) {
            red[warp][8u * t + fm][fn] = acc[t].thread_elements()[0];
            red[warp][8u * t + fm][fn + 1u] = acc[t].thread_elements()[1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < NBLK * 64u * NT; idx += 128u) {
            const uint b = idx / (64u * NT), rem = idx % (64u * NT), nl = rem >> 3, r = rem & 7u;
            const uint nn = (blk * NBLK + b) * (8u * NT) + nl;
            if (r < MR && nn < uint(N)) {
                float v = 0.0f;
                _Pragma("clang loop unroll(full)") for (uint s = 0; s < KS; ++s) v += red[b * KS + s][nl][r];
                y[ulong(r) * N + nn] = static_cast<OT>(v);
            }
        }
    }
"""
_PREP2 = r"""
    const uint t = thread_position_in_grid.x;
    if (t < uint(K)) {
        half v[8];
        for (uint r = 0; r < 8; ++r) v[r] = r < MR ? half(x[ulong(r) * K + t]) : half(0.0f);
        device half4* o = (device half4*)(xt + ulong(t) * 8);
        o[0] = half4(v[0], v[1], v[2], v[3]); o[1] = half4(v[4], v[5], v[6], v[7]);
    }
    if (t < uint(K / 8)) {
        const uint g = t >> 3, r = t & 7u;
        float s = 0.0f;
        if (r < MR) for (uint i = 0; i < 64; ++i) s += float(half(x[ulong(r) * K + g * 64 + i]));
        xs[t] = s;
    }
"""


def prep_x2(x):
    K = x.shape[-1]
    R = x.size // K
    if "prep2" not in _k:
        _k["prep2"] = mx.fast.metal_kernel(name="kyojin_q4mma_prep2", input_names=["x"], output_names=["xt", "xs"], source=_PREP2)
    return _k["prep2"](inputs=[x.reshape(R, K)], template=[("K", K), ("MR", R)], grid=(K, 1, 1), threadgroup=(256, 1, 1),
                       output_shapes=[(K, 8), (K // 8,)], output_dtypes=[mx.float16, mx.float32])


def q4mma2(px, w, scales, biases, rows, nt=4, ks=1, dbg=0):
    """v2: group-wise raw-nibble MMA + per-group scale/bias epilogue. px = prep_x2(x)."""
    N, K = w.shape[0], w.shape[1] * 8
    assert K % 64 == 0 and rows <= 8 and 4 % ks == 0
    if "mma2" not in _k:
        _k["mma2"] = mx.fast.metal_kernel(name="kyojin_q4mma2", input_names=["w", "sc", "bi", "xt", "xs"], output_names=["y"], source=_SRC2,
                                          header="#include <metal_simdgroup_matrix>\n")
    ntg = (N + 8 * nt * (4 // ks) - 1) // (8 * nt * (4 // ks))
    return _k["mma2"](inputs=[w, scales, biases, px[0], px[1]], template=[("K", K), ("N", N), ("NT", nt), ("KS", ks), ("MR", rows), ("OT", scales.dtype), ("DBG", dbg), ("PF", pf)],
                      grid=(ntg * 128, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(rows, N)], output_dtypes=[scales.dtype])[0]



_SRC_FMA = r"""
    constexpr uint VPT = 16, BS = 512, KP = K / 4, NG = K / 64;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint r0 = threadgroup_position_in_grid.x * 8u + sg * 4u;
    const device uint2* ws[4];
    const device SCT* sp[4];
    const device SCT* bp[4];
    _Pragma("clang loop unroll(full)") for (uint rr = 0; rr < 4; ++rr) {
        const uint row = min(r0 + rr, uint(N) - 1u);
        ws[rr] = (const device uint2*)((const device uint16_t*)w + ulong(row) * KP) + lane;
        sp[rr] = sc + ulong(row) * NG + (lane >> 2);
        bp[rr] = bi + ulong(row) * NG + (lane >> 2);
    }
    float result[MR][4];
    _Pragma("clang loop unroll(full)") for (uint r = 0; r < MR; ++r) for (uint rr = 0; rr < 4; ++rr) result[r][rr] = 0.0f;
    const device XT* xp = x + lane * VPT;
    for (uint k0 = 0; k0 < uint(K); k0 += BS) {
        float xt[MR][VPT], xsum[MR];
        _Pragma("clang loop unroll(full)") for (uint r = 0; r < MR; ++r) {
            float sm = 0.0f;
            _Pragma("clang loop unroll(full)") for (uint i = 0; i < VPT; i += 4) {
                const float a0 = float(xp[ulong(r) * K + k0 + i]), a1 = float(xp[ulong(r) * K + k0 + i + 1]);
                const float a2 = float(xp[ulong(r) * K + k0 + i + 2]), a3 = float(xp[ulong(r) * K + k0 + i + 3]);
                sm += a0 + a1 + a2 + a3;
                xt[r][i] = a0; xt[r][i + 1] = a1 * (1.0f / 16.0f); xt[r][i + 2] = a2 * (1.0f / 256.0f); xt[r][i + 3] = a3 * (1.0f / 4096.0f);
            }
            xsum[r] = sm;
        }
        _Pragma("clang loop unroll(full)") for (uint rr = 0; rr < 4; ++rr) {
            const uint2 wv0 = ws[rr][k0 / VPT];
            const float s = float(sp[rr][k0 / 64u]), b = float(bp[rr][k0 / 64u]);
            float acc[MR];
            _Pragma("clang loop unroll(full)") for (uint r = 0; r < MR; ++r) acc[r] = 0.0f;
            _Pragma("clang loop unroll(full)") for (uint p = 0; p < 4; ++p) {
                const uint wd = p < 2 ? wv0.x : wv0.y;
                const uint16_t w16 = uint16_t(p & 1 ? (wd >> 16) : (wd & 0xFFFFu));
                const float q0 = float(w16 & 0x000f), q1 = float(w16 & 0x00f0), q2 = float(w16 & 0x0f00), q3 = float(w16 & 0xf000);
                _Pragma("clang loop unroll(full)") for (uint r = 0; r < MR; ++r)
                    acc[r] += xt[r][4 * p] * q0 + xt[r][4 * p + 1] * q1 + xt[r][4 * p + 2] * q2 + xt[r][4 * p + 3] * q3;
            }
            _Pragma("clang loop unroll(full)") for (uint r = 0; r < MR; ++r) result[r][rr] += s * acc[r] + b * xsum[r];
        }
    }
    _Pragma("clang loop unroll(full)") for (uint r = 0; r < MR; ++r) for (uint rr = 0; rr < 4; ++rr) {
        const float v = simd_sum(result[r][rr]);
        if (lane == 0 && r0 + rr < uint(N)) y[ulong(r) * N + r0 + rr] = static_cast<OT>(v);
    }
"""


def q4fma(x, w, scales, biases):
    """y [R, N] for R in 1..4 rows (K % 512 == 0): qmv_fast-style lanes (4 output rows per simdgroup, 16 k per lane, 256 B contiguous per row), R accumulators
    per row, one x conversion shared by the 4 rows. x [R, K] bf16/fp16."""
    R, K = x.shape
    N = w.shape[0]
    assert K % 512 == 0 and R <= 4 and N % 8 == 0
    if "fma" not in _k:
        _k["fma"] = mx.fast.metal_kernel(name="kyojin_q4fma", input_names=["x", "w", "sc", "bi"], output_names=["y"], source=_SRC_FMA)
    return _k["fma"](inputs=[x, w, scales, biases], template=[("K", K), ("N", N), ("MR", R), ("OT", scales.dtype), ("SCT", scales.dtype), ("XT", x.dtype)],
                     grid=(N // 8 * 64, 1, 1), threadgroup=(64, 1, 1), output_shapes=[(R, N)], output_dtypes=[scales.dtype])[0]



_SRC3 = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint warp = simdgroup_index_in_threadgroup;
    const uint tid = thread_position_in_threadgroup.x;
    const uint blk = threadgroup_position_in_grid.x;
    const uint fm = ((lane >> 4) & 1u) * 4u + ((lane >> 1) & 3u);
    const uint c = ((lane >> 3) & 1u) * 2u + (lane & 1u);
    const uint fn = c * 2u;
    constexpr uint NBLK = 4 / KS, KW = K / 8, NG = K / 64, KB = K / 512;
    const uint kslice = warp % KS, nb = warp / KS;
    const uint n0 = (blk * NBLK + nb) * (8u * NT);

    simdgroup_matrix<float, 8, 8> acc[NT][RB];
    uint rowt[NT];
    _Pragma("clang loop unroll(full)") for (uint t = 0; t < NT; ++t) {
        _Pragma("clang loop unroll(full)") for (uint rb = 0; rb < RB; ++rb) acc[t][rb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        rowt[t] = min(n0 + 8u * t + fm, uint(N) - 1u);
    }
    for (uint kb = kslice; kb < KB; kb += KS) {
        uint4 wv[NT][4]; half2 s2[NT][2], b2[NT][2];
        _Pragma("clang loop unroll(full)") for (uint t = 0; t < NT; ++t) {
            const device uint4* wp = (const device uint4*)(w + ulong(rowt[t]) * KW + kb * 64u) + c * 4u;
            _Pragma("clang loop unroll(full)") for (uint m = 0; m < 4; ++m) wv[t][m] = wp[m];
            const ulong go = ulong(rowt[t]) * NG + kb * 8u + 2u * c;
            _Pragma("clang loop unroll(full)") for (uint h = 0; h < 2; ++h) {
                s2[t][h] = half2(half(float(sc[go + h])));
                b2[t][h] = half2(half(float(bi[go + h])));
            }
        }
        const device half* xb = xt + ulong(kb * 512u + 128u * (fm >> 1) + 4u * (fm & 1u)) * (8u * RB) + fn;
        _Pragma("clang loop unroll(full)") for (uint m = 0; m < 4; ++m) {
            _Pragma("clang loop unroll(full)") for (uint q = 0; q < 4; ++q) {
                _Pragma("clang loop unroll(full)") for (uint j = 0; j < 4; ++j) {
                    simdgroup_matrix<half, 8, 8> B[RB];
                    _Pragma("clang loop unroll(full)") for (uint rb = 0; rb < RB; ++rb) {
                        const half2 bv = *((const device half2*)(xb + (32u * m + 8u * q + j) * (8u * RB) + 8u * rb));
                        B[rb].thread_elements()[0] = bv.x; B[rb].thread_elements()[1] = bv.y;
                    }
                    _Pragma("clang loop unroll(full)") for (uint t = 0; t < NT; ++t) {
                        const uint u = wv[t][m][q];
                        const half2 h = as_type<half2>(((u >> (4u * j)) & 0x000F000Fu) | 0x64006400u) - half2(1024.0h);
                        const half2 w2 = fma(s2[t][m >> 1], h, b2[t][m >> 1]);
                        simdgroup_matrix<half, 8, 8> A;
                        A.thread_elements()[0] = w2.x; A.thread_elements()[1] = w2.y;
                        _Pragma("clang loop unroll(full)") for (uint rb = 0; rb < RB; ++rb) simdgroup_multiply_accumulate(acc[t][rb], A, B[rb], acc[t][rb]);
                    }
                }
            }
        }
    }
    if (KS == 1) {
        _Pragma("clang loop unroll(full)") for (uint t = 0; t < NT; ++t) {
            const uint n = n0 + 8u * t + fm;
            if (n < uint(N)) {
                _Pragma("clang loop unroll(full)") for (uint rb = 0; rb < RB; ++rb) {
                    const uint r = 8u * rb + fn;
                    if (r < MR) y[ulong(r) * N + n] = static_cast<OT>(acc[t][rb].thread_elements()[0]);
                    if (r + 1u < MR) y[ulong(r + 1u) * N + n] = static_cast<OT>(acc[t][rb].thread_elements()[1]);
                }
            }
        }
    } else {
        threadgroup float red[4][8 * NT][8 * RB];
        _Pragma("clang loop unroll(full)") for (uint t = 0; t < NT; ++t) {
            _Pragma("clang loop unroll(full)") for (uint rb = 0; rb < RB; ++rb) {
                red[warp][8u * t + fm][8u * rb + fn] = acc[t][rb].thread_elements()[0];
                red[warp][8u * t + fm][8u * rb + fn + 1u] = acc[t][rb].thread_elements()[1];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < NBLK * 64u * NT * RB; idx += 128u) {
            const uint b = idx / (64u * NT * RB), rem = idx % (64u * NT * RB), nl = rem / (8u * RB), r = rem % (8u * RB);
            const uint nn = (blk * NBLK + b) * (8u * NT) + nl;
            if (r < MR && nn < uint(N)) {
                float v = 0.0f;
                _Pragma("clang loop unroll(full)") for (uint s = 0; s < KS; ++s) v += red[b * KS + s][nl][r];
                y[ulong(r) * N + nn] = static_cast<OT>(v);
            }
        }
    }
"""


_SRC4 = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint warp = simdgroup_index_in_threadgroup;
    const uint tid = thread_position_in_threadgroup.x;
    const uint blk = threadgroup_position_in_grid.x;
    const uint fm = ((lane >> 4) & 1u) * 4u + ((lane >> 1) & 3u);
    const uint c = ((lane >> 3) & 1u) * 2u + (lane & 1u);
    const uint fn = c * 2u;
    constexpr uint NBLK = 4 / KS, KW = K / 8, NG = K / 64, KB = K / 512;
    const uint kslice = warp % KS, nb = warp / KS;
    const uint n0 = (blk * NBLK + nb) * (8u * NT);

    simdgroup_matrix<float, 8, 8> acc[NT];
    uint rowt[NT];
    _Pragma("clang loop unroll(full)") for (uint t = 0; t < NT; ++t) {
        acc[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        rowt[t] = min(n0 + 8u * t + fm, uint(N) - 1u);
    }
    for (uint kb = kslice; kb < KB; kb += KS) {
        uint4 wv[NT][4]; half2 s2[NT][2], b2[NT][2];
        _Pragma("clang loop unroll(full)") for (uint t = 0; t < NT; ++t) {
            const device uint4* wp = (const device uint4*)(w + ulong(rowt[t]) * KW + kb * 64u) + c * 4u;
            _Pragma("clang loop unroll(full)") for (uint m = 0; m < 4; ++m) wv[t][m] = wp[m];
            const ulong go = ulong(rowt[t]) * NG + kb * 8u + 2u * c;
            _Pragma("clang loop unroll(full)") for (uint h = 0; h < 2; ++h) {
                s2[t][h] = half2(half(float(sc[go + h])));
                b2[t][h] = half2(half(float(bi[go + h])));
            }
        }
        const device XT* xb0 = x + ulong(fn) * K + kb * 512u + 128u * (fm >> 1) + 4u * (fm & 1u);
        const device XT* xb1 = x + ulong(fn + 1u) * K + kb * 512u + 128u * (fm >> 1) + 4u * (fm & 1u);
        _Pragma("clang loop unroll(full)") for (uint m = 0; m < 4; ++m) {
            _Pragma("clang loop unroll(full)") for (uint q = 0; q < 4; ++q) {
                _Pragma("clang loop unroll(full)") for (uint j = 0; j < 4; ++j) {
                    simdgroup_matrix<half, 8, 8> B;
                    B.thread_elements()[0] = fn < MR ? half(float(xb0[32u * m + 8u * q + j])) : half(0.0f);
                    B.thread_elements()[1] = fn + 1u < MR ? half(float(xb1[32u * m + 8u * q + j])) : half(0.0f);
                    _Pragma("clang loop unroll(full)") for (uint t = 0; t < NT; ++t) {
                        const uint u = wv[t][m][q];
                        const half2 h = as_type<half2>(((u >> (4u * j)) & 0x000F000Fu) | 0x64006400u) - half2(1024.0h);
                        const half2 w2 = fma(s2[t][m >> 1], h, b2[t][m >> 1]);
                        simdgroup_matrix<half, 8, 8> A;
                        A.thread_elements()[0] = w2.x; A.thread_elements()[1] = w2.y;
                        simdgroup_multiply_accumulate(acc[t], A, B, acc[t]);
                    }
                }
            }
        }
    }
    if (KS == 1) {
        _Pragma("clang loop unroll(full)") for (uint t = 0; t < NT; ++t) {
            const uint n = n0 + 8u * t + fm;
            if (n < uint(N)) {
                if (fn < MR) y[ulong(fn) * N + n] = static_cast<OT>(acc[t].thread_elements()[0]);
                if (fn + 1u < MR) y[ulong(fn + 1u) * N + n] = static_cast<OT>(acc[t].thread_elements()[1]);
            }
        }
    } else {
        threadgroup float red[4][8 * NT][8];
        _Pragma("clang loop unroll(full)") for (uint t = 0; t < NT; ++t) {
            red[warp][8u * t + fm][fn] = acc[t].thread_elements()[0];
            red[warp][8u * t + fm][fn + 1u] = acc[t].thread_elements()[1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < NBLK * 64u * NT; idx += 128u) {
            const uint b = idx / (64u * NT), rem = idx % (64u * NT), nl = rem >> 3, r = rem & 7u;
            const uint nn = (blk * NBLK + b) * (8u * NT) + nl;
            if (r < MR && nn < uint(N)) {
                float v = 0.0f;
                _Pragma("clang loop unroll(full)") for (uint s = 0; s < KS; ++s) v += red[b * KS + s][nl][r];
                y[ulong(r) * N + nn] = static_cast<OT>(v);
            }
        }
    }
"""


def q4mma3(xt, w, scales, biases, rows, nt=4, ks=1):
    """v3: 64 B contiguous per lane (256 B per row per 512 k), full per-weight dequant (half2 fma), 4 MMA slabs per load. K % 512 == 0."""
    N, K = w.shape[0], w.shape[1] * 8
    assert K % 512 == 0 and rows <= 16 and 4 % ks == 0 and xt.shape == (K, 8 if rows <= 8 else 16)
    if "mma3" not in _k:
        _k["mma3"] = mx.fast.metal_kernel(name="kyojin_q4mma3", input_names=["w", "sc", "bi", "xt"], output_names=["y"], source=_SRC3,
                                          header="#include <metal_simdgroup_matrix>\n")
    per = 8 * nt * (4 // ks)
    ntg = (N + per - 1) // per
    return _k["mma3"](inputs=[w, scales, biases, xt], template=[("K", K), ("N", N), ("NT", nt), ("KS", ks), ("MR", rows), ("RB", 1 if rows <= 8 else 2), ("OT", scales.dtype)],
                      grid=(ntg * 128, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(rows, N)], output_dtypes=[scales.dtype])[0]


def q4mma4(x, w, scales, biases, rows, nt=1, ks=1):
    """v4 = v3 reading x [rows, K] (bf16/fp16) directly: no prep launch."""
    N, K = w.shape[0], w.shape[1] * 8
    assert K % 512 == 0 and rows <= 8 and 4 % ks == 0 and x.shape == (rows, K)
    if "mma4" not in _k:
        _k["mma4"] = mx.fast.metal_kernel(name="kyojin_q4mma4", input_names=["w", "sc", "bi", "x"], output_names=["y"], source=_SRC4,
                                          header="#include <metal_simdgroup_matrix>\n")
    per = 8 * nt * (4 // ks)
    ntg = (N + per - 1) // per
    return _k["mma4"](inputs=[w, scales, biases, x], template=[("K", K), ("N", N), ("NT", nt), ("KS", ks), ("MR", rows), ("OT", scales.dtype), ("XT", x.dtype)],
                      grid=(ntg * 128, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(rows, N)], output_dtypes=[scales.dtype])[0]


# ------------------------------------------------------------------------------------------------ NAX (M5 neural accelerator) few-row GEMM
# Rows x K activations (k permuted, zero padded to the M tile) are the left input of a `matmul2d` (MT x 32 x 64); the right input is the
# weight slab of 32 output columns, dequantized in registers straight into the right-input cooperative tensor (the same half dequant as
# q4mma3: fma(scale, q, bias) in half), so the decode ALU runs once per weight for every row and the MMA work leaves the shader ALUs.
# Right-input layout (M5, tools/rows101_layout.py): element i = 16 s + 4 j + c of a 64-k step holds (k' = 16 s + 4 G + c, n = nb + 8 j),
# G, nb from get_multidimensional_index(0). A chunk of 128 U k: lane G owns the physical k range [32 U G, 32 U (G + 1)) of each of its
# 4 rows (U uint4 loads per row), step t < 2 U, word 2 (t % 2) + s / 2 of uint4 t / 2, nibble pair (q, q + 4), q = 2 (s & 1) + c / 2:
#     k' = 128 U ch + 64 t + 16 s + 4 G + c   <->   k = 128 U ch + 32 U G + 32 (t / 2) + 8 (2 (t % 2) + s / 2) + 4 (c & 1) + (c >> 1) + 2 (s & 1)
HEADER_NAX = r"""
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
#define RA mpp::tensor_ops::__tensor_ops_detail::__remove_addrspace_t
"""
_PREPN = r"""
    const uint kp = thread_position_in_grid.x, r = thread_position_in_grid.y;
    if (kp >= uint(K)) return;
    const uint ch = kp / (128u * U), l = kp % (128u * U);
    const uint t = l >> 6, s = (l >> 4) & 3u, G = (l >> 2) & 3u, c = l & 3u;
    const uint k = ch * 128u * U + 32u * U * G + 32u * (t >> 1) + 8u * (2u * (t & 1u) + (s >> 1)) + 4u * (c & 1u) + (c >> 1) + 2u * (s & 1u);
    xp[ulong(r) * K + kp] = r < uint(MR) ? half(x[ulong(r) * K + k]) : half(0.0f);
"""
_SRCN = r"""
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_position_in_threadgroup.x;
    constexpr uint KW = K / 8, NG = K / 64, CH = K / (128 * U), U2 = U >= 2 ? U / 2 : 1;
    const uint nbk = sg / KS, ksi = sg % KS;
    const uint n0 = (threadgroup_position_in_grid.x * NB + nbk) * 32u;
    const uint m0 = threadgroup_position_in_grid.y * MT;
    constexpr auto desc = matmul2d_descriptor(MT, 32, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tA((device half*)(xp + ulong(m0) * K), dextents<int32_t, 2>(int(K), int(MT)));
    auto rB = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto cT = op.template get_destination_cooperative_tensor<RA<decltype(tA)>, RA<decltype(rB)>, float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;
    const ushort i0 = 0;
    const auto ix0 = rB.get_multidimensional_index(i0);
    const uint G = uint(ix0[0]) >> 2, nb = uint(ix0[1]);
    uint rowj[4];
    _Pragma("clang loop unroll(full)") for (uint j = 0; j < 4; ++j) rowj[j] = min(n0 + nb + 8u * j, uint(N) - 1u);
    uint4 wn[4][U]; float sn[4][U2], bn[4][U2];
#define Q4N_LOAD(CHX) _Pragma("clang loop unroll(full)") for (uint j = 0; j < 4; ++j) { \
            const device uint4* wp = (const device uint4*)(w + ulong(rowj[j]) * KW + (CHX) * 16u * U) + G * U; \
            _Pragma("clang loop unroll(full)") for (uint u = 0; u < U; ++u) wn[j][u] = (DBG & 4) ? uint4((CHX) * 7u + u, j, G, nb) : wp[u]; \
            const ulong go = ulong(rowj[j]) * NG + (CHX) * 2u * U + (U * G) / 2u; \
            _Pragma("clang loop unroll(full)") for (uint h = 0; h < U2; ++h) { sn[j][h] = float(sc[go + h]); bn[j][h] = float(bi[go + h]); } }
    if (PF && ksi < CH) { Q4N_LOAD(ksi) }
    for (uint ch = ksi; ch < CH; ch += KS) {
        if (!PF) { Q4N_LOAD(ch) }
        uint4 wv[4][U]; half2 s2[4][U2], b2[4][U2];
        _Pragma("clang loop unroll(full)") for (uint j = 0; j < 4; ++j) {
            _Pragma("clang loop unroll(full)") for (uint u = 0; u < U; ++u) wv[j][u] = wn[j][u];
            _Pragma("clang loop unroll(full)") for (uint h = 0; h < U2; ++h) { s2[j][h] = half2(half(sn[j][h])); b2[j][h] = half2(half(bn[j][h])); }
        }
        if (PF && ch + KS < CH) { Q4N_LOAD(ch + KS) }
        if (DBG & 8) { uint a = 0; _Pragma("clang loop unroll(full)") for (uint j = 0; j < 4; ++j) { _Pragma("clang loop unroll(full)") for (uint u = 0; u < U; ++u) a ^= wv[j][u].x ^ wv[j][u].y ^ wv[j][u].z ^ wv[j][u].w; a ^= as_type<uint>(sn[j][0]) ^ as_type<uint>(bn[j][0]); } cT[i0] += float(a & 7u); continue; }
        _Pragma("clang loop unroll(full)") for (uint t = 0; t < 2 * U; ++t) {
            _Pragma("clang loop unroll(full)") for (uint s = 0; s < 4; ++s) {
                _Pragma("clang loop unroll(full)") for (uint j = 0; j < 4; ++j) {
                    const uint u = wv[j][t >> 1][2 * (t & 1) + (s >> 1)];
                    _Pragma("clang loop unroll(full)") for (uint cp = 0; cp < 2; ++cp) {
                        const uint q = 2 * (s & 1) + cp;
                        const half2 h = (DBG & 2) ? as_type<half2>(u ^ q) : as_type<half2>(((u >> (4u * q)) & 0x000F000Fu) | 0x64006400u) - half2(1024.0h);
                        const half2 w2 = (DBG & 2) ? h : fma(s2[j][t >> 2], h, b2[j][t >> 2]);
                        rB[16 * s + 4 * j + 2 * cp] = w2.x; rB[16 * s + 4 * j + 2 * cp + 1] = w2.y;
                    }
                }
            }
            auto sA = tA.slice((DBG & 16) ? 0 : int((ch * 2u * U + t) * 64u), 0);
            if (DBG & 1) { cT[i0] += float(rB[0]) + float(rB[63]); } else op.run(sA, rB, cT);
        }
    }
    if (KS == 1) {
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) {
            const auto ix = cT.get_multidimensional_index(i);
            const uint n = n0 + uint(ix[0]), m = m0 + uint(ix[1]);
            if (m < uint(MR) && n < uint(N)) y[ulong(m) * N + n] = static_cast<OT>(cT[i]);
        }
    } else {
        threadgroup float red[NB * KS][MT * 32];
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) {
            const auto ix = cT.get_multidimensional_index(i);
            red[sg][uint(ix[1]) * 32u + uint(ix[0])] = cT[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < NB * MT * 32u; idx += 32u * NB * KS) {
            const uint b = idx / (MT * 32u), rem = idx % (MT * 32u), m = m0 + rem / 32u, n = (threadgroup_position_in_grid.x * NB + b) * 32u + rem % 32u;
            if (m < uint(MR) && n < uint(N)) {
                float v = 0.0f;
                _Pragma("clang loop unroll(full)") for (uint s = 0; s < KS; ++s) v += red[b * KS + s][rem];
                y[ulong(m) * N + n] = static_cast<OT>(v);
            }
        }
    }
"""


def nax_mt(rows: int) -> int:
    return 16 if rows <= 16 else 32


def prep_xn(x: mx.array, u: int = 2) -> mx.array:
    """x [..., K] -> half [Mp, K]: k permuted for q4nax (chunks of 128 u), rows zero padded to the M tile (16 up to 16 rows, else 32)."""
    K = x.shape[-1]
    R = x.size // K
    mt = nax_mt(R)
    mp = (R + mt - 1) // mt * mt
    if "prepn" not in _k:
        _k["prepn"] = mx.fast.metal_kernel(name="kyojin_q4nax_prep", input_names=["x"], output_names=["xp"], source=_PREPN)
    return _k["prepn"](inputs=[x.reshape(R, K)], template=[("K", K), ("MR", R), ("U", u)], grid=(K, mp, 1), threadgroup=(256, 1, 1),
                       output_shapes=[(mp, K)], output_dtypes=[mx.float16])[0]


def q4nax(xp: mx.array, w: mx.array, scales: mx.array, biases: mx.array, rows: int, u: int = 2, nb: int = 4, ks: int = 1, dbg: int = 0, pf: int = 1) -> mx.array:
    """y [rows, N] = x @ dequant(w).T on the NAX matmul; xp = prep_xn(x, u). nb column slabs x ks k-splits = simdgroups per threadgroup.
    pf = register prefetch of the next k chunk; dbg (ablation only, wrong results): 1 no MMA, 2 no dequant, 4 synthetic weights, 8 loads only, 16 x slice pinned at k = 0."""
    N, K = w.shape[0], w.shape[1] * 8
    mt = nax_mt(rows)
    assert K % (128 * u) == 0 and N % 32 == 0 and scales.shape[1] == K // 64 and xp.shape[1] == K and xp.shape[0] >= rows
    key = "nax"
    if key not in _k:
        _k[key] = mx.fast.metal_kernel(name="kyojin_q4nax", input_names=["w", "sc", "bi", "xp"], output_names=["y"], source=_SRCN, header=HEADER_NAX)
    ntg = (N + 32 * nb - 1) // (32 * nb)
    return _k[key](inputs=[w, scales, biases, xp], template=[("K", K), ("N", N), ("U", u), ("NB", nb), ("KS", ks), ("MT", mt), ("MR", rows), ("OT", scales.dtype), ("DBG", dbg), ("PF", pf)],
                   grid=(ntg * 32 * nb * ks, (rows + mt - 1) // mt, 1), threadgroup=(32 * nb * ks, 1, 1), output_shapes=[(rows, N)], output_dtypes=[scales.dtype])[0]


# ------------------------------------------------------------------------------------------------ model integration
import os

import mlx.nn as nn

_ENV_MIN = os.environ.get("KYOJIN_Q4MMA_MIN_ROWS")
_DEFAULT_MIN = 5
MIN_ROWS = int(_ENV_MIN or _DEFAULT_MIN)            # default crossover; an explicit env value overrides the per-shape table below
_TUNE: dict = {}                           # (N, K) -> smallest row count where q4mma beats MLX's quantized matmul on this chip (tools/q4mma_tune.py)


def device_name() -> str:
    import mlx.core as mx
    try:
        return str((mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()).get("device_name", ""))
    except Exception:
        return ""


_TUNE_NAX: dict = {}                                # (N, K) -> (lo, hi, u, nb, ks): rows served by q4nax and its launch shape


def load_tuning(path: str | None = None) -> int:
    """Read the per-shape row window of THIS chip: {"devices": {<metal device name>: {"min_rows": {"NxK": lo or [lo, hi]}}}} from path,
    $KYOJIN_Q4MMA_TUNING, or kyojin_mlx/q4mma_tuning.json next to this file. q4mma serves lo <= rows <= hi of that shape (lo > hi: never);
    a bare int is the legacy 8-row cap (hi = 8). Another chip gets no entries (MIN_ROWS..DEFAULT_MAX everywhere) until tools/q4mma_tune.py
    has been run on it. "nax": {"NxK": [lo, hi, u, nb, ks]} gives the rows served by q4nax (checked first; chips with neural accelerators);
    an untuned shape on such a chip gets NAX_DEFAULT. Returns the entry count."""
    import json
    path = path or os.environ.get("KYOJIN_Q4MMA_TUNING") or os.path.join(os.path.dirname(__file__), "q4mma_tuning.json")
    _TUNE.clear(); _TUNE_NAX.clear()
    if os.path.exists(path):
        for k, v in json.load(open(path)).get("devices", {}).get(device_name(), {}).get("min_rows", {}).items():
            n, kk = k.split("x")
            _TUNE[(int(n), int(kk))] = (int(v[0]), int(v[1])) if isinstance(v, list) else (int(v), 8)
        for k, v in json.load(open(path)).get("devices", {}).get(device_name(), {}).get("nax", {}).items():
            n, kk = k.split("x")
            _TUNE_NAX[(int(n), int(kk))] = tuple(int(e) for e in v)
    return len(_TUNE)


MAX_ROWS = 16                                       # kernel limit (two 8-column B tiles)
DEFAULT_MAX = 8                                     # untuned shapes / chips: the 8-row window measured on the M5 Air before the 16-row kernel
_ENV_MAX = os.environ.get("KYOJIN_Q4MMA_MAX_ROWS")  # explicit cap on every shape (A/B switch), clipped to MAX_ROWS


def rows_for(N: int, K: int) -> tuple[int, int]:
    """(lo, hi): the row counts this shape sends to q4mma (env overrides first, then the chip table, then the defaults)."""
    lo, hi = _TUNE.get((N, K), (MIN_ROWS, DEFAULT_MAX))
    if _ENV_MIN is not None or MIN_ROWS != _DEFAULT_MIN:
        lo = MIN_ROWS
    if _ENV_MAX is not None:
        hi = min(hi, int(_ENV_MAX)) if (N, K) in _TUNE else int(_ENV_MAX)
    return lo, min(hi, MAX_ROWS)


NAX_MAX_ROWS = 32                                   # q4nax limit (two 16-row M tiles; above: MLX qmm)
NAX_DEFAULT = (9, 32, 2, 2, 1)                      # untuned shape on a NAX chip: where the 8x8 simdgroup MMA doubles its work (M5 Air table, rows101)
_NAX_ON: list = []


def nax_on() -> bool:
    if not _NAX_ON:
        on = os.environ.get("KYOJIN_Q4NAX", "auto") != "0"
        if on:
            try:
                from kyojin_mlx import metal_nax
                on = metal_nax.nax_available()
            except Exception:
                on = False
        _NAX_ON.append(on)
    return _NAX_ON[0]


def nax_cfg(N: int, K: int, R: int):
    """(u, nb, ks) when q4nax serves R rows of this shape on this chip, else None."""
    if R > NAX_MAX_ROWS or N % 32 or not nax_on():
        return None
    lo, hi, u, nb, ks = _TUNE_NAX.get((N, K), NAX_DEFAULT)
    return (u, nb, ks) if lo <= R <= hi and K % (128 * u) == 0 else None


def min_rows_for(N: int, K: int) -> int:
    return rows_for(N, K)[0]


def serves(N: int, K: int, R: int) -> bool:
    lo, hi = rows_for(N, K)
    return lo <= R <= hi


load_tuning()
NT, KS = 1, 1
ABL = int(os.environ.get('Q4_ABL', '0'))
_memo_d: dict = {}
_memo: list = [None, None]


def _xt(x):
    if _memo[0] is x:
        return _memo[1]
    xt = prep_x(x)
    _memo[0], _memo[1] = x, xt
    return xt


_memo_n: list = [None, 0, None]


def _xn(x, u):
    if _memo_n[0] is x and _memo_n[1] == u:
        return _memo_n[2]
    xp = prep_xn(x, u)
    _memo_n[0], _memo_n[1], _memo_n[2] = x, u, xp
    return xp


def rows_of(x) -> int:
    return x.size // x.shape[-1]


class Q4Linear(nn.Module):
    """Drop-in for an affine 4-bit group-64 nn.QuantizedLinear: the NAX rows window (nax_cfg) through q4nax, the rows window of the shape
    (rows_for) through q4mma, anything else through the original module."""

    def __init__(self, lin):
        super().__init__()
        self.weight, self.scales, self.biases = lin.weight, lin.scales, lin.biases
        if "bias" in lin:
            self.bias = lin.bias
        self.__dict__["_orig"] = lin
        self.__dict__["_K"] = lin.weight.shape[1] * 8

    def __call__(self, x):
        R = rows_of(x)
        c = nax_cfg(self.weight.shape[0], self._K, R) if x.dtype in (mx.bfloat16, mx.float16) else None
        if c is not None:
            y = q4nax(_xn(x, c[0]), self.weight, self.scales, self.biases, R, *c).reshape(*x.shape[:-1], self.weight.shape[0])
            return y + self.bias if "bias" in self else y
        if serves(self.weight.shape[0], self._K, R) and x.dtype in (mx.bfloat16, mx.float16):
            if ABL == 1:
                _xt(x)
                return mx.zeros((*x.shape[:-1], self.weight.shape[0]), dtype=x.dtype)
            if ABL == 2:
                if "abl" not in _memo_d: _memo_d["abl"] = prep_x(x)
                _memo[0], _memo[1] = x, _memo_d["abl"]
            y = q4mma3(_xt(x), self.weight, self.scales, self.biases, R, NT, KS).reshape(*x.shape[:-1], self.weight.shape[0])
            return y + self.bias if "bias" in self else y
        return self._orig(x)


class Q4Head(nn.Module):
    """Tied output head (quantized embedding as_linear) with the same row routing."""

    def __init__(self, emb):
        super().__init__()
        self.__dict__["_e"] = emb

    def __call__(self, h):
        e = self._e
        R = rows_of(h)
        N, K = e.weight.shape[0], e.weight.shape[1] * 8
        c = nax_cfg(N, K, R) if h.dtype in (mx.bfloat16, mx.float16) else None
        if c is not None:
            return q4nax(_xn(h, c[0]), e.weight, e.scales, e.biases, R, *c).reshape(*h.shape[:-1], N)
        if serves(e.weight.shape[0], e.weight.shape[1] * 8, R) and h.dtype in (mx.bfloat16, mx.float16):
            return q4mma3(_xt(h), e.weight, e.scales, e.biases, R, NT, KS).reshape(*h.shape[:-1], e.weight.shape[0])
        return e.as_linear(h)


def _ok(m) -> bool:
    return (isinstance(m, nn.QuantizedLinear) and m.bits == 4 and m.group_size == 64 and getattr(m, "mode", "affine") == "affine"
            and (m.weight.shape[1] * 8) % 512 == 0)


def install(model) -> int:
    """Swap every eligible quantized linear of the language model (and the tied head) for the few-row kernel. Returns the count."""
    from mlx.utils import tree_unflatten
    lm = model.language_model if hasattr(model, "language_model") else model
    import re
    pat = re.compile(os.environ.get('Q4_ONLY', '.'))
    mods = [(n, Q4Linear(m)) for n, m in lm.named_modules() if _ok(m) and pat.search(n)]
    lm.update_modules(tree_unflatten(mods))
    emb = lm.model.embed_tokens
    tied = getattr(getattr(lm, "args", None), "tie_word_embeddings", True) and "lm_head" not in lm   # untied: the lm_head linear was swapped above
    if tied and isinstance(emb, nn.QuantizedEmbedding) and emb.bits == 4 and emb.group_size == 64 and pat.search('lm_head'):
        lm.__dict__["lm_head"] = Q4Head(emb)
        mods.append(("lm_head", None))
    return len(mods)


def install_mtp(mtp) -> int:
    from mlx.utils import tree_unflatten
    mods = [(n, Q4Linear(m)) for n, m in mtp.named_modules() if _ok(m)]
    mtp.update_modules(tree_unflatten(mods))
    return len(mods)
