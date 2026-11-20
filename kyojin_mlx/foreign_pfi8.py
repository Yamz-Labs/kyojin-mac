# SPDX-License-Identifier: Apache-2.0
"""int8 x int8 -> int32 prefill GEMM for the DENSE linears (any stored format: FP8 / FP8 block / NVFP4 / affine / plain fp16) on the M5 Neural Accelerator
(`mpp::tensor_ops::matmul2d`, int8 operands: 116-145 TOPS against 86-99 TFLOPS for the fp16 matmul MLX runs on the same shapes). Flag `KYOJIN_PF_GEMM_I8_DENSE`
(default off, read per call; `kyojin_mlx.flags` entry `pf_gemm_i8_dense`). The routed-expert counterpart is `foreign_pfq8` (`KYOJIN_PF_GEMM_I8`).

Weight side: NO resident copy (rule: no second resident copy of weight data). Every call decodes the stored weight to fp16 (`DevPack.dequant()`, the kernel the fp16 prefill
path runs anyway) and quantizes it in ONE row-quant launch: ONE fp32 scale per output row (amax / 127 over the whole K reduction, round half even). The only approximation is this
8-bit re-grid. The int8 weight + scale are graph temporaries of the call (n * k bytes + 4 n), freed with the call; the extra cost over the fp16 path is one read of the fp16
decode and one write of n * k bytes, amortised over the rows of the chunk.

Activation side: dynamic int8, one fp32 scale per (row, block of QB = 128 consecutive K) exactly like foreign_pfq8 (`quant_rows`, one launch): the kernel runs the matmul2d of
every block into an int32 partial (exact: 128 * 127 * 127) and sums partial * xs[row, block] in fp32, so an outlier channel only costs the 127 other elements of its block.
Epilogue: acc * wscale[col], written as fp16 (clamped to +-65504).

Everything but `linear` / `_kern` is plain mx ops (CPU-testable, see tests/test_pfi8.py)."""
from __future__ import annotations

import os

import mlx.core as mx

from . import foreign_pfq8 as Q8
from . import kernels as KN

BM, BK = 128, 128


def on() -> bool:
    """Flag KYOJIN_PF_GEMM_I8_DENSE (default off): dense prefill linears of >= KYOJIN_PF_GEMM_I8_MIN rows run int8 x int8 -> int32."""
    return os.environ.get("KYOJIN_PF_GEMM_I8_DENSE", "0") not in ("0", "")


def min_rows() -> int:
    """KYOJIN_PF_GEMM_I8_MIN (default 2048): below this the fp16 matmul keeps the call (the per-tile fixed cost does not amortize, the MTP / verify rows stay exact)."""
    return int(os.environ.get("KYOJIN_PF_GEMM_I8_MIN", "2048"))


def hook_rows() -> int:
    """Cheap pre-check of the DevPack hooks: min(1024, KYOJIN_PF_GEMM_I8_MIN). Default behaviour unchanged (1024 < 2048, `why_not` decides); lowering KYOJIN_PF_GEMM_I8_MIN lets short prompts through (quality stress runs)."""
    return min(1024, min_rows())


def min_nk() -> int:
    """KYOJIN_PF_GEMM_I8_MINNK (default 4M weights): smaller matrices (router, compressors, indexer projections) keep the fp16 matmul."""
    return int(os.environ.get("KYOJIN_PF_GEMM_I8_MINNK", str(1 << 22)))


def min_n() -> int:
    """KYOJIN_PF_GEMM_I8_MINN (default 4096): narrower outputs lose to the fp16 matmul (2048 x 4096 x0.93 incl. quant_rows)."""
    return int(os.environ.get("KYOJIN_PF_GEMM_I8_MINN", "4096"))


def min_k() -> int:
    """KYOJIN_PF_GEMM_I8_MINK (default 4096): K = 2048 inputs (the down projections) lose (4096 x 2048 x0.62: the per-block scale flush and quant_rows do not amortize)."""
    return int(os.environ.get("KYOJIN_PF_GEMM_I8_MINK", "4096"))


def why_not(dev, R: int) -> str:
    """Flag audit: why `linear` does not take (dev, R rows) ('' = it takes it)."""
    if not on():
        return "flag_off"
    if getattr(dev, "E", None):
        return "experts"
    if R < min_rows():
        return "rows<min"
    n, k = int(dev.n), int(dev.k)
    if n % 64 or k % 256 or k % BK:
        return "shape"
    if n * k < min_nk() or n < min_n() or k < min_k():
        return "small"
    return ""


def quant_weight(w):
    """w half / float [n, k] -> (int8 [n, k], float32 scale [n]): scale = row amax / 127 (1 for a zero row), round half to even; codes * scale ~= w."""
    wf = w.astype(mx.float32)
    amax = mx.max(mx.abs(wf), axis=-1)
    sc = mx.where(amax > 0, amax / 127.0, mx.ones_like(amax))
    q = mx.clip(mx.round(wf / sc[:, None]), -127, 127).astype(mx.int8)
    return q, sc


def weights(dev):
    """(int8 [n, k], fp32 [n]) of a DevPack's dequantized weight, built ON THE FLY for ONE call (rule: no second resident copy of the weight data). The stored weight is decoded
    to fp16 by the same kernel the fp16 prefill path runs (dense packs: their resident fp16), then ONE row-quant launch (foreign_pfq8.quant_rows with qb == k: per-row amax / 127, round half
    even, == quant_weight) writes the int8 codes and the row scale. The caller frees both after the GEMM (transient n * k bytes + the fp16 decode, which the fp16 path pays too)."""
    n, k = int(dev.n), int(dev.k)
    full = dev.w if dev.fmt == "dense" else dev.dequant()
    if full.dtype != mx.float16:
        full = full.astype(mx.float16)
    q, s = Q8.quant_rows(full, k)
    return q, s.reshape(n)


_DENSE_SRC = r"""
    threadgroup uint4 Wsv[2 * 4 * BK];
    threadgroup int8_t* Ws = (threadgroup int8_t*)Wsv;
    const uint ti = threadgroup_position_in_grid.y, c = threadgroup_position_in_grid.x;
    if (ti >= tb[0]) return;
    const uint rs = ti * uint(BM);
    const int rows = int(min(uint(BM), off[0] - rs));
    const uint sgid = simdgroup_index_in_threadgroup;
    const uint lid = sgid * 32u + thread_index_in_simdgroup;
    const uint nthr = 2u * uint(BM);
    const short tm = short(32 * (sgid / 2u)), tn = short(32 * (sgid % 2u));
    const int rsg = max(0, min(32, rows - int(tm)));
    const uint wpr = uint(BK) / 16u;
    const uint c0 = c * 64u;
    constexpr auto desc = matmul2d_descriptor(32, 32, BK, false, true, false, QBS == 1 ? matmul2d_descriptor::mode::multiply : matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tA((device int8_t*)(x + ulong(rs + uint(tm)) * uint(K)), dextents<int32_t, 2>(K, max(rsg, 1)));
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tB((threadgroup int8_t*)Ws, dextents<int32_t, 2>(BK, 128));
    auto sA0 = tA.slice(0, 0);
    auto sB0 = tB.slice(0, int(tn));
    auto cT = op.template get_destination_cooperative_tensor<decltype(sA0), decltype(sB0), int32_t>();
    constexpr uint CAPN = 32u;
    const uint16_t cap = cT.get_capacity();
    float acc[CAPN];
    ushort rloc[CAPN];
    threadgroup float xsT[2 * BM];
    for (uint i = 0; i < CAPN; ++i) {
        acc[i] = 0.0f; rloc[i] = 0;
        if (i < cap && cT.is_valid_element(uint16_t(i))) { cT[i] = 0; rloc[i] = ushort(tm) + ushort(cT.get_multidimensional_index(uint16_t(i))[1]); }
    }
    #define PFI8_XS(BLK) if (lid < uint(BM)) xsT[((BLK) & 1) * BM + lid] = int(lid) < rows ? xs[ulong(rs + lid) * uint(KB) + uint(BLK)] : 0.0f;
    PFI8_XS(0)
    constexpr uint ITEMS = (64u * uint(BK) / 16u + 2u * uint(BM) - 1u) / (2u * uint(BM));
    uint4 wq[ITEMS];
    #define PFI8_LD(KB_) \
        for (uint u = 0; u < ITEMS; ++u) { \
            const uint it = lid + u * nthr; \
            if (it < 64u * wpr) { \
                wq[u] = *(const device uint4*)(w + ulong(c0 + it / wpr) * uint(K) + uint(KB_) * uint(BK) + (it % wpr) * 16u); \
            } \
        }
    #define PFI8_ST(BUF) \
        for (uint u = 0; u < ITEMS; ++u) { \
            const uint it = lid + u * nthr; \
            if (it < 64u * wpr) Wsv[(uint(BUF) * 64u + it / wpr) * wpr + (it % wpr)] = wq[u]; \
        }
    PFI8_LD(0)
    PFI8_ST(0)
    if (K / BK > 1) { PFI8_LD(1) }
    for (int kb = 0; kb < K / BK; ++kb) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kb % QBS == 0 && kb / QBS + 1 < KB) { PFI8_XS(kb / QBS + 1) }
        if (rsg > 0) {
            auto sA = tA.slice(kb * BK, 0);
            auto sB = tB.slice(0, int(tn) + 64 * (kb & 1));
            op.run(sA, sB, cT);
        }
        if (kb + 1 < K / BK) { PFI8_ST((kb + 1) & 1) }
        if (kb + 2 < K / BK) { PFI8_LD(kb + 2) }
        if ((kb + 1) % QBS == 0) {
            const uint xb = ((kb / QBS) & 1) * uint(BM);
            for (uint i = 0; i < CAPN; ++i) {
                if (i < cap) { acc[i] += float(cT[i]) * xsT[xb + rloc[i]]; if (QBS > 1) cT[i] = 0; }
            }
        }
    }
    if (rsg == 0) return;
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (!cT.is_valid_element(i)) continue;
        const auto ix = cT.get_multidimensional_index(i);
        const uint col = uint(ix[0]), r = uint(ix[1]);
        if (int(r) >= rsg) continue;
        const uint rr = rs + uint(tm) + r, cc = c0 + uint(tn) + col;
        y[ulong(rr) * uint(N) + cc] = half(clamp(acc[i] * wsc[cc], -65504.0f, 65504.0f));
    }
"""


def _kern():
    key = ("pfi8_dense", 1)
    k = KN._cache.get(key)
    if k is None:
        k = KN._cache[key] = mx.fast.metal_kernel(name="kyojin_pfi8_dense", input_names=["x", "w", "wsc", "xs", "tb", "off"], output_names=["y"], header=Q8._hdr(), source=_DENSE_SRC)
    return k


_TBOFF: dict = {}


def _tb_off(S: int):
    v = _TBOFF.get(S)
    if v is None:
        import numpy as np
        v = _TBOFF[S] = (mx.array(np.array([-(-S // BM), 0], np.uint32)), mx.array(np.array([S, 0], np.uint32)))
    return v


def gemm_i8(x_i8, x_scale, w_i8, w_scale, qb: int = BK):
    """x_i8 int8 [S, k] (natural K order), x_scale float32 [S, k / qb], w_i8 int8 [n, k], w_scale float32 [n] -> half [S, n] ~= (x * xs) @ (w * ws).T"""
    S, K = int(x_i8.shape[0]), int(x_i8.shape[1])
    N = int(w_i8.shape[0])
    assert N % 64 == 0 and K % 256 == 0 and qb % BK == 0
    tb, off = _tb_off(S)
    grid = ((N // 64) * 32, -(-S // BM) * 2, BM // 32)
    return _kern()(inputs=[x_i8, w_i8, w_scale, x_scale, tb, off], template=[("BM", BM), ("BK", BK), ("N", N), ("K", K), ("QBS", qb // BK), ("KB", K // qb)],
                   grid=grid, threadgroup=(32, 2, BM // 32), output_shapes=[(S, N)], output_dtypes=[mx.float16])[0]


def linear(dev, x2):
    """x2 half [R, k] @ the DevPack's weight.T as fp16 through the int8 GEMM, or None if (dev, R) is not eligible (`why_not`)."""
    R = int(x2.shape[0])
    if why_not(dev, R):
        return None
    if x2.dtype != mx.float16:
        x2 = mx.clip(x2, -65504.0, 65504.0).astype(mx.float16)
    xi, xsc = Q8.quant_rows(x2, BK)
    w8, ws = weights(dev)                                                                  # transient: lives in this call's graph only (see `weights`)
    return gemm_i8(xi, xsc, w8, ws, BK)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# NATIVE Q8_0 x dynamic int8 activations (flag KYOJIN_PF_GEMM_I8_NATIVE, default off): the stored Q8_0 bytes ARE int8 codes with one fp16 scale per 32 weights, so there is no weight re-grid
# (the `weights()` path above rebuilds a per-row int8 weight from the fp16 decode on every call: ~1.1 % rms per GEMM). Activations are quantised per (row, 32 K) (quant_rows qb 32).
# One matmul2d (int8 x int8 -> int32) per 32 K step, then acc[r, c] += float(int32) * xs[r, blk] * d[c, blk] for every element: the flush is per 32 K instead of per 128 K, so the kernel may be
# bound by the flush rather than the MMA (the micro-benchmark decides). Same tile structure as _DENSE_SRC (BM 128 rows x 64 columns, 2 x 4 simdgroups of 32 x 32, weight tile double-buffered).
_NATIVE_SRC = r"""
    threadgroup uint4 Wsv[2 * 4 * BK];
    threadgroup int8_t* Ws = (threadgroup int8_t*)Wsv;
    threadgroup float4 xsT[2][BM];                                  // [row][K step j]: the four 32-K scales of a 128-K block in one float4
    threadgroup float4 wsT[2][64];
    const uint ti = threadgroup_position_in_grid.y, c = threadgroup_position_in_grid.x;
    if (ti >= tb[0]) return;
    const uint rs = ti * uint(BM);
    const int rows = int(min(uint(BM), off[0] - rs));
    const uint sgid = simdgroup_index_in_threadgroup;
    const uint lid = sgid * 32u + thread_index_in_simdgroup;
    const uint nthr = 2u * uint(BM);
    const short tm = short(32 * (sgid / 2u)), tn = short(32 * (sgid % 2u));
    const int rsg = max(0, min(32, rows - int(tm)));
    const uint wpr = uint(BK) / 16u;
    const uint c0 = c * 64u;
    constexpr auto desc = matmul2d_descriptor(32, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tA((device int8_t*)(x + ulong(rs + uint(tm)) * uint(K)), dextents<int32_t, 2>(K, max(rsg, 1)));
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tB((threadgroup int8_t*)Ws, dextents<int32_t, 2>(BK, 128));
    auto sA0 = tA.slice(0, 0);
    auto sB0 = tB.slice(0, int(tn));
    auto cT0 = op.template get_destination_cooperative_tensor<decltype(sA0), decltype(sB0), int32_t>();
    auto cT1 = op.template get_destination_cooperative_tensor<decltype(sA0), decltype(sB0), int32_t>();
    auto cT2 = op.template get_destination_cooperative_tensor<decltype(sA0), decltype(sB0), int32_t>();
    auto cT3 = op.template get_destination_cooperative_tensor<decltype(sA0), decltype(sB0), int32_t>();
    constexpr uint CAPN = 32u;
    const uint16_t cap = cT0.get_capacity();
    float acc[CAPN];
    ushort rloc[CAPN];
    ushort cloc[CAPN];
    for (uint i = 0; i < CAPN; ++i) {
        acc[i] = 0.0f; rloc[i] = 0; cloc[i] = 0;
        if (i < cap && cT0.is_valid_element(uint16_t(i))) {
            const auto ix = cT0.get_multidimensional_index(uint16_t(i));
            rloc[i] = ushort(tm) + ushort(ix[1]);
            cloc[i] = ushort(tn) + ushort(ix[0]);
        }
    }
    #define NQ_XS(BLK) if (lid < uint(BM)) { xsT[(BLK) & 1][lid] = int(lid) < rows ? *(const device float4*)(xs + ulong(rs + lid) * uint(KB32) + uint(BLK) * 4u) : float4(0.0f); }
    #define NQ_WS(BLK) if (lid < 64u) { wsT[(BLK) & 1][lid] = float4(*(const device half4*)(mt + ulong(c0 + lid) * uint(KB32) + uint(BLK) * 4u)); }
    NQ_XS(0)
    NQ_WS(0)
    constexpr uint ITEMS = (64u * uint(BK) / 16u + 2u * uint(BM) - 1u) / (2u * uint(BM));
    uint4 wq[ITEMS];
    #define PFI8_LD(KB_) \
        for (uint u = 0; u < ITEMS; ++u) { \
            const uint it = lid + u * nthr; \
            if (it < 64u * wpr) { \
                wq[u] = *(const device uint4*)(w + ulong(c0 + it / wpr) * uint(K) + uint(KB_) * uint(BK) + (it % wpr) * 16u); \
            } \
        }
    #define PFI8_ST(BUF) \
        for (uint u = 0; u < ITEMS; ++u) { \
            const uint it = lid + u * nthr; \
            if (it < 64u * wpr) Wsv[(uint(BUF) * 64u + it / wpr) * wpr + (it % wpr)] = wq[u]; \
        }
    PFI8_LD(0)
    PFI8_ST(0)
    if (K / BK > 1) { PFI8_LD(1) }
    for (int kb = 0; kb < K / BK; ++kb) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kb + 1 < K / BK) { NQ_XS(kb + 1) NQ_WS(kb + 1) }
        if (rsg > 0) {
            const int tnb = int(tn) + 64 * (kb & 1);
            auto sA0_ = tA.slice(kb * BK, 0);              // lvalues: op.run takes its operands by non-const reference
            auto sB0_ = tB.slice(0, tnb);
            auto sA1_ = tA.slice(kb * BK + 32, 0);
            auto sB1_ = tB.slice(32, tnb);
            auto sA2_ = tA.slice(kb * BK + 64, 0);
            auto sB2_ = tB.slice(64, tnb);
            auto sA3_ = tA.slice(kb * BK + 96, 0);
            auto sB3_ = tB.slice(96, tnb);
            op.run(sA0_, sB0_, cT0);
            op.run(sA1_, sB1_, cT1);
            op.run(sA2_, sB2_, cT2);
            op.run(sA3_, sB3_, cT3);
            for (uint i = 0; i < CAPN; ++i) {
                if (i < cap) {
                    const float4 xr = xsT[kb & 1][rloc[i]];
                    const float4 wc = wsT[kb & 1][cloc[i]];
                    acc[i] += float(cT0[i]) * (xr.x * wc.x) + float(cT1[i]) * (xr.y * wc.y) + float(cT2[i]) * (xr.z * wc.z) + float(cT3[i]) * (xr.w * wc.w);
                }
            }
        }
        if (kb + 1 < K / BK) { PFI8_ST((kb + 1) & 1) }
        if (kb + 2 < K / BK) { PFI8_LD(kb + 2) }
    }
    if (rsg == 0) return;
    for (uint16_t i = 0; i < cT0.get_capacity(); ++i) {
        if (!cT0.is_valid_element(i)) continue;
        const auto ix = cT0.get_multidimensional_index(i);
        const uint col = uint(ix[0]), r = uint(ix[1]);
        if (int(r) >= rsg) continue;
        const uint rr = rs + uint(tm) + r, cc = c0 + uint(tn) + col;
        y[ulong(rr) * uint(N) + cc] = half(clamp(acc[i] * wsc[cc], -65504.0f, 65504.0f));
    }
"""


def native_on() -> bool:
    return os.environ.get("KYOJIN_PF_GEMM_I8_NATIVE", "0") not in ("0", "")


def native_why_not(dev, R: int) -> str:
    if not native_on():
        return "flag_off"
    if getattr(dev, "E", None) or dev.fmt != "q8_0":
        return "fmt"
    if R < int(os.environ.get("KYOJIN_PF_GEMM_I8_NATIVE_MIN", "256")):
        return "rows<min"
    n, k = int(dev.n), int(dev.k)
    if n % 64 or k % 256:
        return "shape"
    return ""


def _native_kern():
    key = ("pfi8_native", 1)
    k = KN._cache.get(key)
    if k is None:
        k = KN._cache[key] = mx.fast.metal_kernel(name="kyojin_pfi8_native", input_names=["x", "w", "mt", "wsc", "xs", "tb", "off"], output_names=["y"], header=Q8._hdr(), source=_NATIVE_SRC)
    return k


def native_gemm(x_i8, xs32, dev):
    """x_i8 int8 [S, k], xs32 float32 [S, k / 32], dev = dense Q8_0 DevPack (qs int8 codes [n, k] as stored, meta half [n, k / 32], gs float [n]) -> half [S, n]."""
    S, K = int(x_i8.shape[0]), int(x_i8.shape[1])
    N = int(dev.n)
    w = dev.qs
    if w.dtype not in (mx.int8, mx.uint8):
        w = mx.view(w, mx.int8)
    w = w.reshape(N, K)
    tb, off = _tb_off(S)
    grid = ((N // 64) * 32, -(-S // BM) * 2, BM // 32)
    return _native_kern()(inputs=[x_i8, w, dev.meta, dev.gs, xs32, tb, off], template=[("BM", BM), ("BK", BK), ("N", N), ("K", K), ("KB32", K // 32)],
                          grid=grid, threadgroup=(32, 2, BM // 32), output_shapes=[(S, N)], output_dtypes=[mx.float16])[0]


def native_linear(dev, x2):
    """x2 [R, k] -> half [R, n] through the native Q8_0 x int8 GEMM, or None when not eligible."""
    R = int(x2.shape[0])
    if native_why_not(dev, R):
        return None
    if x2.dtype != mx.float16:
        x2 = mx.clip(x2, -65504.0, 65504.0).astype(mx.float16)
    xi, xs = Q8.quant_rows(x2, 32)
    return native_gemm(xi, xs, dev)
