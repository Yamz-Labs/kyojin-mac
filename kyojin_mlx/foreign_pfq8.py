# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 prefill routed-expert MoE GEMM on int8 activations x int8 weight tiles, accumulated in int32 on the M5 Neural Accelerator (Metal 4 `mpp::tensor_ops::matmul2d`,
int8_t x int8_t -> int32_t, ~2x the fp16 rate: measured 125-146 TOPS against 69-82 TFLOPS at the 65536-slot expert shapes). Flag `KYOJIN_GLM_PF_Q8` (default off, registered as
`glm_pf_q8` in kyojin_mlx.flags; fired.fb reasons in `why_not`). STATUS_pfq8.md has the measurements; the full-model PPL / KL gate has NOT been run.

Weight side: the NVFP4 experts stay resident as stored (an int8 pack of the routed experts would be 7.25 GB per layer, 304 GB for 42 MoE layers). `gmm_i8` decodes the stored
nibbles + UE4M3 block scales to int8 per K step in threadgroup memory (table lookup + one integer multiply-add per byte, see the tile source): code = rint(e2m1 * e4 * 127 /
(6 * emax_row)), ONE int8 scale per (expert, output row) over the full K reduction, round half EVEN (v1 rounded ties up: 0.7-0.9 % of the elements, +0.004 code of bias). The E2M1
side is lossless (doubled values are integers in [-12, 12]); the row-wide re-quantization of the 16-wide block scale is the approximation. `row_emax` / `row_fm` (fp32 [E, n]) are
cached on the RESIDENT pack, also when the prefill reads a transient `untiled()` copy (flag glmtile). `pack_i8` / `dequant_i8` are a host reference for tests only.

Activation side: dynamic int8, scale = amax / 127 per (row, block of QB consecutive K), QB = BK = 128 by default (KYOJIN_PFQ8_QB: 128 / 256; one scale per K step of the
kernel). v1 used ONE scale per row: a row with amax / rms = R loses ~ R / 440 of relative accuracy (0.9 % on gaussian rows, 11 % at R = 50: sink / BOS tokens, SwiGLU spikes). Now the
kernel runs the matmul2d of every scale block into an int32 partial (exact: <= 128 * 127 * 127), and sums partial * xs[row, block] in fp32 (acc), so an outlier channel only
costs the 127 other elements of ITS block. `quant_rows` (MoE input) and `silu_quant` (silu(min(a, L)) * clip(b, +-L) fused with the quantization) write the [rows, K / QB] scales
in one launch each; both need k % 256 == 0, which `ok_tile` checks (a shape that fails it falls back instead of asserting). `quant_act` is the mx-op reference (qb=None: the v1
per-row scheme).

Everything down to `quant_act` / `pack_i8` is plain `mx` ops (runs on the CPU, see tests/test_pfq8.py); `gmm_i8`, `quant_rows`, `silu_quant`, `gather_i8` and `mlp` need the GPU."""
from __future__ import annotations

import os

import numpy as np
import mlx.core as mx

from . import kernels as KN
from . import metal_nax as MN

KVALUES_FP4 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], np.float32)   # doubled signed E2M1 (foreign_gguf.KVALUES_FP4, same table)


def on() -> bool:
    """Flag KYOJIN_GLM_PF_Q8 (default off): int8-activation x int8-weight NVFP4 MoE GEMM for GLM-5.3 prefill. Read per call."""
    return os.environ.get("KYOJIN_PF_GEMM_I8", "0") not in ("0", "") or os.environ.get("KYOJIN_GLM_PF_Q8", "0") not in ("0", "")      # op-named flag; KYOJIN_GLM_PF_Q8 = alias


def min_slots() -> int:
    """KYOJIN_GLM_PF_Q8_MIN (default 16384 slots, ~ an 8K-token prompt chunk at K=2): below this the fp16 NAX path already wins (small-M matmul2d calls do
    not amortize the per-tile fixed cost; no measurement yet, a conservative placeholder gate the cut benchmark should tighten)."""
    return int(os.environ.get("KYOJIN_GLM_PF_Q8_MIN", "16384"))


def row_scale() -> bool:
    """Flag KYOJIN_PF_GEMM_I8_ROW (default off, live, sub-flag of pf_gemm_i8): ONE activation scale per row (the v1 scheme: 30.9 ms / layer vs 35.1 for the per-128-K-block scales, but 2-5x the error on outlier rows)."""
    return os.environ.get("KYOJIN_PF_GEMM_I8_ROW", "0") not in ("0", "")


def down_fp16() -> bool:
    """KYOJIN_GLM_PF_Q8_DOWN=fp16 (default i8): keep the down projection (input = SwiGLU output, rows with amax up to ~100 x the median on the real model) on the fp16 path; gate / up stay int8."""
    return os.environ.get("KYOJIN_GLM_PF_Q8_DOWN", "i8") == "fp16"


def _e4m3_mx(b):
    """UE4M3 byte (uint8, sign ignored like the stored block scale: foreign_pack._e4m3 masks the same way) -> float32, mx ops only."""
    b = b.astype(mx.uint32) & mx.array(127, mx.uint32)
    e = (b >> 3).astype(mx.int32)
    m = (b & mx.array(7, mx.uint32)).astype(mx.float32)
    sub = m * float(2.0 ** -9)
    norm = (1.0 + m / 8.0) * (2.0 ** (e.astype(mx.float32) - 7.0))
    v = mx.where(e == 0, sub, norm)
    return mx.where(b == 127, mx.zeros_like(v), v)


def ok(dp) -> bool:
    """A stacked nvfp4 expert pack this lever can pack (no k-alignment requirement beyond the model's own 16-block NVFP4 grid, which k always satisfies)."""
    return dp.fmt == "nvfp4" and bool(dp.E) and not dp.rowgs and int(dp.k) % 16 == 0


def pack_i8(dp):
    """Stacked nvfp4 DevPack (qs uint32 [E,n,k/8], meta uint16 [E,n,k/32], gs [E]) -> (w_i8 int8 [E,n,k], w_scale float16 [E,n]), cached on `dp` (invalidated
    if `dp.qs` identity changes, which never happens for a loaded model's resident packs). ONE int8 scale per (expert, output row) over the full K reduction
    (see the module docstring for the precision trade-off this makes): w_i8[e,c,:].astype(f32) * w_scale[e,c] ~= the fp16 weight the native NVFP4 decode
    produces for row c of expert e (exact on the E2M1 side; the approximation is this file's row-wide re-quantization of the 16-wide block scale)."""
    cache = getattr(dp, "_pfq8", None)
    if cache is not None and cache[0] is dp.qs:
        return cache[1], cache[2]
    E, n, k = int(dp.E), int(dp.n), int(dp.k)
    qs = dp.qs.astype(mx.uint32)                                                  # [E, n, k // 8]
    shifts = mx.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=mx.uint32)
    nib = (qs[..., None] >> shifts) & mx.array(0xF, mx.uint32)                     # [E, n, k // 8, 8]; nibble j of word w is k = 8w + j (natural order)
    codes = nib.reshape(E, n, k).astype(mx.int32)
    lut = mx.array(KVALUES_FP4)
    x2 = lut[codes]                                                               # [E, n, k] float32, exact small integers (doubled E2M1)
    meta_u8 = mx.view(dp.meta, mx.uint8).reshape(E, n, k // 16)                    # one UE4M3 byte per 16-wide block
    e4 = _e4m3_mx(meta_u8)                                                        # [E, n, k // 16]
    e4_full = mx.repeat(e4, 16, axis=-1)                                          # [E, n, k]
    w = (x2 * 0.5) * e4_full * dp.gs.reshape(E, 1, 1)                             # the true dequantized weight (float32), same value the fp16 NAX path decodes
    amax = mx.max(mx.abs(w), axis=-1)                                             # [E, n]: one scale per output row over the whole K reduction
    amax = mx.where(amax > 0, amax, mx.ones_like(amax))
    scale = amax / 127.0
    codes8 = mx.clip(mx.round(w / scale[..., None]), -127, 127).astype(mx.int8)
    scale16 = scale.astype(mx.float16)
    dp._pfq8 = (dp.qs, codes8, scale16)
    return codes8, scale16


def dequant_i8(w_i8, w_scale):
    """Reconstruct the approximate dequantized weight from a pack_i8 pack (float32 [E, n, k]): for tests / a PPL-arm sanity check, not on the hot path."""
    return w_i8.astype(mx.float32) * w_scale.astype(mx.float32)[..., None]


def quant_act(x2, qb: int | None = None):
    """x2 half/float [T, K] -> (codes int8 [T, K], scale float32): dynamic quantization, scale = amax / 127, round half to even. qb None: ONE scale per row (float32 [T], the v1 scheme); qb = block
    size: one scale per (row, block of qb consecutive K) (float32 [T, K / qb], what the kernels `quant_rows` / `silu_quant` write). codes.astype(f32) * scale ~= x2.astype(f32)."""
    xf = x2.astype(mx.float32)
    T, K = xf.shape
    if qb is not None:
        xb = xf.reshape(T, K // qb, qb)
        m = mx.max(mx.abs(xb), axis=-1)
        scale = mx.where(m > 0, m / 127.0, mx.ones_like(m))                           # [T, K / qb]
        codes = mx.clip(mx.round(xb / mx.maximum(scale[..., None], 1e-30)), -127, 127).astype(mx.int8).reshape(T, K)
        return codes, scale
    m = mx.max(mx.abs(xf), axis=-1)
    scale = mx.where(m > 0, m / 127.0, mx.ones_like(m))                           # free fp32 scale (a power-of-two scale wasted up to one bit: 1.3 % -> ~0.9 % rms on Gaussian rows)
    codes = mx.clip(mx.round(xf / mx.maximum(scale[:, None], 1e-30)), -127, 127).astype(mx.int8)
    return codes, scale


_GATHER_I8 = r"""
    const uint g = thread_position_in_grid.x;
    const uint s = g / (uint(KIN) / 16u), p = (g - s * (uint(KIN) / 16u)) * 16u;
    uint4 v = *(device const uint4*)(x + ulong(order[s] / uint(K)) * uint(KIN) + p);
    v = (v & 0xFF0000FFu) | ((v >> 8) & 0x0000FF00u) | ((v << 8) & 0x00FF0000u);                       // perm4: bytes 1 and 2 of every word swap (see gmm_i8)
    *(device uint4*)(o + ulong(s) * uint(KIN) + p) = v;
"""


def gather_i8(x_i8, order, K: int):
    """x_i8 int8 [T, k] -> int8 [S, k]: row s = x_i8[order[s] // K], columns K-permuted by perm4 (the order gmm_i8 consumes)."""
    S, k = int(order.shape[0]), int(x_i8.shape[1])
    assert k % 16 == 0
    kern = KN._kern(("pfq8_gather",), name="kyojin_pfq8_gather", input_names=["x", "order"], output_names=["o"], header="", source=_GATHER_I8)
    return kern(inputs=[x_i8, order], template=[("KIN", k), ("K", K)], grid=(S * k // 16, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(S, k)], output_dtypes=[mx.int8])[0]


def _root(dp):
    """The resident pack a transient `untiled()` copy came from (flag glmtile: prefill reads a fresh stored-layout copy per call, so a cache on the copy would be rebuilt, and synced, on every call)."""
    return getattr(dp, "_parent", dp)


def _key(root):
    return root.tl if getattr(root, "tl", None) is not None else root.qs


def row_emax(dp):
    """fp32 [E, n]: the largest UE4M3 block scale of every (expert, output row) over K, cached on the resident pack (3 x 288 x (2048 | 4096) x 4 B per layer; computed on the first prefill chunk
    only, one mx.eval). The int8 weight tile is NEVER stored: an int8 pack of the routed experts is 7.25 GB per layer, 304 GB for the 42 MoE layers, so `gmm_i8` decodes the stored NVFP4 bytes to
    int8 in threadgroup memory per K step. A block of 16 holds an element of magnitude 6 (the NVFP4 scale is block amax / 6), so the row amax of the dequantized weight is
    6 * emax * gs and the row int8 scale is 6 * emax * gs / 127; the global scale gs cancels in the int8 code: code = rint(e2m1 * e4 * 127 / (6 * emax))."""
    root = _root(dp)
    cache = getattr(root, "_pfq8_em", None)
    if cache is not None and cache[0] is _key(root):
        return cache[1]
    E, n, k = int(dp.E), int(dp.n), int(dp.k)
    em = mx.concatenate([mx.max(_e4m3_mx(mx.view(dp.meta[e:e + 24], mx.uint8).reshape(-1, n, k // 16)), axis=-1) for e in range(0, E, 24)], axis=0)
    em = mx.where(em > 0, em, mx.ones_like(em)).astype(mx.float32)
    fm = (127.0 / 6.0) / em
    mx.eval(em, fm)
    root._pfq8_em = (_key(root), em)
    root._pfq8_fm = (_key(root), fm)
    return em


_FP4_TILE_HDR = None
_K2 = (0, 1, 2, 3, 4, 6, 8, 12)                      # doubled E2M1 magnitudes


def _lut_text():
    """256-entry constant table indexed by one stored byte (two E2M1 nibbles, low nibble = lower k): lane0 | lane1 << 16 with lane = (signed doubled value) + 12 in [0, 24]."""
    def lane(n):
        v = _K2[n & 7] * (-1 if n & 8 else 1)
        return v + 12
    return "constant uint PFQ8_T[256] = {" + ",".join(str(lane(i & 15) | (lane(i >> 4) << 16)) + "u" for i in range(256)) + "};\n"


def _hdr():
    global _FP4_TILE_HDR
    if _FP4_TILE_HDR is None:
        from . import foreign_kqnax as NX
        _FP4_TILE_HDR = NX._mlx_headers() + "\nusing namespace metal;\nusing namespace mpp::tensor_ops;\n" + _lut_text()
    return _FP4_TILE_HDR


def row_fm(dp):
    """fp32 [E, n] = (127 / 6) / row_emax: the per-row factor of the decode (cached with row_emax on the resident pack)."""
    row_emax(dp)
    return _root(dp)._pfq8_fm[1]


def perm4(x):
    """The K permutation of the decoded weight tile: inside every group of 4 consecutive k the middle two swap (0, 2, 1, 3). The activations (gather_i8, silu_quant) carry the
    same permutation, so the dot products are unchanged (a sum over k) while the decoder packs its output bytes straight from the byte-pair lookup (no byte shuffle)."""
    sh = x.shape
    return x.reshape(*sh[:-1], sh[-1] // 4, 4)[..., [0, 2, 1, 3]].reshape(sh)


# Tile schedule of oMLX / foreign_kqnax with int8 operands: a threadgroup (32 x 2 x BM / 32 = 2 BM / 32 simdgroups) owns a BM x 64 output tile of ONE expert; per K step
# (BK) the 64 x BK weight tile is decoded once from the stored NVFP4 bytes + UE4M3 block scales to int8 in threadgroup memory (ONE int8 scale per weight row, applied in
# the epilogue; code = x * e4 * 127 / (6 emax_row), |code| <= 127) and every simdgroup runs `matmul2d` (int8 x int8 -> int32, multiply_accumulate into a persistent
# cooperative accumulator) of its 32 x 32 block with the A rows read from the sorted int8 activations in device memory. Epilogue: acc * (6 / 127 emax[col] gs[expert]).
# DECODE (per item = one 16-wide e4m3 block of one weight row, 8 stored bytes): per byte, ONE constant-table load gives the two signed doubled E2M1 values (+12) in two 16-bit
# lanes, ONE integer multiply-add by Q = round(64 * block factor) puts both on the int8 grid (lane = (v + 12) Q + 63 + 2^15 - 12 Q = v Q + 63 + 2^15, never negative, never above
# 49092), `lane += (lane >> 7) & 1` turns the floor of bits 7..14 into round-half-EVEN (a tie has v Q + 64 = 0 mod 128: the lower candidate is odd exactly when the upper one is even),
# and two shifts / masks pack four codes into one word (order 0, 2, 1, 3: see perm4).
# ACTIVATION SCALES: one per (row, K block of QBS steps). The matmul2d of a block runs into the int32 cooperative tensor (mode multiply when QBS = 1, accumulate + zero otherwise), then
# acc[i] += float(cT[i]) * xs[row, block] (xs of the tile's rows staged per block in threadgroup memory, one block ahead); the epilogue only applies the weight row factor.
# DBUF: two weight buffers, ONE barrier per K step, the decode of step kb + 1 follows the mma issue of step kb.
_TILE_SRC = r"""
    threadgroup uint4 Wsv[2 * 4 * BK];
    threadgroup int8_t* Ws = (threadgroup int8_t*)Wsv;
    const uint ti = threadgroup_position_in_grid.y, c = threadgroup_position_in_grid.x;
    if (ti >= tb[uint(E)]) return;
    if (ti == 0xFFFFFFFFu) y[0] = half(dep[0]);                 // never true: `dep` (the previous GEMM's output) only orders this launch after it (MLX would run two independent launches concurrently: 134 ms for gate + up)
    uint e_lo = 0u, e_hi = uint(E);
    while (e_hi - e_lo > 1u) { const uint e_mid = (e_lo + e_hi) >> 1; if (tb[e_mid] <= ti) e_lo = e_mid; else e_hi = e_mid; }
    const uint ex = e_lo;
    const uint rs = off[ex] + (ti - tb[ex]) * uint(BM);
    const int rows = int(min(uint(BM), off[ex + 1u] - rs));
    const uint sgid = simdgroup_index_in_threadgroup;
    const uint lid = sgid * 32u + thread_index_in_simdgroup;
    const uint nthr = 2u * uint(BM);                                       // threads of the threadgroup (BM / 32 x 2 simdgroups of 32)
    const short tm = short(32 * (sgid / 2u)), tn = short(32 * (sgid % 2u));
    const int rsg = max(0, min(32, rows - int(tm)));
    const uint wpr = uint(BK) / 16u;                                      // blocks per row per K step
    const uint c0 = c * 64u;
    constexpr auto desc = matmul2d_descriptor(32, 32, BK, false, true, false, QBS == 1 ? matmul2d_descriptor::mode::multiply : matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tA((device int8_t*)(x + ulong(rs + uint(tm)) * uint(K)), dextents<int32_t, 2>(K, max(rsg, 1)));
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tB((threadgroup int8_t*)Ws, dextents<int32_t, 2>(BK, 128));
    auto sA0 = tA.slice(0, 0);
    auto sB0 = tB.slice(0, int(tn));
    auto cT = op.template get_destination_cooperative_tensor<decltype(sA0), decltype(sB0), int32_t>();
    // per-K-block activation scales: the int32 partial of every block of QBS K steps is scaled by xs[row, block] and summed in fp32 (acc). xsT = the scales of the block, per tile row
    // (double buffered like the weight tile: written one block ahead, ONE barrier per K step covers both). rloc = the tile row of every accumulator element.
    constexpr uint CAPN = 32u;
    const uint16_t cap = cT.get_capacity();
    float acc[CAPN];
    ushort rloc[CAPN];
    threadgroup float xsT[2 * BM];
    for (uint i = 0; i < CAPN; ++i) {
        acc[i] = 0.0f; rloc[i] = 0;
        if (i < cap && cT.is_valid_element(uint16_t(i))) { cT[i] = 0; rloc[i] = ushort(tm) + ushort(cT.get_multidimensional_index(uint16_t(i))[1]); }
    }
    #define PFQ8_XS(BLK) if (lid < uint(BM)) xsT[((BLK) & 1) * BM + lid] = int(lid) < rows ? xs[ulong(rs + lid) * uint(KB) + uint(BLK)] : 0.0f;
    PFQ8_XS(0)
    // software-pipelined weight loads: per item (row r, 16-wide block b of the K step) 8 stored bytes + 1 scale byte, fetched one K step AHEAD of their decode; the row factor is loop invariant
    constexpr uint ITEMS = (64u * uint(BK) / 16u + 2u * uint(BM) - 1u) / (2u * uint(BM));
    uint2 wq[ITEMS];
    uint wsb[ITEMS];
    float wfm[ITEMS];
    #define PFQ8_ROW(U) (ulong(ex) * uint(N) + c0 + (lid + (U) * nthr) / wpr)
    #define PFQ8_LD(KB) \
        for (uint u = 0; u < ITEMS; ++u) { \
            const uint it = lid + u * nthr; \
            if (it < 64u * wpr) { \
                const uint b = it % wpr; \
                const ulong wrow = PFQ8_ROW(u); \
                const uint kk = uint(KB) * uint(BK) + b * 16u; \
                wsb[u] = uint(sc[wrow * uint(K / 16) + (kk >> 4)]) & 0x7Fu; \
                const device uint32_t* wp = (const device uint32_t*)((const device uint8_t*)w + wrow * uint(K / 2) + (kk >> 1)); \
                wq[u] = uint2(wp[0], wp[1]); \
            } \
        }
    #define PFQ8_DEC(BUF) \
        for (uint u = 0; u < ITEMS; ++u) { \
            const uint it = lid + u * nthr; \
            if (it < 64u * wpr) { \
                const uint r = it / wpr, b = it % wpr; \
                const uint8_t sb8 = uint8_t(wsb[u]); \
                float fb = float(static_cast<bfloat>(*(thread fp8_e4m3*)(&sb8))) * wfm[u]; \
                fb = wsb[u] == 127u ? 0.0f : fb; \
                const uint Q = uint(fb * 64.0f + 0.5f); \
                const uint R2 = (63u + uint(1 - TIE) + 32768u - 12u * Q) * 0x00010001u; \
                uint z[8]; \
                for (uint j = 0; j < 4u; ++j) { \
                    z[j] = PFQ8_T[(wq[u].x >> (8u * j)) & 255u] * Q + R2; \
                    z[4u + j] = PFQ8_T[(wq[u].y >> (8u * j)) & 255u] * Q + R2; \
                    if (TIE) { z[j] += (z[j] >> 7) & 0x00010001u; z[4u + j] += (z[4u + j] >> 7) & 0x00010001u; } \
                } \
                uint4 o0; \
                o0.x = ((z[0] >> 7) & 0x00FF00FFu) | ((z[1] << 1) & 0xFF00FF00u); \
                o0.y = ((z[2] >> 7) & 0x00FF00FFu) | ((z[3] << 1) & 0xFF00FF00u); \
                o0.z = ((z[4] >> 7) & 0x00FF00FFu) | ((z[5] << 1) & 0xFF00FF00u); \
                o0.w = ((z[6] >> 7) & 0x00FF00FFu) | ((z[7] << 1) & 0xFF00FF00u); \
                Wsv[(uint(BUF) * 64u + r) * (uint(BK) / 16u) + b] = o0; \
            } \
        }
    for (uint u = 0; u < ITEMS; ++u) wfm[u] = (lid + u * nthr < 64u * wpr) ? fm[PFQ8_ROW(u)] : 0.0f;
    PFQ8_LD(0)
    PFQ8_DEC(0)
    if (K / BK > 1) { PFQ8_LD(1) }
    for (int kb = 0; kb < K / BK; ++kb) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kb % QBS == 0 && kb / QBS + 1 < KB) { PFQ8_XS(kb / QBS + 1) }
        if (MMAF && rsg > 0) {
            auto sA = tA.slice(kb * BK, 0);
            auto sB = tB.slice(0, int(tn) + 64 * (kb & 1));
            op.run(sA, sB, cT);
        }
        if (kb + 1 < K / BK) { PFQ8_DEC((kb + 1) & 1) }
        if (kb + 2 < K / BK) { PFQ8_LD(kb + 2) }
        if (!MMAF && rsg > 0) {
            auto sA = tA.slice(kb * BK, 0);
            auto sB = tB.slice(0, int(tn) + 64 * (kb & 1));
            op.run(sA, sB, cT);
        }
        if ((kb + 1) % QBS == 0) {                                          // end of a scale block: acc += partial * xs[row, block]
            const uint xb = ((kb / QBS) & 1) * uint(BM);
            for (uint i = 0; i < CAPN; ++i) {
                if (i < cap) { acc[i] += float(cT[i]) * xsT[xb + rloc[i]]; if (QBS > 1) cT[i] = 0; }
            }
        }
    }
    if (rsg == 0) return;
    const float gsc = gs[ex] * (6.0f / 127.0f);
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (!cT.is_valid_element(i)) continue;
        const auto ix = cT.get_multidimensional_index(i);
        const uint col = uint(ix[0]), r = uint(ix[1]);
        if (int(r) >= rsg) continue;
        const uint rr = rs + uint(tm) + r, cc = c0 + uint(tn) + col;
        const float yv = acc[i] * em[ulong(ex) * uint(N) + cc] * gsc;
        y[ulong(SCAT ? orow[rr] : rr) * uint(N) + cc] = half(clamp(yv, -65504.0f, 65504.0f));
    }
"""


def _kern_tile(scat: bool):
    key = ("pfq8_tile", scat, _SRC_TAG)
    k = KN._cache.get(key)
    if k is None:
        k = KN._cache[key] = mx.fast.metal_kernel(name="kyojin_pfq8_tile" + ("_s" if scat else ""), input_names=["x", "w", "sc", "em", "fm", "xs", "gs", "tb", "off", "dep"] + (["orow"] if scat else []),
                                                   output_names=["y"], header=_hdr(), source=_TILE_SRC if scat else _TILE_SRC.replace("SCAT ? orow[rr] : rr", "rr"))
    return k


_SRC_TAG = 0
_PLAN = tuple(int(v) for v in os.environ.get("KYOJIN_PFQ8_PLAN", "128,128").split(","))
BM, BK = _PLAN
MMAF = int(os.environ.get("KYOJIN_PFQ8_MMAF", "1"))          # 1 = issue the mma of step kb before the decode of step kb + 1
QB = int(os.environ.get("KYOJIN_PFQ8_QB", str(BK)))          # activation scale block (K elements per scale, a multiple of BK; 128 = one scale per K step). A row holds K / QB scales
assert QB % BK == 0 and QB in (32, 64, 128, 256), (BK, QB)
TIE = int(os.environ.get("KYOJIN_PFQ8_TIE", "1"))             # 1 = round half even (default); 0 = ties up (v1 decode, kept to time the extra decode ops)
QBS = QB // BK                                                # K steps per scale block


def ok_tile(dp) -> bool:
    return dp.fmt == "nvfp4" and bool(dp.E) and not dp.rowgs and int(dp.k) % BK == 0 and int(dp.k) % 256 == 0 and int(dp.n) % 64 == 0      # k % 256: quant_rows / silu_quant


def gmm_i8(x_i8, x_scale, dp, tb, off, S: int, orow=None, after=None, qb: int | None = None):
    """x_i8 int8 [S, k] sorted by expert and K-permuted (perm4), x_scale float32 [S, k / QB] (sorted order, one per row and K block), dp = stacked nvfp4 pack (qs/meta/gs read in place). Returns y half [S, n] in
    sorted order, or scattered to slot order through `orow` (the down projection). tb / off from foreign_kqgmm.csort(.., MT = BM)."""
    from . import foreign_kqnax as NX
    E, N, K = int(dp.E), int(dp.n), int(dp.k)
    qb = QB if qb is None else qb                                                       # qb == K: ONE scale per row (the v1 scheme, flag pf_gemm_i8_row); the int32 partial then runs over the whole K
    assert qb % BK == 0 and K % qb == 0, (qb, K, BK)
    ins = [x_i8, dp.qs, NX.meta_bytes(dp), row_emax(dp), row_fm(dp), x_scale, dp.gs, tb, off, after if after is not None else tb] + ([orow] if orow is not None else [])
    maxt = S // BM + E
    grid = ((N // 64) * 32, maxt * 2, BM // 32)
    return _kern_tile(orow is not None)(inputs=ins, template=[("BM", BM), ("BK", BK), ("N", N), ("K", K), ("E", E), ("SCAT", orow is not None), ("MMAF", MMAF), ("QBS", qb // BK), ("KB", K // qb), ("TIE", TIE)],
                                          grid=grid, threadgroup=(32, 2, BM // 32), output_shapes=[(S, N)], output_dtypes=[mx.float16])[0]


# silu(min(a, L)) * clip(b, +-L) (rounded to half like silu_perm) and the per-(row, K block) int8 quantization of the result in ONE pass (the mx-op quant_act on [S, I] was 7 ms at 64K slots).
# One threadgroup of 256 threads per row; KIN % 256 == 0; each thread keeps KIN / 256 values in registers. Thread t holds the elements t + 256 j: for a fixed j they sit in ONE
# scale block of QB elements (QB | 256), the block amax is a simd_max plus a max over the QB / 32 simdgroups that cover the block.
_QBLOCK_RED = r"""
    threadgroup float red[8 * PT];
    const uint sg = t >> 5, lane = t & 31u, SGB = uint(QB) / 32u;
    for (uint j = 0; j < PT; ++j) { const float mm = simd_max(abs(hv[j])); if (lane == 0u) red[sg * PT + j] = mm; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint sg0 = (sg / SGB) * SGB;
    for (uint j = 0; j < PT; ++j) {
        float m = red[sg0 * PT + j];
        for (uint q = 1; q < SGB; ++q) m = max(m, red[(sg0 + q) * PT + j]);
        const float sc = m > 0.0f ? m / 127.0f : 1.0f;
        const uint ci = t + 256u * j;
        OUT_CODE
        if ((ci & uint(QB - 1)) == 0u) osc[ulong(s) * uint(KIN / QB) + ci / uint(QB)] = sc;
    }
"""

# per-ROW variant (qb == k): one amax over the whole row (8 simdgroups), ONE scale osc[s]
_ROW_RED = r"""
    threadgroup float red[8];
    const uint sg = t >> 5, lane = t & 31u;
    float mm = 0.0f;
    for (uint j = 0; j < PT; ++j) mm = max(mm, abs(hv[j]));
    mm = simd_max(mm);
    if (lane == 0u) red[sg] = mm;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = red[0];
    for (uint q = 1; q < 8u; ++q) m = max(m, red[q]);
    const float sc = m > 0.0f ? m / 127.0f : 1.0f;
    for (uint j = 0; j < PT; ++j) {
        const uint ci = t + 256u * j;
        OUT_CODE
    }
    if (t == 0u) osc[s] = sc;
"""

_SILU_Q8_PRE = r"""
    const uint s = threadgroup_position_in_grid.x, t = thread_position_in_threadgroup.x;
    constexpr uint PT = uint(KIN) / 256u;
    float hv[PT];
    for (uint j = 0; j < PT; ++j) {
        const ulong i = ulong(s) * uint(KIN) + t + 256u * j;
        float av = float(a[i]), bv = float(b[i]);
        LIMIT_CLAMP
        hv[j] = float(half(av * (1.0f / (1.0f + metal::precise::exp(-av))) * bv));
    }
"""
_SILU_OUT = "o[ulong(s) * uint(KIN) + (ci ^ (3u * ((ci ^ (ci >> 1)) & 1u)))] = char(clamp(rint(hv[j] / sc), -127.0f, 127.0f));"
_SILU_Q8 = _SILU_Q8_PRE + _QBLOCK_RED.replace("OUT_CODE", _SILU_OUT)
_SILU_Q8_ROW = _SILU_Q8_PRE + _ROW_RED.replace("OUT_CODE", _SILU_OUT)


def silu_quant(a, b, limit=None, qb: int | None = None):
    """a, b half [S, k] (gate, up outputs, sorted order) -> (int8 [S, k], float32 [S, k / qb]): perm4(quant_act(silu_perm(a, b, identity, limit), qb)) in one launch (K-permuted for gmm_i8)."""
    S, k = int(a.shape[0]), int(a.shape[1])
    qb = QB if qb is None else qb
    assert k % 256 == 0 and (qb in (32, 64, 128, 256) or qb == k)
    row = qb == k and qb not in (32, 64, 128, 256)
    lim = "" if limit is None else f"av = min(av, {float(limit)!r}f); bv = clamp(bv, -{float(limit)!r}f, {float(limit)!r}f);"
    kern = KN._kern(("pfq8_silu2", limit, row), name="kyojin_pfq8_silu_q8" + ("_row" if row else ""), input_names=["a", "b"], output_names=["o", "osc"], header="", source=(_SILU_Q8_ROW if row else _SILU_Q8).replace("LIMIT_CLAMP", lim))
    return kern(inputs=[a, b], template=[("KIN", k), ("QB", qb)], grid=(S * 256, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(S, k), (S, k // qb)], output_dtypes=[mx.int8, mx.float32])


# per-(row, K block) int8 quantization of the layer input (replaces the mx-op quant_act: 1.9 ms for 8192 x 4096). One threadgroup of 256 threads per row; KIN % 256 == 0.
_QROW = r"""
    const uint s = threadgroup_position_in_grid.x, t = thread_position_in_threadgroup.x;
    constexpr uint PT = uint(KIN) / 256u;
    float hv[PT];
    for (uint j = 0; j < PT; ++j) hv[j] = float(x[ulong(s) * uint(KIN) + t + 256u * j]);
"""
_QROW_OUT = "o[ulong(s) * uint(KIN) + ci] = char(clamp(rint(hv[j] / sc), -127.0f, 127.0f));"
_QROW_ROW = _QROW + _ROW_RED.replace("OUT_CODE", _QROW_OUT)
_QROW = _QROW + _QBLOCK_RED.replace("OUT_CODE", _QROW_OUT)


def quant_rows(x2, qb: int | None = None):
    """x2 half [T, k] -> (int8 [T, k], float32 [T, k / qb]) == quant_act(x2, qb) in one launch (k % 256 == 0)."""
    T, k = int(x2.shape[0]), int(x2.shape[1])
    qb = QB if qb is None else qb
    assert k % 256 == 0 and (qb in (32, 64, 128, 256) or qb == k)
    row = qb == k and qb not in (32, 64, 128, 256)
    kern = KN._kern(("pfq8_qrow2", row), name="kyojin_pfq8_qrow" + ("_row" if row else ""), input_names=["x"], output_names=["o", "osc"], header="", source=_QROW_ROW if row else _QROW)
    return kern(inputs=[x2], template=[("KIN", k), ("QB", qb)], grid=(T * 256, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(T, k), (T, k // qb)], output_dtypes=[mx.int8, mx.float32])


def why_not(glu, S: int) -> str:
    """Flag audit: why `mlp(glu, ...)` did not run for a prompt chunk of S slots (the reason string of fired.fb)."""
    if not on():
        return "flag_off"
    if glu.limit is None:
        return "no_swiglu_limit"
    for nm, p in (("gate", glu.g), ("up", glu.u), ("down", glu.d)):
        if p.fmt != "nvfp4" or not p.E:
            return f"{nm}:" + ("not_experts" if not p.E else "fmt=" + str(p.fmt))
        if p.rowgs:
            return f"{nm}:rowgs"
        if int(p.k) % BK or int(p.k) % 256:
            return f"{nm}:k%{max(BK, 256)}"
        if int(p.n) % 64:
            return f"{nm}:n%64"
    if int(glu.g.E) > 1024:
        return "E>1024"
    if S < min_slots():
        return "slots<min"
    return "?"


def mlp(glu, x2, e32, K: int):
    """ForeignSwitchGLU prefill on the stored rows, int8 path: x2 half [T, H], e32 uint32 [T * K] (slot s = token s // K). Returns y half [S, H] in slot
    order, or None if not eligible (small-shape gate, or a projection this flag cannot run: `why_not` names the reason)."""
    from . import foreign_kqgmm as KQ
    g, u, d = glu.g, glu.u, glu.d
    S = int(e32.shape[0])
    if not (ok_tile(g) and ok_tile(u) and ok_tile(d)) or S < min_slots() or int(g.E) > 1024:
        return None
    E = g.E
    order, tb, off = KQ.csort(e32, E, BM)
    rw = row_scale()
    qg = int(g.k) if rw else None
    xi8, xsc = quant_rows(x2 if x2.dtype == mx.float16 else x2.astype(mx.float16), qg)
    xg = gather_i8(xi8, order, K)
    xsc_s = mx.take(xsc, order // K, axis=0)
    a = gmm_i8(xg, xsc_s, g, tb, off, S, qb=qg)
    b = gmm_i8(xg, xsc_s, u, tb, off, S, after=a, qb=qg)
    if down_fp16():                                                                     # gate / up int8, down projection on the fp16 NAX path (kqgmm tiles of 32 rows)
        tb32, off32 = KQ.plan(mx.take(e32, order), E, 32)
        return KQ.gmm(d, KQ.silu_perm(a, b, KQ.xperm(d.fmt, d.k), glu.limit), tb32, off32, S, 32, orow=order)
    qd = int(d.k) if rw else None
    hi8, hsc = silu_quant(a, b, glu.limit, qd)                                          # natural order (no xperm on this path)
    return gmm_i8(hi8, hsc, d, tb, off, S, orow=order, qb=qd)
