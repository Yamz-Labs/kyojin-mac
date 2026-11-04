# SPDX-License-Identifier: Apache-2.0
"""DS4.1 decode indexer at depth: two Metal kernels instead of ~12 stock launches that move a [heads, keys] tile through memory and fully sort all the keys.

score_kernel   one launch, many threadgroups (split over the keys): s[i] = sum_h rnd(rnd(max(rnd(q_h . k_i), 0)) * w_h) with the stock fp16 / bf16 rounding points
               (q . k on simdgroup 8x8 matrices, fp32 accumulate). A candidate mask skips whole 8-key blocks (the second-level indexer layers score only
               the 2048 blocks kept by the candidate source: constant cost at any depth).
select_kernel  one threadgroup: radix select (two 8-bit histogram passes over an order-preserving 16-bit key) of the K best, ties to the lower index like the
               stable stock argsort, then an ordered compaction (no sort: the output is ascending by construction). MODE 1 does the candidate-source job:
               block max (8 keys), last block pinned, K blocks kept, bool mask over the keys.
Values the same as `fuse_index` (same rounding, same tie rule); the fp32 sum order over the 32 heads differs (4 partial sums), so a near tie can flip.
`scores_packed` is `scores` over the fp4-packed key rows of the cache (KYOJIN_IDX_KEYS_PACKED, E8): 68 B per key instead of 256, unpacked in the kernel."""
from __future__ import annotations

import os

import mlx.core as mx

NSG = 4          # simdgroups per scoring threadgroup
TPT = 16         # 8-key tiles per simdgroup
GT = 4           # tiles per pass over the query matrix (dense path)

_HEADER = """
#include <metal_simdgroup_matrix>
using namespace metal;
template <typename T> inline uint skey(T v) {
    ushort u = as_type<ushort>(v);
    if (u == 0x8000u || u == 0u) { return 0x8000u; }
    return (u & 0x8000u) ? uint((~u) & 0xFFFFu) : uint(u | 0x8000u);
}
inline uint skey(float v) {
    uint u = as_type<uint>(v);
    if (u == 0x80000000u) { u = 0u; }
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
template <typename P> inline float blockmax(P sc, uint nn, uint i) {
    float m = -INFINITY;
    uint e = metal::min(i * 8 + 8, nn);
    for (uint j = i * 8; j < e; ++j) { m = metal::max(m, float(sc[j])); }
    return m;
}
inline uint tg_excl_scan(uint v, threadgroup uint* red, uint sg, uint lane, uint nsg, thread uint& total) {
    uint p = simd_prefix_exclusive_sum(v);
    if (lane == 31) { red[sg] = p + v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = 0; uint tot = 0;
    for (uint s = 0; s < nsg; ++s) { uint r = red[s]; if (s < sg) { base += r; } tot += r; }
    total = tot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return base + p;
}
"""

_SCORE = r"""
    const uint tg = threadgroup_position_in_grid.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tid = sg * 32 + lane;
    threadgroup T Qs[32 * 128];
    threadgroup float Wh[32];
    threadgroup float scr[NSG][8 * 32];
    threadgroup T stage[NSG][8 * 128];
    for (uint i = tid; i < 32 * 128; i += NSG * 32) { Qs[i] = q[i]; }
    if (tid < 32) { Wh[tid] = float(w[tid]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nn = n[0];
    const uint key = lane & 7, qt = lane >> 3;
    // simdgroup sg of threadgroup tg owns TPT tiles of 8 keys; the dense path takes them GT at a time (one pass over the query matrix per GT tiles)
    for (uint j0 = 0; j0 < TPT; j0 += GT) {
        const uint k0 = ((tg * NSG + sg) * TPT + j0) * 8;
        if (k0 >= nn) { break; }
        if (!HASMASK && k0 + 8 * GT <= nn) {
            const device T* kp = km + ulong(k0) * 128;
            simdgroup_matrix<float, 8, 8> acc[GT][4];
            for (uint t = 0; t < GT; ++t) { for (uint hb = 0; hb < 4; ++hb) { acc[t][hb] = simdgroup_matrix<float, 8, 8>(0.0f); } }
            for (uint ks = 0; ks < 16; ++ks) {
                simdgroup_matrix<T, 8, 8> a[GT];
                for (uint t = 0; t < GT; ++t) { simdgroup_load(a[t], kp + ulong(t) * 8 * 128 + ks * 8, 128); }
                for (uint hb = 0; hb < 4; ++hb) {
                    simdgroup_matrix<T, 8, 8> b;
                    simdgroup_load(b, &Qs[hb * 8 * 128 + ks * 8], 128, ulong2(0, 0), true);
                    for (uint t = 0; t < GT; ++t) { simdgroup_multiply_accumulate(acc[t][hb], a[t], b, acc[t][hb]); }
                }
            }
            for (uint t = 0; t < GT; ++t) {
                for (uint hb = 0; hb < 4; ++hb) { simdgroup_store(acc[t][hb], &scr[sg][hb * 8], 32); }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                float s = 0.0f;
                for (uint h = qt * 8; h < qt * 8 + 8; ++h) {
                    const float cf = scr[sg][key * 32 + h];
                    if (F32) { s += metal::max(cf, 0.0f) * Wh[h]; }
                    else { T c = static_cast<T>(cf); c = c > static_cast<T>(0) ? c : static_cast<T>(0); s += float(static_cast<T>(c * static_cast<T>(Wh[h]))); }
                }
                s += simd_shuffle_xor(s, 8u);
                s += simd_shuffle_xor(s, 16u);
                if (lane < 8) { out[k0 + t * 8 + lane] = static_cast<OT>(s); }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            continue;
        }
        for (uint t = 0; t < GT; ++t) {
            const uint kt = k0 + 8 * t;
            if (kt >= nn) { break; }
            if (HASMASK) {
                if (!cand[kt]) { if (lane < 8 && kt + lane < nn) { out[kt + lane] = static_cast<OT>(-INFINITY); } continue; }
            }
            const bool tail = kt + 8 > nn;
            const device T* kp = km + ulong(kt) * 128;
            if (tail) {
                for (uint i = lane; i < 8 * 128; i += 32) { stage[sg][i] = (kt + i / 128 < nn) ? kp[i] : static_cast<T>(0); }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            simdgroup_matrix<float, 8, 8> acc[4];
            for (uint hb = 0; hb < 4; ++hb) { acc[hb] = simdgroup_matrix<float, 8, 8>(0.0f); }
            for (uint ks = 0; ks < 16; ++ks) {
                simdgroup_matrix<T, 8, 8> a;
                if (tail) { simdgroup_load(a, &stage[sg][ks * 8], 128); } else { simdgroup_load(a, kp + ks * 8, 128); }
                for (uint hb = 0; hb < 4; ++hb) {
                    simdgroup_matrix<T, 8, 8> b;
                    simdgroup_load(b, &Qs[hb * 8 * 128 + ks * 8], 128, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(acc[hb], a, b, acc[hb]);
                }
            }
            for (uint hb = 0; hb < 4; ++hb) { simdgroup_store(acc[hb], &scr[sg][hb * 8], 32); }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            float s = 0.0f;
            for (uint h = qt * 8; h < qt * 8 + 8; ++h) {
                const float cf = scr[sg][key * 32 + h];
                if (F32) { s += metal::max(cf, 0.0f) * Wh[h]; }
                else { T c = static_cast<T>(cf); c = c > static_cast<T>(0) ? c : static_cast<T>(0); s += float(static_cast<T>(c * static_cast<T>(Wh[h]))); }
            }
            s += simd_shuffle_xor(s, 8u);
            s += simd_shuffle_xor(s, 16u);
            if (lane < 8 && kt + lane < nn) { out[kt + lane] = static_cast<OT>(s); }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
"""


_HEADER_P = _HEADER + """
constant float FP4_LUT[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
// the value of dim d of key row k stored as fp4 codes (2 per byte, vendor block 32 / power-of-two scale) -> T, bit-identical to fp4pack.unpack of the row
// C / S: MLX binds an input under 8 elements as `constant` (one key: 4 scale bytes), larger ones as `device`
template <typename T, typename C, typename S> inline T fp4_key(C cod, S sca, uint k, uint d, uint DI) {
    const uint b = uint(cod[ulong(k) * (DI / 2) + (d >> 1)]);
    const uint c = (d & 1u) ? (b >> 4) : (b & 15u);
    const float mag = FP4_LUT[c & 7u];
    const float sv = metal::exp2(float(sca[ulong(k) * (DI / 32) + (d >> 5)]) - 127.0f);
    return static_cast<T>(((c & 8u) ? -mag : mag) * sv);
}
"""

# The same scores as _SCORE over the fp4-packed key rows (KeySlots: codes DI / 2 bytes, scales DI / 32 bytes) instead of the fp16 mirror: 68 B per key instead of
# 256. The 8-key tile is unpacked into threadgroup memory and goes through the same simdgroup 8x8 matrices, so the per-head products and the head sum are the same
# FMAs in the same order. A masked 8-key block is skipped before its codes are touched (the consumer layers keep 1/8 of the keys).
_SCORE_P = r"""
    const uint tg = threadgroup_position_in_grid.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tid = sg * 32 + lane;
    threadgroup T Qs[32 * 128];
    threadgroup float Wh[32];
    threadgroup float scr[NSG][8 * 32];
    threadgroup T stage[NSG][8 * 128];
    for (uint i = tid; i < 32 * 128; i += NSG * 32) { Qs[i] = q[i]; }
    if (tid < 32) { Wh[tid] = float(w[tid]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nn = n[0];
    const uint key = lane & 7, qt = lane >> 3;
    for (uint j = 0; j < TPT; ++j) {
        const uint kt = ((tg * NSG + sg) * TPT + j) * 8;
        if (kt >= nn) { break; }
        if (HASMASK && !cand[kt]) {
            if (lane < 8 && kt + lane < nn) { out[kt + lane] = static_cast<OT>(-INFINITY); }
            continue;
        }
        for (uint i = lane; i < 8 * 128; i += 32) {
            const uint r = i >> 7, d = i & 127u, kk = kt + r;
            stage[sg][i] = (kk < nn) ? fp4_key<T>(cod, sca, kk, d, 128u) : static_cast<T>(0);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_matrix<float, 8, 8> acc[4];
        for (uint hb = 0; hb < 4; ++hb) { acc[hb] = simdgroup_matrix<float, 8, 8>(0.0f); }
        for (uint ks = 0; ks < 16; ++ks) {
            simdgroup_matrix<T, 8, 8> a;
            simdgroup_load(a, &stage[sg][ks * 8], 128);
            for (uint hb = 0; hb < 4; ++hb) {
                simdgroup_matrix<T, 8, 8> b;
                simdgroup_load(b, &Qs[hb * 8 * 128 + ks * 8], 128, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc[hb], a, b, acc[hb]);
            }
        }
        for (uint hb = 0; hb < 4; ++hb) { simdgroup_store(acc[hb], &scr[sg][hb * 8], 32); }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float s = 0.0f;
        for (uint h = qt * 8; h < qt * 8 + 8; ++h) {
            const float cf = scr[sg][key * 32 + h];
            if (F32) { s += metal::max(cf, 0.0f) * Wh[h]; }
            else { T c = static_cast<T>(cf); c = c > static_cast<T>(0) ? c : static_cast<T>(0); s += float(static_cast<T>(c * static_cast<T>(Wh[h]))); }
        }
        s += simd_shuffle_xor(s, 8u);
        s += simd_shuffle_xor(s, 16u);
        if (lane < 8 && kt + lane < nn) { out[kt + lane] = static_cast<OT>(s); }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

# Rows: the same scoring as _SCORE for every row of a batched pass (the verify round of dsv4_core.core, R rows of one or more sessions), one
# row per threadgroup block. A row brings its own query [32, 128], head weights, valid count, session (which key buffer it reads) and candidate
# mask; the simdgroup matrix arithmetic, the head reduction and the fp16 rounding points are _SCORE's, so row u of a call gives the values
# dsa_decode.scores(q[u], w[u], keys[sess[u]], lens[u], cand[u]) gives. The indexer scores of the batched core were the one thread per key chain
# (dsv4_core._ISCORE): ~2.5 x the keys of the layer on the plain step, R x that on a verify round, at fp32 FMA throughput. Key tile per
# threadgroup: NSG * TPT * 8 = 512 columns.
_SCORE_ROWS = r"""
    const uint tg = threadgroup_position_in_grid.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tid = sg * 32 + lane;
    const uint Tm = ip[0];
    const uint Ck = ip[1];
    const uint Tc = ip[2];
    const uint nt = (Tm + NSG * TPT * 8 - 1) / (NSG * TPT * 8);
    const uint u = tg / nt;
    const uint gt = tg % nt;
    threadgroup T Qs[32 * 128];
    threadgroup float Wh[32];
    threadgroup float scr[NSG][8 * 32];
    threadgroup T stage[NSG][8 * 128];
    for (uint i = tid; i < 32 * 128; i += NSG * 32) { Qs[i] = q[ulong(u) * 32 * 128 + i]; }
    if (tid < 32) { Wh[tid] = float(wts[u * 32 + tid]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nn = metal::min(uint(lens[u]), Tm);
    const device T* km = keys + ulong(sess[u]) * Ck * 128;
    const device bool* cm = CAND ? (cand + ulong(u) * Tc) : cand;
    const uint key = lane & 7, qt = lane >> 3;
    for (uint j0 = 0; j0 < TPT; j0 += GT) {
        const uint k0 = ((gt * NSG + sg) * TPT + j0) * 8;
        if (k0 >= nn) { break; }
        if (!HASMASK && k0 + 8 * GT <= nn) {
            const device T* kp = km + ulong(k0) * 128;
            simdgroup_matrix<float, 8, 8> acc[GT][4];
            for (uint t = 0; t < GT; ++t) { for (uint hb = 0; hb < 4; ++hb) { acc[t][hb] = simdgroup_matrix<float, 8, 8>(0.0f); } }
            for (uint ks = 0; ks < 16; ++ks) {
                simdgroup_matrix<T, 8, 8> a[GT];
                for (uint t = 0; t < GT; ++t) { simdgroup_load(a[t], kp + ulong(t) * 8 * 128 + ks * 8, 128); }
                for (uint hb = 0; hb < 4; ++hb) {
                    simdgroup_matrix<T, 8, 8> b;
                    simdgroup_load(b, &Qs[hb * 8 * 128 + ks * 8], 128, ulong2(0, 0), true);
                    for (uint t = 0; t < GT; ++t) { simdgroup_multiply_accumulate(acc[t][hb], a[t], b, acc[t][hb]); }
                }
            }
            for (uint t = 0; t < GT; ++t) {
                for (uint hb = 0; hb < 4; ++hb) { simdgroup_store(acc[t][hb], &scr[sg][hb * 8], 32); }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                float s = 0.0f;
                for (uint h = qt * 8; h < qt * 8 + 8; ++h) {
                    const float cf = scr[sg][key * 32 + h];
                    T c = static_cast<T>(cf); c = c > static_cast<T>(0) ? c : static_cast<T>(0); s += float(static_cast<T>(c * static_cast<T>(Wh[h])));
                }
                s += simd_shuffle_xor(s, 8u);
                s += simd_shuffle_xor(s, 16u);
                if (lane < 8) { out[ulong(u) * Tm + k0 + t * 8 + lane] = static_cast<OT>(s); }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            continue;
        }
        for (uint t = 0; t < GT; ++t) {
            const uint kt = k0 + 8 * t;
            if (kt >= nn) { break; }
            if (HASMASK && !cm[kt]) { if (lane < 8 && kt + lane < Tm) { out[ulong(u) * Tm + kt + lane] = static_cast<OT>(-INFINITY); } continue; }
            const bool tail = kt + 8 > nn;
            const device T* kp = km + ulong(kt) * 128;
            if (tail) {
                for (uint i = lane; i < 8 * 128; i += 32) { stage[sg][i] = (kt + i / 128 < nn) ? kp[i] : static_cast<T>(0); }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            simdgroup_matrix<float, 8, 8> acc[4];
            for (uint hb = 0; hb < 4; ++hb) { acc[hb] = simdgroup_matrix<float, 8, 8>(0.0f); }
            for (uint ks = 0; ks < 16; ++ks) {
                simdgroup_matrix<T, 8, 8> a;
                if (tail) { simdgroup_load(a, &stage[sg][ks * 8], 128); } else { simdgroup_load(a, kp + ks * 8, 128); }
                for (uint hb = 0; hb < 4; ++hb) {
                    simdgroup_matrix<T, 8, 8> b;
                    simdgroup_load(b, &Qs[hb * 8 * 128 + ks * 8], 128, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(acc[hb], a, b, acc[hb]);
                }
            }
            for (uint hb = 0; hb < 4; ++hb) { simdgroup_store(acc[hb], &scr[sg][hb * 8], 32); }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            float s = 0.0f;
            for (uint h = qt * 8; h < qt * 8 + 8; ++h) {
                const float cf = scr[sg][key * 32 + h];
                T c = static_cast<T>(cf); c = c > static_cast<T>(0) ? c : static_cast<T>(0); s += float(static_cast<T>(c * static_cast<T>(Wh[h])));
            }
            s += simd_shuffle_xor(s, 8u);
            s += simd_shuffle_xor(s, 16u);
            if (lane < 8 && kt + lane < nn) { out[ulong(u) * Tm + kt + lane] = static_cast<OT>(s); }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
"""

_HEADER_NAX = _HEADER + """
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""

# Dense scoring on the M5 neural accelerators: one `matmul2d` (MM keys x 32 heads x 128) per task, results through threadgroup memory, one lane per key for the head reduction.
_SCORE_NAX = r"""
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tg = threadgroup_position_in_grid.x;
    const uint tid = sg * 32 + lane;
    threadgroup half Qs[32 * 128];
    threadgroup float Wh[32];
    threadgroup float scr[NSG][MM * 33];
    for (uint i = tid; i < 32 * 128; i += NSG * 32) { Qs[i] = q[i]; }
    if (tid < 32) { Wh[tid] = float(w[tid]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nn = n[0];
    constexpr auto desc = matmul2d_descriptor(MM, 32, 128, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroup> op;
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tQ((threadgroup half*)Qs, dextents<int32_t, 2>(128, 32));
    for (uint j = 0; j < TPT; ++j) {
        const uint k0 = ((tg * NSG + sg) * TPT + j) * MM;
        if (k0 >= nn) { break; }
        const int rem = int(metal::min(uint(MM), nn - k0));
        tensor<device half, dextents<int32_t, 2>, tensor_inline> tK((device half*)(km + ulong(k0) * 128), dextents<int32_t, 2>(128, rem));
        auto cT = op.template get_destination_cooperative_tensor<decltype(tK), decltype(tQ), float>();
        op.run(tK, tQ, cT);
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
            if (cT.is_valid_element(i)) {
                const auto ix = cT.get_multidimensional_index(i);
                scr[sg][uint(ix[1]) * 33 + uint(ix[0])] = cT[i];
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < uint(rem)) {
            float s = 0.0f;
            for (uint h = 0; h < 32; ++h) {
                const float cf = scr[sg][lane * 33 + h];
                if (F32) { s += metal::max(cf, 0.0f) * Wh[h]; }
                else { half c = static_cast<half>(cf); c = c > static_cast<half>(0) ? c : static_cast<half>(0); s += float(static_cast<half>(c * static_cast<half>(Wh[h]))); }
            }
            out[k0 + lane] = static_cast<OT>(s);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

# Rows on the neural accelerators: _SCORE_NAX with the row axis of _SCORE_ROWS (one row per threadgroup block), which is the kernel the solo
# decode step serves (`scores` takes it whenever there is no candidate mask and the device has matmul2d). Row u holds scores(q[u], w[u],
# keys[sess[u]], lens[u]) with no mask, so a plain step and a verify round score a position the same way.
_SCORE_ROWS_NAX = r"""
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tg = threadgroup_position_in_grid.x;
    const uint tid = sg * 32 + lane;
    const uint Tm = ip[0];
    const uint Ck = ip[1];
    const uint nt = (Tm + NSG * TPT * MM - 1) / (NSG * TPT * MM);
    const uint u = tg / nt;
    const uint gt = tg % nt;
    threadgroup half Qs[32 * 128];
    threadgroup float Wh[32];
    threadgroup float scr[NSG][MM * 33];
    for (uint i = tid; i < 32 * 128; i += NSG * 32) { Qs[i] = q[u * 32 * 128 + i]; }
    if (tid < 32) { Wh[tid] = float(wts[u * 32 + tid]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nn = metal::min(uint(lens[u]), Tm);
    const device half* km = keys + ulong(sess[u]) * Ck * 128;
    constexpr auto desc = matmul2d_descriptor(MM, 32, 128, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroup> op;
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tQ((threadgroup half*)Qs, dextents<int32_t, 2>(128, 32));
    for (uint j = 0; j < TPT; ++j) {
        const uint k0 = ((gt * NSG + sg) * TPT + j) * MM;
        if (k0 >= nn) { break; }
        const int rem = int(metal::min(uint(MM), nn - k0));
        tensor<device half, dextents<int32_t, 2>, tensor_inline> tK((device half*)(km + ulong(k0) * 128), dextents<int32_t, 2>(128, rem));
        auto cT = op.template get_destination_cooperative_tensor<decltype(tK), decltype(tQ), float>();
        op.run(tK, tQ, cT);
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
            if (cT.is_valid_element(i)) {
                const auto ix = cT.get_multidimensional_index(i);
                scr[sg][uint(ix[1]) * 33 + uint(ix[0])] = cT[i];
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < uint(rem)) {
            float s = 0.0f;
            for (uint h = 0; h < 32; ++h) {
                const float cf = scr[sg][lane * 33 + h];
                half c = static_cast<half>(cf); c = c > static_cast<half>(0) ? c : static_cast<half>(0);
                s += float(static_cast<half>(c * static_cast<half>(Wh[h])));
            }
            out[ulong(u) * Tm + k0 + lane] = static_cast<half>(s);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

# Rows over the fp4-packed index keys (flag dec_keys_codes, keyscodes.py): _SCORE_ROWS with the key tile read from the codes. The cdecpack view / the stackpack stack
# hold a key row as 64 code bytes + 4 ue8m0 scale bytes (68 B) and the core used to unpack the WHOLE layer ([S, cap, 128] fp16, 256 B per key written and read back) for
# every score call of every step. Here a threadgroup unpacks the 8-key tile it is about to multiply (the same `sign * LUT[code] * 2^(scale - 127)` as fp4pack.unpack /
# fp4_key, one exp2 per byte pair) into threadgroup memory and runs the simdgroup 8x8 matrices of _SCORE_ROWS on it: the per-key products, the head reduction and the
# fp16 rounding points are _SCORE_ROWS's (the `nax=False` form), so a row holds the values scores_rows(nax=False) gives over the unpacked keys. A masked 8-key block is
# skipped before its codes are read (consumer layers: 1/8 of the keys). Row u reads the codes of buffer slot sess[u] (Plan.sess, NOT the session rank), `Ck` rows per slot.
_SCORE_ROWS_P = r"""
    const uint tg = threadgroup_position_in_grid.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tid = sg * 32 + lane;
    const uint Tm = ip[0];
    const uint Ck = ip[1];
    const uint Tc = ip[2];
    const uint nt = (Tm + NSG * TPT * 8 - 1) / (NSG * TPT * 8);
    const uint u = tg / nt;
    const uint gt = tg % nt;
    threadgroup T Qs[32 * 128];
    threadgroup float Wh[32];
    threadgroup float scr[NSG][8 * 32];
    threadgroup T stage[NSG][8 * 128];
    for (uint i = tid; i < 32 * 128; i += NSG * 32) { Qs[i] = q[ulong(u) * 32 * 128 + i]; }
    if (tid < 32) { Wh[tid] = float(wts[u * 32 + tid]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nn = metal::min(uint(lens[u]), Tm);
    const ulong kbase = ulong(sess[u]) * Ck;
    const device bool* cm = CAND ? (cand + ulong(u) * Tc) : cand;
    const uint key = lane & 7, qt = lane >> 3;
    for (uint j = 0; j < TPT; ++j) {
        const uint kt = ((gt * NSG + sg) * TPT + j) * 8;
        if (kt >= nn) { break; }
        if (HASMASK && !cm[kt]) { if (lane < 8 && kt + lane < Tm) { out[ulong(u) * Tm + kt + lane] = static_cast<OT>(-INFINITY); } continue; }
        for (uint i = lane; i < 8 * 64; i += 32) {
            const uint r = i >> 6, bi = i & 63u, kk = kt + r;
            T v0 = static_cast<T>(0), v1 = static_cast<T>(0);
            if (kk < nn) {
                const ulong row = kbase + kk;
                const uint b = uint(cod[row * 64 + bi]);
                const float sv = metal::exp2(float(sca[row * 4 + (bi >> 4)]) - 127.0f);
                const uint c0 = b & 15u, c1 = b >> 4;
                const float m0 = FP4_LUT[c0 & 7u], m1 = FP4_LUT[c1 & 7u];
                v0 = static_cast<T>(((c0 & 8u) ? -m0 : m0) * sv);
                v1 = static_cast<T>(((c1 & 8u) ? -m1 : m1) * sv);
            }
            stage[sg][r * 128 + 2 * bi] = v0;
            stage[sg][r * 128 + 2 * bi + 1] = v1;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_matrix<float, 8, 8> acc[4];
        for (uint hb = 0; hb < 4; ++hb) { acc[hb] = simdgroup_matrix<float, 8, 8>(0.0f); }
        for (uint ks = 0; ks < 16; ++ks) {
            simdgroup_matrix<T, 8, 8> a;
            simdgroup_load(a, &stage[sg][ks * 8], 128);
            for (uint hb = 0; hb < 4; ++hb) {
                simdgroup_matrix<T, 8, 8> b;
                simdgroup_load(b, &Qs[hb * 8 * 128 + ks * 8], 128, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc[hb], a, b, acc[hb]);
            }
        }
        for (uint hb = 0; hb < 4; ++hb) { simdgroup_store(acc[hb], &scr[sg][hb * 8], 32); }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float s = 0.0f;
        for (uint h = qt * 8; h < qt * 8 + 8; ++h) {
            const float cf = scr[sg][key * 32 + h];
            T c = static_cast<T>(cf); c = c > static_cast<T>(0) ? c : static_cast<T>(0); s += float(static_cast<T>(c * static_cast<T>(Wh[h])));
        }
        s += simd_shuffle_xor(s, 8u);
        s += simd_shuffle_xor(s, 16u);
        if (lane < 8 && kt + lane < nn) { out[ulong(u) * Tm + kt + lane] = static_cast<OT>(s); }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

_SELECT = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    constexpr uint NT = 1024;
    constexpr uint NS = NT / 32;
    const uint nn = n[0];
    const uint cnt = (MODE == 1) ? ((nn + 7) / 8) : nn;
    const uint Kk = metal::min(uint(K), cnt);
    const uint lastb = (nn - 1) / 8;
    threadgroup atomic_uint hist[256];
    threadgroup uint sh[8];
    threadgroup uint red[NS];
    #define ITEM_KEY(i) ((MODE == 1) ? ((uint(i) == lastb) ? 0xFFFFu : skey(static_cast<T>(blockmax(sc, nn, uint(i))))) : ((HASV && !vmask[i]) ? 0u : skey(sc[i])))
    // radix select, NP passes of 8 bits from the top: after pass p `pre` is the top 8 (p + 1) bits of the K-th key, `above` the number of keys strictly above it
    uint pre = 0, above = 0;
    for (uint p = 0; p < uint(NP); ++p) {
        const uint shift = 8u * (uint(NP) - 1u - p);
        if (tid < 256) { atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < cnt; i += NT) {
            const uint k = ITEM_KEY(i);
            if (p == 0 || (k >> (shift + 8u)) == pre) { atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            const uint need = Kk - above;
            uint cum = 0, b = 255;
            for (;;) { uint c = atomic_load_explicit(&hist[b], memory_order_relaxed); if (cum + c >= need || b == 0) { break; } cum += c; --b; }
            sh[0] = b; sh[1] = above + cum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        pre = (p == 0 ? 0u : (pre << 8)) | sh[0];
        above = sh[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint T0 = pre;
    const uint needTie = Kk - above;
    const uint R = (cnt + NT - 1) / NT;
    const uint lo = tid * R, hi_i = metal::min(lo + R, cnt);
    uint gt = 0, eq = 0;
    for (uint i = lo; i < hi_i; ++i) { uint k = ITEM_KEY(i); gt += (k > T0); eq += (k == T0); }
    uint tot;
    const uint eqPref = tg_excl_scan(eq, red, sg, lane, NS, tot);
    const uint tieTake = (eqPref >= needTie) ? 0u : metal::min(eq, needTie - eqPref);
    if (MODE == 0) {
        uint total2;
        uint pos = tg_excl_scan(gt + tieTake, red, sg, lane, NS, total2);
        uint seen = 0;
        for (uint i = lo; i < hi_i; ++i) {
            uint k = ITEM_KEY(i);
            if (k > T0) { out[pos++] = int(i); }
            else if (k == T0) { if (seen < tieTake) { out[pos++] = int(i); } seen++; }
        }
    } else {
        uint seen = 0;
        for (uint i = lo; i < hi_i; ++i) {
            uint k = ITEM_KEY(i);
            bool sel = (k > T0);
            if (k == T0) { sel = seen < tieTake; seen++; }
            sel = sel && (blockmax(sc, nn, i) > -INFINITY);
            uint e = metal::min(i * 8 + 8, nn);
            for (uint j = i * 8; j < e; ++j) { out[j] = sel; }
        }
    }
"""

_k: dict = {}
USE_NAX = os.environ.get("KYOJIN_DSA_NAX", "1") != "0"
NAX_NSG, NAX_TPT, NAX_MM = 4, 4, 32


def _nax() -> bool:
    if "nax" not in _k:
        from .metal_nax import nax_available
        _k["nax"] = bool(nax_available())
    return _k["nax"]


def _nomask() -> mx.array:
    if "nomask" not in _k:
        _k["nomask"] = mx.zeros((8,), dtype=mx.bool_)
    return _k["nomask"]


def scores(q: mx.array, w: mx.array, keys: mx.array, n: mx.array, n_host: int, cand: mx.array | None = None, f32: bool = False) -> mx.array:
    """q [32, 128] (rotated, fp4 round trip), w [32], keys [cap, 128] (the whole key buffer), n int32 [1] -> scores [n_pad] in q.dtype (-inf on masked blocks).
    f32=True: no fp16 rounding of the product, fp32 output (GLM indexer: relu(q.k) * w summed in fp32)."""
    dt = q.dtype
    odt = mx.float32 if f32 else dt
    if cand is None and USE_NAX and dt == mx.float16 and _nax():
        key = ("score_nax",)
        k = _k.get(key)
        if k is None:
            k = _k[key] = mx.fast.metal_kernel(name="kyojin_dsa_score_nax", input_names=["q", "w", "km", "n"], output_names=["out"], source=_SCORE_NAX, header=_HEADER_NAX)
        per = NAX_NSG * NAX_TPT * NAX_MM
        ntg = -(-n_host // per)
        return k(inputs=[q, w, keys, n], template=[("NSG", NAX_NSG), ("TPT", NAX_TPT), ("MM", NAX_MM), ("F32", f32), ("OT", odt)], grid=(ntg * NAX_NSG * 32, 1, 1),
                 threadgroup=(NAX_NSG * 32, 1, 1), output_shapes=[(ntg * per,)], output_dtypes=[odt])[0]
    key = ("score", str(dt), f32)
    has = cand is not None
    if cand is None:
        cand = _nomask()
    k = _k.get(key)
    if k is None:
        names = ["q", "w", "km", "n", "cand"]
        k = _k[key] = mx.fast.metal_kernel(name="kyojin_dsa_score", input_names=names, output_names=["out"], source=_SCORE, header=_HEADER)
    per = NSG * TPT * 8
    ntg = -(-n_host // per)
    return k(inputs=[q, w, keys, n, cand],
             template=[("T", dt), ("NSG", NSG), ("TPT", TPT), ("GT", 1 if has else GT), ("HASMASK", has), ("F32", f32), ("OT", odt)],
             grid=(ntg * NSG * 32, 1, 1), threadgroup=(NSG * 32, 1, 1), output_shapes=[(ntg * per,)], output_dtypes=[odt])[0]


def scores_packed(q: mx.array, w: mx.array, codes: mx.array, scales: mx.array, n: mx.array, n_host: int,
                  cand: mx.array | None = None) -> mx.array:
    """`scores` over key rows stored as fp4 codes (kyojin_mlx.fp4pool.KeySlots: BLOCK 32, power-of-two scale): codes uint8 [cap, 64], scales uint8 [cap, 4],
    n int32 [1]. Same values as reading the fp16 / bf16 rows `fp4pack.unpack(codes, scales, 32, False, dtype)` gives, which is what the cache holds."""
    dt = q.dtype
    assert dt in (mx.float16, mx.bfloat16), "packed key scores are fp16 / bf16 only"
    has = cand is not None
    if cand is None:
        cand = _nomask()
    k = _k.get(("score_p", str(dt)))
    if k is None:
        k = _k[("score_p", str(dt))] = mx.fast.metal_kernel(name="kyojin_dsa_score_p", input_names=["q", "w", "cod", "sca", "n", "cand"],
                                                            output_names=["out"], source=_SCORE_P, header=_HEADER_P)
    per = NSG * TPT * 8
    ntg = -(-n_host // per)
    return k(inputs=[q, w, codes, scales, n, cand],
             template=[("T", dt), ("NSG", NSG), ("TPT", TPT), ("HASMASK", has), ("F32", False), ("OT", dt)],
             grid=(ntg * NSG * 32, 1, 1), threadgroup=(NSG * 32, 1, 1), output_shapes=[(ntg * per,)], output_dtypes=[dt])[0]


def rows_supported(nh: int, di: int, dt) -> bool:
    """Geometry of `scores_rows`: the solo scoring kernel of one row (32 heads x 128 channels, fp16 / bf16)."""
    return nh == 32 and di == 128 and dt in (mx.float16, mx.bfloat16)


def scores_rows(q: mx.array, w: mx.array, keys: mx.array, lens: mx.array, t_max: int, cand: mx.array | None = None,
                sess: mx.array | None = None, nax: bool | None = None) -> mx.array:
    """`scores` for every row of a pass in one launch: q [N, 32, 128] (rotated, fp4 round trip), w [N, 32], keys [S, C, 128], lens int32 [N] (valid
    keys per row), sess int32 [N] (which key buffer the row reads) -> scores [N, t_max] in q's dtype, -inf where t >= lens[u] or the candidate mask
    drops the key. Row u holds exactly the values `scores(q[u], w[u], keys[sess[u]], lens[u])` holds (same simdgroup matrix arithmetic, same
    rounding points, same head reduction); columns >= t_max are not written. `nax` picks the kernel: None = the neural accelerators when the device has
    them and the row is fp16 without a mask, True / False forces it (False: the simdgroup form, which a caller that must hold the masked path's values
    bit for bit takes - see idxblocks)."""
    N, NH, DI = q.shape
    dt = q.dtype
    assert rows_supported(NH, DI, dt) and w.shape == (N, NH) and lens.shape == (N,) and t_max <= keys.shape[1]
    has = cand is not None
    tc = cand.shape[-1] if has else 1
    assert cand is None or tc >= t_max
    if cand is None:
        cand = _nomask()
    if sess is None:
        sess = mx.arange(N, dtype=mx.int32)
    nax = ((not has) and dt == mx.float16 and USE_NAX and _nax()) if nax is None else (bool(nax) and not has and dt == mx.float16)
    key = ("rows_nax",) if nax else ("rows", str(dt), has)
    k = _k.get(key)
    if k is None:
        if nax:
            k = _k[key] = mx.fast.metal_kernel(name="kyojin_dsa_score_rows_nax", input_names=["q", "wts", "keys", "lens", "cand", "sess", "ip"],
                                               output_names=["out"], source=_SCORE_ROWS_NAX, header=_HEADER_NAX)
        else:
            k = _k[key] = mx.fast.metal_kernel(name="kyojin_dsa_score_rows", input_names=["q", "wts", "keys", "lens", "cand", "sess", "ip"],
                                               output_names=["out"], source=_SCORE_ROWS, header=_HEADER)
    tmpl = ([("NSG", NAX_NSG), ("TPT", NAX_TPT), ("MM", NAX_MM)] if nax else
            [("T", dt), ("NSG", NSG), ("TPT", TPT), ("GT", 1 if has else GT), ("HASMASK", has), ("CAND", has), ("OT", dt)])
    ntg, nt_t = (NAX_NSG, NAX_NSG * NAX_TPT * NAX_MM) if nax else (NSG, NSG * TPT * 8)
    nt = -(-t_max // nt_t)
    return k(inputs=[q, w, keys, lens, cand, sess, mx.array([t_max, keys.shape[1], tc], dtype=mx.uint32)],
             template=tmpl, grid=(N * nt * ntg * 32, 1, 1), threadgroup=(ntg * 32, 1, 1),
             output_shapes=[(N, t_max)], output_dtypes=[dt])[0]


def rows_codes_supported(nh: int, di: int, dt, block: int, e4m3: bool) -> bool:
    """Geometry of `scores_rows_codes`: 32 heads x 128 channels, fp16 / bf16 queries, index keys on the ue8m0 / block 32 codec (KeySlots)."""
    return rows_supported(nh, di, dt) and block == 32 and not e4m3


def scores_rows_codes(q: mx.array, w: mx.array, codes: mx.array, scales: mx.array, lens: mx.array, t_max: int, cand: mx.array | None = None,
                      sess: mx.array | None = None) -> mx.array:
    """`scores_rows(nax=False)` over fp4-packed key rows: q [N, 32, 128], w [N, 32], codes uint8 [S, Ck, 64], scales uint8 [S, Ck, 4], lens int32 [N], sess [N]
    (the buffer slot of each row, `Plan.sess`) -> scores [N, t_max] in q's dtype, -inf where the candidate mask drops the key; columns >= lens[u] are not written.
    The same values `scores_rows(q, w, fp4pack.unpack(codes, scales, 32, False, q.dtype), lens, t_max, cand, sess, nax=False)` holds, with 68 B per key read
    instead of a 256 B dense layer written and read back."""
    N, NH, DI = q.shape
    dt = q.dtype
    assert rows_codes_supported(NH, DI, dt, 32, False) and w.shape == (N, NH) and lens.shape == (N,)
    assert codes.dtype == mx.uint8 and scales.dtype == mx.uint8 and codes.shape[-1] == DI // 2 and scales.shape[-1] == DI // 32
    assert codes.shape[:2] == scales.shape[:2] and t_max <= codes.shape[1]
    has = cand is not None
    tc = cand.shape[-1] if has else 1
    assert cand is None or tc >= t_max
    if cand is None:
        cand = _nomask()
    if sess is None:
        sess = mx.arange(N, dtype=mx.int32)
    key = ("rows_codes", str(dt), has)
    k = _k.get(key)
    if k is None:
        k = _k[key] = mx.fast.metal_kernel(name="kyojin_dsa_score_rows_codes", input_names=["q", "wts", "cod", "sca", "lens", "cand", "sess", "ip"],
                                           output_names=["out"], source=_SCORE_ROWS_P, header=_HEADER_P)
    nt = -(-t_max // (NSG * TPT * 8))
    return k(inputs=[q, w, codes, scales, lens, cand, sess, mx.array([t_max, codes.shape[1], tc], dtype=mx.uint32)],
             template=[("T", dt), ("NSG", NSG), ("TPT", TPT), ("HASMASK", has), ("CAND", has), ("OT", dt)],
             grid=(N * nt * NSG * 32, 1, 1), threadgroup=(NSG * 32, 1, 1), output_shapes=[(N, t_max)], output_dtypes=[dt])[0]


def select(sc: mx.array, n: mx.array, n_host: int, topk: int, valid: mx.array | None = None) -> mx.array:
    """Indices (int32, ascending) of the min(topk, n) largest scores, ties to the lower index. `valid` (bool [n]) sends invalid keys to the bottom."""
    has = valid is not None
    key = ("sel0", str(sc.dtype), has)
    k = _k.get(key)
    if k is None:
        k = _k[key] = mx.fast.metal_kernel(name="kyojin_dsa_select", input_names=["sc", "n", "vmask"], output_names=["out"], source=_SELECT, header=_HEADER)
    np_ = 4 if sc.dtype == mx.float32 else 2
    return k(inputs=[sc, n, valid if has else _nomask()], template=[("T", sc.dtype), ("K", topk), ("MODE", 0), ("NP", np_), ("HASV", has)], grid=(1024, 1, 1), threadgroup=(1024, 1, 1),
             output_shapes=[(min(topk, n_host),)], output_dtypes=[mx.int32])[0]


def candidate_mask(sc: mx.array, n: mx.array, n_host: int, topk_blocks: int, block: int = 8) -> mx.array:
    """The stock `select_candidate_blocks` for one query: bool [n]."""
    assert block == 8
    key = ("sel1", str(sc.dtype))
    k = _k.get(key)
    if k is None:
        k = _k[key] = mx.fast.metal_kernel(name="kyojin_dsa_candidates", input_names=["sc", "n", "vmask"], output_names=["out"], source=_SELECT, header=_HEADER)
    return k(inputs=[sc, n, _nomask()], template=[("T", sc.dtype), ("K", topk_blocks), ("MODE", 1), ("NP", 2), ("HASV", False)], grid=(1024, 1, 1), threadgroup=(1024, 1, 1),
             output_shapes=[(n_host,)], output_dtypes=[mx.bool_])[0]
