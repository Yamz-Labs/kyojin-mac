# SPDX-License-Identifier: Apache-2.0
"""GLM-5.x DSA attend of B sessions x L rows in a handful of well-parallel launches (flag `KYOJIN_DSA_ATTEND_ROWS`, V2-VERIFY).

The stock attend of a DSA layer after the indexer is, per row, embed_q + `mx.fast.scaled_dot_product_attention` (D = 512 is outside MLX's fused kernel: q * scale, QK^T
gemm, mask select, precise softmax, split-K P @ V + accumulate pass = ~8 launches, the split-K P @ V alone ~90 us at 2051 keys) + unembed_out: ~150 us per layer and row in situ.
`glm_mla` (G10d) fused it into one flash kernel per (session, row) that ran 2-4x SLOWER than stock (a threadgroup of ONE simdgroup with 26 KB of threadgroup memory: ~1-2
resident simdgroups per core, hi / lo split of P doubling the PV mma). Here the same arithmetic as the stock path, split so that every launch has thousands of independent simdgroups
and no threadgroup memory worth the name:
    scores    S[u, h, k] = (q * scale) . key_k          one simdgroup per (32 keys, 8 heads, row): 8x8 mma over 512 dims, fp32 accumulate, stored fp32
    softmax   P = softmax(mask(half(S))) in fp32, P rounded to the model dtype (the stock rounding points: scores to fp16, precise softmax, P to fp16)
    PV        partial[sp] = P . V over a split of the keys, one simdgroup per (split, 8 heads, 64 output columns, row): fp32 accumulate
    sum       out = dtype(sum_sp partial[sp])
embed_q / unembed_out stay the stock MultiLinear calls, one per row (the shape of the stock call: a matmul over M rows does not give row j the bits of the M = 1 call). Row invariance: every row is reduced on its own, in an order that does
not depend on how many rows / sessions share the launch, so verify row i == plain decode row i bit for bit when the flag is on for both. Versus the stock path only the fp32 summation
order of the two matmuls differs (a fp16 rounding flips only when a sum lands on a rounding boundary: ~1 element in 8000), i.e. 1-ulp-class logits.

q [B, 64, NQ, 512] (embed_q output), rows [B * NQ * KS, 512] / vm from `glm_mla.gather_pad` (KS = K rounded up to 32; rows >= K zero and invalid) or `glm_dsa.gather_rows`.
"""
from __future__ import annotations

import os

import mlx.core as mx

from . import fired as FD

ENABLED = os.environ.get("KYOJIN_DSA_ATTEND_ROWS", "0") not in ("0", "")
CPS = int(os.environ.get("KYOJIN_DSA_ATTEND_CPS", "5"))              # 32-key chunks per PV split
MAX_U = 16                                                          # (session, row) pairs per call
STATS = {"fused": 0}
_k: dict = {}

_SCORES = r"""
    const uint ch = threadgroup_position_in_grid.x;
    const uint hg = threadgroup_position_in_grid.y;
    const uint u = threadgroup_position_in_grid.z;
    const uint lane = thread_index_in_simdgroup;
    const uint b = u / NQ, j = u % NQ;
    threadgroup float scr[32 * 8];
    simdgroup_matrix<float, 8, 8> acc[4];
    for (int t = 0; t < 4; ++t) { acc[t] = simdgroup_matrix<float, 8, 8>(0.0f); }
    const device T* qp = q + (((size_t)b * 64 + hg * 8) * NQ + j) * 512;
    const device T* kp = rows + ((size_t)u * KS + ch * 32) * 512;
    for (int ks = 0; ks < 64; ++ks) {
        simdgroup_matrix<T, 8, 8> qb;
        simdgroup_load(qb, qp + ks * 8, (ulong)NQ * 512, ulong2(0, 0), true);
        for (int t = 0; t < 4; ++t) {
            simdgroup_matrix<T, 8, 8> ka;
            simdgroup_load(ka, kp + t * 8 * 512 + ks * 8, 512);
            simdgroup_multiply_accumulate(acc[t], ka, qb, acc[t]);
        }
    }
    for (int t = 0; t < 4; ++t) { simdgroup_store(acc[t], &scr[t * 64], 8); }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lane; i < 256; i += 32) {
        const uint key = i / 8, hh = i % 8;
        S[((size_t)u * 64 + hg * 8 + hh) * KS + ch * 32 + key] = scr[i];
    }
"""

_SOFTMAX = r"""
    const uint row = threadgroup_position_in_grid.x;
    const uint u = row / 64;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    threadgroup float red[8];
    float v[NPT];
    float m = -INFINITY;
    for (uint i = 0; i < NPT; ++i) {
        const uint k = tid + 256 * i;
        float x = -INFINITY;
        if (k < KS && vm[u * KS + k]) { x = float(static_cast<T>(S[(size_t)row * KS + k])); }
        v[i] = x;
        m = metal::max(m, x);
    }
    m = simd_max(m);
    if (lane == 0) { red[sg] = m; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mm = red[0];
    for (uint i = 1; i < 8; ++i) { mm = metal::max(mm, red[i]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.0f;
    for (uint i = 0; i < NPT; ++i) {
        const float e = (v[i] == -INFINITY) ? 0.0f : metal::exp(v[i] - mm);
        v[i] = e;
        s += e;
    }
    s = simd_sum(s);
    if (lane == 0) { red[sg] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint i = 0; i < 8; ++i) { tot += red[i]; }
    if (!(tot > 0.0f)) { tot = 1.0f; }
    for (uint i = 0; i < NPT; ++i) {
        const uint k = tid + 256 * i;
        if (k < KS) { P[(size_t)row * KS + k] = static_cast<T>(v[i] / tot); }
    }
"""

_PV = r"""
    const uint sp = threadgroup_position_in_grid.x;
    const uint hg = threadgroup_position_in_grid.y / 8, cg = threadgroup_position_in_grid.y % 8;
    const uint u = threadgroup_position_in_grid.z;
    const uint b = u / NQ, j = u % NQ;
    simdgroup_matrix<float, 8, 8> acc[8];
    for (int c = 0; c < 8; ++c) { acc[c] = simdgroup_matrix<float, 8, 8>(0.0f); }
    for (int c = 0; c < CPS; ++c) {
        const int ch = int(sp) * CPS + c;
        if (ch >= NCH) { break; }
        for (int t = 0; t < 4; ++t) {
            const size_t k0 = (size_t)ch * 32 + t * 8;
            simdgroup_matrix<T, 8, 8> pa;
            simdgroup_load(pa, P + ((size_t)u * 64 + hg * 8) * KS + k0, KS);
            const device T* vp = rows + ((size_t)u * KS + k0) * 512 + cg * 64;
            for (int cb = 0; cb < 8; ++cb) {
                simdgroup_matrix<T, 8, 8> vb;
                simdgroup_load(vb, vp + cb * 8, 512);
                simdgroup_multiply_accumulate(acc[cb], pa, vb, acc[cb]);
            }
        }
    }
    device float* dst = part + ((((size_t)sp * BB + b) * 64 + hg * 8) * NQ + j) * 512 + cg * 64;
    for (int cb = 0; cb < 8; ++cb) { simdgroup_store(acc[cb], dst + cb * 8, (ulong)NQ * 512); }
"""

_SUM = r"""
    const uint i = thread_position_in_grid.x;
    if (i >= NTOT) { return; }
    float a = 0.0f;
    for (uint s = 0; s < NSP; ++s) { a += part[(size_t)s * NTOT + i]; }
    out[i] = static_cast<T>(a);
"""


def ks_of(K: int) -> int:
    return -(-K // 32) * 32


def usable(self, q, topk, kv_cache, w_embed, w_unembed) -> bool:
    return bool(ENABLED and q.shape[1] == 64 and q.shape[3] == 256 and w_embed.shape == (64, 512, 256) and w_unembed.shape == (64, 256, 512)
                and w_embed.dtype == q.dtype == w_unembed.dtype and q.dtype in (mx.float16, mx.bfloat16) and q.shape[0] * q.shape[2] <= MAX_U and CPS >= 1)


def _kernels(dt, B: int, NQ: int, KS: int, nsp: int, cps: int):
    from . import dsa_decode as D
    key = (str(dt), B, NQ, KS, nsp, cps)
    k = _k.get(key)
    if k is None:
        nch = KS // 32
        k1 = mx.fast.metal_kernel(name="kyojin_dsa_att_scores", input_names=["q", "rows"], output_names=["S"], header=D._HEADER, source=_SCORES)
        k2 = mx.fast.metal_kernel(name="kyojin_dsa_att_softmax", input_names=["S", "vm"], output_names=["P"], header=D._HEADER, source=_SOFTMAX)
        k3 = mx.fast.metal_kernel(name="kyojin_dsa_att_pv", input_names=["P", "rows"], output_names=["part"], header=D._HEADER, source=_PV)
        k4 = mx.fast.metal_kernel(name="kyojin_dsa_att_sum", input_names=["part"], output_names=["out"], header=D._HEADER, source=_SUM)
        k = _k[key] = (k1, k2, k3, k4, nch)
    return k


def core(q_s, rows, vm, B: int, NQ: int, K: int, cps: int = 0):
    """q_s [B, 64, NQ, 512] = embed_q(q) * scale (model dtype), rows [B * NQ * KS, 512], vm [B * NQ * KS] -> attention output o [B, 64, NQ, 512] (model dtype), the input of unembed_out."""
    dt = q_s.dtype
    cps = cps or CPS
    KS = ks_of(K)
    U = B * NQ
    nch = KS // 32
    nsp = -(-nch // cps)
    k1, k2, k3, k4, _ = _kernels(dt, B, NQ, KS, nsp, cps)
    S = k1(inputs=[q_s, rows], template=[("T", dt), ("NQ", NQ), ("KS", KS)], grid=(nch * 32, 8, U), threadgroup=(32, 1, 1), output_shapes=[(U * 64 * KS,)], output_dtypes=[mx.float32])[0]
    npt = -(-KS // 256)
    P = k2(inputs=[S, vm], template=[("T", dt), ("KS", KS), ("NPT", npt)], grid=(U * 64 * 256, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(U * 64 * KS,)], output_dtypes=[dt])[0]
    part = k3(inputs=[P, rows], template=[("T", dt), ("NQ", NQ), ("KS", KS), ("CPS", cps), ("NCH", nch), ("BB", B)], grid=(nsp * 32, 64, U), threadgroup=(32, 1, 1),
              output_shapes=[(nsp * B * 64 * NQ * 512,)], output_dtypes=[mx.float32])[0]
    ntot = B * 64 * NQ * 512
    return k4(inputs=[part], template=[("T", dt), ("NTOT", ntot), ("NSP", nsp)], grid=(ntot, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(B, 64, NQ, 512)], output_dtypes=[dt])[0]


def _embed(self, q, start: int):
    """embed_q per row (the stock call shape): [B, 64, nq, 512]"""
    outs = [self.embed_q(q[:, :, i:i + 1]) for i in range(start, int(q.shape[2]))]
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)


def finish(self, o, B: int, NQ: int):
    """unembed_out + the layout o_proj reads: [B, 64, NQ, 512] -> [B, NQ, 64 * 256]."""
    outs = [self.unembed_out(o[:, :, j:j + 1]) for j in range(NQ)]          # one call per row like the stock path: a matmul with M rows does not give row j the bits of the M = 1 call
    out = outs[0] if NQ == 1 else mx.concatenate(outs, axis=2)
    return out.transpose(0, 2, 1, 3).reshape(B, NQ, -1)


def attend_batch(self, q, topk, kv_cache, start: int, last_only: bool):
    """Dense latent of B sessions (LatentBatch / BatchKVCache): q [B, 64, Lr, 256], topk [B, Lr, K] -> o [B, nq, 64 * 256]."""
    from . import glm_mla as MLA
    B, _, Lr, _ = q.shape
    nq = Lr - start
    K = int(topk.shape[-1])
    rows, vm = MLA.gather_pad(kv_cache.keys, topk[:, start:].reshape(-1), B, nq, K, int(kv_cache._idx))
    q_lat = _embed(self, q, start)
    STATS["fused"] += 1
    FD.fire("dsa_attend_rows")
    return finish(self, core(q_lat * self.scale, rows, vm, B, nq, K), B, nq)


def attend_rows(self, q, topk, kv_cache, start: int):
    """ONE session whose latent is a LatentTQCache: q [1, 64, Lr, 256], topk [1, Lr, K]."""
    from . import glm_dsa as G
    nq = int(q.shape[2]) - start
    K = int(topk.shape[-1])
    KS = ks_of(K)
    tk = mx.pad(topk[0, start:], [(0, 0), (0, KS - K)], constant_values=-1) if KS != K else topk[0, start:]
    rows, vm = G.gather_rows(kv_cache, tk.reshape(-1), q.dtype)
    q_lat = _embed(self, q, start)
    STATS["fused"] += 1
    FD.fire("dsa_attend_rows")
    return finish(self, core(q_lat * self.scale, rows, vm, 1, nq, K), 1, nq)
