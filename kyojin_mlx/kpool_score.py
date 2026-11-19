# SPDX-License-Identifier: Apache-2.0
"""Flag kpool_score (KYOJIN_KPOOL_SCORE, default off, read per call): the indexer scores of a block-sparse (kpool) layer in ONE multi-threadgroup launch, any query rows.

Stock scorer (foreign_skeleton_kpool.select_pools): kq = relu(iq @ pk^T) as fp32 [B, H, L, J], times the head weights, summed over H -> score [B, L, J], then a full-row argsort. The
[B, H, L, J] fp32 temporaries are H times the size of the score matrix (a 2048-row chunk at 64K tokens: 16 x 2048 x 16384 x 4 B = 2 GiB per pass, 12 indexer layers), and the decode
kernel of kpool_fuse (one threadgroup per query row, one pool per thread, scalar d-loops over device memory) leaves the GPU idle at long context.
Here, score[b, l, j] = sum_h relu(iq[b, h, l] . pk[b, j]) * w[b, l, h] with the dot on fp32 simdgroup 8 x 8 matrix tiles (fp32 operands and accumulation: the stock matmul's
precision, only the order of the d-sum differs, ~1e-6 relative):
  one simdgroup = one 16-key unit x RTP row tiles of 8 rows; its K fragments (16 d-blocks x 2 key tiles) stay in registers across all heads and row tiles;
  per head: 16 MMAs per key tile, the 8 x 8 product goes through threadgroup memory, each lane applies relu * w[row, h] to its 2 elements and adds them to its accumulator;
  the visibility test of the stock path (pool j visible to the token at position q = off + l: j < q // kp if select_tail else (q + 1) // kp, and valid[j]) is applied at the write (-inf).
Row tiles whose rows see no pool of the unit write -inf without any MMA (causal prompt chunks skip about half of the tiles).
Generic by shape: H / D come from the arrays (template constants), kp / select_tail are template constants. fp32 iq / pk / w, Metal only.
Top-k of the result: idxfast.topk_mt (exact, lower pool first on ties = the stable argsort rule) for prompt chunks; for decode rows kpool_fuse.select takes the scores as `pre`."""
from __future__ import annotations

import mlx.core as mx

ON = False                                                           # bound by flags.set (attrs); read per call
NSG = 4                                                              # simdgroups per threadgroup (each a 16-key unit)
RTP = 4                                                              # row tiles (8 rows) a simdgroup walks with its K fragments in registers

_HEADER = r"""#pragma METAL fp contract(off)
#include <metal_simdgroup_matrix>
#define KS_NSG 4
"""

_SRC = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint b = threadgroup_position_in_grid.z;
    const uint unit = threadgroup_position_in_grid.x * KS_NSG + sg;
    const uint Lp = par[0], L = par[1], J = par[2], Jp = par[3], RT = par[4];
    threadgroup float tile[KS_NSG * 2 * 64];
    const uint j0 = unit * 16u;
    if (j0 >= Jp) return;
    const uint rt0 = threadgroup_position_in_grid.y * RTP_;
    const uint rt1 = min(rt0 + RTP_, RT);
    threadgroup float* tl = tile + sg * 128;
    const uint r = lane / 4u, c = (lane % 4u) * 2u;
    const int offb = off[b];
    const float inf = metal::numeric_limits<float>::infinity();
    simdgroup_float8x8 kf[2][D_ / 8];
    bool loaded = false;
    for (uint rt = rt0; rt < rt1; ++rt) {
        const int qlast = offb + int(min(rt * 8u + 7u, L - 1u));
        const uint nvmax = min(uint(KSEL_ ? qlast / KP_ : (qlast + 1) / KP_), J);
        float acc[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
        if (j0 < nvmax) {
            if (!loaded) {
                for (uint t = 0; t < 2; ++t) {
                    const auto kb = pk + (ulong(b) * Jp + j0 + t * 8u) * D_;
                    for (uint i = 0; i < D_ / 8; ++i) { simdgroup_load(kf[t][i], kb + i * 8, D_, ulong2(0, 0), true); }
                }
                loaded = true;
            }
            for (uint h = 0; h < H_; ++h) {
                simdgroup_float8x8 s0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                simdgroup_float8x8 s1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                const device float* qb = iq + ((ulong(b) * H_ + h) * Lp + rt * 8u) * D_;
                for (uint i = 0; i < D_ / 8; ++i) {
                    simdgroup_float8x8 qf;
                    simdgroup_load(qf, qb + i * 8, D_);
                    simdgroup_multiply_accumulate(s0, qf, kf[0][i], s0);
                    simdgroup_multiply_accumulate(s1, qf, kf[1][i], s1);
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_store(s0, tl, 8);
                simdgroup_store(s1, tl + 64, 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                const float wv = w[(ulong(b) * Lp + rt * 8u + r) * H_ + h];
                for (uint t = 0; t < 2; ++t) {
                    acc[t][0] += metal::max(tl[t * 64 + r * 8 + c], 0.0f) * wv;
                    acc[t][1] += metal::max(tl[t * 64 + r * 8 + c + 1], 0.0f) * wv;
                }
            }
        }
        const uint row = rt * 8u + r;
        if (row < L) {
            const int q = offb + int(row);
            const uint nv = min(uint(KSEL_ ? q / KP_ : (q + 1) / KP_), J);
            for (uint t = 0; t < 2; ++t) {
                for (uint e = 0; e < 2; ++e) {
                    const uint j = j0 + t * 8u + c + e;
                    if (j < J) { S[(ulong(b) * L + row) * J + j] = (j < nv && valid[ulong(b) * J + j]) ? acc[t][e] : -inf; }
                }
            }
        }
    }
"""

_k: dict = {}


def use(L: int, B: int = 1) -> bool:
    return ON and mx.metal.is_available() and mx.default_device() == mx.gpu


def _kernel(H: int, D: int, kp: int, select_tail: bool):
    key = (H, D, kp, bool(select_tail))
    k = _k.get(key)
    if k is None:
        src = (_SRC.replace("H_", str(H)).replace("D_", str(D)).replace("RTP_", str(RTP)).replace("KP_", str(kp)).replace("KSEL_", str(int(bool(select_tail)))))
        k = _k[key] = mx.fast.metal_kernel(name=f"kyojin_kpool_score_{H}_{D}_{kp}_{int(bool(select_tail))}", input_names=["iq", "w", "pk", "valid", "off", "par"],
                                           output_names=["S"], header=_HEADER, source=src)
    return k


def scores(iq, w, pk, valid, off, kp: int, select_tail: bool):
    """iq [B, H, L, D] (roped at the token positions), w [B, L, H], pk [B, J, D], valid [B, J] bool, off int / int32 array [B] (cache offset before the L rows)
    -> fp32 [B, L, J]: the stock score with invisible pools = -inf."""
    B, H, L, D = iq.shape
    J = pk.shape[1]
    assert D % 8 == 0, D
    Lp, Jp = -(-L // 8) * 8, -(-J // 16) * 16
    iq = iq.astype(mx.float32)
    w = w.astype(mx.float32)
    if pk.dtype != mx.float16:                                       # an fp16 pooled-key cache (KYOJIN_HY4_FIXED_PKCACHE_F16) is read as is: half the bytes, the fragments widen to fp32 on load
        pk = pk.astype(mx.float32)
    if Lp != L:
        iq = mx.pad(iq, [(0, 0), (0, 0), (0, Lp - L), (0, 0)])
        w = mx.pad(w, [(0, 0), (0, Lp - L), (0, 0)])
    if Jp != J:
        pk = mx.pad(pk, [(0, 0), (0, Jp - J), (0, 0)])
    offv = mx.array(off, dtype=mx.int32) if not isinstance(off, mx.array) else off.astype(mx.int32)
    offv = mx.broadcast_to(offv.reshape(-1), (B,)) if offv.size == 1 else offv
    if valid.shape[0] != B:
        valid = mx.broadcast_to(valid, (B, J))
    RT = Lp // 8
    par = mx.array([Lp, L, J, Jp, RT], dtype=mx.uint32)
    gx, gy = -(-(Jp // 16) // NSG), -(-RT // RTP)
    return _kernel(H, D, kp, select_tail)(inputs=[iq, w, pk, valid, offv, par], grid=(gx * 32 * NSG, gy, B), threadgroup=(32 * NSG, 1, 1),
                                            output_shapes=[(B, L, J)], output_dtypes=[mx.float32])[0]


def select_pools_fast(iq, w, pk, valid, off, S, kp, top_k, select_tail, left=False):
    """foreign_skeleton_kpool.select_pools for batch 1 prompt chunks on the fused scorer + exact multi-threadgroup top-k (the same dict, `top` ascending by pool index)."""
    from . import foreign_skeleton_kpool as KP
    from . import idxfast as IX
    B, H, L, _ = iq.shape
    J = pk.shape[1]
    pad, off = KP.row_pad(off, L, S, B)
    if left:
        pad = mx.zeros_like(pad)
    qpos = off[:, None] + KP.ar(L)[None, :]
    score = scores(iq, w, pk, valid, off, kp, select_tail)                                       # [B, L, J], invisible = -inf
    j = KP.ar(J)[None, None, :]
    nvis = qpos // kp if select_tail else (qpos + 1) // kp
    vis = (j < nvis[..., None]) & valid[:, None, :]
    n_top = min(J, top_k // kp - int(bool(select_tail)))
    top = IX.topk_mt(score.reshape(L, J), mx.full((L,), J, dtype=mx.int32), n_top)[None].astype(mx.int32)   # [1, L, n_top]
    sel = mx.put_along_axis(mx.zeros(score.shape, dtype=mx.bool_), top, mx.ones(top.shape, dtype=mx.bool_), axis=-1) & vis
    q = qpos[..., None]
    gstart = (q // kp) * kp if select_tail else q - (q + 1) % kp + 1
    return dict(sel=sel, top=top, pad=pad, qpos=qpos, gstart=gstart, J=J, S=S, kp=kp)
