# SPDX-License-Identifier: Apache-2.0
"""Flag kpool_fuse (KYOJIN_KPOOL_FUSE, default off, read per call): the decode-time pool selection of a block-sparse (kpool) layer as ONE launch.

Stock (foreign_skeleton_kpool.select_pools + keep_mask, ~50 elementwise launches per layer and token): score = sum_h relu(iq_h . pk_j) * w_h per pool j,
visibility, argsort(-score)[:n_top] (stable: equal scores keep the LOWEST pool index), scatter to a pool mask, expand to the [B, 1, L, S] slot mask
with the forced tail group. Here one kernel does all of it for rows <= ROWS_MAX (one threadgroup per (batch row, query row)):
  1 scores   thread-strided over the pools, h-sequential sum of relu(dot) * w (the stock formula; the dot is a plain d-loop, so values differ from the
             matmul in the last bits only), invisible pools = -inf; scores go to a device scratch output.
  2 top-k    exact radix select (32 bit passes over an order-preserving uint key of the score, -0.0 = +0.0) of the n_top-th largest key T; keys > T are
             chosen, keys == T are taken in ascending pool index (contiguous chunk per thread + exclusive scan) = the stable argsort tie rule.
  3 mask     slot s holds position s - pad; keep = (its pool is chosen | position >= tail start) & 0 <= position <= q, written as bool [B, 1, L, S].
Generic by shape: n_idx_heads / d / J / L / S are read from the arrays, kp / top_k / select_tail are template constants; no model constants.
Blueprint by shape (MIT, attribution): llama.cpp ggml_lightning_indexer_rows (scores read straight from the pooled keys) + top_k + ggml_top_k_mask
(ggml/src/ggml-metal, ggml-metal-ops.cpp:1389 / 1502; graph hy-v4-flash.cpp:851-947). NaN scores are not handled (stock argsort order is undefined too).
"""
from __future__ import annotations

import os

import mlx.core as mx

ON = False                                                           # bound by flags.set (attrs); read per call
ROWS_MAX = 8                                                         # decode rows (1 .. 8 query rows)
TG = 256                                                             # threads per threadgroup (multiple of 32, <= 1024)

_HEADER = r"""#pragma METAL fp contract(off)
inline uint kp_key(float s) {
    if (s == 0.0f) { s = 0.0f; }                                     // -0.0 == +0.0
    const uint u = as_type<uint>(s);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);               // larger score = larger key
}
"""

_SRC = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint l = threadgroup_position_in_grid.y;
    const uint b = threadgroup_position_in_grid.z;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint NSG = TG / 32;
    const uint H = iq_shape[1], D = iq_shape[3];
    const uint S = par[0], J = par[2], L = par[3];
    const int offb = off[b];
    const int q = offb + int(l);
    const int pad = par[1] != 0u ? 0 : int(S) - int(L) - offb;                // par[1]: left-aligned fixed-state buffer (slot == position)
    const uint nv = min(uint(KSEL ? q / KP : (q + 1) / KP), J);
    const ulong row = ulong(b) * L + l;
    const float inf = metal::numeric_limits<float>::infinity();
    // ---- 1 scores
    if (PRE == 0) for (uint j = tid; j < J; j += TG) {
        float s = -inf;
        if (j < nv && valid[ulong(b) * J + j]) {
            const auto pj = pk + (ulong(b) * J + j) * D;
            s = 0.0f;
            for (uint h = 0; h < H; ++h) {
                const auto qh = iq + ((ulong(b) * H + h) * L + l) * D;
                float dot = 0.0f;
                for (uint d = 0; d < D; ++d) { dot = dot + qh[d] * pj[d]; }
                s = s + metal::max(dot, 0.0f) * w[row * H + h];
            }
        }
        scores[row * J + j] = s;
    }
    if (PRE != 0) { for (uint j = tid; j < J; j += TG) { scores[row * J + j] = pre[row * J + j]; } }       // the scores output carries the pre-computed values too
    auto scr = [&](ulong i) -> float { return PRE != 0 ? pre[i] : scores[i]; };   // (a lambda: `pre` may be bound as constant or device memory, `scores` is device) PRE: the scores come from kpool_score (multi-threadgroup launch), stage 1 is skipped
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    // ---- 2 top-k: the NT-th largest key by radix select
    const uint NT = min(uint(TOPN), J);
    threadgroup uint red[32];
    threadgroup uint cnt[TG];
    uint prefix = 0u, k = NT;
    for (int bit = 31; bit >= 0 && NT > 0; --bit) {
        const uint cand = prefix | (1u << uint(bit));
        const uint him = ~0u << uint(bit);
        uint c = 0;
        for (uint j = tid; j < J; j += TG) { c += ((kp_key(scr(row * J + j)) & him) == cand) ? 1u : 0u; }
        c = simd_sum(c);
        if (lane == 0) { red[sg] = c; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint tot = 0;
        for (uint i = 0; i < NSG; ++i) { tot += red[i]; }
        if (tot >= k) { prefix = cand; } else { k -= tot; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // ties (key == prefix) are taken in ascending pool index: contiguous chunk per thread, exclusive scan of the chunk counts
    const uint chunk = (J + TG - 1) / TG;
    const uint lo = min(tid * chunk, J), hi = min(lo + chunk, J);
    uint tc = 0;
    for (uint j = lo; j < hi; ++j) { tc += (kp_key(scr(row * J + j)) == prefix) ? 1u : 0u; }
    cnt[tid] = tc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint before = 0;
    for (uint i = 0; i < tid; ++i) { before += cnt[i]; }
    for (uint j = lo; j < hi; ++j) {
        const uint ky = kp_key(scr(row * J + j));
        bool pick = false;
        if (NT > 0) {
            if (ky > prefix) { pick = true; }
            else if (ky == prefix) { pick = (before < k); before += 1; }
        }
        sel[row * J + j] = pick && j < nv && valid[ulong(b) * J + j];
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    // ---- 3 slot mask
    const int gs = KSEL ? (q / KP) * KP : q - (q + 1) % KP + 1;
    for (uint s = tid; s < S; s += TG) {
        const int kpos = int(s) - pad;
        bool kp_ = false;
        if (kpos >= 0 && kpos <= q) {
            const uint kj = uint(kpos) / KP;
            kp_ = (kj < J && sel[row * J + kj]) || kpos >= gs;
        }
        keep[row * S + s] = kp_;
    }
"""

_SCORE_SRC = r"""
    const uint j = thread_position_in_grid.x;
    const uint l = thread_position_in_grid.y;
    const uint b = thread_position_in_grid.z;
    const uint H = iq_shape[1], L = iq_shape[2], D = iq_shape[3], J = pk_shape[1];
    if (j >= J) return;
    const int q = off[b] + int(l);
    const uint nv = min(uint(KSEL ? q / KP : (q + 1) / KP), J);
    const ulong row = ulong(b) * L + l;
    float s = -metal::numeric_limits<float>::infinity();
    if (j < nv && valid[ulong(b) * J + j]) {
        const auto pj = pk + (ulong(b) * J + j) * D;
        s = 0.0f;
        for (uint h = 0; h < H; ++h) {
            const auto qh = iq + ((ulong(b) * H + h) * L + l) * D;
            float dot = 0.0f;
            for (uint d = 0; d < D; ++d) { dot = dot + qh[d] * pj[d]; }
            s = s + metal::max(dot, 0.0f) * w[row * H + h];
        }
    }
    scores[row * J + j] = s;
"""

PRE_J = 1024                                                         # pools from which the scores come from the multi-threadgroup kernel (one thread per pool); below: inside the selection launch
_k = None
_ks = None


def on() -> bool:
    return ON


MAXS = int(os.environ.get("KYOJIN_KPOOL_FUSE_MAXS", "1000000000") or 0)   # key slots above which the stock selection runs instead (set by measurement; default: no limit)


def use(L: int, S: int | None = None) -> bool:
    """The fused launch applies: flag on, Metal, 1 .. ROWS_MAX query rows, and (S given) at most MAXS key slots."""
    return ON and 1 <= L <= ROWS_MAX and mx.metal.is_available() and (S is None or S <= MAXS)


def _kernel():
    global _k
    if _k is None:
        _k = mx.fast.metal_kernel(name="kyojin_kpool_fuse", input_names=["iq", "w", "pk", "valid", "off", "par", "pre"],
                                  output_names=["keep", "scores", "sel"], header=_HEADER, source=_SRC)
    return _k


def _score_kernel():
    global _ks
    if _ks is None:
        _ks = mx.fast.metal_kernel(name="kyojin_kpool_score", input_names=["iq", "w", "pk", "valid", "off"], output_names=["scores"], header=_HEADER, source=_SCORE_SRC)
    return _ks


def select_ref(iq, w, pk, valid, off, S, kp, top_k, select_tail):
    """The stock composition (the reference of the Metal test): -> keep [B, 1, L, S], sel [B, L, J]."""
    from . import foreign_skeleton_kpool as KP
    s = KP.select_pools(iq, w, pk, valid, off, S, kp, top_k, select_tail)
    return KP.keep_mask(s), s["sel"]


def _score_pre(iq, w, pk, valid, off, kp, select_tail):
    """Flag kpool_score off, J >= PRE_J pools: the scalar scorer as one thread per (pool, query row) in its own launch (None below PRE_J: stage 1 stays inside the selection launch)."""
    B, H, L, _ = iq.shape
    J = pk.shape[1]
    if J < PRE_J:
        return None
    offv = mx.array(off, dtype=mx.int32) if not isinstance(off, mx.array) else off.astype(mx.int32)
    offv = mx.broadcast_to(offv.reshape(-1), (B,)) if offv.size == 1 else offv
    return _score_kernel()(inputs=[iq.astype(mx.float32), w.astype(mx.float32), pk.astype(mx.float32), valid, offv], template=[("KP", int(kp)), ("KSEL", int(bool(select_tail)))],
                           grid=(-(-J // 128) * 128, L, B), threadgroup=(128, 1, 1), output_shapes=[(B, L, J)], output_dtypes=[mx.float32])[0]


def select(iq, w, pk, valid, off, S, kp, top_k, select_tail, left=False):
    """Flag kpool_score on: the scores come from kpool_score.scores (one multi-threadgroup launch), this kernel then does the top-k + slot mask (stage 1 skipped)."""
    from . import kpool_score as KS
    pre = KS.scores(iq, w, pk, valid, off, kp, select_tail) if KS.use(iq.shape[2]) else _score_pre(iq, w, pk, valid, off, kp, select_tail)
    return _select(iq, w, pk, valid, off, S, kp, top_k, select_tail, left, pre)


def _select(iq, w, pk, valid, off, S, kp, top_k, select_tail, left, pre):
    """left: the slots are left-aligned (fixed-state KV buffer: slot s holds position s, as select_pools(left=True)). iq [B, H, L, d] (roped at the token positions), w [B, L, H], pk [B, J, d], valid [B, J], off int / [B] (cache offset before this step), S slots
    -> keep bool [B, 1, L, S], scores fp32 [B, L, J] (-inf = invisible), sel bool [B, L, J] (chosen and visible pools). One launch."""
    B, H, L, d = iq.shape
    J = pk.shape[1]
    offv = mx.array(off, dtype=mx.int32) if not isinstance(off, mx.array) else off.astype(mx.int32)
    offv = mx.broadcast_to(offv.reshape(-1), (B,)) if offv.size == 1 else offv
    top_n = top_k // kp - int(bool(select_tail))
    if pre is not None:
        ins = [mx.zeros((1, 1, 1, 1), dtype=mx.float32), mx.zeros((1, 1, 1), dtype=mx.float32), mx.zeros((1, 1, 1), dtype=mx.float32)]
    else:
        ins = [iq.astype(mx.float32), w.astype(mx.float32), pk.astype(mx.float32)]
    return _kernel()(inputs=ins + [valid, offv, mx.array([S, int(bool(left)), J, L], dtype=mx.uint32), pre if pre is not None else mx.zeros((1,), dtype=mx.float32)],
                     template=[("TG", TG), ("KP", int(kp)), ("TOPN", max(top_n, 0)), ("KSEL", int(bool(select_tail))), ("PRE", int(pre is not None))],
                     grid=(TG, L, B), threadgroup=(TG, 1, 1),
                     output_shapes=[(B, 1, L, S), (B, L, J), (B, L, J)], output_dtypes=[mx.bool_, mx.float32, mx.bool_])
