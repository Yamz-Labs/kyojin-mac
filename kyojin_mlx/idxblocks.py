# SPDX-License-Identifier: Apache-2.0
"""Consumer-layer index scores over the kept blocks only (lever KYOJIN_ISCORE_BLOCKS, default 0).

Why. A pass of the batched core scores the index layers once per row: three ratio-2 layers and the candidate source over all their keys, and the four
consumer layers over the keys the source layer's block mask kept (`dsv4_core.core`, `cand = hand.candidates[:, 0, :]`). The mask keeps
`candidate_topk_blocks` blocks of `candidate_block_size` keys - 16384 keys of any width - so a consumer layer's ANSWER is constant in the context, but
its kernel is not: `dsa_decode._SCORE_ROWS` with a mask walks every 8-key block of every key of the pass, reads its mask byte and writes -inf, and the
selection walks the whole width too. Measured on Metal (tools/flatdec3_terms.py, 131072 keys, 6 rows): 8 plain layers cost 1.05 ms on the matrix kernel
while 4 consumer layers cost 1.19 ms, i.e. a consumer layer costs ~2.3x a full plain layer for 1/8 of its keys. None of that grows with the answer: it
grows with the key count the layer does not need.

What. The kept blocks are compacted on the device into an ascending position table, the kept keys are gathered into one dense [rows, 16384, dim] tile,
the same scoring kernel scores that tile (no mask), and the top-k of the gathered scores is mapped back through the table. The score and the select are
then functions of 16384 keys, not of the context.

Exactness. The score of a kept key is computed by the same kernel from the same numbers: the matrix kernel is per key (a key's dot products, the
rounding of its tile, its head reduction and its head sum do not depend on which other keys share a tile), so a compacted key holds bit-for-bit what it
holds in the full-width call. The selection sees the same finite scores in the same ascending key order as the full-width call, so the same tie rule
("value descending, lower index first") picks the same indices and the same -1 tail. The compaction is exact too: a block ranks in when the mask keeps
it AND it holds a key below the row's count (the mask past `lens` is undefined by contract, so it is never read there), and a gathered key past `lens`
is dropped by the same rule the full-width select applies. `tests/test_iscore_blocks.py` asserts the selection and the scores are equal to the masked
full-width path, and that both hold what the stock chain selects except for a near tie (the chain and the matrix kernel round the head sum differently -
the same drift `iscore_mma` carries).

Scope. Consumer layers only (a candidate mask, not the candidate source; the source decides the mask and the plain layers have no mask), fp16 / bf16,
32 index heads of 128 channels (the geometry `dsa_decode.scores_rows` serves), and the runtime-width path. Gated on the key width
(`iscore_blocks_min`, 262144 by default) because below that the full-width kernel is already cheap and the compaction adds fixed launches.
"""
from __future__ import annotations

import os

import mlx.core as mx

ENABLED = os.environ.get("KYOJIN_ISCORE_BLOCKS", "0") not in ("0", "")   # flags.py iscore_blocks (switched at call time)
MIN_KEYS = int(os.environ.get("KYOJIN_ISCORE_BLOCKS_MIN", "262144"))      # key width from which a consumer layer takes the gathered path

HITS = {"ok": 0, "table_keys": 0}


def applies(idxr, hand_keys, cand, q, t_max: int) -> bool:
    """Is the gathered path worth taking for this layer? `cand` = the source layer's block mask [n, >= t_max] (None on a plain / source layer)."""
    if not ENABLED or cand is None or not mx.metal.is_available():
        return False
    if getattr(idxr, "is_candidate_source", False) or not getattr(idxr, "uses_candidates", False):
        return False
    if hand_keys is None or getattr(hand_keys, "dtype", None) != q.dtype or q.dtype not in (mx.float16, mx.bfloat16):
        return False
    from . import dsa_decode as D
    if not D.rows_supported(idxr.n_heads, idxr.index_head_dim, q.dtype):
        return False
    bs = int(getattr(idxr, "candidate_block_size", 0) or 0)
    nbk = int(getattr(idxr, "candidate_topk_blocks", 0) or 0)
    return (bs > 0 and nbk >= 2 and t_max >= MIN_KEYS and t_max >= nbk * bs and hand_keys.shape[1] >= t_max
            and cand.shape[0] == q.shape[0] and cand.shape[-1] >= t_max)


def kept(cand: mx.array, lens: mx.array, t_max: int, bs: int, nbk: int):
    """Ascending global positions of the kept keys of every row, [n, w * bs] int32, with a bool mask of the entries that are real keys of the row.
    The kept blocks are packed in ascending block order, so a row's real entries come first and are strictly increasing - which is what keeps the tie
    rule of the select ("lower index first") identical to the full-width call."""
    n = cand.shape[0]
    nb = t_max // bs
    blk = cand[:, :nb * bs].reshape(n, nb, bs)[:, :, 0]                       # a block ranks in when its FIRST key's mask bit is set - the masked
    blk = blk & (mx.arange(nb, dtype=mx.int32)[None, :] * bs < lens[:, None])  # kernel's own rule; blocks past the row's count are never read
    cnt = blk.sum(axis=-1).astype(mx.int32)
    slot = mx.where(blk, mx.cumsum(blk.astype(mx.int32), axis=-1) - 1, 0)
    ids = mx.broadcast_to(mx.arange(nb, dtype=mx.int32)[None, :], blk.shape)
    packed = mx.zeros((n, nb + 1), dtype=mx.int32)
    packed = mx.put_along_axis(packed, mx.where(blk, slot, nb), ids, axis=-1)  # kept block ids at their slot, the dropped ones in the spare column
    w = min(nb, nbk)
    pos = (packed[:, :w, None] * bs + mx.arange(bs, dtype=mx.int32)[None, None, :]).reshape(n, w * bs)
    real = mx.arange(w * bs, dtype=mx.int32)[None, :] < (mx.minimum(cnt, w) * bs)[:, None]
    return mx.where(real, pos, 0), real & (pos < lens[:, None])


def select(q: mx.array, w: mx.array, keys: mx.array, lens: mx.array, cand: mx.array, sess: mx.array, t_max: int, k: int,
           bs: int, nbk: int) -> mx.array:
    """The rt-form top-k ([n, k] int32, ascending, -1 tail) of the index scores of consumer-layer rows q [n, NH, DI] / w [n, NH] against keys
    [S, C, DI] (row u reads keys[sess[u]], first lens[u] keys), with the source layer's candidate mask `cand` [n, >= t_max] (blocks of `bs` keys, at
    most `nbk` of them kept)."""
    from . import dsa_decode as D, idxfast as IX
    n = q.shape[0]
    pos, real = kept(cand, lens, t_max, bs, nbk)
    W = pos.shape[1]
    gk = keys[sess[:, None], pos]                                            # [n, W, DI]: the kept keys, ascending, one row's slice each
    s = D.scores_rows(q, w, gk, real.sum(axis=-1).astype(mx.int32), W, None, mx.arange(n, dtype=mx.int32), nax=False)
    s = mx.where(real, s, mx.array(-mx.inf, dtype=q.dtype))                  # slots the kernel did not write (and keys past the row's count)
    ci = IX.topk_mt(s, mx.full((n,), W, dtype=mx.int32), k)
    idx = mx.take_along_axis(pos, mx.maximum(ci, 0), axis=1)
    HITS["ok"] += 1
    HITS["table_keys"] += W                                     # the gathered width, one table per layer
    return mx.where(ci >= 0, idx, mx.array(-1, dtype=mx.int32)).astype(mx.int32)


def layer(idxr, hand_keys, cand, q, w, em, t_max: int, sess: mx.array) -> mx.array | None:
    """The selection of one consumer layer through the gathered path, rt form, or None when the layer does not take it (/ the stock path keeps it)."""
    if not applies(idxr, hand_keys, cand, q, t_max):
        return None
    return select(q, w, hand_keys, em.lens_arr, cand, sess, t_max, idxr.index_topk,
                  int(idxr.candidate_block_size), int(idxr.candidate_topk_blocks))
