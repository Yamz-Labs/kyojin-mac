# SPDX-License-Identifier: Apache-2.0
"""Block-summary first pass for the decode index scores (lever KYOJIN_DEC_BLK_SUM, default 0; KYOJIN_DEC_BLK_SUM_MULT, default 8).

Why. One `dsv4_core.core` pass scores every row against its session's whole index-key buffer (the T1 term of docs/reports/flatdec3_REPORT.md):
the cost is the key bytes read, linear in the context, and it is the term that still grows at 1M once the select is flat. The top-k keys of a row
live in a few blocks of 8 consecutive keys, so most of that read is spent on blocks that cannot win.

What. Every 8-key block keeps one summary key, the mean of its keys (fp16, [S, nblk, 128], 1/8 of the key bytes, held per layer). The summaries are
scored by the same row kernel as the keys (`dsa_decode.scores_rows`, no new kernel): 1/8 of the keys, so ~1/8 of T1. The `MULT * topk` best blocks
by summary score (the candidate source: `max(candidate_topk_blocks, topk) * MULT`; the last visible block pinned; ties to the lower block -
`pf_flat.pick_blocks`) are gathered per row from its own session's buffer and rescored by the exact kernel, and the top-k of those scores mapped
back through the position table is the answer. A candidate-source layer rebuilds its kept block mask from the exact block maxima of the picked
blocks (`pf_flat.rerank_blocks`); a consumer layer ranks only the blocks its mask kept. Same gather / rescore / select tail as `decidx8`.

Exactness. NOT bitwise, by design. The rescored values are the full-width path's values, so the answer is exact whenever every true top-k key sits
in a picked block; the summary ranking is a heuristic, MULT is the shield. `tests/test_dec_blk_sum.py` reports the top-k recall against the exact
selection (>= 0.99 required at the default MULT on keys with block locality; on keys with NO block locality - i.i.d. - a mean summary has nothing
to rank, the test pins that the recall then rises to 1 with MULT instead). Quality must be re-measured on real keys (`tools/blk_recall.py` style:
`KYOJIN_DEC_SEL_CHECK` like overlap) before the flag is served. Rows with fewer finite keys than `index_topk` get a -1 tail, as in `decidx8`.

Memory. 256 B per block = 32 B per key per index-key layer in fp16 (2.5 L keys on DS4.1 at 1M: ~80 MB). Complete committed blocks are summarized
once; the blocks an open pass may still change are re-summarized at every call (a rolled-back count re-summarizes).

Scope. fp16 / bf16 keys and queries of 32 heads x 128 channels (`dsa_decode.rows_supported`), the runtime-width path, and `KYOJIN_DEC_BLK_SUM_MIN`
(262144) index keys; a candidate mask only with the 8-key block size it was built with. Anything else keeps the exact path: `layer` returns None.
Model-independent: any Engram-style index with that geometry. Nothing changes at flag 0.
"""
from __future__ import annotations

import os

import mlx.core as mx

from . import pf_flat as PF

ENABLED = os.environ.get("KYOJIN_DEC_BLK_SUM", "0") not in ("0", "")       # flags.py dec_blk_sum (switched at call time)
MULT = int(os.environ.get("KYOJIN_DEC_BLK_SUM_MULT", "8"))                # blocks picked per block the answer needs
MIN_KEYS = int(os.environ.get("KYOJIN_DEC_BLK_SUM_MIN", "262144"))        # index keys from which a layer takes the first pass

BS = PF.BS
NEG = mx.array(-mx.inf, dtype=mx.float32)
HITS = {"layer": 0, "source": 0, "masked": 0, "sum_blocks": 0, "sum_open": 0}


# ----------------------------------------------------------------------------------------------------------------------- scoring (Metal / CPU)
def _score(q, w, keys, lens, t_max: int, sess):
    """Index scores [n, t_max] of rows q [n, NH, DI] / w [n, NH] against keys[sess[u]] (first lens[u] keys), -inf past lens. The row kernel on Metal;
    the reference formula sum_h w_h * relu(q_h . k) in float32 without it (tests, no GPU)."""
    from . import dsa_decode as D
    if mx.metal.is_available():
        return D.scores_rows(q, w, keys, lens, t_max, None, sess, nax=False)
    k = keys[sess[:, None], mx.arange(t_max, dtype=mx.int32)[None, :]].astype(mx.float32)           # [n, t, DI]
    s = (mx.maximum(mx.einsum("nhd,ntd->nht", q.astype(mx.float32), k), 0) * w.astype(mx.float32)[:, :, None]).sum(axis=1)
    return mx.where(mx.arange(t_max, dtype=mx.int32)[None, :] < lens[:, None], s, -mx.inf).astype(q.dtype)


def _score_sum(q, w, sm, lens_b, nb: int, sess):
    """The same on the block summaries (the fast matrix kernel on Metal: only the ranking is used, the answer is rescored exactly)."""
    from . import dsa_decode as D
    if mx.metal.is_available():
        return D.scores_rows(q, w, sm, lens_b, nb, None, sess, nax=None)
    return _score(q, w, sm, lens_b, nb, sess)


def _topk(sc, lens, k: int):
    if mx.metal.is_available():
        from . import idxfast as IX
        return IX.topk_mt(sc, lens, k)
    cols = mx.arange(sc.shape[1])[None, :]
    s = mx.where(cols < lens[:, None], sc, -mx.inf)
    ids = mx.sort(mx.argsort(-s, axis=-1)[:, :k], axis=-1).astype(mx.int32)
    return mx.where(ids < lens[:, None], ids, -1)


# ----------------------------------------------------------------------------------------------------------------------- summaries
def summarize(keys: mx.array, n: int) -> mx.array:
    """Block means [ceil(n / 8), DI] (fp16 / the keys' dtype) of keys [>= n, DI]; the last block averages the keys it has."""
    nb = -(-n // BS)
    k = keys[:n].astype(mx.float32)
    if nb * BS > n:
        k = mx.concatenate([k, mx.zeros((nb * BS - n, k.shape[-1]), dtype=mx.float32)], axis=0)
    cnt = mx.minimum(mx.array(n, dtype=mx.int32) - mx.arange(nb, dtype=mx.int32) * BS, BS).astype(mx.float32)
    return (k.reshape(nb, BS, -1).sum(axis=1) / cnt[:, None]).astype(keys.dtype)


class BlkSum:
    """The block summaries of every session of one layer: sm [S, cap_blocks, DI]. Complete blocks of the committed rows are summarized once
    (`have[s]` blocks), the blocks touching rows of an open pass at every call (a rejected row is overwritten by the next pass)."""

    def __init__(self, S: int):
        self.sm = None
        self.have = [0] * S

    def get(self, keys, committed, T: int):
        S, C, DI = keys.shape
        nbT = -(-T // BS)
        if self.sm is None or self.sm.shape[1] < nbT:
            cap = -(-nbT // 512) * 512
            sm = mx.zeros((S, cap, DI), dtype=keys.dtype)
            if self.sm is not None:
                sm[:, :self.sm.shape[1]] = self.sm
            self.sm = sm
        for s in range(S):
            nc = min(max(int(committed[s]), 0), T)
            full = nc // BS
            if self.have[s] > full:
                self.have[s] = 0
            if self.have[s] < full:
                a = self.have[s] * BS
                self.sm[s, self.have[s]:full] = summarize(keys[s, a:full * BS], full * BS - a)
                HITS["sum_blocks"] += full - self.have[s]
                self.have[s] = full
            if T > full * BS:                                                     # the open region: the committed tail block and the rows of this pass
                self.sm[s, full:nbT] = summarize(keys[s, full * BS:T], T - full * BS)
                HITS["sum_open"] += nbT - full
        return self.sm


# ----------------------------------------------------------------------------------------------------------------------- selection
def select(q, keys, w, lens, sess, t_max: int, topk: int, store: BlkSum, committed, cand=None, kb: int = 0, mult: int | None = None):
    """(top-k [n, topk] int32 ascending with the -1 tail, mask [n, t_max] bool or None) of the index scores of rows q [n, NH, DI] / w [n, NH] against
    keys [S, K, DI] (row u reads keys[sess[u]], keys below lens[u]; `cand` [n, >= t_max] bool of 8-key blocks drops whole blocks), through the block
    summaries. `kb` > 0 = the candidate source layer: the top-k AND its kept block mask."""
    n = q.shape[0]
    m = MULT if mult is None else mult
    nb = -(-t_max // BS)
    cl = mx.minimum(lens.astype(mx.int32), mx.array(int(t_max), dtype=mx.int32))
    sess = sess.astype(mx.int32)
    sm = store.get(keys, committed, t_max)
    bm = _score_sum(q, w, sm, (cl + (BS - 1)) // BS, nb, sess).astype(mx.float32)
    if cand is not None:
        bm = mx.where(cand[:, ::BS][:, :nb] & (mx.arange(nb, dtype=mx.int32)[None, :] * BS < cl[:, None]), bm, NEG)
    pick = min(nb, max(kb, topk) * m)
    ids = PF.pick_blocks(bm, cl, pick, True)                                     # the last visible block is always rescored (recency)
    fin = mx.take_along_axis(bm, mx.maximum(ids, 0), axis=1) > NEG
    ids = _compact(mx.where((ids >= 0) & fin, ids, mx.array(-1, dtype=mx.int32)), nb)
    P = ids.shape[1]
    W = P * BS
    bok = ids >= 0
    gpos = (mx.where(bok, ids, 0)[:, :, None] * BS + mx.arange(BS, dtype=mx.int32)).reshape(n, W)
    real = mx.repeat(bok, BS, axis=1) & (gpos < cl[:, None])
    gk = keys[sess[:, None], gpos]
    nreal = real.sum(axis=-1).astype(mx.int32)
    sc = _score(q, w, gk, nreal, W, mx.arange(n, dtype=mx.int32))
    sc = mx.where(real, sc, mx.array(-mx.inf, dtype=q.dtype))
    ci = _topk(sc, nreal, topk)
    idx = mx.where(ci >= 0, mx.take_along_axis(gpos, mx.maximum(ci, 0), axis=1), mx.array(-1, dtype=mx.int32)).astype(mx.int32)
    if not kb:
        HITS["masked" if cand is not None else "layer"] += 1
        return idx, None
    mask = PF.mask_from_ids(PF.rerank_blocks(sc, ids, cl, kb), nb, t_max) & (mx.arange(t_max, dtype=mx.int32)[None, :] < cl[:, None])
    HITS["source"] += 1
    return idx, mask


def _compact(ids, nb: int):
    r, P = ids.shape
    flag = mx.zeros((r, nb + 1), dtype=mx.bool_)
    flag = mx.put_along_axis(flag, mx.where(ids >= 0, ids, nb), mx.ones((r, P), dtype=mx.bool_), axis=1)[:, :nb]
    return PF.ids_from_mask(flag, P)


def applies(idxr, hand_keys, cand, q, w, lens, sess, t_max: int) -> bool:
    """Geometry the row kernel serves (fp16 / bf16, 32 heads x 128), the runtime width, enough keys, a mask of the 8-key block size."""
    if not ENABLED or not mx.metal.is_available() or hand_keys is None:
        return False
    if q.ndim != 3 or hand_keys.ndim != 3 or w.ndim != 2 or q.dtype not in (mx.float16, mx.bfloat16) or hand_keys.dtype != q.dtype:
        return False
    from . import dsa_decode as D
    if not D.rows_supported(idxr.n_heads, idxr.index_head_dim, q.dtype):
        return False
    n = q.shape[0]
    if w.shape != (n, idxr.n_heads) or tuple(lens.shape) != (n,) or tuple(sess.shape) != (n,) or hand_keys.shape[-1] != q.shape[-1]:
        return False
    if not max(MIN_KEYS, 32) <= t_max <= hand_keys.shape[1] or -(-t_max // BS) * BS > hand_keys.shape[1]:
        return False
    k = int(getattr(idxr, "index_topk", 0) or 0)
    if not 1 <= k <= t_max:
        return False
    if cand is not None and (cand.ndim != 2 or cand.shape[0] != n or cand.shape[-1] < t_max or int(getattr(idxr, "candidate_block_size", 0) or 0) != BS):
        return False
    if getattr(idxr, "is_candidate_source", False):
        kb = int(getattr(idxr, "candidate_topk_blocks", 0) or 0)
        if cand is not None or not 2 <= kb <= t_max:
            return False
    return True


def layer(st, l: int, idxr, hand_keys, cand, q, w, em, t_max: int, sess):
    """The summary first pass for one index layer of `core`: (top-k [n, topk] int32 rt form, mask [n, t_max] bool or None) or None (exact path)."""
    if not applies(idxr, hand_keys, cand, q, w, em.lens_arr, sess, t_max):
        return None
    S = hand_keys.shape[0]
    store = getattr(st, "blks", None)
    if store is None:
        store = st.blks = {}
    bs = store.get(l)
    if bs is None or len(bs.have) != S or bs.sm is not None and bs.sm.shape[0] != S:
        bs = store[l] = BlkSum(S)
    committed = st.kcnt[l] if getattr(st, "kcnt", None) is not None else [0] * S
    src = bool(getattr(idxr, "is_candidate_source", False))
    return select(q, hand_keys, w, em.lens_arr, sess, t_max, int(idxr.index_topk), bs, committed, cand,
                  kb=int(idxr.candidate_topk_blocks) if src else 0)
