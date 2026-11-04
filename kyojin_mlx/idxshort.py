# SPDX-License-Identifier: Apache-2.0
"""Indexer skip while the pooled keys fit in the top-k (flag `idx_short`, default 0).

An indexer layer scores every pooled key of a row, selects the `index_topk` best and, on the candidate-source layer, derives a block mask. When a
row has `lens <= index_topk` pooled keys there is nothing to rank: the exact selection (index_topk_rt, idxfuse, idxfast, the stock
sort(argsort) chain) takes EVERY key t < lens, whatever its score (the `all` branch of dsv4_core._SELECT: nd <= K), writes the indices in increasing
order and fills the row with -1 up to K. The scores, the wq_b projection of the indexer, the head weights, the rotary of the query and the select
are computed for a result that is a function of `lens` alone. This module builds that result directly:

    idx[u, j] = j if j < lens[u] else -1          (width K = index_topk on the rt / fused paths, min(K, t_max) on the stock path)

Candidate-source layers also hand a block mask to the consumer layers. Below `candidate_topk_blocks` blocks (ceil(lens / block) <= topk_blocks, the
pinned last block included) every block ranks in, and the kernel keeps a block when its max score is finite, so the mask is True on t < lens
(the one exception is a block whose scores are all -inf, which needs an fp16 overflow of the head sum; the skip treats it as finite). The skip
then hands an all-True mask of the same shape; a consumer layer is gated on the same lens and skips as well.

What is still computed: the compressor, the indexer key projection and the key pool writes (a later token, past the threshold, scores them).
Not covered: the solo path (fuse_index, dsa_decode), the prefill indexer, and any pass with the explicit state / verify replay (the recorded
command stream must not change at the crossover position: `canon` gates the skip off).

The decision is made on the host from the pass's own lens (Plan.emit(r).tmax, python ints), per layer, so a context that crosses the threshold
mid-request switches layer by layer with no recompilation."""
from __future__ import annotations

import os

import mlx.core as mx

ENABLED = os.environ.get("KYOJIN_IDX_SHORT", "1") not in ("0", "")        # flags.py idx_short (switched at call time)

_const: dict = {}


def applies(tmax: int, topk: int, *, canon: bool = False, enabled: bool | None = None, is_source: bool = False, topk_blocks: int = 0,
            block_size: int = 1, ratios_uniform: bool = True) -> bool:
    """May this layer skip its indexer? tmax = most pooled keys of any row of the pass (host int), topk = index_topk.
    A candidate-source layer (is_source) also needs ceil(tmax / block_size) <= topk_blocks and the consumers on the same key index space."""
    if not (ENABLED if enabled is None else enabled) or canon or tmax <= 0 or topk <= 0:
        return False
    if tmax > topk:
        return False
    if is_source:
        if topk_blocks < 2 or block_size < 1 or not ratios_uniform:
            return False
        if -(-tmax // block_size) > topk_blocks:
            return False
    return True


def ratios_uniform(cfg, layer_idx: int | None = None) -> bool:
    """The layers that consume the candidate mask (the index-source layers after the candidate source, LEVERS: 24, 28, 32, 36 on DS4.1) share the
    compress ratio of the source layer, so their lens, hence their skip decision, equal the source's. Layers before the source (2, 8, 14: ratio 2
    against the source's 1) never read its mask. False when the config does not say."""
    try:
        ratios = list(cfg.compress_ratios)
        ids = list(cfg.index_source_layer_ids)
        src = int(cfg.candidate_source_layer_id)
        return src >= 0 and all(ratios[i] == ratios[src] for i in ids if i > src)
    except (AttributeError, TypeError, IndexError, ValueError):
        return False


def decide(idxr, cfg, em, rt: bool, t_max: int, n: int, canon: bool):
    """The whole skip decision of one indexer layer, pure (no model needed): (topk_idxs [n, 1, width], candidates [n, 1, t_max] | None) when the layer
    skips, None when it runs the stock path. rt: the path the layer would take builds the fixed width index_topk (fused / rt kernels), else min(K, t_max).
    Attributes a layer may not carry (candidate_*) are read only for a candidate-source layer."""
    if not ENABLED:
        return None
    topk = int(idxr.index_topk)
    src = bool(getattr(idxr, "is_candidate_source", False))
    if not applies(em.tmax, topk, canon=canon, is_source=src, topk_blocks=int(getattr(idxr, "candidate_topk_blocks", 0) or 0) if src else 0,
                   block_size=int(getattr(idxr, "candidate_block_size", 1) or 1) if src else 1, ratios_uniform=ratios_uniform(cfg) if src else True):
        return None
    return idx_for(em, topk if rt else min(topk, t_max))[:, None, :], (all_mask(n, t_max) if src else None)


def short_idx(lens: mx.array, width: int) -> mx.array:
    """The selection of a row whose pooled keys all fit: [N, width] int32, j for j < lens[u], -1 after. Identical to index_topk_rt / idxfuse /
    where(sort(argsort(-scores)[:, :width]) < lens, ., -1) for lens <= width."""
    ar = mx.arange(width, dtype=mx.int32)[None, :]
    return mx.where(ar < lens.astype(mx.int32)[:, None], ar, mx.array(-1, dtype=mx.int32))


def idx_for(em, width: int) -> mx.array:
    """short_idx cached on the pass's emit record (one per compress ratio and pass): the layers of a ratio share it, two small launches per ratio
    per token instead of one selection per layer."""
    c = getattr(em, "_short_idx", None)
    if c is None:
        c = em._short_idx = {}
    a = c.get(width)
    if a is None:
        a = c[width] = short_idx(em.lens_arr, width)
    return a


def all_mask(n: int, t_max: int) -> mx.array:
    """The candidate mask of a source layer whose blocks all rank in: bool [n, 1, t_max], True. Cached by shape (materialised by the first graph
    that uses it, free afterwards)."""
    a = _const.get((n, t_max))
    if a is None:
        if len(_const) >= 64:
            _const.clear()
        a = _const[(n, t_max)] = mx.ones((n, 1, t_max), dtype=mx.bool_)
    return a
