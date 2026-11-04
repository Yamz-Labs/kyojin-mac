# SPDX-License-Identifier: Apache-2.0
"""Blocked top-k of the prefill indexer: the query rows of a prompt chunk are reduced in blocks of at most `KYOJIN_IDX_QBLOCK` rows (default 512).

The stock indexer builds the whole score matrix of a chunk, then `-scores` and a full-width `argsort` over it. The top-k is a row-wise reduction: a query row selects
from its own `Tc` scores and nothing in that row depends on the other rows or on where a row block starts. So the same selection can run block by block, each block
evaluated (and its score tile freed) before the next one is built:

    stock, per indexer layer:   scores 4 T Tc + negation 4 T Tc + argsort 4 T Tc  = 12 T Tc bytes  (+ the matmul product)
    blocked, per indexer layer: the same three passes over 4 Q Tc only             = 12 Q Tc bytes  (+ the matmul product)

with `Q = min(QBLOCK, T)`. The indices are the same values in the same order: `argsort` is stable, so equal scores come out in ascending index order, and every row
is reduced on its own. A block boundary is therefore invisible in the output (see tests/test_idx_topk.py).

`scan` also runs the per-block side jobs of the same scores (the candidate-source block maxima), so one pass over the scores serves both.
"""
from __future__ import annotations

import os

import mlx.core as mx

from . import fired as FD

QBLOCK_ENV = "KYOJIN_IDX_QBLOCK"
QBLOCK_DEFAULT = 512


def qblock(n_rows: int, override: int | None = None) -> int:
    """Rows per reduction block; 0 means one block, i.e. the stock single-pass shape (a chunk no larger than one block keeps it)."""
    q = int(os.environ.get(QBLOCK_ENV, QBLOCK_DEFAULT)) if override is None else int(override)
    if q <= 0 or q >= n_rows:
        return 0
    return q


def plan(n_rows: int, q: int) -> list[tuple[int, int]]:
    """(t0, n) of every block of `n_rows` rows; one block when `q` is 0."""
    if not q or q >= n_rows:
        return [(0, n_rows)]
    return [(t0, min(q, n_rows - t0)) for t0 in range(0, n_rows, q)]


def topk(sc: mx.array, topk: int) -> mx.array:
    """[n, Tc] scores -> [n, min(topk, Tc)] int32 ascending: the stock `sort(argsort(-sc)[..., :topk])` on one block of rows."""
    return mx.sort(mx.argsort(-sc, axis=-1)[..., :topk], axis=-1).astype(mx.int32)


def partition(sc: mx.array, topk: int) -> mx.array:
    """[n, Tc] scores -> [n, topk] int32 in `argpartition` order: the deepseek_v32 / GLM indexer selection on one block of rows."""
    return mx.argpartition(sc, kth=-topk, axis=-1)[..., -topk:]


def scan(make_block, n_rows: int, q: int, fn):
    """Reduce `n_rows` rows block by block: `make_block(t0, n)` -> scores [n, Tc] of those rows, `fn(t0, scores)` -> array or tuple of arrays.

    Every block's outputs are evaluated before the next block is built, which is where the memory goes: the score tile of a block (and the negation / argsort
    scratch over it) is dead at that point instead of being kept alive by the graph of the whole chunk. `fn` may return a tuple; the blocks of each position are
    concatenated along the row axis. Returns the array (or tuple) for the whole chunk."""
    outs = []
    for t0, n in plan(n_rows, q):
        if q:
            FD.hit("idx_qblock")
        r = fn(t0, make_block(t0, n))
        if isinstance(r, (tuple, list)):
            mx.eval(*r)
            outs.append(tuple(r))
        else:
            mx.eval(r)
            outs.append(r)
    if not outs:
        raise ValueError("scan: no rows")
    if isinstance(outs[0], tuple):
        return tuple(outs[0][i] if len(outs) == 1 else mx.concatenate([o[i] for o in outs], axis=0) for i in range(len(outs[0])))
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
