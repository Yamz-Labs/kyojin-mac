# SPDX-License-Identifier: Apache-2.0
"""GLM-5 indexer decode path on the dsa_decode kernels (flag `KYOJIN_DSA_INDEX_GLM`, default off).

Replaces the pooled-key scoring + argpartition inside `_exact_pool_select` for one decode query (batch 1, 32 heads x 128): one scoring launch (fp32 accumulate and
fp32 head sum like the stock path) and one radix select over the fp32 scores. Prefill / batches / other shapes use the stock code."""
from __future__ import annotations

import os

import mlx.core as mx
from .modeldesc import KERNEL as KG

from . import dsa_decode as D

ENABLED = os.environ.get("KYOJIN_DSA_INDEX_GLM", "0") == "1"


def install() -> bool:
    if not ENABLED:
        return False
    from . import archs as _AR; L = _AR.lang(_AR.GLM5)
    if getattr(L, "_kyojin_dsa", False):
        return True
    stock = L._exact_pool_select

    def patched(q, pool_keys, weights, pool_ends, pool_valid, query_positions, select_k, scale, chunk_size=512):
        if (q is None or q.shape[0] != 1 or q.shape[1] != 1 or q.shape[2] != KG.IDX_HEADS or q.shape[3] != KG.IDX_DIM or pool_keys.shape[-1] != KG.IDX_DIM
                or q.dtype not in (mx.float16, mx.bfloat16) or pool_keys.dtype != q.dtype or select_k >= pool_keys.shape[1]):
            return stock(q, pool_keys, weights, pool_ends, pool_valid, query_positions, select_k, scale, chunk_size)
        return select_pools(q, pool_keys, weights, pool_ends, pool_valid, query_positions, select_k, scale)

    L._exact_pool_select = patched
    L._kyojin_dsa = True
    return True


def select_pools(q, pool_keys, weights, pool_ends, pool_valid, query_positions, select_k, scale):
    pc = pool_keys.shape[1]
    cand = pool_valid[:, None] & (pool_ends[:, None] <= query_positions[None, 0:1, None])      # [1, 1, pc]
    w = (weights[0, 0].astype(mx.float32) * scale)
    n = mx.array([pc], dtype=mx.int32)
    sc = D.scores(q[0, 0], w, pool_keys[0], n, pc, f32=True)
    sel = D.select(sc, n, pc, select_k, valid=cand.reshape(-1))
    selected = sel[None, None, :]
    return selected, mx.take_along_axis(cand, selected, axis=-1)
