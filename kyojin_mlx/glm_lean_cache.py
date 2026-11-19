# SPDX-License-Identifier: Apache-2.0
"""Lean served cache path for the stock `mlx_vlm.models.cache` batch caches (flag `KYOJIN_GLM_LEAN_CACHE`, default off).

BP-IDX (STATUS_BPIDX.md) found the DSA indexer's OWN compute is already ~5-8 dispatches/call (near the
brief's <=3 target once a found-but-unshipped redundant scale multiply is folded); the real 46-66
launches/call it measured come almost entirely from stock `BatchKVCache` / `BatchPoolingCache` bookkeeping,
NOT DSA math. G2g independently found this same shape of problem engine-wide: ~8.4 ms/step (60% of the
13.9 ms decode gap) is "a long tail of hundreds of tiny launch-bound kernels", a near-fixed per-dispatch
floor (9-25 us), not bytes.

Of that bookkeeping, G10c's `glm_fuse_dsa.accumulate_fast` / `pool_compress_b` already fuse
`BatchPoolingCache.accumulate_windows` (the ring-buffer insert) and the pool-compression softmax into one
launch each, on the decode hot path (`acc_ok`). BP-IDX explicitly flagged the ONE remaining, concretely
proven, unconditional-every-call Python per-session loop that neither of those touch:
`BatchPoolingCache.update_and_fetch`'s final pooled-buffer insert --

    for i in range(B):
        nc = new_counts[i]
        if nc > 0:
            pl = self._pool_lengths[i]
            self.pooled[i, pl : pl + nc] = px[i, :nc]
            self._pool_lengths[i] = pl + nc

-- one `SliceUpdate` dispatch per session, every DSA-layer indexer call that completed >= 1 pool window
this step (the served lane always runs sessions=4). This module replaces that B-iteration Python loop
with ONE vectorized MLX scatter (advanced integer indexing on both the session and pool-slot axes: MLX
compiles `arr[rows, cols] = values` into a single `Scatter` graph node regardless of B), observably
identical to the stock loop.

Why "observably identical" rather than byte-for-byte on the WHOLE buffer: the stock loop only ever writes
`px[i, :nc]` into the visible region `[pl, pl + nc)` of session i and leaves every other row's
not-yet-promoted tail slots (`j >= new_counts[j]`, for sessions that did not complete a window this call)
untouched at their old value. The vectorized scatter below writes `px[i, :max_new]` for every session
unconditionally, so a session with `new_counts[i] < max_new` gets its own not-yet-promoted slot
`pooled[i, pl_i + j]` (`new_counts[i] <= j < max_new`) overwritten with that session's own freshly computed
(not another session's) pool-compression output for this call -- which is deterministic ZERO for a
session that did not complete a window (`glm_fuse_dsa.pool_compress_b`'s `accumulate_fast` kernel zeros
`ork`/`org` for any session whose running remainder did not cross the ratio this call, so the softmax-
weighted sum over all-zero keys/gates for that session is also a defined, deterministic value -- never a
read of uninitialized memory). That slot is NEVER read until a later call promotes it (every reader --
`make_mask`'s length check, the index kernels' per-row pool count -- gates on `_pool_lengths`, exactly the
counter this module advances the same way stock does), and whichever call eventually promotes it
overwrites it again with the then-correct value. So the only state this module's output can ever legally
be read at (`pooled[i, : _pool_lengths[i]]` for every i, plus `_pool_lengths` itself) is bit-identical to
stock after any sequence of calls, including speculative commit / abort (which only rewind host-side lists
and `pooled`'s WIDTH via `mx.contiguous(self.pooled[:, :pooled_length])`, both already in the stock path's
`BatchPoolingCache.commit_speculation` / `abort_speculation` and untouched by this module). Tests below
check exactly that invariant (plus the full greedy-identity gate every sibling track uses).
"""
from __future__ import annotations

import os

import mlx.core as mx

from . import fired as FD

ENABLED = os.environ.get("KYOJIN_GLM_LEAN_CACHE", "0") == "1"

STATS: dict = {"pool_insert_fast": 0, "pool_insert_stock": 0}

_arange_cache: dict = {}


def _rows(b: int) -> mx.array:
    """`mx.arange(b)[:, None]` memoized per batch size (the served lane only ever runs B=1 or B=4, same
    idiom as `glm_dsa._arr` / `glm_fuse_dsa._arr`'s small-constant cache): one real dispatch total for
    the whole process's life per B, not one per call. Census (`bpcache/census_pool.py`, B=4): without this
    the scatter path is 2 real dispatches/call (Arange + Scatter, still fewer than stock's 4 SliceUpdate);
    with it, 1 (Scatter only) for any B -- strictly fewer than stock's B SliceUpdate dispatches at B >= 2."""
    a = _arange_cache.get(b)
    if a is None:
        a = _arange_cache[b] = mx.arange(b)[:, None]
        mx.eval(a)
    return a


def pool_insert_b(pooled: mx.array, px: mx.array, pool_lengths, new_counts) -> None:
    """In-place: `pooled[i, pool_lengths[i] : pool_lengths[i] + new_counts[i]] = px[i, :new_counts[i]]`
    for every session i, in ONE scatter dispatch instead of B slice-update dispatches.

    `pooled` [B, P, D] must already be grown to fit every session's `new_counts[i]` (the caller's job,
    same growth check as stock). `px` [B, N, D] with N >= max(new_counts). `pool_lengths` / `new_counts`:
    plain python lists of length B (host-known, no array read needed to size anything).
    """
    max_new = max(new_counts)
    if max_new <= 0:
        return
    rows = _rows(pooled.shape[0])
    cols = mx.array([[pl + j for j in range(max_new)] for pl in pool_lengths])
    pooled[rows, cols] = px[:, :max_new]


def pool_update_and_fetch(self, px: mx.array):
    """Drop-in replacement for the stock `BatchPoolingCache.update_and_fetch` (same semantics, same
    growth policy; only the per-session insert loop becomes one scatter)."""
    if not ENABLED:
        STATS["pool_insert_stock"] += 1
        return self._kyojin_stock_update_and_fetch(px)

    B, N, D = px.shape
    if N == 0:
        FD.fire("lean_cache_pool")
        if self.pooled is None:
            return mx.zeros((B, 0, D), dtype=px.dtype)
        return self.pooled

    new_counts = [
        (self._processed[i] - self.remainder[i]) // self.ratio - self._pool_lengths[i]
        for i in range(B)
    ]
    max_new = max(new_counts)
    if max_new == 0:
        FD.fire("lean_cache_pool")
        if self.pooled is None:
            return mx.zeros((B, 0, D), dtype=px.dtype)
        return self.pooled

    max_pool = max(self._pool_lengths) + max_new

    if self.pooled is None:
        self.pooled = mx.zeros((B, max_pool, D), dtype=px.dtype)
    elif self.pooled.shape[1] < max_pool:
        pad = mx.zeros((B, max_pool - self.pooled.shape[1], D), dtype=px.dtype)
        self.pooled = mx.concatenate([self.pooled, pad], axis=1)

    pool_insert_b(self.pooled, px, self._pool_lengths, new_counts)
    for i in range(B):
        if new_counts[i] > 0:
            self._pool_lengths[i] += new_counts[i]

    STATS["pool_insert_fast"] += 1
    FD.fire("lean_cache_pool")
    return self.pooled


def install() -> bool:
    """Monkeypatch `mlx_vlm.models.cache.BatchPoolingCache.update_and_fetch` once per process.

    Installed unconditionally (like every other `install()` in this package) so `STATS` and the
    `KYOJIN_GLM_LEAN_CACHE=0` <-> `=1` A/B can flip in one running server without a reload; `ENABLED`
    (read inside the wrapper, not here) gates the actual behaviour change.
    """
    from mlx_vlm.models.cache import BatchPoolingCache as C

    if getattr(C, "_kyojin_lean_cache", False):
        return ENABLED
    C._kyojin_stock_update_and_fetch = C.update_and_fetch
    C.update_and_fetch = pool_update_and_fetch
    C._kyojin_lean_cache = True
    return ENABLED
