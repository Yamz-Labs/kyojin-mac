# SPDX-License-Identifier: Apache-2.0
"""Depth-flat decode (DS4.1 and any model whose cache keeps its pool / index keys in `fp4pool.PackedSlots`).

The packed slots store the pooled KV rows and the index keys as fp4 codes + scales. A reader that needs the dense fp16 rows (the indexer scores
read `cache.keys[l]`; the chained verify R>=2 reads the pool through `PackedView.dense()` once per reuse layer; short pools before the PackedView
threshold) used to unpack the WHOLE layer on every step: ~12 un-fused elementwise ops that move ~40 bytes per element, so the cost grew with the
context (about 650 MB of traffic per 128K-key layer per step) and was paid again after every append (the memo was dropped on append, and never kept
above KYOJIN_FP4_MEMO_MB).

KYOJIN_DEPTHFLAT=1 (default 0, exact):
    mirror   every packed layer keeps a dense mirror in a preallocated buffer (same dtype as the stock unpack). A dense read unpacks only the rows that
             were appended since the last read (rows and blocks are independent: bit-identical to unpacking the whole layer) and returns a prefix slice
             of the buffer. Per-step cost O(new rows) instead of O(context). The packed codes stay the ground truth (checkpoints, copies, kvdisk never see
             the mirror); `cow` aliases it, a new `_Pk` never shares a Python array object.
    slack    pool / key code buffers grow with a 1/8 slack (as KYOJIN_LONGCTX_GROW) so a full buffer copy happens every n/8 rows, not every 256.
Memory: a mirror costs n x D x itemsize per layer (fp16 pool 1 KiB per row, keys 256 B per row); at 1M tokens that is ~2.5 GiB of pool mirrors (layers
2, 8, 14 at 0.5M rows, layer 20 at 1M rows) plus ~0.6 GiB of key mirrors. A layer above KYOJIN_DEPTHFLAT_MIRROR_MB (default 2048) keeps the stock path.
HITS counts the paths taken, so a flag that never reached the served path is visible in the gate output."""
from __future__ import annotations

import os

import mlx.core as mx

ENABLED = os.environ.get("KYOJIN_DEPTHFLAT", "1") not in ("0", "")
MIRROR_BYTES = int(float(os.environ.get("KYOJIN_DEPTHFLAT_MIRROR_MB", "2048")) * 2 ** 20)
SLACK_SHIFT = 3
HITS = {"build": 0, "tail": 0, "hit": 0, "grow": 0, "view": 0, "skip": 0, "cap": 0}
_DTYPES = (mx.float16, mx.bfloat16, mx.float32)


def capacity(need: int, step: int) -> int:
    """Buffer capacity for `need` rows: a 1/8 slack, rounded to the step."""
    HITS["cap"] += 1
    need = need + (need >> SLACK_SHIFT)
    return -(-need // step) * step


def mirror_read(slots, e):
    """Dense rows [B, e.n, D] of the packed layer `e` (`fp4pool._Pk`) through its mirror, or None when the layer is outside the mirror budget.
    `slots` supplies the codec (BLOCK, E4M3)."""
    from . import fp4pack
    n = e.n
    B, D = e.codes.shape[0], e.codes.shape[2] * 2
    if e.dtype not in _DTYPES or B * n * D * (4 if e.dtype == mx.float32 else 2) > MIRROR_BYTES:
        HITS["skip"] += 1
        return None
    m, mn = e.mirror, e.mn
    if m is not None and mn > n:                                    # cannot happen with append-only layers; rebuild rather than trust it
        m, mn = None, 0
    if m is not None and mn == n:
        HITS["hit"] += 1
        return m[:, :n]
    if m is None:
        HITS["build"] += 1
        d = fp4pack.unpack(e.codes[:, :n], e.scales[:, :n], slots.BLOCK, slots.E4M3, e.dtype)
        cap = -(-(n + (n >> SLACK_SHIFT)) // slots.STEP) * slots.STEP
        m = d if cap <= n else mx.concatenate([d, mx.zeros((B, cap - n, D), dtype=e.dtype)], axis=1)
        e.mirror, e.mn = m, n
        return m[:, :n]
    if m.shape[1] < n:                                              # grow: copy the valid prefix once per n/8 rows
        HITS["grow"] += 1
        cap = -(-(n + (n >> SLACK_SHIFT)) // slots.STEP) * slots.STEP
        m = mx.concatenate([m[:, :mn], mx.zeros((B, cap - mn, D), dtype=e.dtype)], axis=1)
    HITS["tail"] += 1
    m[:, mn:n] = fp4pack.unpack(e.codes[:, mn:n], e.scales[:, mn:n], slots.BLOCK, slots.E4M3, e.dtype)
    e.mirror, e.mn = m, n
    return m[:, :n]
