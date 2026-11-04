# SPDX-License-Identifier: Apache-2.0
"""Sliding-window ring of the DS4.1 attention (`cache.window[layer]`, [B, win, D]) read and written with slices instead of a gather and a scatter.

The vendor `_window_part` (and the two kyojin copies of it, `fused_group` S5 and `attn_prefill.window_part`) reads the history of a chunk with
`mx.take(buffer, arange(base, start_pos) % win, axis=1)` (a Gather launch per layer per chunk) and stores the chunk's last `retained` rows with
`buffer[:B, arange(end - retained, end) % win] = tail` (a Scatter on a copy of the ring, per layer per chunk). Both index lists are runs of consecutive
slots modulo `win`: at most two contiguous pieces. Here each piece is a slice, so the ops are Slice + Concatenate: the same rows, copied, no arithmetic.

`read` returns exactly `mx.take(buffer[:B], mx.arange(base, start_pos) % win, axis=1)`; `write` returns exactly the array
`buffer[:B, mx.arange(end - r, end) % win] = tail` leaves behind (rows >= B of the ring kept, the tail cast to the ring dtype as the assignment does).
Switch: `ENABLED` (env KYOJIN_PF_WIN_SLICE, default 0, flags.py `pf_win_slice`, read per call)."""
from __future__ import annotations

import os

import mlx.core as mx

ENABLED = os.environ.get("KYOJIN_PF_WIN_SLICE", "0") not in ("0", "")
HITS = {"read": 0, "write": 0}


def _cat(parts, axis):
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=axis)


def read(buffer: mx.array, batch: int, base: int, start_pos: int, win: int) -> mx.array:
    """Rows of absolute positions base .. start_pos - 1 from the ring [B, win, D]: [batch, start_pos - base, D]."""
    HITS["read"] += 1
    n = start_pos - base
    s = base % win
    b = buffer[:batch]
    if s + n <= win:
        return b[:, s:s + n]
    return mx.concatenate([b[:, s:], b[:, :s + n - win]], axis=1)


def write(buffer: mx.array, batch: int, tail: mx.array, end_pos: int, win: int) -> mx.array:
    """The ring after rows `tail` [batch, r, D] (r <= win) take the slots of positions end_pos - r .. end_pos - 1."""
    HITS["write"] += 1
    r = tail.shape[1]
    s = (end_pos - r) % win
    tail = tail.astype(buffer.dtype)
    first = min(r, win - s)                                  # rows of the tail that land on slots s .. s + first - 1, the rest on slots 0 ..
    head, rest = buffer[:batch], r - first
    pieces = []
    if rest:
        pieces.append(tail[:, first:])                       # slots 0 .. rest - 1
        pieces.append(head[:, rest:s]) if s > rest else None
    else:
        pieces.append(head[:, :s]) if s else None
    pieces.append(tail[:, :first])                           # slots s .. s + first - 1
    if s + first < win:
        pieces.append(head[:, s + first:])
    out = _cat([p for p in pieces if p is not None and p.shape[1]], 1)
    return out if buffer.shape[0] == batch else mx.concatenate([out, buffer[batch:]], axis=0)
