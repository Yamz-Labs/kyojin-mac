"""Command-buffer billing of the dense (non-routed) weights: KYOJIN_DENSE_VIEWS.

MLX closes a command buffer when the data_size of the arrays bound so far passes its limit, and bills every bound array once per buffer by its
data_size, not by what the kernel touches. The routed expert banks already bind a 256 KiB prefix view (moe.bank_view, KYOJIN_MOE_VIEWS); the dense
trellis words of a decode step (wq_a, wkv, wq_b, wo_a, wo_b, shared expert, head, ...) are bound whole, so a model whose dense weights run to GBs pays
tens of extra commits per step, each a pipeline bubble. The kernels read the trellis through a raw pointer (no `trellis_shape` in any source), so the
prefix view of the same buffer gives the same bytes: the output is bit-identical.

bind(words, view) is what a decode-path kernel gets for a weight array: the array itself (flag off, or no view), else the prefix view the weight's holder
built with make_view at construction (never evaluated at bind time, so a compiled region or a replay recording sees no eval)."""
from __future__ import annotations

import os

import mlx.core as mx

from . import fired as FD

DENSE_VIEWS = os.environ.get("KYOJIN_DENSE_VIEWS", "0") == "1"      # bind a prefix view of each large dense weight instead of the whole array (bit-identical)
VIEW_WORDS = 1 << 16                                                   # >= 16 KiB: a smaller array takes a slow inline path (moe.VIEW_WORDS)
MIN_BYTES = int(os.environ.get("KYOJIN_DENSE_VIEW_MIN_MB", "1")) << 20  # arrays below this stay as they are: their bill is small next to the buffer limit

def want_view(nbytes: int, size: int, enabled: bool | None = None, min_bytes: int | None = None, view_words: int | None = None) -> bool:
    """Policy (pure): a view only pays for an array whose bill matters, and only if the view is smaller than the array and not in the inline-copy size class."""
    on = DENSE_VIEWS if enabled is None else enabled
    vw = VIEW_WORDS if view_words is None else view_words
    return bool(on) and nbytes >= (MIN_BYTES if min_bytes is None else min_bytes) and size > vw


def make_view(words: mx.array) -> mx.array | None:
    """Prefix view of `words` for bind(), built where the weight is built (an eager slice, no copy: the same buffer, data_size of VIEW_WORDS elements). None
    for an array whose bill does not matter (under MIN_BYTES, or not larger than the view). Built whatever the flag says so that a runtime switch never
    evaluates inside a compiled region or a replay recording."""
    if not want_view(words.nbytes, words.size, enabled=True):
        return None
    v = words.reshape(-1)[:VIEW_WORDS]
    mx.eval(v)
    return v


def bind(words: mx.array, view: mx.array | None) -> mx.array:
    """What a decode kernel binds for the weight `words`: its view when the flag is on and one exists, else the array. Never evaluates."""
    if DENSE_VIEWS:
        if view is not None:
            FD.hit("dense_views")
            return view
        FD.hit("dense_views_whole")                                          # flag on, array under MIN_BYTES / not larger than the view: bound whole
    return words
