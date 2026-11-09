# SPDX-License-Identifier: Apache-2.0
"""Flag pf_cut_merge: one prefill pass across the LAST chat cut, with the cut state captured inside the pass.

The chat layer passes `cuts` (end of the system block, end of the conversation before the generation prompt). The engine ends a chunk at a cut and keeps a
snapshot there, so a turn with new content runs two passes: the new tokens up to the last cut, then the ~28 tokens of the generation prompt. Every pass pays a fixed
cost (the MoE decodes the trellis of every routed expert its rows touch), so the second pass costs ~150 ms for 28 rows. This module lets the engine run ONE pass over
[start, end) and still keep the snapshot at the cut.

Design: the cut state of a position is overwritten by the rows past the cut only in the FIXED-size and append-only slots of the per-session cache (window ring,
compressor group state, compressed pool, index keys). Only the attention touches them. So the attention of every layer is run in two segments inside the pass,
rows [0, k) and rows [k, n) (k = cut - start), with the cache offset moved between them, and the layer's slots are captured between the segments. Everything else
(hyper-connection mixes, Engram, the MoE and the shared expert, which hold the fixed cost) runs ONCE over all n rows. One generic place: the `attn.__call__`
wrapper `_call`; every attention writer (stock, attn_prefill, compressor_pf, fuse_attn, fused_group, glue paths) runs under it unchanged, because they all read
`cache.offset` as the position of their first row and a row's attention does not depend on later rows.

  captured per layer after the first segment (what the split path holds at the cut): window ring, kv_state / score_state, compress, keys (packed slots: `cow`, an
  alias, so the live buffer is copied on its next append: transient, one copy per layer)
  taken from the end of the pass: Engram history (prefix slice `[:, :cut]`), offset (= cut); the cross-layer handoff is reset by `snapshot()`
  drafter: the taps of the pass are kept for 64 more rows (`taps_keep`), the ring at the cut is the ring before the pass fed with the taps of the rows < cut
  cross-layer handoff (index_k, candidates, compress_kv, topk_idxs): a source layer writes it for its segment; both segments keep their own copy and the layers
  after it read the one of their segment.

Rounding class: the attention projections of the two segments run with another M than the split path's two passes (the first segment is the split path's first
pass M), and the MoE with M = n instead of two passes; tests compare to rounding.

Mode 3 (CED pass kept): the pass runs with `ced_tail = tail + t` (t = rows past the cut), so the decoder half sees the split path's CED tail [cut - tail, cut) and the
t rows after it. The decoder layers' attention runs in two segments of tail / t rows (`_call`, second segment with history from the ring), the midpoint (which calls
`ced._attention_mid` outside `attn.__call__`) through `mid_attention`: the split path's CED chunk over [start, cut), then its own attention at the cut. Slots are captured
after segment 0 as in the encoder half. A pass of n <= tail + t rows is no CED pass (the split path's two chunks are not either when k <= tail). The result after the cut
matches the split path's two chunks; a stale (skipped-decoder) snapshot before the pass is read only if the first chunk is a CED chunk, as in the split path.

Not covered (the engine falls back to the split path, see `VlmBackend.cut_merge_ok`): CED passes unless the flag is >= 2 (then the pass runs `ced.exact`: the CED midpoint
calls `_attention_mid` directly, outside the attention call), a TurboQuant cache (KYOJIN_TQ), a prompt with images, a batched / rows cache, a rebuilt session, a cut
a pf_share taker is waiting for, a tail > 64 tokens, a chunk that would cross a grid point."""
from __future__ import annotations

import os

import mlx.core as mx

TAIL_MAX = 64                       # longest tail after the cut that is merged (the generation prompt is ~28 tokens)
SLOTS = ("window", "compress", "keys", "kv_state", "score_state")


def mode() -> int:
    """KYOJIN_PF_CUT_MERGE (read per call): 0 = off, 1 = merge the passes that are not CED passes, 2 = also CED passes (run exact: the decoder half sees every row),
    3 = also CED passes, kept CED: the decoder half runs the last `tail + rows past the cut` rows (the split path's tail before the cut, then the short chunk)."""
    return int(os.environ.get("KYOJIN_PF_CUT_MERGE", "0") or 0)


class Cut:
    """State of one merged pass, hung on the cache entry as `_cutm` while the model runs."""

    def __init__(self, k: int, n: int, dec_n: int = 0, dec_k: int = 0):
        self.k, self.n = k, n
        self.dec_n, self.dec_k = dec_n, dec_k   # mode 3, CED pass: rows of the decoder half (tail + rows past the cut) and how many of them lie before the cut (= tail)
        self.grab: dict = {}             # layer -> {slot: captured entry}
        self.h = [None, None]            # cross-layer handoff of segment 0 / 1


def _handoff(cache) -> dict:
    return {a: getattr(cache, a) for a in cache.HANDOFF}


def _set_handoff(cache, h: dict) -> None:
    for a, v in h.items():
        setattr(cache, a, v)


def _grab(cache, l: int) -> dict:
    out = {}
    for name in SLOTS:
        slots = getattr(cache, name)
        if hasattr(slots, "cow"):                       # fp4pool.PackedSlots: the raw entry, aliased (never unpacked)
            slots._settle(l)
            out[name] = slots.cow(list.__getitem__(slots, l))
        else:
            v = slots[l]
            out[name] = None if v is None else v[...]
    return out


def _call(attn, orig, x, cache):
    cm = getattr(cache, "_cutm", None)
    if cm is None or x.ndim != 3 or x.shape[0] != 1 or x.shape[1] not in (cm.n, cm.dec_n):
        return orig(attn, x, cache)
    dec = cm.dec_n > 0 and x.shape[1] == cm.dec_n              # a decoder layer of a CED pass: its rows are the last dec_n of the pass
    l, k = attn.layer_idx, cm.dec_k if dec else cm.k
    if cm.h[0] is None:
        cm.h = [_handoff(cache), _handoff(cache)]
    _set_handoff(cache, cm.h[0])
    o1 = orig(attn, x[:, :k], cache)
    cm.h[0] = _handoff(cache)
    cm.grab[l] = _grab(cache, l)
    off, nh = cache.offset, getattr(cache, "_ced_nohist", False)
    cache.offset = off + k
    cache._ced_nohist = False                                   # the rows past the cut read the ring the first segment wrote (the split path's second, short chunk)
    _set_handoff(cache, cm.h[1])
    try:
        o2 = orig(attn, x[:, k:], cache)
    finally:
        cache.offset = off
        cache._ced_nohist = nh
    cm.h[1] = _handoff(cache)
    return mx.concatenate([o1, o2], axis=1)


def mid_attention(attn, x_full, cache, ctx, cm: Cut):
    """Mode 3: the attention of the CED midpoint (it is called outside `attn.__call__`) in the two segments of the pass. Segment 0 is the split path's CED chunk
    [start, cut): compressed KV and index keys of all its rows, queries / window / indexer over its last `tail` rows (no history before them); the layer's slots are
    captured after it. Segment 1 is the split path's short chunk after the cut: the layer's own attention at offset cut, window ring from segment 0. Returns the
    outputs of the decoder rows (the last dec_n rows of the pass, tail rows before the cut then the rows after it)."""
    from . import ced
    l, k, tk = attn.layer_idx, cm.k, cm.dec_k
    if cm.h[0] is None:
        cm.h = [_handoff(cache), _handoff(cache)]
    _set_handoff(cache, cm.h[0])
    nh = getattr(cache, "_ced_nohist", False)
    cache._ced_nohist = True
    try:
        o1 = ced._attention_mid(attn, x_full[:, :k], x_full[:, k - tk:k], cache, ctx.s0, ctx.s0 + k - tk)
    finally:
        cache._ced_nohist = nh
    cm.h[0] = _handoff(cache)
    cm.grab[l] = _grab(cache, l)
    off = cache.offset
    cache.offset = ctx.s0 + k
    cache._ced_nohist = False                                   # (the caller set it for the whole midpoint)
    _set_handoff(cache, cm.h[1])
    try:
        o2 = attn(x_full[:, k:], cache)
    finally:
        cache.offset = off
        cache._ced_nohist = nh
    cm.h[1] = _handoff(cache)
    return mx.concatenate([o1, o2], axis=1)


def install(lm) -> int:
    """Wrap the attention call of every layer of `lm` (a language model). Idempotent; a wrapped call without a `_cutm` on the cache is a plain call."""
    from .ced import _override
    n = 0
    for blk in lm.layers:
        attn = getattr(blk, "attn", None)
        if attn is None or getattr(attn, "_kyojin_cutm", False):
            continue
        _override(attn, "__call__", _call)
        attn._kyojin_cutm = True
        n += 1
    return n


def build(cache: list, cm: Cut, start: int, fork) -> list:
    """The cache as the split path holds it at the cut: a fork of the cache after the pass, with the slots captured between the segments, the Engram history cut at
    the cut and the offset there. `fork` = the backend's fork (rows.fork). Lazy: the caller evaluates it with the pass."""
    out = fork(cache)
    e = out[0]
    cut = start + cm.k
    for l, g in cm.grab.items():
        for name, v in g.items():
            slots = getattr(e, name)
            if hasattr(slots, "set_raw"):
                slots.set_raw(l, v)
            else:
                slots[l] = v
    e._compress_buffers = [None] * e.n_layers
    e._key_buffers = [None] * e.n_layers
    e.offset = cut
    if e.engram is not None:
        e.engram = e.engram[:, :cut]
    e.reset_handoff()
    return out
