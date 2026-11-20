# SPDX-License-Identifier: Apache-2.0
"""Flag KYOJIN_VERIFY_CAPTURE (default off): a speculative verify pass keeps the per-row recurrent state of every fixed-state layer, so a partial accept SELECTS
the state after `keep` rows instead of restoring the snapshot and running the trunk again on the accepted rows (fnmtp._rollback's replay = a whole trunk forward).

While `SINK` is set (fnmtp.spec_verify_bc, S > 1 rows, B = 1), a fused gated-delta mixer (foreign_gdn_fast, gdn_fuse steps=True) records for its cache, per row count
keep = 1..S, the conv state and the recurrent state after `keep` rows; ArraysCache.update_window (the n-gram / short-conv windows of the PLE layers) records the window
that ends `S - keep` rows before its end. A cache whose slots are not all recorded (a mixer on the stock path, an unknown slot writer) is reported incomplete and the
caller replays as before. A compiled layer program (step_compile) returns the recorded arrays as extra outputs and re-registers them against the real cache."""
from __future__ import annotations

import os

import mlx.core as mx

ACTIVE = os.environ.get("KYOJIN_VERIFY_CAPTURE", "0") not in ("0", "")
DRAFT = os.environ.get("KYOJIN_VERIFY_COMPILE_DRAFT", "0") not in ("0", "")     # fnmtp: the MTP head layer's mlp (+ layer wrapper) through step_compile
# keep < MINKEEP replays: a replay of one row runs the S = 1 decode kernels (= plain greedy), the recorded state comes from the S = d+1 fused loop and differs in the last bits
MINKEEP = int(os.environ.get("KYOJIN_VERIFY_CAPTURE_MINKEEP", "2"))
SINK = None
STATS = {"captured": 0, "incomplete": 0}
_PATCHED = []


class Sink:
    __slots__ = ("S", "items")

    def __init__(self, S: int):
        self.S, self.items = S, {}            # id(cache) -> (cache, {slot: [array for keep = 1..S]})

    def put(self, cache, slot: int, variants: list):
        self.items.setdefault(id(cache), (cache, {}))[1][slot] = variants

    def select(self, cache, keep: int):
        """The slot list of `cache` after `keep` rows, or None when a live slot was not recorded."""
        ent = self.items.get(id(cache))
        got = ent[1] if ent else {}
        out = []
        for i, a in enumerate(cache.cache):
            if a is None:
                out.append(None)
            elif i in got and len(got[i]) == self.S:
                out.append(mx.contiguous(got[i][keep - 1]))   # a copy: a view would keep the whole d+1-state buffer alive in the cache / prefix store
            else:
                return None
        return out


def begin(S: int):
    global SINK
    SINK = Sink(S) if ACTIVE and S > 1 else None
    if SINK is not None:
        _patch()
    return SINK


def end():
    global SINK
    s, SINK = SINK, None
    return s


def _patch():
    if _PATCHED:
        return
    from mlx_vlm.models.cache import ArraysCache
    orig = ArraysCache.update_window

    def update_window(self, index, source, width, *, lengths=None):
        r = orig(self, index, source, width, lengths=lengths)
        s = SINK
        if s is not None and lengths is None:
            n, w = source.shape[1], int(width)
            if n - w >= s.S:                                   # the window ending `S - keep` rows before the end of `source`
                s.put(self, index, [source[:, n - (s.S - k) - w:n - (s.S - k)] for k in range(1, s.S + 1)])
        return r
    ArraysCache.update_window = update_window
    _PATCHED.append(orig)
