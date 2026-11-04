"""Prompt-lookup drafts (flag spec_lookup) on synthetic token streams and a fake backend: 4-gram match / longest suffix /
caps / gate / EMA drift, the fixed-size index, the eager MTP round path (lookup rows in the same verify pass), prefix-cache
restore, sampled sessions and multi-session rounds. No model is loaded."""
import os
import threading
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from kyojin_mlx import pld as PLD
from kyojin_mlx.serve.backends import LookupDrafts
from kyojin_mlx.serve.engine import Engine
from kyojin_mlx.serve.sampling import Sampling
from test_serve_engine import Fake, V, collect, onehot


def nxt8(hist):
    """Next token depends on the last 8 tokens only: an earlier identical suffix really does predict the continuation."""
    h = hist[-8:]
    return (sum((i + 3) * t for i, t in enumerate(h)) * 31 + len(h)) % V


class SpecPass:
    def __init__(self, logits, starts, bc, token_lists):
        self.logits, self.starts, self.bc, self.token_lists = logits, starts, bc, token_lists

    def commit(self, accepted, drafts=None, p0s=None):
        for k, (cache, toks) in enumerate(zip(self.bc, self.token_lists)):
            if drafts is not None and drafts[k] is not None:
                drafts[k]["fed"] += accepted[k] + 1
            cache["h"].extend(toks[:accepted[k] + 1])


class FakeLookup(Fake):
    """FakeSpec-shaped backend whose transitions depend on the last 8 tokens, so a repeated prompt span gives a lookup a
    continuation that verifies exactly. `hooks` counts verify passes that ran through the (dummy) ablit hook."""
    kind = "fakelookup"
    supports_spec = True
    max_spec_rows = 24

    def __init__(self):
        self.passes, self.hooks, self.depth_asked = [], 0, []

    def new_draft(self):
        return {"fed": 0}

    def draft_snapshot(self, d):
        return dict(d)

    def draft_restore(self, snap):
        return dict(snap)

    def draft_nbytes(self, d):
        return 8

    def draft_arrays(self, d):
        return []

    def prefill_chunk(self, cache, ids, draft=None):
        if draft is not None:
            draft["fed"] += len(ids)
        cache["h"].extend(ids)
        return onehot([nxt8(cache["h"])])

    def replay(self, cache, tokens, draft=None):
        cache = {"h": cache["h"] + list(tokens)}
        if draft is not None:
            draft["fed"] += len(tokens)
        return cache

    def draft(self, d, pending, p0, depth, history):
        ids, out = history
        assert out[-1] == pending and p0 == len(ids) + len(out) - 1
        self.depth_asked.append(depth)
        h, res = ids + out, []
        for _ in range(depth):
            t = nxt8(h)
            res.append(t)
            h = h + [t]
        return res

    def step(self, bc, tokens):
        for c, t in zip(bc, tokens.tolist()):
            c["h"].append(int(t))
        return onehot([nxt8(c["h"]) for c in bc])

    def spec_verify_bc(self, bc, token_lists):
        from kyojin_mlx import ablit as AB
        self.hooks += 1                                          # where a real backend's model blocks (and their ablit hook) run
        blk = SimpleNamespace(_ablit=SimpleNamespace(wa=0.5, wm=0.0, R=mx.array([[1.0, 0.0]], dtype=mx.float32)))
        y = mx.array([[2.0, 3.0]] * max(len(token_lists), 1), dtype=mx.float32)
        assert AB.attn_out(blk, y).shape == y.shape
        self.passes.append(sum(len(t) for t in token_lists))
        rows, starts = [], []
        for cache, toks in zip(bc, token_lists):
            starts.append(len(rows))
            rows += [nxt8(cache["h"] + toks[:j + 1]) for j in range(len(toks))]
        return SpecPass(onehot(rows), starts, bc, token_lists)


def repeated_prompt(n=20, tail=8):
    """S + C + S, with C the fake's own continuation after S: the model's pending token after the prompt is C[0], and the
    tokens that followed the first S are exactly the lookup draft."""
    s = [(i * 7 + 3) % V for i in range(n)]
    c, h = [], list(s)
    for _ in range(tail):
        c.append(nxt8(h))
        h.append(c[-1])
    return s + c + s


def run(reqs, backend=None, sessions=1, delay=0.0, **kw):
    be = backend or FakeLookup()
    kw.setdefault("ctx", 4096)
    eng = Engine(be, sessions=sessions, chunk=kw.pop("chunk", 4), block=kw.pop("block", 8),
                 prefix_bytes=kw.pop("prefix_bytes", 0), **kw)
    eng.start()
    out = {}

    def worker(i, name):
        if delay:
            import time
            time.sleep(delay * i)
        ids, sub = reqs[name]
        out[name] = collect(eng.submit(ids, **sub))
    ths = [threading.Thread(target=worker, args=(i, n)) for i, n in enumerate(reqs)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    st = eng.status()
    eng.stop()
    return out, st, be


# ------------------------------------------------------------------------------------------------ index / match
def test_match_latest_occurrence_and_suffix():
    ix = PLD.Index(64)
    ix.extend([1, 2, 3, 4, 10, 50, 0, 11, 12, 13, 1, 2, 3, 4, 77, 5])
    ix.extend([60, 0, 11, 12, 13, 1, 2, 3])
    m = ix.match(4, 3)
    assert m is not None and m.draft == [77, 5, 60] and m.suffix == 8
    assert ix.match(9, 3) is None


def test_match_best_takes_the_longest_agreeing_suffix():
    ix = PLD.Index(64)
    ix.extend([9, 9, 9, 1, 2, 3, 4, 50, 51, 52, 53, 54, 7, 1, 2, 3, 4, 60, 61, 70, 9, 9, 9, 1, 2, 3])
    latest = ix.match(4, 4)
    assert latest.draft == [60, 61, 70, 9] and latest.suffix == 4
    best = ix.match_best(4, 4)
    assert best.draft == [50, 51, 52, 53] and best.suffix == 7
    # Alone, or with the latest already best, it is the latest.
    one = PLD.Index(64)
    one.extend([1, 2, 3, 4, 5, 6, 1, 2, 3])
    assert one.match(4, 3).draft == one.match_best(4, 3).draft


def test_index_is_incremental_and_bounded():
    ix = PLD.Index(64)
    assert ix.extend(list(range(36))) == 32                      # positions 4..35
    n, ins = ix.n, ix.inserts
    ix.extend([7, 7, 7, 7])                                       # one more 4-gram, no rebuild
    assert ix.n == n + 4 and ix.inserts == ins + 4
    ix.extend(range(1000))                                        # past the bound: fixed arrays, no growth, no loop
    assert ix.n == ix.cap == 64 and ix.toks.size == 64 and ix.prev.size == 64
    assert ix.head.size >= 64 and ix.inserts <= 64
    assert ix.toks.nbytes + ix.prev.nbytes + ix.head.nbytes <= 64 * 4 * 2 + ix.head.size * 4


# ------------------------------------------------------------------------------------------------ gate / ema
def _draft(n=40):
    return list(range(n))


@pytest.mark.parametrize("suffix,streak,expect", [
    (PLD.MIN_SUFFIX - 1, True, 0),
    (PLD.MIN_SUFFIX, True, PLD.MAX_DRAFT),
    (PLD.STRONG_SUFFIX - 1, True, PLD.MAX_DRAFT),
    (PLD.STRONG_SUFFIX, True, 23),                                # max verify rows 24, less the pending row
])
def test_gate_caps(suffix, streak, expect):
    m = PLD.Match(_draft(), suffix)
    assert PLD.gate(m, 100, 8.4, 4.6, 6, streak, lookup_ms=10.0, mtp_ms=30.0, max_rows=24) == expect


def test_gate_prices_measured_costs_and_falls_back_to_fit_units():
    strong = PLD.Match(_draft(), PLD.STRONG_SUFFIX)
    ordinary = PLD.Match(_draft(), PLD.MIN_SUFFIX)
    # measured: a cheap lookup wins at 6 drafts, a dear one loses.
    assert PLD.gate(ordinary, 100, 4.0, 5.3, 6, False, lookup_ms=20.0, mtp_ms=31.5, max_rows=24) == 6
    assert PLD.gate(strong, 100, 9.0, 1.0, 3, False, lookup_ms=200.0, mtp_ms=21.45, max_rows=24) == 0
    # cold (no measurement): the fit costs decide, and one measured side only sets the unit.
    cold = PLD.gate(strong, 100, 8.4, 4.6, 6, True, max_rows=24)
    assert cold == 23
    assert PLD.gate(strong, 100, 8.4, 4.6, 6, True, mtp_ms=3.7 * PLD.mtp_fit_cost(6), max_rows=24) == cold


def test_drift_retries_lookups():
    ema = 0.0
    for _ in range(300):
        ema = PLD.drift_step(ema)
    assert ema > 6.5                                            # drifts back toward the ordinary cap
    assert PLD.ema_step(ema, 7) > ema
    m = PLD.Match(_draft(), PLD.MIN_SUFFIX)
    assert PLD.gate(m, 100, 0.0, 0.0, 0, False, lookup_ms=30.0, mtp_ms=30.0, max_rows=24) == 0
    assert PLD.gate(m, 100, ema, 0.0, 0, False, lookup_ms=30.0, mtp_ms=30.0, max_rows=24) == PLD.MAX_DRAFT


def test_lookup_drafts_merge_interface():
    rows = {1: [11, 12, 13]}
    lk = LookupDrafts(rows, [3, 3], base_rows={0: [7, 8, 9]})
    assert lk.toks.shape == (2, 3) and lk.slot == {0: 0, 1: 1}
    assert lk.tolist() == [[7, 8, 9], [11, 12, 13]]
    assert lk.keep == [3, 3] and lk.counts == [3, 3]


class _LazyBase:
    """The duck-typed part of LazyDrafts a LookupDrafts reads: the MTP rows of the sessions without a lookup."""
    depths = [1, 2]
    slot = {0: 0}
    keep = None
    toks = mx.array([[5, 0], [6, 7]], dtype=mx.int32)

    def tolist(self):
        self.keep = list(self.depths)
        return [[5], [6, 7]]


def test_lookup_drafts_merge_lazy_base():
    lk = LookupDrafts({1: [11, 12]}, [1, 2], base=_LazyBase())
    assert lk.slot == {0: 0, 1: 1} and lk.counts == [1, 2]
    assert lk.toks.tolist() == [[5, 0], [11, 12]]                # the MTP row keeps its block, padded to the lookup width
    assert lk.tolist() == [[5], [11, 12]] and lk.keep == [1, 2]


# ------------------------------------------------------------------------------------------------ engine
def test_lookup_plan_reclamps_to_the_room_left(monkeypatch):
    """A step in flight commits a token before the round: _lookup_plan(clamp=True) re-cuts the drafts to the reply room and
    the lookup draft is truncated with the depth (no verify rows past max_new)."""
    monkeypatch.setenv("KYOJIN_SPEC_LOOKUP", "1")
    ids = repeated_prompt()
    eng = Engine(FakeLookup(), sessions=1, ctx=4096, chunk=4, block=8, prefix_bytes=0, mtp_depth=4)
    s = SimpleNamespace(ids=ids, rids=ids, out=[nxt8(ids)], n_prompt=len(ids), n_gen=1, max_new=3, last=nxt8(ids),
                        lk_ema=0.0, mt_ema=0.0, lk_streak=True, lookup_index=None)
    depths, lookups = eng._lookup_plan([s], [4], clamp=True)      # 1 token left to emit: at most 1 draft
    assert depths == [1] and len(lookups[0][1]) == 1
    assert s.lookup_index is not None and len(s.lookup_index) == len(ids)     # prompt + output[:-1], indexed once


def test_lookup_rounds_beat_plain_and_keep_greedy_output(monkeypatch):
    ids = repeated_prompt()
    monkeypatch.setenv("KYOJIN_SPEC_LOOKUP", "1")
    got, st, be = run({"a": (ids, dict(max_new=96))}, mtp_depth=4)
    assert got["a"][1] == "length" and len(got["a"][0]) == 96
    assert st["spec_lookup_rounds"] > 0 and st["spec_lookup_drafted"] > PLD.MAX_DRAFT
    assert st["spec_lookup_accepted"] > 0 and be.hooks > 0
    assert any(p > PLD.MAX_DRAFT + 1 for p in be.passes)         # the >= 16 suffix took the strong cap (rows > 8)
    monkeypatch.setenv("KYOJIN_SPEC_LOOKUP", "0")
    plain, _, _ = run({"a": (ids, dict(max_new=96))}, mtp_depth=0)
    mpt, _, _ = run({"a": (ids, dict(max_new=96))}, mtp_depth=4)
    assert got["a"][0] == plain["a"][0] == mpt["a"][0]


def test_lookup_prefix_restore_and_new_turn(monkeypatch):
    ids = repeated_prompt()
    monkeypatch.setenv("KYOJIN_SPEC_LOOKUP", "1")
    be = FakeLookup()
    eng = Engine(be, sessions=1, ctx=4096, chunk=4, block=8, prefix_bytes=1 << 20, mtp_depth=4)
    eng.start()
    try:
        a = collect(eng.submit(ids, max_new=48))
        b = collect(eng.submit(ids, max_new=48))                # prefix hit: the index takes the cached prefix in one insert
        st = eng.status()
    finally:
        eng.stop()
    assert a[0] == b[0]
    assert b[2]["cached_tokens"] > 0 and st["prefix_hits"] >= 1
    assert st["spec_lookup_rounds"] > 0


def test_lookup_multi_session_and_sampled(monkeypatch):
    ids = repeated_prompt()
    other = [((i * 11 + 5) % V) for i in range(24)]
    monkeypatch.setenv("KYOJIN_SPEC_LOOKUP", "1")
    s = Sampling(0.9, 1.0, 0, 0.0, 7)
    reqs = {"g": (ids, dict(max_new=64)), "s": (other, dict(max_new=64, sampling=s))}
    got, st, be = run(reqs, sessions=2, mtp_depth=4, delay=0.01)
    assert got["g"][1] == got["s"][1] == "length"
    assert st["spec_lookup_rounds"] > 0 and st["spec_rows"] == sum(be.passes)
    assert max(be.passes) <= be.max_spec_rows
    monkeypatch.setenv("KYOJIN_SPEC_LOOKUP", "0")
    want, _, _ = run({"g": (ids, dict(max_new=64))}, mtp_depth=0)
    assert got["g"][0] == want["g"][0]                          # a greedy session equals serial decode with a sampled neighbour


def test_full_index_does_not_draft_from_a_stale_tail(monkeypatch):
    monkeypatch.setenv("KYOJIN_SPEC_LOOKUP", "1")
    eng = Engine(FakeLookup(), sessions=1, ctx=4096, chunk=4, block=8, prefix_bytes=0, mtp_depth=4)
    idx = PLD.Index(16)
    idx.extend([1, 2, 3, 4, 5, 6, 7, 8] * 2)                     # full: its tail (6, 7, 8) is not the session's tail
    s = SimpleNamespace(lookup_index=idx, n_prompt=100, out=[], n_gen=0, max_new=64, last=1, lk_ema=0.0, mt_ema=0.0, lk_streak=True)
    assert eng._lookup_depth(s, 0) == (0, None)                  # 16 of 100 committed tokens indexed
    s2 = SimpleNamespace(**{**vars(s), "n_prompt": 16})          # same index, up to date: the match is real
    assert eng._lookup_depth(s2, 0)[0] > 0
