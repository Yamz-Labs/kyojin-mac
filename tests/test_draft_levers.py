"""Draft-cost and round-gate levers on stubs (no model, no Metal): draft_cand (dspark markov tail over K candidates), spec_skip (a round whose
drafts were all cut by the gate takes the plain step), spec_lookup_long (measured, tiered lookup widths)."""
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from kyojin_mlx import dspark as DS

V, RANK, H, L, S = 200, 8, 6, 5, 3


def stub(seed=0):
    mx.random.seed(seed)
    lm = SimpleNamespace(markov_embed=nn.Embedding(V, RANK), markov_head=nn.Linear(RANK, V, bias=False),
                         confidence=nn.Linear(H + RANK, 1, bias=False))
    return SimpleNamespace(lm=lm, block=L, _conf_w=None, _tail_conf=lambda *a: DS.Drafter._tail_conf(drafter, *a)), lm


drafter, _lm = stub()


def tail(cand, scale=1.0, keep=False, ids=None):
    global drafter
    DS.CAND = cand
    drafter, lm = stub()
    drafter._tail_conf = lambda *a: DS.Drafter._tail_conf(drafter, *a)
    mx.random.seed(1)
    x = mx.random.normal((S * L, 1, H))
    logits = mx.random.normal((S, L, V)) * scale
    pend = mx.array([3, 4, 5], dtype=mx.int32)
    out = DS.Drafter._rows_tail(drafter, x, logits, pend, S, keep=keep, ids=ids)
    mx.eval(out[0], out[1])
    return out


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    from kyojin_mlx import rowinv
    monkeypatch.setattr(rowinv, "rowmm", lambda x, w: mx.matmul(x, w.T))      # the Metal row-invariant product has no CPU path
    old = DS.CAND
    yield
    DS.CAND = old


def test_cand_equals_the_full_chain_when_k_covers_the_vocab():
    a, b = tail(0), tail(V)
    assert a[0].tolist() == b[0].tolist() and np.allclose(a[1], b[1])


def test_cand_equals_the_full_chain_when_the_bias_cannot_lift_a_token_past_the_kth():
    a, b = tail(0, scale=50.0), tail(16, scale=50.0)             # base logits spread ~50, bigram bias ~O(1): the argmax is inside the top 16
    assert a[0].tolist() == b[0].tolist()
    assert np.allclose(a[1], b[1], atol=1e-5)
    assert b[0].shape == (S, L) and b[1].shape == (S, L)


def test_cand_keeps_the_exact_chain_when_logits_or_teacher_ids_are_asked():
    full = tail(0, keep=True)
    kept = tail(16, keep=True)
    assert full[2] is not None and kept[2] is not None and full[0].tolist() == kept[0].tolist()


def test_cand_rows_do_not_depend_on_the_other_sessions():
    DS.CAND = 8
    d, _ = stub()
    d._tail_conf = lambda *a: DS.Drafter._tail_conf(d, *a)
    mx.random.seed(2)
    x = mx.random.normal((S * L, 1, H))
    lg = mx.random.normal((S, L, V)) * 20
    pend = mx.array([3, 4, 5], dtype=mx.int32)
    allr = DS.Drafter._rows_tail(d, x, lg, pend, S)
    one = DS.Drafter._rows_tail(d, x[L:2 * L], lg[1:2], pend[1:2], 1)
    assert allr[0][1].tolist() == one[0][0].tolist()


# ------------------------------------------------------------------------------------------------ spec_skip (engine, fake backend)
def _cut_backend():
    from test_serve_spec import FakeSpec

    class Cut(FakeSpec):
        """A draft head whose gate cuts every proposal (prose): draft_many answers [] for every job; step_feed is the plain step."""
        steps = 0

        def draft(self, d, pending, p0, depth, history):
            self.depth_asked.append(depth)
            return []

        def step_feed(self, bc, tokens, drafts, p0s):
            self.steps += 1
            return self.step(bc, tokens)
    return Cut(err=0)


def test_spec_skip_replaces_one_row_verifies_by_plain_steps_and_keeps_the_output(monkeypatch):
    from test_serve_spec import run
    monkeypatch.setenv("KYOJIN_SPEC_LOOKUP", "0")
    reqs = {"g": ([1, 2, 3, 4], dict(max_new=40))}
    monkeypatch.setenv("KYOJIN_SPEC_SKIP", "0")
    base, st0, be0 = run(reqs, _cut_backend(), mtp_depth=3, depth_override=3)
    monkeypatch.setenv("KYOJIN_SPEC_SKIP", "1")
    got, st1, be1 = run(reqs, _cut_backend(), mtp_depth=3, depth_override=3)
    assert got["g"][0] == base["g"][0] and len(got["g"][0]) == 40
    assert st0["spec_skip_rounds"] == 0 and len(be0.passes) > 0 and all(p == 1 for p in be0.passes)
    assert st1["spec_skip_rounds"] > 0 and len(be1.passes) == 0 and be1.steps > 0
    assert len(be1.depth_asked) < len(be0.depth_asked)           # the resting head is not called every round
    assert st1["spec_skip_held"] > 0


def test_spec_skip_keeps_a_session_that_drafts_well_on_the_draft_path(monkeypatch):
    from test_serve_spec import FakeSpec, run
    monkeypatch.setenv("KYOJIN_SPEC_LOOKUP", "0")
    monkeypatch.setenv("KYOJIN_SPEC_SKIP", "1")
    got, st, be = run({"g": ([1, 2, 3, 4], dict(max_new=40))}, FakeSpec(err=0), mtp_depth=3, depth_override=3)
    assert st["spec_skip_rounds"] == 0 and any(p > 1 for p in be.passes) and len(got["g"][0]) == 40


# ------------------------------------------------------------------------------------------------ spec_lookup_long
def test_lookup_costs_price_measured_counts_and_unlock_tier_by_tier():
    from kyojin_mlx import pld as PLD
    c = PLD.LookupCosts()
    assert PLD.tier_caps(24) == [7, 14, 23] and PLD.tier_caps(8) == [7]
    assert c.unlocked(24) == 7 and c.estimate(10) is None
    for k, ms in [(7, 30.0), (7, 30.0), (7, 31.0), (7, 31.0), (3, 20.0), (3, 20.0), (3, 21.0)]:    # the first sample of each count is the graph build
        c.observe(k, ms)
    assert c.median(7) == 30.0 or c.median(7) == 31.0
    assert c.unlocked(24) == 14                                  # a >= 7 round was measured: the middle tier opens, not the top
    assert c.estimate(14) > c.median(7)                          # extrapolated up from the measured slope
    for ms in (60.0, 61.0, 61.0):
        c.observe(14, ms)
    assert c.unlocked(24) == 23 and c.estimate(14) == c.median(14)


def test_gate_with_tiers_takes_the_widest_draft_that_still_pays():
    from kyojin_mlx import pld as PLD
    m = PLD.Match(list(range(40)), PLD.STRONG_SUFFIX)
    cheap = lambda k: 10.0 + 1.0 * k                              # 14 -> 24 ms: more tokens per ms than 7 -> 17 ms
    dear = lambda k: 10.0 if k <= 7 else 200.0                    # the jump past the fused rows: 14 and 23 lose to 7
    kw = dict(streak=True, mtp_ms=40.0, max_rows=24)
    assert PLD.gate(m, 100, 8.0, 3.0, 3, tiers=[7, 14, 23], lookup_ms=cheap, **kw) == 23
    assert PLD.gate(m, 100, 8.0, 3.0, 3, tiers=[7, 14], lookup_ms=cheap, **kw) == 14
    assert PLD.gate(m, 100, 8.0, 3.0, 3, tiers=[7], lookup_ms=cheap, **kw) == 7
    assert PLD.gate(m, 100, 8.0, 3.0, 3, tiers=[7, 14, 23], lookup_ms=dear, **kw) == 7
    ordinary = PLD.Match(list(range(40)), PLD.MIN_SUFFIX)       # a weak match never leaves MAX_DRAFT, tiers or not
    assert PLD.gate(ordinary, 100, 8.0, 3.0, 3, tiers=[7, 14, 23], lookup_ms=cheap, **kw) == 7


def test_lookup_long_engine_keeps_greedy_output_and_measures_lookup_rounds(monkeypatch):
    from test_spec_lookup import repeated_prompt, run as run_lookup
    ids = repeated_prompt()
    monkeypatch.setenv("KYOJIN_SPEC_LOOKUP", "1")
    monkeypatch.setenv("KYOJIN_SPEC_LOOKUP_LONG", "0")
    off, _, be0 = run_lookup({"a": (ids, dict(max_new=96))}, mtp_depth=4)
    monkeypatch.setenv("KYOJIN_SPEC_LOOKUP_LONG", "1")
    on, st, be1 = run_lookup({"a": (ids, dict(max_new=96))}, mtp_depth=4)
    assert on["a"][0] == off["a"][0] and len(on["a"][0]) == 96
    assert st["spec_lookup_rounds"] > 0
    assert max(be1.passes) <= be1.max_spec_rows
