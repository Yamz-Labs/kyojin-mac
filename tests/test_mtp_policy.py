"""The MTP depth policy on synthetic timings (no model, no GPU work): the engine's own _plan_depth / _spec_update / _learn_k driven round by round
with a fixed plain-step cost and a chosen drafted-round cost, so the depth choice is checked against the arithmetic it should do.
KYOJIN_MTP_POLICY=robust (median spec_k, bounded probe gap) and the default policy side by side; the predictor x cost policies run with
KYOJIN_SPEC_RATE=0, the solo default (KYOJIN_SPEC_RATE=1: the cap unless its measured rate loses to plain) has its own tests at the end."""
import random

import pytest

from kyojin_mlx.serve.engine import Engine, PROBE_CAP, Session
from kyojin_mlx.serve.sampling import Sampling
from test_serve_spec import FakeSpec, prompts, run, solo_all

PLAIN_MS = 25.0


def engine(monkeypatch, robust: bool, depth: int = 3, rate: str = "0") -> Engine:
    monkeypatch.setenv("KYOJIN_SPEC_RATE", rate)
    if robust:
        monkeypatch.setenv("KYOJIN_MTP_POLICY", "robust")
    else:
        monkeypatch.setenv("KYOJIN_MTP_POLICY", "constant")
    eng = Engine(FakeSpec(err=0), sessions=1, ctx=1 << 20, chunk=4, block=8, mtp_depth=depth)
    eng.plain_ms = PLAIN_MS
    return eng


def session() -> Session:
    return Session(sid=0, ids=[1, 2, 3], max_new=1 << 20, sampling=Sampling(0.0, 1.0, 0, 0.0, 0), stop_ids=frozenset(), handle=None)


def simulate(eng: Engine, p: float, round_ms, rounds: int, seed: int = 0) -> list:
    """round_ms(i, d) -> ms of drafted round i at depth d (the verify + draft cost). Each draft position lands with probability p. Returns the
    planned depth of every round."""
    s, rng, plan = session(), random.Random(seed), []
    eng.members = [s]
    for i in range(rounds):
        d = eng._plan_depth(s)
        plan.append(d)
        if d == 0:
            continue
        a = 0
        while a < d and rng.random() < p:
            a += 1
        eng._spec_update(s, d, a)
        eng._spec_obs = (round_ms(i, d), d, a + 1)        # (ms, depth, emitted) as engine._spec_tick stores it
        eng._learn_k()
    return plan


def cost(k: float):
    """A drafted round that costs k times the formula's round (1 + row_cost * d + draft_cost plain steps)."""
    return lambda i, d: k * PLAIN_MS * (1.0 + 0.4 * d + 0.15)


@pytest.mark.parametrize("robust", [False, True])
def test_cheap_verify_and_good_acceptance_draft(monkeypatch, robust):
    plan = simulate(engine(monkeypatch, robust), 0.85, cost(1.0), 400)
    assert sum(d >= 2 for d in plan[20:]) >= 0.95 * len(plan[20:])


@pytest.mark.parametrize("robust", [False, True])
def test_todays_slow_verify_stays_plain(monkeypatch, robust):
    """Served 28/10: D=3 rounds ~4x a plain step (k ~ 1.74 against the formula) at 0.69 acceptance per position: plain is faster, D=0 is right.
    The probes stay rare (the robust bound: one D=1 round per PROBE_CAP x probe_every rounds at most once the floor is reached)."""
    eng = engine(monkeypatch, robust)
    plan = simulate(eng, 0.69, cost(1.74), 4000)
    tail = plan[200:]
    assert sum(d > 0 for d in tail) <= len(tail) // 40
    assert max(tail) <= eng.mtp_depth                                # only probes (D = 1, or the cap while the fit lacks its samples)
    if robust:
        assert sum(d > 0 for d in tail) >= len(tail) // (PROBE_CAP * eng.probe_every) - 1


def test_one_slow_round_does_not_switch_drafting_off(monkeypatch):
    """A one-off slow drafted round (10x: a stall, a cold graph, a page fault) in a fast stretch: the robust median ignores it."""
    slow = {30}
    ms = lambda i, d: cost(10.0 if i in slow else 1.0)(i, d)
    plan = simulate(engine(monkeypatch, True), 0.85, ms, 300)
    assert all(d > 0 for d in plan[20:])


def test_default_policy_is_switched_off_by_one_slow_round(monkeypatch):
    """The failure mode the robust policy fixes (documented, default unchanged): the capped EMA takes spec_k to ~2.5 and D=0 with long probe gaps.
    Formula costs (KYOJIN_SPEC_FIT=0): the measured fit prices rounds from per-depth medians, which one slow round does not move."""
    monkeypatch.setenv("KYOJIN_SPEC_FIT", "0")
    slow = {30}
    ms = lambda i, d: cost(10.0 if i in slow else 1.0)(i, d)
    plan = simulate(engine(monkeypatch, False), 0.85, ms, 300)
    assert sum(d == 0 for d in plan[32:]) > 100


@pytest.mark.parametrize("fit", ["0", "1"])
def test_slow_first_sample_recovers_in_bounded_rounds(monkeypatch, fit):
    """The first cost sample is slow (spec_k at its cap of 4: D=0), the next ones are fast: the bounded probe floor lets the robust policy
    take a second sample and draft again within two probe floors, with the formula's costs and with the measured fit (one sample of a depth
    is not a point of the fit)."""
    monkeypatch.setenv("KYOJIN_SPEC_FIT", fit)
    eng = engine(monkeypatch, True)
    def ms(i, d):                                                    # the first round of a depth is a warm-up (not a sample): slow = the first real sample
        return cost(10.0 if eng._k_seen == 0 and d in eng._k_warm else 1.0)(i, d)
    plan = simulate(eng, 0.85, ms, 1200)
    off = next(i for i in range(len(plan)) if i > 2 and plan[i] == 0)
    back = next(i for i in range(off, len(plan)) if plan[i] >= 2)
    assert back - off <= 2 * PROBE_CAP * eng.probe_every + 8
    assert sum(d >= 2 for d in plan[back + 20:]) >= 0.95 * len(plan[back + 20:])


class FlushSpec(FakeSpec):
    """FakeSpec with the backend's deferred-tap flush: counts the calls the robust policy makes before it times a drafted round."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.flushes = 0

    def flush_drafts(self, ds):
        self.flushes += 1


@pytest.mark.parametrize("robust", [False, True])
def test_robust_policy_flushes_outside_the_timed_round_and_keeps_tokens(monkeypatch, robust):
    if robust:
        monkeypatch.setenv("KYOJIN_MTP_POLICY", "robust")
    else:
        monkeypatch.setenv("KYOJIN_MTP_POLICY", "constant")
    reqs = prompts(3)
    want = solo_all(reqs)
    got, st, be = run(reqs, FlushSpec(err=3), sessions=1, mtp_depth=3)
    for n in reqs:
        assert got[n][0] == want[n][0], n
    drafted = sum(v for k, v in st.items() if k.startswith("spec_d") and k[6:].isdigit() and k != "spec_d0")
    assert (be.flushes > 0) == robust and (not robust or be.flushes >= drafted)


def test_robust_is_the_default_policy(monkeypatch):
    monkeypatch.delenv("KYOJIN_MTP_POLICY", raising=False)
    eng = Engine(FakeSpec(err=0), sessions=1, ctx=1 << 20, chunk=4, block=8, mtp_depth=3)
    assert eng.robust
    monkeypatch.setenv("KYOJIN_MTP_POLICY", "constant")
    assert not Engine(FakeSpec(err=0), sessions=1, ctx=1 << 20, chunk=4, block=8, mtp_depth=3).robust


def test_fit_drafts_when_rows_are_cheap(monkeypatch):
    """DS4.1 on the Ultra (30/10): a drafted round = plain + a fixed ~0.5 step + ~0.1 step per row (D=1 1.6x plain, D=3 1.8x). At 0.65
    acceptance per position D=3 pays (2.35 tokens / 1.8) and D=1 does not; the formula, scaled on the D=1 rounds it sees, charges D=3
    2.4x and stays plain. The measured fit (cap probes give it two depths) drafts deep."""
    ms = lambda i, d: PLAIN_MS * (1.5 + 0.1 * d)
    plan = simulate(engine(monkeypatch, True), 0.65, ms, 600)
    assert sum(d >= 2 for d in plan[100:]) >= 0.8 * len(plan[100:])
    monkeypatch.setenv("KYOJIN_SPEC_FIT", "0")
    plan = simulate(engine(monkeypatch, True), 0.65, ms, 600)
    assert sum(d >= 2 for d in plan[100:]) < 0.2 * len(plan[100:])


def test_fit_still_stays_plain_on_slow_verify(monkeypatch):
    """Fixed 3 at 4x a plain step and 0.69 acceptance: plain is faster, the fit agrees."""
    plan = simulate(engine(monkeypatch, True), 0.69, lambda i, d: PLAIN_MS * (1.0 + d), 2000)
    assert sum(d > 0 for d in plan[200:]) <= len(plan[200:]) // 20



def rate_engine(monkeypatch, depth: int = 3) -> Engine:
    return engine(monkeypatch, True, depth, rate="1")


def test_rate_drafts_at_the_cap_when_it_pays(monkeypatch):
    plan = simulate(rate_engine(monkeypatch), 0.85, cost(1.0), 400)
    assert sum(d == 3 for d in plan) >= 0.97 * len(plan)


def test_rate_served_shape_drafts_from_a_cold_start(monkeypatch):
    """Served 30/10 (DS4.1, Ultra): fixed 3 = 1.75 plain steps, 0.70 acceptance on the hardest text (35.0 tok/s vs 33.9 plain), a 10x first
    sample: the cost policies sat at D = 0 for ~300 rounds there; the rate rule drafts at the cap from round 0."""
    ms = lambda i, d: PLAIN_MS * (10.0 if i < 3 else 1.5 + 0.083 * d)
    for seed in range(5):
        plan = simulate(rate_engine(monkeypatch), 0.70, ms, 400, seed)
        assert sum(d == 3 for d in plan) >= 0.95 * len(plan), seed


def test_rate_goes_plain_on_low_acceptance_and_probes_rarely(monkeypatch):
    plan = simulate(rate_engine(monkeypatch), 0.2, lambda i, d: PLAIN_MS * (1.5 + 0.083 * d), 4000)
    assert sum(d > 0 for d in plan[:40]) <= 12                        # plain after the first window (+ the probes)
    tail = plan[400:]
    assert sum(d > 0 for d in tail) <= len(tail) // 50                # probes: 2 cap rounds per PROBE_CAP x probe_every rounds
    assert set(tail) <= {0, 3}


def test_rate_todays_slow_verify_stays_plain(monkeypatch):
    plan = simulate(rate_engine(monkeypatch), 0.69, cost(1.74), 4000)
    tail = plan[200:]
    assert sum(d > 0 for d in tail) <= len(tail) // 40


def test_rate_recovers_when_acceptance_returns(monkeypatch):
    """Hostile text then easy text in one session: back at the cap within the probe gap of the longest plain stretch."""
    eng, s, rng, plan = rate_engine(monkeypatch), session(), random.Random(0), []
    eng.members = [s]
    for i in range(1200):
        p = 0.1 if i < 600 else 0.9
        d = eng._plan_depth(s)
        plan.append(d)
        if d == 0:
            continue
        a = 0
        while a < d and rng.random() < p:
            a += 1
        eng._spec_update(s, d, a)
        eng._spec_obs = (PLAIN_MS * (1.5 + 0.083 * d), d, a + 1)
        eng._learn_k()
    assert sum(d > 0 for d in plan[100:600]) <= 30
    first = next(i for i in range(600, 1200) if plan[i] == 3 and plan[i + 1] == 3 and plan[i + 2] == 3)
    assert first - 600 <= PROBE_CAP * eng.probe_every + 4
    assert sum(d == 3 for d in plan[first:]) >= 0.97 * len(plan[first:])


def test_rate_is_off_with_the_flag(monkeypatch):
    """KYOJIN_SPEC_RATE=0: the predictor x cost policy (today's slow verify stays plain; the cap rule would too: a different plan)."""
    plan = simulate(engine(monkeypatch, True, rate="0"), 0.85, cost(1.0), 400)
    assert set(plan[20:]) - {3}


def test_rate_noisy_plain_samples_do_not_drop_a_paying_cap(monkeypatch):
    """Served 30/10: plain rounds read 26-35 ms (true ~29.5) and the EMA's low reading switched two 0.71-acceptance texts to plain (-9 %):
    the median of the plain rounds and the 0.9 margin keep the cap."""
    for seed in range(5):
        eng = rate_engine(monkeypatch)
        eng.plain_ms, eng._plain_ring = 26.0, [26.0, 35.0, 29.5, 28.0, 30.5]
        plan = simulate(eng, 0.71, lambda i, d: 29.5 * (1.5 + 0.083 * d), 600, seed)
        assert sum(d == 3 for d in plan) >= 0.97 * len(plan), seed
