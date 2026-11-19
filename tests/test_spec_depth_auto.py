"""Flag spec_depth_auto (KYOJIN_SPEC_DEPTH_AUTO, SPEC-DEEP): the speculation depth follows the recent per-position acceptance. Policy unit tests (pure python) and the engine
on the scripted backend of test_glm_adaptive (head right / wrong on a schedule, the verify rolls back): greedy text identical to depth 0 for any depth the policy picks,
deep drafts on a head that keeps landing, d1 / plain on a head that misses, a crowd capped at depth 1. No GPU."""
import random

import pytest

import test_glm_adaptive as TGA
import test_spec_lookup as TSL
from kyojin_mlx import flags as F
from kyojin_mlx import spec_depth as SD
from kyojin_mlx.serve import engine as E
from kyojin_mlx.serve.sampling import Sampling

ENV = "KYOJIN_SPEC_DEPTH_AUTO"


# ------------------------------------------------------------------------------------------------ the policy
def test_registered_default_off_and_runtime():
    f = F.FLAGS["spec_depth_auto"]
    assert f.env == ENV and f.default == "0" and f.runtime


def test_prob_extrapolates_and_tokens_sum_the_run():
    st = SD.State()
    assert st.prob(1) == pytest.approx(SD.PRIOR_P)
    assert st.prob(3) == pytest.approx(SD.PRIOR_P * SD.EXTRAP ** 2)
    st.p = [1.0, 0.5, 0.5]
    assert st.tokens(0) == 1.0
    assert st.tokens(3) == pytest.approx(1 + 1.0 + 0.5 + 0.25)
    assert st.prob(4) == pytest.approx(0.5 * SD.EXTRAP)


def test_observe_counts_only_the_reached_positions():
    st = SD.State()
    st.observe(4, 2)                               # drafts 1, 2 landed, 3 missed, 4 never looked at
    assert len(st.p) == 3
    st2 = SD.State()
    st2.observe(4, 4)
    assert len(st2.p) == 4 and all(p > SD.PRIOR_P for p in st2.p)
    st3 = SD.State()
    st3.observe(3, 0)
    assert len(st3.p) == 1 and st3.p[0] < SD.PRIOR_P
    st3.observe(0, 0)                              # no draft: nothing learned
    assert len(st3.p) == 1


CHEAP = "1,1.4,1.7,1.95,2.2,2.5,2.8,3,3.3"                  # a backend whose 5th row is as cheap as the 4th (the GLM table has a step there)


def run_policy(accs, cap=4, d_ms=None, plain_ms=None, start=None, prior=None):
    """Feed the policy a sequence of per-round landing counts (clipped to the planned depth): the depths it planned."""
    st = start or SD.State()
    out = []
    for a in accs:
        d = SD.choose(st, cap, plain_ms, d_ms or {}, prior)
        out.append(d)
        if d:
            st.observe(d, min(a, d))
    return out, st


def test_a_head_that_keeps_landing_climbs_one_row_per_round_to_the_cap():
    out, _ = run_policy([9] * 12, prior=(1.0, 1.4, 1.7, 1.95, 2.2, 2.5, 2.8, 3, 3.3))
    assert out[:5] == [1, 2, 3, 4, 4] and set(out[4:]) == {4}
    glm, _ = run_policy([9] * 12)                                      # the measured GLM-5.3 shape: the 5th row costs 0.67 plain steps, d3 is the top
    assert glm[:4] == [1, 2, 3, 3] and set(glm[3:]) == {3}


def test_an_unvisited_low_estimate_is_tried_again():
    """A position that missed once and was never drafted again must not pin the depth for ever: its estimate drifts toward the one before it."""
    st = SD.State()
    st.p = [0.95, 0.2]
    st.last = 1
    seen = []
    for _ in range(40):
        d = SD.choose(st, 3, None, {})
        seen.append(d)
        st.observe(d, min(d, 9))
    assert max(seen) >= 2


def test_misses_bring_the_depth_down_at_once():
    out, st = run_policy([9] * 8 + [0] * 16, prior=(1.0, 1.4, 1.7, 1.95, 2.2, 2.5, 2.8, 3, 3.3))
    assert out[7] == 4 and out[-1] <= 1 and min(out[8:]) <= 2


def test_prose_stays_shallow():
    rng = random.Random(3)
    accs = [1 if rng.random() < 0.55 else 0 for _ in range(300)]           # first draft lands 55 %, a second (given the first) 0.5 of those
    accs = [a + (1 if a and rng.random() < 0.4 else 0) for a in accs]
    out, _ = run_policy(accs)
    ds = [d for d in out[20:] if d]
    assert ds and sum(ds) / len(ds) < 2.2, sum(ds) / len(ds)


def test_a_head_that_never_lands_goes_plain_and_probes():
    out, st = run_policy([0] * 80)
    assert out.count(0) > 50
    probes = [i for i, d in enumerate(out) if d == 1 and i > 10]
    assert probes and all(b - a >= SD.PROBE for a, b in zip(probes, probes[1:]))
    assert st.gap > SD.PROBE                       # a failed probe doubles the gap


def test_a_probe_that_lands_brings_drafting_back():
    st = SD.State()
    out, st = run_policy([0] * 40, start=st)
    out2, _ = run_policy([9] * 40, start=st)
    assert out2.count(0) <= SD.PROBE + 1 and out2[-1] == 3


def test_cost_prior_scaled_by_the_measured_depth():
    ms, ref = SD.cost_table(4, 20.0, {})
    assert ms[0] == 20.0 and ms[1] == pytest.approx(20.0 * SD.PRIOR_COST[1]) and ref == 20.0
    ms, ref = SD.cost_table(4, 20.0, {2: [40.0, 40.0, 41.0]})
    assert ms[2] == 40.0 and ms[3] == pytest.approx(SD.PRIOR_COST[3] * 40.0 / SD.PRIOR_COST[2])
    ms, _ = SD.cost_table(4, None, {1: [30.0], 2: [40.0]})                       # one sample is not a measurement (a stall)
    assert ms[2] == pytest.approx(SD.PRIOR_COST[2])


def test_a_measured_expensive_deep_row_keeps_the_depth_down():
    st = SD.State()
    st.p = [0.95, 0.9, 0.9, 0.9]
    st.last = 4
    cheap = SD.choose(st, 4, 17.0, {}, (1.0, 1.4, 1.7, 1.95, 2.2))
    dear = {1: [25.0, 25.0], 2: [60.0, 60.0], 3: [95.0, 95.0], 4: [130.0, 130.0]}          # steep cost: every extra row costs a plain step and more
    st.last = 4
    steep = SD.choose(st, 4, 17.0, dear, (1.0, 1.4, 1.7, 1.95, 2.2))
    assert cheap == 4 and steep < cheap


def test_env_cost_shape(monkeypatch):
    monkeypatch.setenv(SD.COST_ENV, "1,2,3")
    assert SD.prior_cost() == (1.0, 2.0, 3.0)
    monkeypatch.setenv(SD.COST_ENV, "junk")
    assert SD.prior_cost() == SD.PRIOR_COST


# ------------------------------------------------------------------------------------------------ the engine
def _engine(monkeypatch, flag, n_members, depth=4, **kw):
    monkeypatch.setenv(ENV, flag)
    monkeypatch.setenv("KYOJIN_GLM_ADAPTIVE", "0")
    monkeypatch.setenv("KYOJIN_GLM_PLAN", "0")
    monkeypatch.setenv("KYOJIN_GLM_MULTI_PLAIN", "0")
    monkeypatch.setenv("KYOJIN_SPEC_LOOKUP", "0")
    eng = E.Engine(TGA.Scripted(), sessions=4, mtp_depth=depth, ctx=4096, chunk=4, block=8, prefix_bytes=0, **kw)
    eng.members = [E.Session(sid=i, ids=[1, 2, 3, 4, 5], max_new=200, sampling=Sampling(), stop_ids=frozenset(), handle=None) for i in range(n_members)]
    return eng


def test_plan_solo_goes_through_the_policy_and_the_ceiling_is_mtp_depth(monkeypatch):
    monkeypatch.setenv(SD.COST_ENV, CHEAP)
    eng = _engine(monkeypatch, "1", 1)
    s = eng.members[0]
    seq = []
    for _ in range(6):
        d = eng._plan_depth(s)
        seq.append(d)
        eng._spec_update(s, d, d, planned=d)                 # every draft lands
    assert seq[:4] == [1, 2, 3, 4] and max(seq) == 4
    assert eng.stats["spec_pos4_try"] > 0 and eng.stats["spec_pos1_acc"] == eng.stats["spec_pos1_try"]
    eng2 = _engine(monkeypatch, "1", 1, depth=2)
    for _ in range(6):
        d = eng2._plan_depth(eng2.members[0])
        eng2._spec_update(eng2.members[0], d, d, planned=d)
        assert d <= 2


def test_flag_off_is_the_old_plan(monkeypatch):
    on = _engine(monkeypatch, "1", 1)
    assert on._spec_depth_auto
    off = _engine(monkeypatch, "0", 1)
    assert off._plan_depth(off.members[0]) == 4              # the rate rule drafts at the cap until it measures plain
    assert not off._spec_depth_auto and off.members[0].sd is None


def test_a_crowd_is_capped_at_depth_one_and_the_served_rows(monkeypatch):
    for n, most in ((2, 1), (3, 0), (4, 0)):                 # 4 rows: 2 sessions x (1 + 1), none left for 3 or 4
        eng = _engine(monkeypatch, "1", n)
        assert max(eng._plan_depth(s) for s in eng.members) <= most


def test_a_fixed_depth_wins(monkeypatch):
    eng = _engine(monkeypatch, "1", 1, depth_override=2)
    assert eng._plan_depth(eng.members[0]) == 2 and not eng._spec_depth_auto


def test_no_room_no_draft(monkeypatch):
    eng = _engine(monkeypatch, "1", 1)
    eng.members[0].max_new = 1
    assert eng._plan_depth(eng.members[0]) == 0


def run(monkeypatch, flag, sessions=1, n=60, depth=4, miss_every=3, override=None):
    monkeypatch.setenv(ENV, flag)
    monkeypatch.setenv("KYOJIN_GLM_ADAPTIVE", "0")
    monkeypatch.setenv("KYOJIN_GLM_PLAN", "0")
    monkeypatch.setenv("KYOJIN_GLM_MULTI_PLAIN", "0")
    monkeypatch.setenv("KYOJIN_SPEC_LOOKUP", "0")
    reqs = {f"r{i}": ([80 + i, 5, 9, 11, 13 + i], {"max_new": n}) for i in range(sessions)}
    kw = {} if override is None else {"depth_override": override}
    return TSL.run(reqs, backend=TGA.Scripted(miss_every), sessions=sessions, mtp_depth=depth, **kw)


def toks(out):
    return {k: (v[0], v[1]) for k, v in out.items()}


def test_greedy_text_equals_plain_for_every_miss_pattern(monkeypatch):
    for me in (2, 3, 5, 11, 10 ** 9):
        base, _, _ = run(monkeypatch, "0", miss_every=me, override=0)
        out, st, _ = run(monkeypatch, "1", miss_every=me)
        assert toks(out) == toks(base), me
        assert st["spec_rounds"] > 0


def test_deep_drafts_when_the_head_keeps_landing(monkeypatch):
    monkeypatch.setenv(SD.COST_ENV, CHEAP)
    out, st, _ = run(monkeypatch, "1", n=120, miss_every=10 ** 9)
    base, _, _ = run(monkeypatch, "0", n=120, miss_every=10 ** 9, override=0)
    assert toks(out) == toks(base)
    assert st["spec_auto_d4"] > st["spec_auto_d1"] + 5, {k: v for k, v in st.items() if k.startswith("spec_auto")}
    assert st["spec_accepted"] == st["spec_drafted"] and st["spec_rounds"] < 120 / 3        # >= 3 tokens per round on average


def test_shallow_when_the_head_keeps_missing(monkeypatch):
    out, st, _ = run(monkeypatch, "1", n=120, miss_every=1)
    base, _, _ = run(monkeypatch, "0", n=120, miss_every=1, override=0)
    assert toks(out) == toks(base)
    deep = st.get("spec_auto_d3", 0) + st.get("spec_auto_d4", 0)
    assert deep <= 2, {k: v for k, v in st.items() if k.startswith("spec_auto")}


def test_two_and_three_sessions_keep_the_text(monkeypatch):
    for sessions in (2, 3):
        base, _, _ = run(monkeypatch, "0", sessions=sessions, override=0)
        out, st, be = run(monkeypatch, "1", sessions=sessions)
        assert toks(out) == toks(base)
