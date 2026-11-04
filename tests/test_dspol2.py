"""Flag dspol_v2: the calibrated draft policy (kyojin_mlx.dspol2) with the measured per-row cost, its trace, and its engine wiring on a fake backend.
Pure math tests need no mlx; the engine tests need it like every serve test."""
import json
import math
import os
import random

import pytest

from kyojin_mlx import dspol2 as D


# ----------------------------------------------------------------------------------------------------------------------- axis, calibration
def test_axis_is_dense_near_one():
    assert D.xof(0.0) == 0.0 and abs(D.xof(0.9) - 1.0) < 1e-9 and abs(D.xof(0.99) - 2.0) < 1e-9
    assert D.xof(1.0) == D.X_MAX and D.xof(0.99999) == D.X_MAX
    cal = D.Calibration()
    assert cal.bin_of(0.90) != cal.bin_of(0.99) != cal.bin_of(0.999)        # the 0.9-1 mass is split over several bins (linear bins: one)
    assert [cal.bin_of(c) for c in (0.0, 0.3, 0.6, 0.9, 0.95, 0.99, 0.999, 1.0)] == sorted(cal.bin_of(c) for c in (0.0, 0.3, 0.6, 0.9, 0.95, 0.99, 0.999, 1.0))


def test_empty_calibration_is_the_prior_and_is_monotone():
    cal = D.Calibration(prior=(-0.2, 1.9))
    ps = [cal.prob(c) for c in (0.1, 0.5, 0.9, 0.99, 0.999)]
    assert all(0 < p < 1 for p in ps) and ps == sorted(ps)
    assert abs(cal.prob(0.9) - 0.85) < 0.05


def test_reached_positions_only():
    cal = D.Calibration()
    cal.observe([0.9, 0.9, 0.9, 0.9, 0.9], drafted=5, accepted=2)          # positions 0, 1 accepted, 2 is the first miss, 3 and 4 never reached
    b = cal.bin_of(0.9)
    assert abs(cal.trial[b] - 3.0) < 0.1 and abs(cal.succ[b] - 2.0) < 0.1
    cal2 = D.Calibration()
    cal2.observe([0.9] * 5, drafted=3, accepted=3)                          # every verified draft landed: 3 trials, 3 successes, no trial past the verified rows
    assert abs(cal2.trial[b] - 3.0) < 0.1 and abs(cal2.succ[b] - 3.0) < 0.1


def test_calibration_learns_a_miscalibrated_head_and_forgets():
    rng = random.Random(1)
    cal = D.Calibration()
    for _ in range(600):                                                    # a head that says 0.95 and is right 40 % of the time
        a = 0
        while a < 3 and rng.random() < 0.4:
            a += 1
        cal.observe([0.95, 0.95, 0.95], 3, a)
    assert abs(cal.prob(0.95) - 0.4) < 0.1, cal.prob(0.95)
    for _ in range(300):                                                    # the text changes: now it is right 90 % of the time
        a = 0
        while a < 3 and rng.random() < 0.9:
            a += 1
        cal.observe([0.95, 0.95, 0.95], 3, a)
    assert cal.prob(0.95) > 0.8, cal.prob(0.95)
    assert abs(cal.prob(0.3) - cal.prior(D.xof(0.3) // cal.bin_w * cal.bin_w + cal.bin_w / 2)) < 0.15     # an unvisited bin is still near its prior


def test_unvisited_bins_do_not_forget():
    cal = D.Calibration()
    cal.observe([0.99], 1, 1)
    b = cal.bin_of(0.99)
    before = (cal.succ[b], cal.trial[b])
    for _ in range(50):
        cal.observe([0.5], 1, 0)
    assert (cal.succ[b], cal.trial[b]) == before


# ----------------------------------------------------------------------------------------------------------------------- cost table
def test_cost_table_is_the_ledger_sweep_at_the_reference_step():
    c = D.RowCost()
    for w, ms in {2: 36.8, 3: 42.0, 4: 46.7, 5: 52.1, 6: 58.1, 8: 63.0}.items():
        assert c.ms(w, 20.57) == pytest.approx(ms)
        assert c.ms(w) == pytest.approx(ms)
    assert c.ms(1, 20.57) == pytest.approx(20.57)
    assert c.ms(7, 20.57) == pytest.approx((58.1 + 63.0) / 2)
    assert c.ms(10, 20.57) > c.ms(8, 20.57)


def test_cost_scales_with_the_live_plain_step_and_plain_is_a_row_count_of_one():
    c = D.RowCost()
    assert c.ms(1, 30.0) == 30.0                                            # R = 1 is the live plain step itself
    assert c.ms(4, 41.14) == pytest.approx(46.7 * 2.0)
    ms = [c.ms(r, 25.0) for r in range(1, 10)]
    assert ms == sorted(ms) and len(set(ms)) == len(ms)                    # not one slope: the increments differ
    inc = [b - a for a, b in zip(ms, ms[1:])]
    assert max(inc) - min(inc) > 1.0


def test_cost_parse():
    c = D.RowCost.parse("2:30,4:40")
    assert c.ms(2) == 30 and c.ms(3) == 35 and c.ms(4) == 40
    assert D.RowCost.parse("1:10,2:15,3:18").ms(1) == 10
    assert D.RowCost.parse("not a table").ms(2) == pytest.approx(36.8)      # never raises: the defaults


# ----------------------------------------------------------------------------------------------------------------------- choice
def test_expected_tokens():
    assert D.expected_tokens([0.5, 0.5, 0.5], 1) == 1.0
    assert D.expected_tokens([0.5, 0.5, 0.5], 3) == pytest.approx(1 + 0.5 + 0.25)
    assert D.expected_tokens([1.0] * 4, 5) == 5.0


def test_choose_rows_widens_with_confidence_and_falls_to_a_plain_step():
    c = D.RowCost()
    hi = D.choose_rows([0.97] * 5, c, 20.57)
    mid = D.choose_rows([0.8, 0.75, 0.7, 0.65, 0.6], c, 20.57)
    lo = D.choose_rows([0.4, 0.4, 0.4, 0.4, 0.4], c, 20.57)
    assert hi.rows == 6 and 1 < mid.rows <= 6 and lo.rows == 1            # prose: speculation does not pay, the plain step
    assert D.choose_rows([0.7], c, 20.57).rows == 1                          # one draft at 0.7 costs 36.8 ms for 1.7 tokens: plain is 20.57 for 1


def test_break_even_matches_the_table():
    c = D.RowCost()
    assert D.choose_rows([0.75], c, 20.57).rows == 1                         # 1.75 / 36.8 < 1 / 20.57
    assert D.choose_rows([0.85], c, 20.57).rows == 2                         # 1.85 / 36.8 > 1 / 20.57


def test_ties_within_two_percent_go_to_the_cheaper_rows():
    c = D.RowCost(table={1: 10.0, 2: 20.0, 3: 30.0})
    probs = [1.0, 0.0]
    ch = D.choose_rows(probs, c, None)                                      # R=2: 2/20 = 0.1, R=3: 2/30; plain 0.1: a three-way pick
    assert ch.rows == 1                                                      # plain 1/10 = 0.1 ties R=2 (2/20 = 0.1): the cheaper wins
    probs = [1.0, 1.0]
    c2 = D.RowCost(table={1: 10.0, 2: 20.0, 3: 29.6})
    assert D.choose_rows(probs, c2, None).rows == 1                          # R=3: 3/29.6 = 0.1014, within 2 % of plain 0.1: the cheaper one
    c3 = D.RowCost(table={1: 10.0, 2: 20.0, 3: 28.0})
    assert D.choose_rows(probs, c3, None).rows == 3                          # 0.107 is 7 % over: wider wins


def test_max_rows_caps_the_choice():
    assert D.choose_rows([0.99] * 5, D.RowCost(), 20.57, max_rows=3).rows == 3


# ----------------------------------------------------------------------------------------------------------------------- simulated accept curve
def _true_p(x):
    return min(0.97, max(0.05, 0.40 + 0.20 * x))             # the head's real accept rate: NOT the prior's curve


def _sim(policy, rounds=3000, seed=7, plain_ms=20.57, draft_ms=4.2, eager_ms=0.0):
    """Segments of 40 rounds alternate between predictable text (high confidences) and novel text (low). Returns (tokens, ms).
    eager_ms: wait the lazy draft hides behind the verify build, paid by every round that drafts in mode 2 (the width table was measured lazy)."""
    rng = random.Random(seed)
    cost = D.RowCost()
    cal = D.Calibration()
    track = D.Track()
    tok = ms = 0.0
    for r in range(rounds):
        base = 2.6 if (r // 40) % 2 == 0 else 0.7
        xs = [max(0.0, base + rng.gauss(0, 0.3) - 0.08 * i) for i in range(5)]
        conf = [1.0 - 10.0 ** (-x) for x in xs]
        if policy == "plan":
            rows = D.choose_rows(track.probs(cal, 5), cost, plain_ms, 6).rows if track.ready else 6
            drafted = rows > 1
        elif policy == "round":                                              # plan-time decision to draft, then the cut from this round's confidences
            plan = D.choose_rows(track.probs(cal, 5), cost, plain_ms, 6).rows if track.ready else 6
            drafted = plan > 1
            rows = D.choose_rows(cal.probs(conf[:5]), cost, plain_ms, 6).rows if drafted else 1
        else:
            rows, drafted = policy, policy > 1
        a = 0
        while a < rows - 1 and rng.random() < _true_p(xs[a]):
            a += 1
        tok += a + 1
        ms += cost.ms(rows, plain_ms) + (draft_ms if (drafted and rows == 1) else 0.0) + (eager_ms if (drafted and policy == "round") else 0.0)
        if drafted:
            track.update(conf)
            if rows > 1:
                cal.observe(conf, rows - 1, a)
        elif r % 16 == 15:                                                   # a probe round: D = 1 reads the head again
            track.update(conf)
    return tok, ms


def test_policy_beats_every_fixed_width_on_a_regime_switching_accept_curve():
    fixed = {w: (lambda t, m: t / m)(*_sim(w)) for w in range(1, 7)}
    best_fixed = max(fixed.values())
    for pol in ("plan", "round"):
        t, m = _sim(pol)
        assert t / m > best_fixed * 1.03, (pol, t / m, fixed)


def test_mode2_still_wins_when_it_pays_the_whole_draft_wait_on_every_drafted_round():
    """Mode 2 reads the draft before building the verify (no lazy overlap); the width table was measured lazy. Charge the ledger's 4.2 ms draft on
    every drafted round: the per-round choice then only ties the best fixed width (the synthetic numbers: 59.3 against 59.2 tok/s-equivalent, mode 1
    62.5), so mode 1 is the arm to run first. The assertion pins "not worse by more than 2 %"; it does not claim a win."""
    best_fixed = max((lambda t, m: t / m)(*_sim(w)) for w in range(1, 7))
    t, m = _sim("round", eager_ms=4.2)
    t1, m1 = _sim("plan")
    assert t / m > best_fixed * 0.98, (t / m, best_fixed)
    assert t / m < t1 / m1 * 1.2                                            # and the overlap it gives up is a real price, not noise


def test_policy_does_not_lose_on_a_novel_only_curve():
    """On text where speculation does not pay, the policy falls to the plain step instead of paying for rows."""
    def sim(policy):
        rng = random.Random(3)
        cost, cal, track = D.RowCost(), D.Calibration(), D.Track()
        tok = ms = 0.0
        for r in range(2000):
            xs = [max(0.0, 0.5 + rng.gauss(0, 0.2)) for _ in range(5)]
            conf = [1.0 - 10.0 ** (-x) for x in xs]
            rows = (D.choose_rows(track.probs(cal, 5), cost, 20.57, 6).rows if track.ready else 6) if policy == "plan" else policy
            a = 0
            while a < rows - 1 and rng.random() < 0.25:
                a += 1
            tok += a + 1
            ms += cost.ms(rows, 20.57)
            if rows > 1:
                track.update(conf)
                cal.observe(conf, rows - 1, a)
            elif r % 16 == 15:
                track.update(conf)
        return tok / ms
    assert sim("plan") >= sim(1) * 0.97
    assert sim("plan") > max(sim(w) for w in range(2, 7)) * 1.2


# ----------------------------------------------------------------------------------------------------------------------- trace analysis
def _trace(n, pred_fn, real_fn, seed=2):
    rng = random.Random(seed)
    recs = []
    for _ in range(n):
        pk = [pred_fn(i) for i in range(4)]
        a = 0
        while a < 4 and rng.random() < real_fn(a):
            a += 1
        recs.append({"k": "r", "pk": pk, "acc": a, "tok": a + 1, "et": D.expected_tokens(pk, 5), "lk": 0})
    return recs


def test_calibration_report_sees_a_calibrated_and_a_miscalibrated_policy():
    good = D.calibration_report(_trace(4000, lambda i: 0.8, lambda i: 0.8))
    bad = D.calibration_report(_trace(4000, lambda i: 0.9, lambda i: 0.5))
    assert good["ece"] < 0.03 and abs(good["et"]["pred"] - good["et"]["real"]) < 0.08
    assert bad["ece"] > 0.3 and bad["et"]["pred"] > bad["et"]["real"] + 0.5
    assert good["pos"][0]["n"] > good["pos"][3]["n"]                        # deeper positions are reached less often


def test_calibration_report_skips_plain_and_lookup_rounds():
    rep = D.calibration_report([{"k": "p", "pk": [0.9], "acc": 0, "tok": 1}, {"k": "r", "pk": [0.9], "acc": 1, "tok": 2, "lk": 1}, {"k": "r", "acc": 1, "tok": 2}])
    assert rep["et"]["rounds"] == 0 and rep["ece"] == 0.0


def test_fit_prior_recovers_a_logistic_head():
    rng = random.Random(4)
    samples = []
    for _ in range(4000):
        x = rng.uniform(0.0, 3.0)
        p = 1 / (1 + math.exp(-(-0.8 + 1.5 * x)))
        samples.append((1.0 - 10.0 ** (-x), rng.random() < p))
    a, b = D.fit_prior(samples)
    assert abs(a + 0.8) < 0.25 and abs(b - 1.5) < 0.2
    assert D.fit_prior(samples[:5]) == (D.PRIOR_A, D.PRIOR_B)               # too few: the default


def test_samples_from_trace_use_the_reached_verified_positions():
    recs = [{"k": "r", "cf": [0.9, 0.8, 0.7, 0.6, 0.5], "draft": 4, "cut": 0, "acc": 2}]
    assert D.samples_from_trace(recs) == [(0.9, True), (0.8, True), (0.7, False)]


def test_online_refit_moves_unvisited_bins_toward_what_the_visited_ones_learned():
    rng = random.Random(5)
    cal = D.Calibration(prior=(-0.2, 1.9))
    cold = cal.prob(0.999)
    for _ in range(400):                                                   # a head whose confidence says little: right 45 % of the time at every conf
        c = rng.choice([0.3, 0.5, 0.7, 0.9])
        a = 0
        while a < 2 and rng.random() < 0.45:
            a += 1
        cal.observe([c, c], 2, a)
    assert cal.refits > 0
    assert cal.prob(0.999) < cold - 0.1, (cal.prob(0.999), cold)           # bin 0.999 was never visited: it follows the refit curve, not the cold start
    assert cal.b < 1.9 and cal.b >= 0.1


def test_refit_needs_data_and_two_classes():
    cal = D.Calibration()
    assert cal.refit() is False
    for _ in range(100):
        cal.observe([0.9], 1, 1)                                           # one class only
    assert cal.refit() is False and (cal.a, cal.b) == (D.PRIOR_A, D.PRIOR_B)


def test_flag_default_is_off(monkeypatch):
    monkeypatch.delenv(D.ENV, raising=False)
    assert D.enabled() == 0
    monkeypatch.setenv(D.ENV, "2")
    assert D.enabled() == 2
    monkeypatch.setenv(D.ENV, "junk")
    assert D.enabled() == 0
    from kyojin_mlx import flags
    assert flags.FLAGS["dspol_v2"].default == "0" and flags.FLAGS["dspol_v2"].env == D.ENV
