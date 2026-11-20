"""GLM-5.3 spec cost model and planner (kyojin_mlx.glm_cost): the measured tables reproduce the served rates, the planner drafts depth 1 at one session
and refuses rows that cannot pay, and the n-gram extension only grows on a strong agreement."""
import pytest

from kyojin_mlx import glm_cost as GC

C = GC.Costs()


def rate1(q, d):
    ps = [q] * 4
    return 1000 * GC.tokens(ps, d) / C.round_ms([d])


def test_model_reproduces_served_single_session_rates():
    # served (glmnextn_ab, depth 1 vs plain): code 43.2, fr 41.0, iso 39.3, math 44.5 at acceptance 0.91 / 0.81 / 0.75 / 0.95
    for q, served in ((0.91, 43.2), (0.81, 41.0), (0.75, 39.3), (0.95, 44.5)):
        assert rate1(q, 1) == pytest.approx(served, rel=0.06)
    assert 1000 / C.round_ms([0]) == pytest.approx(32.2, rel=0.02)                  # plain


def test_depth_one_is_the_plan_at_one_session_and_two_rows_do_not_pay_below_095():
    for q in (0.75, 0.81, 0.91):
        assert GC.plan_depths([[q, q * 0.9, q * 0.8]], 3, C) == [1]
    assert GC.plan_depths([[0.97, 0.95, 0.93]], 3, C)[0] >= 2                       # only near-certain text goes deeper
    assert GC.plan_depths([[0.30, 0.3, 0.3]], 3, C) == [0]                          # a head that does not land: plain


def test_planner_stops_drafting_as_sessions_share_the_pass():
    ps = [0.9, 0.8, 0.7]
    d1 = GC.plan_depths([ps], 3, C)
    d4 = GC.plan_depths([ps] * 4, 3, C)
    assert d1 == [1]
    assert sum(d4) <= 4 and len(d4) == 4
    assert GC.plan_depths([ps] * 8, 3, C) == [0] * 8                                # many sessions: the plain batch step wins


def test_row_cap_is_respected_and_best_is_never_worse_than_plain():
    ds = GC.plan_depths([[0.95, 0.9, 0.9]] * 2, 3, C, max_rows=3)
    assert 2 + sum(ds) <= 3
    ps = [[0.6, 0.5, 0.5]] * 3
    ds = GC.plan_depths(ps, 3, C)
    r = sum(GC.tokens(p, d) for p, d in zip(ps, ds)) / C.round_ms(ds)
    assert r >= 3 / C.round_ms([0, 0, 0]) - 1e-12


def test_acc_prior_and_update():
    assert GC.Acc.p([], [], 0) == GC.PRIOR[0]
    assert GC.Acc.p([9.0], [10.0], 0) == pytest.approx((9 + 0.85 * 2) / 12)


def test_chain_extension_needs_agreement_strength():
    assert GC.chain_extra(0.9, 0.95, [1], 0, C, 7) >= 3                              # a strong agreement keeps adding free rows (each row ~12 ms, 0.9 x 0.95^j accepted)
    assert GC.chain_extra(0.9, 0.45, [1], 0, C, 7) == 0                              # an index that is right under half the time does not pay a row
    assert GC.chain_extra(0.9, 0.95, [1], 0, C, 7, max_rows=3) <= 1
