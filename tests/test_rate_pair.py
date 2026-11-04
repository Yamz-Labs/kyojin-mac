"""The paired rate rule (flag rate_pair): both sides of the comparison are measured on the same rounds, tokens over ms."""
import os
from types import SimpleNamespace

import pytest

from kyojin_mlx import flags
from kyojin_mlx.serve.engine import Engine, PAIR_MIN_D, PAIR_MIN_P
from kyojin_mlx.serve.sampling import Sampling
from test_serve_spec import FakeSpec


@pytest.fixture(autouse=True)
def _off(monkeypatch):
    monkeypatch.setenv("KYOJIN_RATE_PAIR", "0")
    yield
    os.environ.pop("KYOJIN_RATE_PAIR", None)


def _eng(monkeypatch, **kw):
    monkeypatch.setenv("KYOJIN_RATE_PAIR", "1")
    eng = Engine(FakeSpec(err=0), sessions=4, mtp_depth=3, ctx=4096, chunk=4, block=8, **kw)
    eng.members = [object()]                                     # the solo path
    s = SimpleNamespace(sampling=Sampling(), max_new=500, n_gen=0, n_prompt=10, acc_succ=0.0, acc_trial=0.0,
                        idle_rounds=0, rate_gap=0, probing=False, cap_n=10, cap_tok=0.0)
    eng.plain_ms = 20.0
    return eng, s


def test_flag_is_in_the_table_and_off():
    assert flags.FLAGS["rate_pair"].env == "KYOJIN_RATE_PAIR"
    assert flags.FLAGS["rate_pair"].default == "0"
    assert flags.FLAGS["rate_pair"].runtime


def test_pair_rate_sums_the_same_rounds():
    f = Engine._pair_rate
    assert f([], PAIR_MIN_D) is None
    assert f([(3, 30.0)], PAIR_MIN_D) is None                     # too few rounds: no decision
    assert f([(3, 30.0), (0, 50.0), (6, 70.0)], PAIR_MIN_D) == pytest.approx(9 / 150.0)
    assert f([(3, 0.0)], 1) is None                               # nothing measured


def test_a_faster_engine_does_not_plan_less(monkeypatch):
    eng, s = _eng(monkeypatch)
    eng._rp_win = [(1, 20.0)] * PAIR_MIN_P                        # plain 0.05 tok/ms
    eng._rd_win = [(2, 40.0)] * PAIR_MIN_D                        # 0.05 tok/ms: a tie still drafts (margin 0.9)
    assert eng._plan_depth(s) == 3
    eng._rd_win = [(3, 40.0)] * PAIR_MIN_D                        # the same plan 25 % faster per round: drafts
    assert eng._plan_depth(s) == 3
    eng._rd_win = [(1, 40.0)] * PAIR_MIN_D                        # 0.025 tok/ms: below the plain rate
    s.idle_rounds, s.rate_gap = 0, 0
    assert eng._plan_depth(s) == 0


def test_cap_tok_bias_does_not_lift_the_rate(monkeypatch):
    """spec_rate multiplies cap_tok (full-proposal rounds only) by plain_ms; a session with a huge cap_tok must still go plain when the
    paired window says the rounds it served were slower than plain."""
    eng, s = _eng(monkeypatch)
    s.cap_tok = 99.0
    eng._rp_win = [(1, 20.0)] * PAIR_MIN_P
    eng._rd_win = [(1, 60.0)] * PAIR_MIN_D
    assert eng._plan_depth(s) == 0


def test_probe_does_not_re_decide_from_one_round(monkeypatch):
    eng, s = _eng(monkeypatch)
    eng._rp_win = [(1, 20.0)] * PAIR_MIN_P
    eng._rd_win = [(1, 60.0)] * PAIR_MIN_D                        # plain is right
    s.idle_rounds, s.rate_gap = 16, 16
    assert eng._plan_depth(s) == 3                                # the probe round
    assert s.idle_rounds == 0 and s.rate_gap == 32
    # the probe round was fast and carried many tokens: it joins the window, the window (still >= PAIR_MIN_D rounds) decides
    eng._rd_win = (eng._rd_win + [(8, 40.0)])[-12:]
    assert eng._plan_depth(s) == 3                                # 0.137 tok/ms > 0.9 x 0.05: the faster engine wins the window
    eng._rd_win = [(1, 60.0)] * PAIR_MIN_D
    s.idle_rounds = 0
    assert eng._plan_depth(s) == 0


def test_pair_rule_needs_a_measured_plain_side(monkeypatch):
    eng, s = _eng(monkeypatch)
    eng.plain_ms = None
    eng._rd_win = [(3, 30.0)] * PAIR_MIN_D
    assert eng._plan_depth(s) == 0                                # two plain rounds measure the plain side first
    assert eng._plan_depth(s) == 0
    assert eng._plan_depth(s) == 3                                # then it drafts (both sides unmeasured: the cap)
