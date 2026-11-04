"""Adaptive confidence gate (flag gate_adapt): the candidate thresholds are priced from the round that just ran, so a failing round narrows the
next one and a run whose rows all land is not narrowed."""
import os

import pytest

from kyojin_mlx.serve.engine import Engine
from test_serve_spec import FakeSpec


def _eng():
    os.environ.setdefault("KYOJIN_GATE_ADAPT", "1")
    return Engine(FakeSpec(err=0), sessions=1, ctx=4096, chunk=4, block=8, mtp_depth=5)


class S:
    gate_thr = 0.5
    gate_rate: list = []

    def __init__(self):
        self.gate_rate = []


def test_failing_round_narrows_the_gate():
    e = _eng()
    s = S()
    for _ in range(6):
        e._gate_learn(s, [0.92, 0.88, 0.66, 0.58, 0.52], acc=1, paid_rows=6, plan=5, ms=60.0)
    assert s.gate_thr > 0.5, s.gate_thr                  # the rows past the first accepted one were a loss: propose fewer
    assert e.stats["spec_gate_adapt_rounds"] == 6


def test_confident_round_keeps_the_width():
    e = _eng()
    s = S()
    for _ in range(6):
        e._gate_learn(s, [0.95, 0.93, 0.91, 0.90, 0.89], acc=5, paid_rows=6, plan=5, ms=60.0)
    assert s.gate_thr <= 0.5, s.gate_thr


def test_plan_caps_the_candidates():
    e = _eng()
    s = S()
    for _ in range(6):
        e._gate_learn(s, [0.9, 0.9, 0.9, 0.9, 0.9], acc=1, paid_rows=2, plan=1, ms=38.0)
    assert s.gate_thr <= 0.5                             # one row was the whole plan: nothing to narrow
