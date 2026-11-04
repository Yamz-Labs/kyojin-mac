"""flag pf_floor_catchup (default 0): a decoder behind pf_floor over its own window gets its deficit repaid by the next burst, within a cap. CPU, fake backend."""
import time
from types import SimpleNamespace

import pytest

from kyojin_mlx import flags as FL
from kyojin_mlx.serve.engine import Engine
from cmulti_fakes import GridFake


def mk(monkeypatch, catchup, n_members=1):
    monkeypatch.setenv("KYOJIN_PF_FLOOR", "15")
    monkeypatch.setenv("KYOJIN_PF_FLOOR_MAX", "60")
    monkeypatch.setenv("KYOJIN_PF_FLOOR_PERIOD_MS", "1000")
    monkeypatch.setenv("KYOJIN_PF_FLOOR_CATCHUP", str(catchup))
    eng = Engine(GridFake(), sessions=4, ctx=1 << 16, chunk=200, block=400, prefix_bytes=0)
    eng.members = [SimpleNamespace(n_gen=0, fl_t0=0.0, fl_n0=0) for _ in range(n_members)]
    eng._rd_s, eng._tpr, eng._floor_k = 0.05, 1.0, 1.0           # duty 15 * 0.05 = 0.75 -> capped at 0.6
    calls = []
    eng._decode_phase = lambda: (calls.append(1), time.sleep(0.001))
    return eng, calls


def test_flag_default():
    f = FL.FLAGS["pf_floor_catchup"]
    assert f.default == "30" and f.env == "KYOJIN_PF_FLOOR_CATCHUP"


def test_deficit_is_per_session_window(monkeypatch):
    eng, _ = mk(monkeypatch, 80, 2)
    a, b = eng.members
    now = time.perf_counter()
    assert eng._floor_deficit(now) == 0.0 and a.fl_t0 and b.fl_t0       # first look opens the windows
    a.fl_t0 = b.fl_t0 = now - 2.0
    a.n_gen = 30                                                         # on the floor (15 * 2 s)
    b.n_gen = 10                                                         # 20 tokens behind
    assert eng._floor_deficit(now) == pytest.approx(20 * 0.05, rel=0.05)
    b.n_gen = 40
    assert eng._floor_deficit(now) == 0.0


def run_yield(eng, elapsed):
    eng._chunk_t0 = time.perf_counter() - elapsed
    eng._inline_s = 0.0
    eng._mix_debt = 0.0
    eng._floor_yield()


def test_off_waits_for_the_period_on_bursts_for_the_deficit(monkeypatch):
    eng, calls = mk(monkeypatch, 0)
    eng.members[0].fl_t0, eng.members[0].fl_n0 = time.perf_counter() - 1.0, 0
    run_yield(eng, 0.2)                                                  # 0.2 s of prefill: owed 0.3 s < period * r = 1.5 s -> no burst
    assert not calls and eng.stats["floor_bursts"] == 0
    eng, calls = mk(monkeypatch, 80)
    eng.members[0].fl_t0, eng.members[0].fl_n0 = time.perf_counter() - 1.0, 0
    run_yield(eng, 0.2)                                                  # deficit 15 tok = 0.75 s of rounds, cap 80 % of 0.2 s = 0.8 s: still under the period
    assert not calls
    eng.members[0].fl_t0 = time.perf_counter() - 3.0
    run_yield(eng, 1.5)                                                  # deficit 45 tok = 2.25 s, base owed 2.25 s, both over the period
    assert eng.stats["floor_bursts"] == 1 and calls


def test_catchup_cap_bounds_the_burst(monkeypatch):
    eng, calls = mk(monkeypatch, 50)
    eng.members[0].fl_t0, eng.members[0].fl_n0 = time.perf_counter() - 30.0, 0   # a huge deficit
    run_yield(eng, 0.4)
    assert not calls                                                     # 50 % of 0.4 s = 0.4 s of rounds < period * r: no burst on a deficit alone
