"""cloud/cmulti-2, N long sessions under a KV budget: flag kv_admit. Fake backend with 8 bytes per cached token (nbytes), so budgets are exact."""
import time

import pytest

from kyojin_mlx.serve.engine import Engine
from cmulti_fakes import GridFake
from test_serve_engine import collect

CH = 16


def solo(ids, max_new):
    eng = Engine(GridFake(), sessions=1, ctx=1 << 15, chunk=CH, block=1 << 14, prefix_bytes=0)
    eng.start()
    try:
        return collect(eng.submit(ids, max_new=max_new))[0]
    finally:
        eng.stop()


def mk(n, plen=1000):
    return [[(13 * k + 3 * i + 1) % 90 for i in range(plen)] for k in range(n)]


def run(monkeypatch, admit, ps, budget, max_new=200, sessions=6, hook=None):
    monkeypatch.setenv("KYOJIN_KV_ADMIT", "1" if admit else "0")
    be = GridFake()
    eng = Engine(be, sessions=sessions, ctx=1 << 15, chunk=CH, block=1 << 14, prefix_bytes=0, kv_budget=budget, kv_refresh=1)
    hs = [eng.submit(p, max_new=max_new) for p in ps]
    if hook:
        hook(be, eng)
    eng.start()
    outs = [collect(h, timeout=60) for h in hs]
    st = eng.status()
    eng.stop()
    width = max([c[1] for c in be.calls if c[0] == "step"], default=0)
    return outs, st, width


def test_reservation_queues_instead_of_running_out(monkeypatch):
    ps = mk(6)
    need = 8 * (1000 + 200)                                  # 9600 bytes per session at its reply limit
    budget = 2 * need + 1000                                 # two fit, a third would not
    on, st_on, w_on = run(monkeypatch, True, ps, budget)
    off, st_off, w_off = run(monkeypatch, False, ps, budget)
    print("decode width on/off:", w_on, w_off, "peak kv on/off:", st_on["kv_peak_bytes"], st_off["kv_peak_bytes"], "budget", budget)
    assert w_on == 2 and w_off > 2                           # off: the budget is not a bound, every session is admitted
    assert st_on["kv_peak_bytes"] <= budget < st_off["kv_peak_bytes"]
    assert [o[0] for o in on] == [solo(p, 200) for p in ps]  # queued sessions give the text of their solo runs
    assert st_on.get("kv_over_budget", 0) == 0 and st_on.get("kv_drops", 0) == 0


def test_cold_start_holds_the_second_session_until_bytes_per_token_is_known(monkeypatch):
    ps = mk(3, plen=400)
    on, st, _ = run(monkeypatch, True, ps, 8 * 3 * 700)
    assert all(o[1] == "length" for o in on)
    assert st["kv_peak_bytes"] <= 8 * 3 * 700


def test_a_request_that_cannot_fit_alone_is_refused_at_submit(monkeypatch):
    monkeypatch.setenv("KYOJIN_KV_ADMIT", "1")
    eng = Engine(GridFake(), sessions=2, ctx=1 << 15, chunk=CH, block=1 << 14, prefix_bytes=0, kv_budget=8 * 1500)
    eng.start()
    try:
        assert collect(eng.submit(mk(1, 300)[0], max_new=20))[1] == "length"           # measures bytes per token
        with pytest.raises(ValueError, match="KV budget"):
            eng.submit(mk(1, 2000)[0], max_new=20)
    finally:
        eng.stop()


def test_a_request_unknown_at_submit_is_refused_when_admitted(monkeypatch):
    ps = mk(2, plen=300) + mk(1, plen=2000)
    on, st, _ = run(monkeypatch, True, ps, 8 * 1500, max_new=20)
    assert on[0][1] == "length" and on[1][1] == "length" and on[2][1] == "refused" and st["kv_refused"] == 1
    assert on[0][0] == solo(ps[0], 20)


def test_estimate_wrong_parks_the_youngest_and_rebuilds_it_exactly(monkeypatch):
    ps = mk(3, plen=200)

    def hook(be, eng):
        orig = be.step
        state = {"n": 0}

        def step(bc, tokens):
            state["n"] += 1
            if state["n"] == 40:
                be.bytes_per_token = 16                       # the cache turns out 2x dearer per token than measured
            return orig(bc, tokens)
        be.step = step
    on, st, _ = run(monkeypatch, True, ps, 8 * 3 * 400, max_new=150, hook=hook)
    print({k: st.get(k) for k in ("kv_parks", "kv_drops", "rebuilds", "replay_tokens", "kv_over_budget")})
    assert st["kv_parks"] >= 1 and st["kv_drops"] >= 1 and st["rebuilds"] >= 1
    assert [o[0] for o in on] == [solo(p, 150) for p in ps]       # the dropped session's reply is its solo reply: prompt re-prefilled on the grid, tokens replayed


def test_flag_off_and_single_session_unchanged(monkeypatch):
    p = mk(1, 500)[0]
    traces = []
    for admit in (False, True):
        monkeypatch.setenv("KYOJIN_KV_ADMIT", "1" if admit else "0")
        be = GridFake()
        eng = Engine(be, sessions=1, ctx=1 << 15, chunk=CH, block=1 << 14, prefix_bytes=0, kv_budget=1 << 20)
        eng.start()
        txt = collect(eng.submit(p, max_new=30))[0]
        eng.stop()
        traces.append((be.calls, txt))
    assert traces[0] == traces[1]
