"""cmudec: (1) SIMULATION of off / pf_srpt / pf_floor on the measured M5 Ultra numbers (cmudec_sim.py, no GPU, not a timing); (2) the engine flag on a fake backend."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import cmudec_sim as S
from kyojin_mlx.serve.engine import Engine
from cmulti_fakes import GridFake
from test_serve_engine import collect
from test_mu2_sched import long_prompt, solo_text, CHUNK


def _rows():
    T, o = S.fit()
    return {p.name: S.simulate(p, T, o) for p in S.POLICIES}


def test_sim_reproduces_the_measured_arms():
    r = _rows()
    off, srpt = r["off"], r["pf_srpt"]
    assert 52.5 <= min(off["ttft"]) and max(off["ttft"]) <= 55.5 and abs(off["wall"] - 61.4) < 0.2 and abs(off["agg_prefill"] - 2424) < 40   # measured: 52.7-55.1, 61.4, 2424
    assert abs(srpt["wall"] - 69.7) < 0.2 and abs(srpt["mean_ttft"] - 39.4) < 2.0 and 2.4 <= srpt["floor_min"] <= 5.7                          # measured: 69.7, 39.4, 2.4-5.7


def test_sim_floor_policy_vs_off_and_srpt():
    r = _rows()
    print("\nSIMULATION (no GPU, fitted to the measured off and pf_srpt arms):")
    for k, v in r.items():
        print("  %-18s mean TTFT %5.1f s  wall %5.1f s  agg prefill %4.0f  min decode rate while others prefill %4.1f tok/s" % (k, v["mean_ttft"], v["wall"], v["agg_prefill"], v["floor_min"]))
    off, srpt, fl = r["off"], r["pf_srpt"], r["pf_floor=15"]
    assert fl["floor_min"] >= 15.0                       # the target
    assert fl["mean_ttft"] < 0.75 * off["mean_ttft"]     # keeps the TTFT gain of srpt
    assert fl["mean_ttft"] <= srpt["mean_ttft"] + 0.5
    assert fl["wall"] < srpt["wall"]                     # recovers part of the wall loss ...
    assert fl["wall"] > off["wall"]                      # ... not all of it: decoding unbatched is dearer per token (model: 18.9 ms at 1 row vs 12 at 4)


def _run(monkeypatch, env):
    for k in ("KYOJIN_PF_FLOOR", "KYOJIN_PF_SRPT", "KYOJIN_MIX_SLICE_MS", "KYOJIN_MIX_DECODE"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    eng = Engine(GridFake(pf_ms=0.1, step_ms=1.0, layers=10), sessions=2, ctx=1 << 16, chunk=CHUNK, block=CHUNK * 2, prefix_bytes=0)
    a, b = long_prompt(2000), long_prompt(5000)[::-1]
    ha, hb = eng.submit(a, max_new=40), eng.submit(b, max_new=40)
    eng.start()
    try:
        ta = collect(ha)[0]
        tb = collect(hb)[0]
    finally:
        eng.stop()
    return ta, tb, eng.stats, a, b


def test_engine_floor_keeps_texts_and_default_is_off(monkeypatch):
    ta0, tb0, st0, a, b = _run(monkeypatch, {"KYOJIN_PF_FLOOR": "0"})
    assert st0["floor_bursts"] == 0 and ta0 == solo_text(a, 40) and tb0 == solo_text(b, 40)
    ta1, tb1, st1, _, _ = _run(monkeypatch, {"KYOJIN_PF_FLOOR": "300", "KYOJIN_PF_FLOOR_PERIOD_MS": "1"})
    assert ta1 == ta0 and tb1 == tb0
    assert st1["floor_bursts"] > 0
    print("\nfloor engine stats:", {k: st1[k] for k in ("floor_bursts", "floor_burst_rounds", "mix_slice_rounds", "decode_ticks")})


def test_tool_rate_window_is_capped_by_max_new():
    """The tool's window (first token -> last user's first token) cannot show more than max_new / window: 144 tokens over ~60 s is 2.4 tok/s
    even for a user served at 15 tok/s while it decoded. decode_rates(active=True) ends the window at the user's own last token."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    import cmudec_mac as T
    ts = [10.0 + i / 15.0 for i in range(144)]
    res = [(ts, "", 0.0), ([60.0, 60.5, 61.0], "", 0.0)]
    old, new = T.decode_rates(res, 60.0), T.decode_rates(res, 60.0, active=True)
    assert len(old) == 1 and old[0] < 3.0
    assert len(new) == 1 and abs(new[0] - 15.0) < 0.2
    r2 = [([10.0 + i / 2.0 for i in range(100)], "", 0.0), ([60.0], "", 0.0)]
    assert abs(T.decode_rates(r2, 60.0)[0] - T.decode_rates(r2, 60.0, active=True)[0]) < 0.1
