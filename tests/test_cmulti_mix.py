"""cloud/cmulti-2, decode during prefill: flags mix_decode (time share between chunks) and mix_slice_ms (decode rounds inside a chunk, at the layer bounds).
Fake backend with a simulated wall cost per prefill row / decode step: the numbers are of the scheduler on that model, not of Metal."""
import os
import time

import pytest

from kyojin_mlx.serve.engine import Engine
from cmulti_fakes import GridFake
from test_serve_engine import collect

PF_MS, STEP_MS, CHUNK, LAYERS = 0.1, 1.5, 200, 10      # a chunk = 20 ms of prefill in 10 layer groups, a decode round = 1.5 ms


def scenario(monkeypatch, mix=0, slice_ms=0, long_len=2000, new_a=4000):
    monkeypatch.setenv("KYOJIN_PF_FLOOR", "0")    # the mix and slice levers alone: pf_floor replaces them
    monkeypatch.setenv("KYOJIN_MIX_DECODE", str(mix))
    monkeypatch.setenv("KYOJIN_MIX_SLICE_MS", str(slice_ms))
    be = GridFake(pf_ms=PF_MS, step_ms=STEP_MS, layers=LAYERS)
    eng = Engine(be, sessions=2, ctx=1 << 16, chunk=CHUNK, block=CHUNK * 2, prefix_bytes=0)
    eng.start()
    a_ids = [(3 * i + 1) % 90 for i in range(20)]
    b_ids = [(7 * i + 5) % 90 for i in range(long_len)]
    ha = eng.submit(a_ids, max_new=new_a)
    n_before = 0
    while n_before < 20:                                    # A is decoding
        assert ha.events.get(timeout=30)[0] == "tok"
        n_before += 1
    eng.stats["tok_gap_pf_max_s"] = 0.0
    hb = eng.submit(b_ids, max_new=3)
    during = 0
    b_toks = []
    # count A's tokens until B's first token is out
    while not b_toks:
        while not ha.events.empty():
            ev = ha.events.get_nowait()
            during += ev[0] == "tok"
        if not hb.events.empty():
            ev = hb.events.get_nowait()
            if ev[0] == "tok":
                b_toks.append(ev[1])
        time.sleep(0.0005)
    st = dict(eng.stats)
    ha.cancel()
    rest_b = collect(hb)[0]
    eng.stop()
    return dict(a_during=during, gap=st.get("tok_gap_pf_max_s", 0.0) * 1000, b=b_toks + rest_b, slice_rounds=st.get("mix_slice_rounds", 0),
                pf_chunks=st.get("n_pf_chunk", 0), a_ids=a_ids, b_ids=b_ids)


def solo_text(ids, max_new):
    eng = Engine(GridFake(), sessions=1, ctx=1 << 16, chunk=CHUNK, block=CHUNK * 2, prefix_bytes=0)
    eng.start()
    try:
        return collect(eng.submit(ids, max_new=max_new))[0]
    finally:
        eng.stop()


def test_today_one_round_per_chunk(monkeypatch):
    r = scenario(monkeypatch)
    n_chunks = 2000 // CHUNK
    print("off:", {k: v for k, v in r.items() if k not in ("a_ids", "b_ids", "b")})
    assert r["a_during"] <= n_chunks + 3                    # decode_per_chunk = 1: about one A token per prefill chunk


def test_mix_decode_gives_the_decoder_its_share(monkeypatch):
    off = scenario(monkeypatch)
    on = scenario(monkeypatch, mix=50)
    print("mix_decode=50:", {k: v for k, v in on.items() if k not in ("a_ids", "b_ids", "b")})
    assert on["a_during"] > 4 * off["a_during"]
    # cost on the prefill rate by construction: B's prefill takes about (1 + 50/50) = 2x the wall. Check through A's decode time share instead of a clock:
    # 50 % of the wall is decode, decode round = 1.5 ms, so about (20 ms * 0.5 / 0.5) / 1.5 = 13 rounds between two chunks
    assert 8 <= on["a_during"] / (2000 // CHUNK) <= 16


def test_mix_slice_bounds_the_gap_inside_a_chunk(monkeypatch):
    off = scenario(monkeypatch)
    on = scenario(monkeypatch, slice_ms=4)
    print("mix_slice_ms=4:", {k: v for k, v in on.items() if k not in ("a_ids", "b_ids", "b")})
    assert off["gap"] >= 18                                 # today: the gap is a whole chunk (20 ms)
    assert on["gap"] < 0.8 * off["gap"]
    assert on["slice_rounds"] > 0


@pytest.mark.parametrize("mix,slice_ms", [(50, 0), (0, 4), (60, 4)])
def test_texts_equal_solo_with_the_levers_on(monkeypatch, mix, slice_ms):
    r = scenario(monkeypatch, mix=mix, slice_ms=slice_ms, long_len=800)
    assert r["b"] == solo_text(r["b_ids"], 3)[:len(r["b"])] and len(r["b"]) == 3


def test_a_decoder_text_is_the_solo_text(monkeypatch):
    monkeypatch.setenv("KYOJIN_MIX_DECODE", "50"); monkeypatch.setenv("KYOJIN_MIX_SLICE_MS", "3")
    be = GridFake(pf_ms=PF_MS, step_ms=STEP_MS, layers=LAYERS)
    eng = Engine(be, sessions=2, ctx=1 << 16, chunk=CHUNK, block=CHUNK * 2, prefix_bytes=0)
    eng.start()
    a_ids = [(3 * i + 1) % 90 for i in range(20)]
    b_ids = [(7 * i + 5) % 90 for i in range(1000)]
    ha = eng.submit(a_ids, max_new=120)
    time.sleep(0.05)
    hb = eng.submit(b_ids, max_new=10)
    ta, tb = collect(ha)[0], collect(hb)[0]
    eng.stop()
    assert ta == solo_text(a_ids, 120) and tb == solo_text(b_ids, 10)


def test_single_session_unchanged_with_the_levers_on(monkeypatch):
    p = list(range(1, 900))
    traces = []
    for mix, sl in ((0, 0), (70, 2)):
        monkeypatch.setenv("KYOJIN_MIX_DECODE", str(mix)); monkeypatch.setenv("KYOJIN_MIX_SLICE_MS", str(sl))
        be = GridFake(pf_ms=0.01, step_ms=0.2, layers=LAYERS)
        eng = Engine(be, sessions=1, ctx=1 << 16, chunk=CHUNK, block=CHUNK * 2, prefix_bytes=0)
        eng.start()
        txt = collect(eng.submit(p, max_new=30))[0]
        eng.stop()
        traces.append((be.calls, txt, eng.stats.get("mix_slice_rounds", 0)))
    assert traces[0][0] == traces[1][0] and traces[0][1] == traces[1][1] and traces[1][2] == 0
