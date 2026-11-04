"""perf/mu2, several sessions on one lane: flag pf_fair (prefill turns by chunk, so a short request is not queued behind a whole long prompt),
the slice on a chunk that skips its decoder half (ced_skip_dec), and flag ttft_stages (the time to first token split by stage).
Fake backend with a simulated wall cost per prefill row / decode step: the numbers are of the scheduler on that model, not of Metal."""
import os
import time
from types import SimpleNamespace

import pytest

from kyojin_mlx.serve import engine as engine_mod
from kyojin_mlx.serve.engine import Engine
from cmulti_fakes import GridFake
from test_serve_engine import collect

PF_MS, STEP_MS, CHUNK, LAYERS = 0.1, 1.0, 200, 10      # a chunk = 20 ms of prefill in 10 layer groups, a decode step = 1 ms


class SkipFake(GridFake):
    """A backend whose chunks skip their decoder half when the next chunk is long (`ced_skip_dec`), as DS4.1 CED does at depth."""

    def ced_can_skip(self, cache, next_rows: int) -> int:
        return 2 if next_rows >= CHUNK else 0


def long_prompt(n):
    return [(7 * i + 5) % 90 for i in range(n)]


def run(monkeypatch, fair, long_len=4000, short_len=60, sessions=2, backend=None, max_new=3):
    """Long request first, short one right behind it: returns (short ttft ms, short text, long text, engine, backend)."""
    monkeypatch.setenv("KYOJIN_PF_FAIR", "1" if fair else "0")
    be = backend if backend is not None else GridFake(pf_ms=PF_MS, step_ms=STEP_MS, layers=LAYERS)
    eng = Engine(be, sessions=sessions, ctx=1 << 16, chunk=CHUNK, block=CHUNK * 2, prefix_bytes=0)
    a, b = long_prompt(long_len), long_prompt(short_len)[::-1]
    ha = eng.submit(a, max_new=max_new)
    hb = eng.submit(b, max_new=max_new)
    eng.start()
    try:
        tb, _, stb = collect(hb)
        ta = collect(ha)[0]
    finally:
        eng.stop()
    return stb["ttft_s"] * 1e3, tb, ta, eng, be, a, b


def solo_text(ids, max_new, backend=None):
    eng = Engine(backend if backend is not None else GridFake(), sessions=1, ctx=1 << 16, chunk=CHUNK, block=CHUNK * 2, prefix_bytes=0)
    eng.start()
    try:
        return collect(eng.submit(ids, max_new=max_new))[0]
    finally:
        eng.stop()


def test_fifo_makes_the_short_request_wait_for_the_whole_long_prompt(monkeypatch):
    """The measurement the flag is for: today's queue is strict FIFO, so the short request's first token is the end of the long prefill."""
    ttft, _, _, eng, _, a, b = run(monkeypatch, False)
    n_chunks = (len(a) + CHUNK - 1) // CHUNK
    print("\nFIFO: short ttft %.0f ms for %d chunks of long" % (ttft, n_chunks))
    assert ttft > 0.8 * n_chunks * PF_MS * CHUNK


def test_pf_fair_serves_the_short_request_in_about_one_chunk(monkeypatch):
    fifo, _, _, _, _, a, b = run(monkeypatch, False)
    fair, tb, ta, eng, _, _, _ = run(monkeypatch, True)
    print("pf_fair: short ttft %.0f ms against %.0f ms FIFO" % (fair, fifo))
    assert fair < 0.3 * fifo                                   # turns by chunk: one chunk of the long prompt, then the short one
    assert tb == solo_text(b, 3) and ta == solo_text(a, 3)      # same texts as the solo runs: the order is all that moved


def test_pf_fair_leaves_one_session_alone(monkeypatch):
    p = list(range(1, 900))
    seen = []
    for fair in ("0", "1"):
        monkeypatch.setenv("KYOJIN_PF_FAIR", fair)
        be = GridFake(pf_ms=0.01, step_ms=0.2, layers=LAYERS)
        eng = Engine(be, sessions=1, ctx=1 << 16, chunk=CHUNK, block=CHUNK * 2, prefix_bytes=0)
        eng.start()
        txt = collect(eng.submit(p, max_new=30))[0]
        eng.stop()
        seen.append((be.calls, txt))
    assert seen[0] == seen[1]                                   # one session: no queue, identical call trace and text


def test_pf_fair_keeps_the_shared_prefix_of_a_child(monkeypatch):
    """Flag pf_share + pf_fair: a child whose seed is still ahead of the leader is held in the queue, so the shared prefix is still handed over
    instead of being prefilled again (without the guard the child starts from 0 and the share hits are all lost)."""
    pre = [(5 * i + 2) % 90 for i in range(77)]
    kids = [pre + [(31 * k + 7 * j + 1) % 90 for j in range(5 + k)] for k in range(4)]
    out = []
    for fair in ("0", "1"):
        monkeypatch.setenv("KYOJIN_PF_SHARE", "1")
        monkeypatch.setenv("KYOJIN_PF_FAIR", fair)
        be = GridFake()
        eng = Engine(be, sessions=8, ctx=1 << 14, chunk=8, block=1024, prefix_bytes=0)
        hs = [eng.submit(p, max_new=6) for p in [pre] + kids]
        eng.start()
        texts = [collect(h)[0] for h in hs]
        st = eng.status()
        eng.stop()
        out.append((st, texts, be.forks))
    (s0, t0, f0), (s1, t1, f1) = out
    assert t0 == t1
    assert s1["share_hits"] == s0["share_hits"] == len(kids)
    assert s1["prefill_tokens"] == s0["prefill_tokens"] < len(pre) + sum(len(k) for k in kids)
    assert (f1, s1["share_held"] > 0) == (f0, s1["share_held"] > 0)


def test_a_chunk_that_skips_its_decoder_half_is_sliced(monkeypatch):
    """mix_slice_ms used to skip the `ced_skip_dec` chunks: they are the largest ones, and the yield runs after a bound's eval (no arithmetic)."""
    monkeypatch.setenv("KYOJIN_PF_FLOOR", "0")    # the slice path alone: pf_floor replaces it with bursts
    monkeypatch.setenv("KYOJIN_MIX_SLICE_MS", "2")
    monkeypatch.setenv("KYOJIN_MIX_DECODE", "0")
    be = SkipFake(pf_ms=PF_MS, step_ms=STEP_MS, layers=LAYERS)
    eng = Engine(be, sessions=2, ctx=1 << 16, chunk=CHUNK, block=CHUNK * 2, prefix_bytes=0)
    eng.start()
    a = long_prompt(20)
    b = long_prompt(1200)
    ha = eng.submit(a, max_new=40)
    n = 0
    while n < 20:                                               # A decodes
        assert ha.events.get(timeout=30)[0] == "tok"
        n += 1
    hb = eng.submit(b, max_new=3)
    tb = collect(hb)[0]
    st = eng.status()
    ha.cancel()
    collect(ha)
    eng.stop()
    assert st["pf_skip_dec"] > 0                                # the chunks did take the skip path ...
    assert st["mix_slice_rounds"] > 0                            # ... and were sliced inside
    assert tb == solo_text(b, 3)


def test_ttft_stages_are_reported_and_off_costs_nothing(monkeypatch):
    monkeypatch.delenv("KYOJIN_TTFT_STAGES", raising=False)      # on by default
    be = GridFake(pf_ms=PF_MS, step_ms=STEP_MS, layers=LAYERS)
    eng = Engine(be, sessions=1, ctx=1 << 16, chunk=CHUNK, block=CHUNK * 2, prefix_bytes=0)
    eng.start()
    _, _, st = collect(eng.submit(long_prompt(500), max_new=4, stages={"render_ms": 1.0, "tokenize_ms": 0.5}))
    agg = dict(eng.status())
    eng.stop()
    s = st["stages"]
    for k in ("render_ms", "tokenize_ms", "queue_ms", "lookup_ms", "restore_ms", "first_chunk_ms", "prefill_ms", "sample0_ms", "ttft_ms"):
        assert k in s, k
    assert s["render_ms"] == 1.0 and s["tokenize_ms"] == 0.5
    assert s["first_chunk_ms"] <= s["prefill_ms"] <= s["ttft_ms"]
    assert s["queue_ms"] <= s["ttft_ms"] and s["lookup_ms"] + s["restore_ms"] < s["ttft_ms"]
    assert agg["t_ttft_ttft"] == pytest.approx(s["ttft_ms"])     # one request: the sum over the finished requests
    assert agg["t_ttft_ttft_max"] == pytest.approx(s["ttft_ms"])
    assert agg["t_ttft_first_chunk_max"] == pytest.approx(s["first_chunk_ms"])

    monkeypatch.setenv("KYOJIN_TTFT_STAGES", "0")                # off: no dict, no aggregates
    eng = Engine(GridFake(pf_ms=PF_MS, step_ms=STEP_MS, layers=LAYERS), sessions=1, ctx=1 << 16, chunk=CHUNK, block=CHUNK * 2, prefix_bytes=0)
    eng.start()
    _, _, st = collect(eng.submit(long_prompt(500), max_new=4))
    agg = dict(eng.status())
    eng.stop()
    assert "stages" not in st and not [k for k in agg if k.startswith("t_ttft_")]


def test_ttft_stages_track_the_prefix_lookup(monkeypatch):
    """A second request on the same prompt reports the tokens the lookup restored and the chunk it did not have to run."""
    monkeypatch.delenv("KYOJIN_TTFT_STAGES", raising=False)
    be = GridFake(pf_ms=PF_MS, step_ms=STEP_MS, layers=LAYERS)
    eng = Engine(be, sessions=2, ctx=1 << 16, chunk=CHUNK, block=CHUNK, prefix_bytes=1 << 24)
    eng.start()
    p = long_prompt(600)
    collect(eng.submit(p, max_new=3))
    _, _, st = collect(eng.submit(p, max_new=3))
    eng.stop()
    assert st["stages"]["prefix_tokens"] > 0 and st["stages"]["chunks"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-q"])


def test_pf_srpt_finishes_equal_prompts_in_turn(monkeypatch):
    """Flag pf_srpt: two equal prompts run one after the other, so one first token comes at about half the shared wall time; each text is the solo text."""
    a, b = long_prompt(4000), long_prompt(4000)[::-1]
    solo_a, solo_b = solo_text(a, 3), solo_text(b, 3)
    ttft = {}
    for srpt in ("0", "1"):
        monkeypatch.setenv("KYOJIN_PF_FLOOR", "0")
        monkeypatch.setenv("KYOJIN_PF_FAIR", "1")
        monkeypatch.setenv("KYOJIN_PF_SRPT", srpt)
        eng = Engine(GridFake(pf_ms=PF_MS, step_ms=STEP_MS, layers=LAYERS), sessions=2, ctx=1 << 16, chunk=CHUNK, block=CHUNK * 2, prefix_bytes=0)
        ha, hb = eng.submit(a, max_new=3), eng.submit(b, max_new=3)
        eng.start()
        try:
            ta, _, sta = collect(ha)
            tb, _, stb = collect(hb)
        finally:
            eng.stop()
        assert ta == solo_a and tb == solo_b
        ttft[srpt] = min(sta["ttft_s"], stb["ttft_s"])
    assert ttft["1"] < 0.75 * ttft["0"]


def test_pf_srpt_short_goes_first_and_long_cannot_starve(monkeypatch):
    monkeypatch.setenv("KYOJIN_PF_SRPT", "1")
    _, tb, ta, _, _, a, b = run(monkeypatch, True)
    assert tb == solo_text(b, 3) and ta == solo_text(a, 3)
    now = time.perf_counter()
    old = SimpleNamespace(ids=[0] * 100000, pos=0, t_start=now - 120)
    new = SimpleNamespace(ids=[0] * 50, pos=0, t_start=now)
    key = lambda x: (len(x.ids) - x.pos) - engine_mod.PF_SRPT_AGE * (now - x.t_start)
    assert key(old) < key(new)
