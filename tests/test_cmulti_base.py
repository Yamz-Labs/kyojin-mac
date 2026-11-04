"""cloud/cmulti-2 base: the grid-sensitive fake sees what the plain one cannot, and the single-session path is pinned."""
import time

from kyojin_mlx.serve.engine import Engine
from cmulti_fakes import GridFake
from test_serve_engine import collect


def run(prompt, chunk=4, block=8, cuts=(), max_new=12, prefix_bytes=1 << 20, **ekw):
    be = GridFake()
    eng = Engine(be, sessions=1, ctx=4096, chunk=chunk, block=block, prefix_bytes=prefix_bytes, **ekw)
    eng.start()
    try:
        return collect(eng.submit(prompt, max_new=max_new, cuts=cuts))[0], be
    finally:
        eng.stop()


def test_gridfake_sees_the_chunk_grid():
    p = list(range(1, 30))
    assert run(p, chunk=4)[0] != run(p, chunk=8)[0]


def test_single_session_trace_is_fixed():
    """The call trace of one session: the reference every flagged run is compared with (flag off = today)."""
    p = list(range(1, 23))
    _, be = run(p, chunk=8, block=16, max_new=3)
    assert [c for c in be.calls if c[0] == "pf"] == [("pf", 0, 8), ("pf", 8, 8), ("pf", 16, 6)]


def test_existing_offgrid_cut_snapshot_reuse():
    """Pre-existing behaviour, measured: a session whose prompt has a cut c leaves a snapshot at c; another prompt with the same first c tokens but no cut there
    hits it and prefills the rest from c, so its chunk partition (c, then the grid) differs from its own solo run's. On the grid-sensitive fake the text differs."""
    base = list(range(1, 38))                      # chunk 8, cut at 11
    a = base + [50, 51, 52]
    b = base + [60, 61, 62]
    be = GridFake()
    eng = Engine(be, sessions=1, ctx=4096, chunk=8, block=16, prefix_bytes=1 << 20)
    eng.start()
    try:
        collect(eng.submit(a, max_new=2, cuts=(11,)))
        tb = collect(eng.submit(b, max_new=8))[0]
        hit = eng.prefix.hits
    finally:
        eng.stop()
    solo_b = run(b, chunk=8, block=16, max_new=8)[0]
    print("cut-snapshot hit:", hit, "text equals solo:", tb == solo_b)
    assert hit >= 1
