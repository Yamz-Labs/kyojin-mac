"""Flag engram_dec_queue on a table on disk (no model): a decode pass does not wait behind a prefill read, and a decode round keeps the chunk's jobs."""
import os
import sys
import threading

import mlx.core as mx
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from kyojin_mlx import engram_mlx as EM, engram_rows as ER
import test_engram_rows as TE


@pytest.fixture
def tables(tmp_path):
    sp, rws = TE.spec_with_rows()
    TE.make_shards(tmp_path, sp, rws)
    t = ER.EngramTables(str(tmp_path), sp)
    t.ahead = EM.EngramAhead(t)
    yield t, rws
    t.close()


def hashes(rws, b, n, seed=0):
    return mx.array(np.stack([np.random.default_rng(seed + p).integers(0, rws[p], (b, n, 24)) for p in range(2)], axis=2))


def block(ex, n=1):
    """Hold every worker of `ex` until the event is set."""
    go = threading.Event()
    for _ in range(n):
        ex.submit(go.wait, 30)
    return go


def test_decode_jobs_pass_a_busy_prefill_queue(tables, monkeypatch):
    t, rws = tables
    monkeypatch.setattr(ER, "DEC_QUEUE", True)
    go = block(t._pre)
    try:
        h = hashes(rws, 6, 1)
        t.ahead.start(h)
        shape, uniq, inv, w, s = t.ahead.jobs[0][1].result(timeout=10)
        want = t.read_rows(0, uniq)
        assert np.array_equal(w, want[0]) and np.array_equal(s, want[1])
    finally:
        go.set()


def test_flag_off_keeps_one_queue(tables, monkeypatch):
    t, rws = tables
    monkeypatch.setattr(ER, "DEC_QUEUE", False)
    go = block(t._pre)
    try:
        t.ahead.start(hashes(rws, 6, 1))
        job = t.ahead.jobs[0][1]
        with pytest.raises(TimeoutError):
            job.result(timeout=0.3)
    finally:
        go.set()
    job.result(timeout=10)


def test_prefill_jobs_stay_on_their_queue(tables, monkeypatch):
    t, rws = tables
    monkeypatch.setattr(ER, "DEC_QUEUE", True)
    go = block(t._pre)
    try:
        t.ahead.start(hashes(rws, 1, EM.EngramAhead.MIN_ROWS))
        job = t.ahead.jobs[0][1]
        with pytest.raises(TimeoutError):
            job.result(timeout=0.3)
    finally:
        go.set()
    job.result(timeout=10)


def test_small_read_passes_a_busy_pool(tables, monkeypatch):
    t, rws = tables
    monkeypatch.setattr(ER, "DEC_QUEUE", True)
    go = block(t._pool, t._nthreads)
    try:
        box = []
        th = threading.Thread(target=lambda: box.append(t.read_rows(0, np.arange(8))))
        th.start()
        th.join(10)
        assert box, "a small read waited behind the big pool"
    finally:
        go.set()
    monkeypatch.setattr(ER, "DEC_QUEUE", False)
    off = t.read_rows(0, np.arange(8))
    assert np.array_equal(box[0][0], off[0]) and np.array_equal(box[0][1], off[1])


@pytest.mark.parametrize("on,kept", [(True, True), (False, False)])
def test_round_keeps_the_chunk_jobs(tables, monkeypatch, on, kept):
    t, rws = tables
    monkeypatch.setattr(ER, "DEC_QUEUE", on)
    a = t.ahead
    a.start(hashes(rws, 1, a.MIN_ROWS))
    chunk = dict(a.jobs)
    seen = []

    def round_():
        a.start(hashes(rws, 6, 1, seed=5))
        seen.append(set(a.jobs))
        for _, j in a.jobs.values():
            j.result(timeout=10)
    a.nested(round_)()
    assert seen == [{0, 1}]
    assert (a.jobs == chunk) is kept
    for _, j in chunk.values():
        j.result(timeout=10)
