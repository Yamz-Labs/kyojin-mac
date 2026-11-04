"""pf_moe_hoist: control flow of `MoePrefill.__call__` for a call of several passes, with the GPU parts stubbed (the kernels themselves are unchanged and need Metal):
- default 0: the old pass loop (`_chunk` per pass, `mx.eval(y)` after each);
- on: every pass planned first, ONE evaluation of all the offsets, each pass built from its own plan and counts and submitted with async_eval, the output the concatenation;
- a pass with a hot expert (None) takes the host plan; MOE_NOSYNC and a single pass keep their own paths."""
import mlx.core as mx
import numpy as np
import pytest

from kyojin_mlx import flags as FL
from kyojin_mlx import moe_prefill as P

E, TOPK, CHUNK = 4, 2, 8


class Proxy:
    """`mx` with evaluation calls recorded in order."""

    def __init__(self, log):
        self._log = log

    def eval(self, *a):
        self._log.append(("eval", len(a)))
        return mx.eval(*a)

    def async_eval(self, *a):
        self._log.append(("async_eval", len(a)))
        return mx.async_eval(*a)

    def __getattr__(self, k):
        return getattr(mx, k)


def make(log, hot=()):
    mp = P.MoePrefill.__new__(P.MoePrefill)
    mp.fused, mp.h, mp._chunk_rows, mp.bailed = True, True, CHUNK, 0

    def plan(ids, wts):
        log.append(("plan", int(ids.shape[0])))
        counts = np.bincount(np.array(ids).reshape(-1), minlength=E).astype(np.uint32)
        cum = mx.array(np.concatenate([[0], np.cumsum(counts)]).astype(np.uint32))
        return (None, None, None, None, cum)

    def glu(x, ids, wts, plan=None, counts=None):
        log.append(("glu", int(x.shape[0]), plan is not None, None if counts is None else counts.tolist()))
        if x.shape[0] in hot:
            return None
        return x * 2

    def host(x, ids, wts):
        log.append(("host", int(x.shape[0])))
        return x * 2

    mp._glu_plan, mp._chunk_glu, mp._chunk_host = plan, glu, host
    return mp


def run(monkeypatch, hoist, T=20, nosync=False, hot=()):
    log = []
    monkeypatch.setattr(P, "mx", Proxy(log))
    monkeypatch.setattr(P, "HOIST", hoist)
    monkeypatch.setattr(P, "MOE_NOSYNC", nosync)
    monkeypatch.setattr(P, "GLU", True)
    mp = make(log, hot)
    x = mx.arange(T * 3, dtype=mx.float32).reshape(T, 3)
    ids = mx.array((np.arange(T * TOPK) % E).reshape(T, TOPK).astype(np.uint32))
    wts = mx.ones((T, TOPK), dtype=mx.float32)
    y = mp(x, ids, wts)
    mx.eval(y)
    return y, x, log


def test_default_is_the_pass_loop(monkeypatch):
    y, x, log = run(monkeypatch, hoist=False)
    assert np.array_equal(np.array(y), np.array(x) * 2)
    assert [e[0] for e in log if e[0] in ("glu", "eval", "async_eval", "plan")] == ["glu", "eval", "glu", "eval", "glu", "eval"]
    assert all(e[2] is False for e in log if e[0] == "glu")       # no plan handed in: each pass plans and syncs by itself


def test_hoisted_plans_first_one_sync_then_async_passes(monkeypatch):
    y, x, log = run(monkeypatch, hoist=True)
    assert np.array_equal(np.array(y), np.array(x) * 2)
    kinds = [e[0] for e in log if e[0] in ("glu", "eval", "async_eval", "plan")]
    # passes of 8, 8, 4 rows: three plans, ONE eval of the three offset arrays, then build + submit per pass
    assert kinds[:5] == ["plan", "plan", "plan", "eval", "glu"]
    assert log[3] == ("eval", 3)
    assert [e[0] for e in log if e[0] in ("glu", "async_eval")] == ["glu", "async_eval", "glu", "async_eval", "glu", "async_eval"]
    glus = [e for e in log if e[0] == "glu"]
    assert [g[1] for g in glus] == [8, 8, 4] and all(g[2] for g in glus)
    # each pass got its own counts (rows per expert), summing to its pairs
    assert [sum(g[3]) for g in glus] == [16, 16, 8]


def test_hoisted_hot_pass_takes_the_host_plan(monkeypatch):
    y, x, log = run(monkeypatch, hoist=True, hot=(4,))
    assert np.array_equal(np.array(y), np.array(x) * 2)
    assert ("host", 4) in log and ("host", 8) not in log


def test_single_pass_and_nosync_keep_their_paths(monkeypatch):
    y, x, log = run(monkeypatch, hoist=True, T=6)
    assert not any(e[0] in ("plan", "async_eval") for e in log)
    y, x, log = run(monkeypatch, hoist=True, nosync=True)
    assert not any(e[0] == "plan" for e in log) and sum(e[0] == "glu" for e in log) == 3


def test_flag_registered_default_off_and_runtime():
    f = FL.FLAGS["pf_moe_hoist"]
    assert f.default == "0" and f.runtime and P.HOIST is False
