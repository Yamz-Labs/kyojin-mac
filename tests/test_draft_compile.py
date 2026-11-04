"""Flag draft_compile on stubs (no model, no Metal): the rows draft runs as one mx.compile region, same values as the eager function, eager first call,
retrace on a flag switch, eager fallback on a trace error, replay keeps its own region."""
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from kyojin_mlx import dspark as DS, flags as FL

V, RANK, H, L = 200, 8, 6, 5


def make(seed=0, host_read=False):
    """A Drafter shell whose rows functions are a small head + the real `_rows_tail` (bigram chain, confidences): enough elementwise / matmul work for
    mx.compile to fuse, and a counter of python executions (a traced call does not run the body again)."""
    mx.random.seed(seed)
    lm = SimpleNamespace(markov_embed=nn.Embedding(V, RANK), markov_head=nn.Linear(RANK, V, bias=False), confidence=nn.Linear(H + RANK, 1, bias=False))
    w_head = mx.random.normal((H, V)) * 0.5
    w_x = mx.random.normal((V, H)) * 0.5
    d = DS.Drafter.__new__(DS.Drafter)
    d.lm, d.block, d._conf_w = lm, L, None
    d._tail_conf = lambda *a: DS.Drafter._tail_conf(d, *a)
    d._rows_c, d._rows_c_off, d._rows_c_warm, d._rows_c_gen = {}, set(), set(), FL.GEN
    d.calls = {"": 0, "q": 0, "r": 0, "rq": 0}

    def run(kind, ids, pend, pos, rings, keep=False, teacher=False):
        d.calls[kind] += 1
        if host_read:
            int(mx.max(pos).item())                                   # a host read inside the region: not traceable
        S = ids.shape[0] // L
        x = w_x[ids[:, 0]] + mx.stack(list(rings)).sum(0)[:S * L, :H] if rings else w_x[ids[:, 0]]       # [S * L, H]
        x = mx.tanh(x) * mx.sigmoid(x.astype(mx.float32)).astype(x.dtype) + pos[:, None].astype(x.dtype) * 1e-3
        logits = (x @ w_head).reshape(S, L, V)
        return DS.Drafter._rows_tail(d, x[:, None], logits, pend, S, keep=keep, ids=(ids.reshape(S, L) if teacher else None))

    d._rows_fn = lambda ids, pend, pos, *r: run("", ids, pend, pos, r)[:2]
    d._rows_fn_q = lambda ids, pend, pos, *r: run("q", ids, pend, pos, r, keep=True)
    d._rows_fn_r = lambda ids, pend, pos, *r: run("r", ids, pend, pos, r, teacher=True)[:2]
    d._rows_fn_rq = lambda ids, pend, pos, *r: run("rq", ids, pend, pos, r, teacher=True, keep=True)
    for reg, fn in DS.Drafter._ROWS_FN.values():
        setattr(d, reg, getattr(d, fn))
    return d


def inputs(S, seed):
    mx.random.seed(seed)
    ids = mx.random.randint(0, V, (S * L, 1))
    pend = mx.random.randint(0, V, (S,)).astype(mx.int32)
    pos = mx.arange(S * L).astype(mx.int32)
    rings = [mx.random.normal((S * L, H))]
    return ids, pend, pos, rings


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    from kyojin_mlx import rowinv, replay as RP
    monkeypatch.setattr(rowinv, "rowmm", lambda x, w: mx.matmul(x, w.T))      # the Metal row-invariant product has no CPU path
    monkeypatch.setattr(RP, "ENABLED", False)
    monkeypatch.setattr(DS, "DRAFT_COMPILE", True)


def same(a, b):
    assert len(a) == len(b)
    for u, v in zip(a, b):
        if u is None or v is None:
            assert u is None and v is None
        else:
            assert np.array_equal(np.array(u), np.array(v))


def test_flag_registered_off_by_default_and_switchable():
    f = FL.FLAGS["draft_compile"]
    assert f.default == "0" and f.runtime and ("dspark", "DRAFT_COMPILE") == f.attrs[0][:2]


def test_off_runs_the_eager_function_every_call(monkeypatch):
    monkeypatch.setattr(DS, "DRAFT_COMPILE", False)
    d = make()
    for k in range(4):
        d._rows_call("", *inputs(2, k)[:3], inputs(2, k)[3])
    assert d.calls[""] == 4 and not d._rows_c


def test_compiled_equals_eager_bitwise_for_every_kind_and_session_count():
    d = make()
    ref = make()
    for kind in ("", "q", "r", "rq"):
        for S in (1, 2, 3):
            for k in range(4):                                         # call 1 of a (kind, rows) is eager, the others compiled
                ids, pend, pos, rings = inputs(S, 10 * S + k)
                got = d._rows_call(kind, ids, pend, pos, rings)
                reg = DS.Drafter._ROWS_FN[kind][1]
                want = getattr(ref, reg)(ids, pend, pos, *rings)
                same(got, want)
    assert not d._rows_c_off and DS.STATS["rows_compiled"] > 0


def test_compiled_call_does_not_rebuild_the_graph_in_python():
    d = make()
    for k in range(6):
        ids, pend, pos, rings = inputs(2, k)
        d._rows_call("", ids, pend, pos, rings)
    # call 1 eager (+1), call 2 traces (+1), calls 3.. replay the compiled graph: the python body is not run again
    assert d.calls[""] == 2


def test_first_call_per_session_count_is_eager():
    d = make()
    for S in (1, 2):
        for k in range(3):
            d._rows_call("", *inputs(S, k)[:3], inputs(S, k)[3])
    assert d.calls[""] == 4                                              # per rows count: one eager warm call, one trace; the third call replays
    assert ("", L) in d._rows_c_warm and ("", 2 * L) in d._rows_c_warm


def test_trace_error_falls_back_to_eager_and_stays_there():
    d = make(host_read=True)
    ref = make()
    outs = []
    for k in range(4):
        ids, pend, pos, rings = inputs(2, k)
        outs.append((d._rows_call("", ids, pend, pos, rings), ref._rows_fn(ids, pend, pos, *rings)))
    for got, want in outs:
        same(got, want)
    assert "" in d._rows_c_off and "" not in d._rows_c
    n = d.calls[""]
    d._rows_call("", *inputs(2, 9)[:3], inputs(2, 9)[3])
    assert d.calls[""] == n + 1                                       # eager each call, no new trace attempt


def test_flag_switch_retraces(monkeypatch):
    d = make()
    for k in range(4):
        d._rows_call("", *inputs(2, k)[:3], inputs(2, k)[3])
    assert "" in d._rows_c
    monkeypatch.setattr(FL, "GEN", FL.GEN + 1)
    d._rows_call("", *inputs(2, 5)[:3], inputs(2, 5)[3])
    assert d._rows_c_gen == FL.GEN and "" in d._rows_c               # the compiled draft was rebuilt for the new switches


def test_replay_keeps_its_own_region(monkeypatch):
    from kyojin_mlx import replay as RP
    monkeypatch.setattr(RP, "ENABLED", True)
    d = make()
    for k in range(4):
        d._rows_call("", *inputs(2, k)[:3], inputs(2, k)[3])
    assert d.calls[""] == 4 and not d._rows_c


def test_refine_and_logits_route_to_their_own_functions():
    d = make()
    for kind in ("q", "r", "rq"):
        d._rows_call(kind, *inputs(1, 1)[:3], inputs(1, 1)[3])
    assert d.calls["q"] == 1 and d.calls["r"] == 1 and d.calls["rq"] == 1 and d.calls[""] == 0
