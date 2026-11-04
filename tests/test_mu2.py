"""Multi-session speculative verify: the round-plan tuner (pure python) and the shared-prefix verify attention (tq.TQKVCache.attend_shared, needs the NAX tiles)."""
import time

import numpy as np
import pytest

from kyojin_mlx.serve.mu_tuner import MultiTuner


def _drive(rate_cost, probs, n=8, dmax=4, ticks=400, **kw):
    t = MultiTuner(window=3, skip=1, hold=50, **kw)
    clock = [0.0]
    real = time.perf_counter
    time.perf_counter = lambda: clock[0]
    try:
        picks = []
        for _ in range(ticks):
            a = t.plan(n, dmax, ("k",), probs)
            clock[0] += rate_cost[a]
            t.done(("k",), 0)
            picks.append(a)
    finally:
        time.perf_counter = real
    return t, picks


COST = {0: 0.041, 1: 0.058, 2: 0.075, 3: 0.082, 4: 0.09}


@pytest.mark.parametrize("probs,want", [([0.8, 0.7, 0.6, 0.5], 3), ([0.4, 0.3, 0.2, 0.1], 0), ([0.95, 0.9, 0.9, 0.9], 4)])
def test_tuner_picks_the_arm_with_most_tokens_per_second(probs, want):
    t, picks = _drive(COST, probs)
    assert picks[-20:] == [want] * 20
    assert t.best(8, 4) == want


def test_tuner_follows_the_acceptance_and_keeps_costs_per_n():
    t, _ = _drive(COST, [0.4, 0.3, 0.2, 0.1])
    assert t.plan(8, 4, ("k",), [0.95, 0.9, 0.9, 0.9]) in (0, 4)            # the open window ends first; the next choice follows the new acceptance
    t.arm = None
    assert t.plan(8, 4, ("k",), [0.95, 0.9, 0.9, 0.9]) == 4
    assert set(t.cost) == {8}
    assert t.plan(4, 4, ("j",), [0.8] * 4) in range(5)                       # a new N measures its own arms
    assert 4 in t.cost


def test_tuner_respects_dmax():
    t, picks = _drive(COST, [0.95, 0.9, 0.9, 0.9], dmax=2)
    assert max(picks) <= 2 and picks[-1] == 2


mx = pytest.importorskip("mlx.core")
from kyojin_mlx import kvwin as KW, tq as TQ                               # noqa: E402
from kyojin_mlx.sdpa_dec import nax_ok                                     # noqa: E402

needs_nax = pytest.mark.skipif(not mx.metal.is_available() or not nax_ok(), reason="needs the NAX tiles (M5)")


@needs_nax
@pytest.mark.parametrize("lens", [[7, 7, 7], [5, 12, 9, 20]])
def test_attend_shared_matches_per_session_attend(lens):
    """Forks of one prefix cache verify QT rows each: the shared pool pass + per-session fp16 parts + one merge equals each session's own attend."""
    KW.install()
    d, HQ, HK, P, QT = 256, 8, 2, 2700, 3
    r = np.random.default_rng(5)
    mk = lambda n: mx.array(r.standard_normal((1, HK, n, d)).astype(np.float32) * 0.5).astype(mx.float16)
    base = TQ.TQKVCache(6, 6, 4, 256)
    for s in range(0, P, 300):
        base.update_and_fetch(mk(min(300, P - s)), mk(min(300, P - s)))
    caches = []
    for n in lens:
        c = base.fork()
        c.update_and_fetch(mk(n), mk(n))
        caches.append(c)
    q = mx.array(r.standard_normal((len(lens), HQ, QT, d)).astype(np.float32)).astype(mx.float16)
    for c in caches:
        c.update_and_fetch(mk(QT), mk(QT))
    scale = d ** -0.5
    TQ.STATS.clear()
    out = TQ.TQKVCache.attend_shared(caches, q, scale, P)
    assert out is not None and TQ.STATS["shared"] == 1
    for j, c in enumerate(caches):
        ref = c.attend(q[j:j + 1], scale)
        err = mx.max(mx.abs(out[j:j + 1].astype(mx.float32) - ref.astype(mx.float32))).item()
        assert err < 0.02 * max(mx.max(mx.abs(ref.astype(mx.float32))).item(), 1e-3), err


@needs_nax
def test_attend_shared_declines_when_it_does_not_apply():
    KW.install()
    d, HQ, HK = 256, 8, 2
    r = np.random.default_rng(6)
    mk = lambda n: mx.array(r.standard_normal((1, HK, n, d)).astype(np.float32)).astype(mx.float16)
    base = TQ.TQKVCache(6, 6, 4, 256)
    base.update_and_fetch(mk(600), mk(600))
    cs = [base.fork() for _ in range(3)]
    q = mx.array(r.standard_normal((3, HQ, 3, d)).astype(np.float32)).astype(mx.float16)
    for c in cs:
        c.update_and_fetch(mk(3), mk(3))
    assert TQ.TQKVCache.attend_shared(cs, q, d ** -0.5, 600) is None           # a short pool (< ROWS_MIN keys): the per-session kernel stays
    assert TQ.TQKVCache.attend_shared(cs, q[:, :, :1], d ** -0.5, 600) is None  # one row per session (decode)
    assert TQ.TQKVCache.attend_shared(cs, q, d ** -0.5, 0) is None              # no shared prefix recorded
