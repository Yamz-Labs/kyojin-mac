"""step_compile: a fixed-state layer cache is one object per request. The compiled program of a layer must be keyed on shapes (never on the cache object), reused by every
later request, and must not keep any request's KV buffers alive. KYOJIN_STEP_COMPILE_LEAK=1 restores the old key (escape hatch). CPU only."""
import gc
import weakref

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import ArraysCache

from kyojin_mlx import step_compile as SC

mx.set_default_device(mx.cpu)

ROWS = 4096                                            # buffer rows per request cache (4096 x 64 fp32 = 1 MiB)
STEPS = 4                                              # per request: warm, trace, replay, replay


class _Layer(nn.Module):
    def __call__(self, x, cache=None):
        buf = cache.cache[0]
        cache.cache[0] = buf.at[0:1].add(x)            # a cache update is an output of the program
        return x * 2.0 + buf[0:1].sum()


class _Holder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [_Layer()]


def _setup(monkeypatch, leak):
    monkeypatch.setenv(SC.LEAK_ENV, "1" if leak else "0")
    monkeypatch.setattr(SC, "ENABLED", True)
    monkeypatch.setattr(SC, "ACTIVE", True)
    h = _Holder()
    assert SC.install(h, log=lambda *_: None, force=True) == 1
    SC.STATS.update(compiled=0, eager_warm=0, stock=0)
    return h.layers[0]


def _request(layer, rows=ROWS):
    c = ArraysCache(size=1)
    c.cache[0] = mx.zeros((rows, 64))
    x = mx.ones((1, 64))
    out = None
    for _ in range(STEPS):
        out = layer(x, cache=c)
        mx.eval(out, c.cache[0])
    return out, c


def _run(monkeypatch, leak, n_req=5):
    layer = _setup(monkeypatch, leak)
    mem, outs, sizes = [], [], []
    for _ in range(n_req):
        out, c = _request(layer)
        outs.append(out.tolist())
        del c, out
        gc.collect()
        mem.append(mx.get_active_memory())
        sizes.append(len(type(layer)._step_programs))
    return mem, outs, sizes


def test_memory_is_flat_and_programs_are_reused(monkeypatch):
    mem, _, sizes = _run(monkeypatch, leak=False)
    assert mem[-1] - mem[0] <= 64 * 1024                         # no request cache outlives its request
    assert SC.STATS["eager_warm"] == 1                           # one warm-up in total, not one per request
    assert SC.STATS["compiled"] == 3 + 4 * 4                     # first request: steps 2..4; every later request: all 4 steps replay the same program
    assert sizes == [1] * 5                                      # one program, the key does not carry the cache object


def test_program_cache_does_not_grow_with_request_count(monkeypatch):
    layer = _setup(monkeypatch, leak=False)
    for _ in range(3 * SC.MAX_KEYS):                             # far more requests than MAX_KEYS
        _request(layer)
    assert len(type(layer)._step_programs) == 1
    assert SC.STATS["stock"] == 0                                # the per-layer key limit is never reached by request count
    assert SC.STATS["eager_warm"] == 1
    assert SC.STATS["compiled"] == 3 + (3 * SC.MAX_KEYS - 1) * STEPS


def test_new_shape_adds_one_program_not_one_per_request(monkeypatch):
    layer = _setup(monkeypatch, leak=False)
    for rows in (ROWS, 2 * ROWS, ROWS, 2 * ROWS, ROWS):          # a cache buffer of another size is another shape: 2 programs in all
        _request(layer, rows)
    assert len(type(layer)._step_programs) == 2


def test_first_request_cache_is_collectable_after_the_second(monkeypatch):
    layer = _setup(monkeypatch, leak=False)
    _, c1 = _request(layer)
    ref = weakref.ref(c1)
    base = mx.get_active_memory()
    del c1
    _request(layer)                                              # the second request runs the program built by the first
    gc.collect()
    assert ref() is None                                         # the program cache holds no reference to the first request's cache object
    assert mx.get_active_memory() - base <= 64 * 1024            # nor to its buffers (1 MiB each)


def test_same_results_as_leak_mode(monkeypatch):
    _, a, _ = _run(monkeypatch, leak=True, n_req=3)
    _, b, _ = _run(monkeypatch, leak=False, n_req=3)
    assert a == b


def test_escape_hatch_restores_the_old_key(monkeypatch):
    mem, _, sizes = _run(monkeypatch, leak=True)
    assert sizes == [1, 2, 3, 4, 5]                              # one program per request cache object
    assert mem[-1] - mem[0] >= 3 * ROWS * 64 * 4 * 0.9           # one KV per request stays
