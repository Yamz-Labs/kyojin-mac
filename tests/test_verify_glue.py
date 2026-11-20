"""V1-VERIFY: POOLSEL static-slice remainder selection == mlx_vlm's `_select_pooling_remainder` bit for bit, and the PoolingCache commit path with the flag on == off."""
import itertools

import mlx.core as mx
import numpy as np
import pytest

C = pytest.importorskip("mlx_vlm.models.cache")
from kyojin_mlx import verify_glue as GV  # noqa: E402


def _arrs(batch, lv, d, ratio, dt, with_init, seed):
    rng = np.random.RandomState(seed)
    vals = mx.array(rng.randn(batch, lv, d).astype(np.float32)).astype(dt)
    init = mx.array(rng.randn(batch, ratio, d).astype(np.float32)).astype(dt) if with_init else None
    return init, vals


@pytest.mark.parametrize("dt", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("ratio", [4, 3, 8])
def test_fast_equals_original(dt, ratio):
    GV.install()
    orig = GV._ORIG
    n = 0
    for batch, lv in itertools.product((1, 3), (1, 2, 3, 5)):
        for r in range(ratio):
            for v in range(0, lv + 1):
                for with_init in (True, False):
                    init, vals = _arrs(batch, lv, 16, ratio, dt, with_init, r * 31 + v)
                    args = (init, vals, [r] * batch, [0] * batch, [v] * batch, ratio)
                    ref = orig(*args)
                    out = GV._fast(*args)
                    assert out is not None, (r, v, lv)
                    n += 1
                    assert out.shape == ref.shape and out.dtype == ref.dtype, (r, v, lv)
                    assert mx.array_equal(out, ref).item(), (batch, lv, r, v, with_init)
    assert n > 100


def test_fast_refuses_ragged_and_dtype():
    init, vals = _arrs(2, 3, 8, 4, mx.float16, True, 0)
    assert GV._fast(init, vals, [0, 1], [0, 0], [2, 2], 4) is None
    assert GV._fast(init.astype(mx.float32), vals, [1, 1], [0, 0], [2, 2], 4) is None
    assert GV._fast(init, vals, [1, 1], [0, 0], [9, 9], 4) is None


def test_poolingcache_commit_flag_on_equals_off(monkeypatch):
    GV.install()
    ratio = 4

    def run(flag, rem_tokens, L, keep):
        monkeypatch.setattr(GV, "POOLSEL", flag)
        pc = C.PoolingCache(ratio)
        pc.buf_kv = mx.array(np.random.RandomState(5).randn(1, ratio, 8).astype(np.float32)).astype(mx.float16)
        pc.buf_gate = mx.array(np.random.RandomState(6).randn(1, ratio, 8).astype(np.float32)).astype(mx.float16)
        pc.remainder = rem_tokens
        pc.pooled = mx.zeros((1, 3, 8), dtype=mx.float16)
        pc.start_speculation(L)
        kv = mx.array(np.random.RandomState(7).randn(1, L, 8).astype(np.float32)).astype(mx.float16)
        gate = mx.array(np.random.RandomState(8).randn(1, L, 8).astype(np.float32)).astype(mx.float16)
        pc._record_speculative_inputs(kv, gate)
        pc.commit_speculation([keep])
        return pc.buf_kv, pc.buf_gate, pc.remainder

    for rem_tokens, L in itertools.product(range(ratio), (1, 2, 3)):
        for keep in range(0, L + 1):
            a = run(False, rem_tokens, L, keep)
            b = run(True, rem_tokens, L, keep)
            assert a[2] == b[2]
            for x, y in zip(a[:2], b[:2]):
                if x is None or y is None:
                    assert x is None and y is None
                else:
                    assert mx.array_equal(x, y).item(), (rem_tokens, L, keep)
    assert GV.STATS["fast"] > 0
