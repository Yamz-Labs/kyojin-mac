import sys, os
sys.path.insert(0, os.path.expanduser("~/kyojin-mlx"))
sys.path.insert(0, os.path.expanduser("~/kyojin-mlx/vlm_pkg"))
import mlx.core as mx, numpy as np
from mlx_lm.models.deepseek_v41 import _fake_quant_fp4
from kyojin_mlx import fp4pool, latent_tq


def test_latent_gather_equals_dense_rows():
    rng = np.random.default_rng(2)
    c = latent_tq.LatentTQCache(bits=6, sink=4, window=64)
    for n in (300, 1, 1, 200):
        c.update_and_fetch(mx.array(rng.standard_normal((1, 1, n, 512)).astype(np.float32)).astype(mx.float16), mx.zeros((1, 1, n, 0), mx.float16))
    dense = c._dense(mx.float16)
    idx = np.concatenate([np.arange(0, 6), np.arange(c.offset - 70, c.offset), rng.integers(0, c.offset, 200)]).astype(np.int32)
    got = c.gather(mx.array(idx)[None, None, :, None], mx.float16)
    ref = mx.take_along_axis(dense, mx.array(idx)[None, None, :, None], axis=2)
    assert np.array_equal(np.array(got.view(mx.uint16)), np.array(ref.view(mx.uint16)))


def test_packed_view_slices_and_dense():
    rng = np.random.default_rng(3)
    x = _fake_quant_fp4(mx.array(rng.standard_normal((1, 40000, 512)).astype(np.float32)).astype(mx.float16), block_size=16, e4m3_scale=True)
    slots = fp4pool.PoolSlots([x])
    assert not isinstance(slots[0], fp4pool.PackedView)
    slots.lazy = True
    v = slots[0]
    assert isinstance(v, fp4pool.PackedView) and v.shape == (1, 40000, 512)
    w = v[:1, :1234]
    assert w.shape == (1, 1234, 512)
    assert bool(mx.array_equal(w.dense(), x[:, :1234]))
