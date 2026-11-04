import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import mlx.core as mx
import numpy as np
from kyojin_mlx.latent_tq import LatentTQCache, wrap_latent_layers
from kyojin_mlx import rows as R
from mlx_vlm.models.cache import KVCache, CacheList, BatchKVCache


def lat(n, seed=0, D=512):
    return mx.array(np.random.default_rng(seed).standard_normal((1, 1, n, D)).astype(np.float32) * 0.7).astype(mx.bfloat16)


def e(v):
    return mx.zeros((1, 1, v.shape[2], 0), dtype=v.dtype)


def test_overlay_and_error():
    c = LatentTQCache(6, 4, 32)
    x = lat(300)
    out = None
    for a in range(0, 300, 100):
        out, _ = c.update_and_fetch(x[:, :, a:a + 100], e(x[:, :, a:a + 100]))
    assert out.shape == (1, 1, 300, 512) and c.offset == 300
    assert mx.array_equal(out[:, :, :4], x[:, :, :4]) and mx.array_equal(out[:, :, -32:], x[:, :, -32:])
    mid = (out[:, :, 4:-32].astype(mx.float32) - x[:, :, 4:-32].astype(mx.float32))
    rel = float(mx.sqrt((mid ** 2).mean() / (x[:, :, 4:-32].astype(mx.float32) ** 2).mean()))
    assert 0 < rel < 0.03, rel


def test_chunking_invariant_and_small():
    x = lat(40)
    a = LatentTQCache(6, 4, 16); b = LatentTQCache(6, 4, 16)
    oa, _ = a.update_and_fetch(x, e(x))
    for i in range(40):
        ob, _ = b.update_and_fetch(x[:, :, i:i + 1], e(x[:, :, i:i + 1]))
    assert mx.array_equal(oa[:, :, :-1], ob[:, :, :-1]) or float(mx.abs(oa - ob).max()) < 0.2   # one-token and one-chunk writes round the same rows
    assert mx.array_equal(oa[:, :, 4:24], ob[:, :, 4:24])
    s = LatentTQCache(6, 4, 256)
    o, _ = s.update_and_fetch(x, e(x))
    assert mx.array_equal(o, x)                                   # shorter than sink + window: exact


def test_fork_state_trim_merge():
    c = LatentTQCache(6, 4, 32)
    x = lat(200, 1)
    c.update_and_fetch(x, e(x))
    f = R.fork([CacheList(c, KVCache())])[0].caches[0]
    y = lat(10, 2)
    of, _ = f.update_and_fetch(y, e(y))
    assert c.offset == 200 and f.offset == 210
    g = LatentTQCache(6, 4, 32)
    g.state = c.state
    assert g.offset == 200
    d1, d2 = c._dense(mx.bfloat16), g._dense(mx.bfloat16)
    assert mx.array_equal(d1, d2)
    assert f.trim(10) == 10 and f.offset == 200 and mx.array_equal(f._dense(mx.bfloat16)[:, :, :168], d1[:, :, :168])
    b = LatentTQCache.merge([c, g])
    assert isinstance(b, BatchKVCache) and b.keys.shape[0] == 2
    assert c.nbytes < 200 * 512 * 2 * 0.6


def test_rope_part_refused():
    c = LatentTQCache()
    try:
        c.update_and_fetch(lat(4), mx.zeros((1, 1, 4, 8), dtype=mx.bfloat16))
    except ValueError:
        return
    raise AssertionError


def test_pool_rows_equal_fake_quant():
    from kyojin_mlx import tq
    x = lat(64, 3)
    c = LatentTQCache(6, 0, 0)
    out, _ = c.update_and_fetch(x, e(x))
    parts = []
    for o in (0, 256):
        y = tq.rotate(x[..., o:o + 256].reshape(-1, 256), 256)
        w, s = tq.tq_write(y, 6)
        parts.append(tq.unrotate(tq.tq_dequant(w, s, 6, 256, mx.float32), 256).reshape(1, 1, 64, 256))
    ref = mx.concatenate(parts, -1).astype(x.dtype)
    assert mx.array_equal(out, ref)
