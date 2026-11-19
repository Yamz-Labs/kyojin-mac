import os, sys
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import mlx.core as mx
import numpy as np
from kyojin_mlx.latent_tq import LatentTQCache, wrap_latent_layers
from kyojin_mlx import latent_tq as LT
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


def _row(n, seed, window=32, dtype=mx.float16):
    c = LatentTQCache(6, 4, window)
    x = lat(n, seed).astype(dtype)
    c.update_and_fetch(x, e(x))
    return CacheList(c, KVCache()), x


def test_merge_keeps_every_row_whatever_the_first_type():
    """Engine join path: row 0 was batched and extracted (dense), row 1 is fresh from prefill (LatentTQCache). type(first).merge must not zero row 1."""
    r0, _ = _row(300, 1)
    r1, _ = _row(260, 2)
    first = R.batch([[r0]])[0]
    old = first.extract(0)
    assert not isinstance(old.caches[0], LatentTQCache)                      # the row left the batch as a dense latent
    mixed = R.batch([[old], [r1]])[0].caches[0]
    d1 = r1.caches[0]._dense(mx.float16)
    n0, n1 = 300, 260
    k = mixed.keys
    assert k.dtype == mx.float16 and k.shape[0] == 2 and k.shape[2] == 300
    assert mx.array_equal(k[1, :, 300 - n1:], d1[0])                           # the joiner's rows are intact, left-padded
    assert float(mx.abs(k[1, :, :300 - n1]).max()) == 0.0
    assert mx.array_equal(k[0], old.caches[0].keys[0])
    assert mx.array_equal(mixed.left_padding, mx.array([0, 40]))
    # the other order, and every row of a three-way mix survive a second extract
    again = R.batch([[r1], [old]])[0].caches[0]
    assert mx.array_equal(again.keys[1], k[0]) or mx.array_equal(again.keys[1, :, :n0], old.caches[0].keys[0])
    r2, _ = _row(200, 3)
    third = R.batch([[R.batch([[old], [r1]])[0].extract(1)], [r2]])[0].caches[0]
    assert float(mx.abs(third.keys[0]).max()) > 0 and float(mx.abs(third.keys[1]).max()) > 0


def test_to_dense_keeps_the_written_dtype():
    """fp16 latents are not narrowed to bf16 by the merge (it rounded every sink / window row: batched row != solo row)."""
    r, x = _row(100, 4, window=256)                                           # shorter than window: all rows exact
    b = R.batch([[r]])[0].caches[0]
    assert b.keys.dtype == mx.float16 and mx.array_equal(b.keys[0], x[0])


def test_batched_rows_round_the_rows_that_leave_the_window_like_solo():
    """G6b: a row merged next to a longer one kept the window rows exact for ever, while its solo run rounds one more row to the pool every decode step
    (6 bits): the short row diverged from solo (logit diff 0.7 over 40 steps on the GLM-5.3 cut, [1500, 3800] tokens, every key selected)."""
    W, steps = 16, 12
    n = (90, 150)
    ex = [lat(n[i] + steps, 10 + i).astype(mx.float16) for i in range(2)]
    def fresh(i):
        c = LatentTQCache(6, 4, W)
        c.update_and_fetch(ex[i][:, :, :n[i]], e(ex[i][:, :, :n[i]]))
        return c
    solo2, batch_rows = [fresh(0), fresh(1)], [fresh(0), fresh(1)]
    ref = [[c.update_and_fetch(ex[i][:, :, n[i] + t:n[i] + t + 1], e(ex[i][:, :, :1]))[0] for t in range(steps)] for i, c in enumerate(solo2)]
    b = R.batch([[CacheList(c, KVCache())] for c in batch_rows])[0].caches[0]
    for t in range(steps):
        step_k = mx.concatenate([ex[i][:, :, n[i] + t:n[i] + t + 1] for i in range(2)], axis=0)
        out, _ = b.update_and_fetch(step_k, mx.zeros((2, 1, 1, 0), dtype=step_k.dtype))
        for i in range(2):
            pad = int(b.left_padding[i])
            assert mx.array_equal(out[i:i + 1, :, pad:], ref[i][t]), (t, i)
    # a row that leaves the batch goes on decoding as it would have solo
    assert b.tq_cfg == (6, 4, W)
    d = b.extract(0)
    assert d.tq_cfg == (6, 4, W) and d.offset == n[0] + steps
    more = lat(5, 77).astype(mx.float16)
    s0 = solo2[0]
    for t in range(5):
        a, _ = d.update_and_fetch(more[:, :, t:t + 1], e(more[:, :, t:t + 1]))
        r, _ = s0.update_and_fetch(more[:, :, t:t + 1], e(more[:, :, t:t + 1]))
        assert mx.array_equal(a, r), t


def test_batched_rows_with_a_row_inside_the_window_and_sink_stay_exact():
    W = 16
    short, long_ = _row(10, 5, window=W)[0], _row(60, 6, window=W)[0]            # 10 rows: sink + window only, nothing to round yet
    b = R.batch([[short], [long_]])[0].caches[0]
    x = lat(3, 9).astype(mx.float16)
    out = None
    for t in range(3):
        out, _ = b.update_and_fetch(mx.concatenate([x[:, :, t:t + 1]] * 2, axis=0), mx.zeros((2, 1, 1, 0), dtype=x.dtype))
    pad = int(b.left_padding[0])
    assert pad == 50 and float(mx.abs(out[0, :, :pad]).max()) == 0.0              # padding stays zero
    ref = short.caches[0]._dense(mx.float16)
    assert mx.array_equal(out[0:1, :, pad:pad + 10], ref)                          # a row shorter than window + sink is exact


@pytest.mark.parametrize("rolled,pad", [([10, 10], [0, 0]), ([10, 14], [0, 3]), ([40, 40], [0, 0]), ([10, 10], [0, 40])])
def test_roll_host_mask_matches_the_array_formula(rolled, pad):
    """`_roll` decides on the host which (row, column) pairs roll; same result as the select over the arrays (all true / mixed / none)."""
    mx.random.seed(3)
    cfg = (6, 4, 256)
    base = (mx.random.normal((2, 1, 64, 512)) * 2).astype(mx.float16)
    lo, hi = 12, 20
    a, b = mx.array(base), mx.array(base)
    LT._roll(a, cfg, lo, hi, rolled, mx.array(pad))
    cols = b[:, :, lo:hi]
    q = LT._roundtrip(cols, 6)
    c = mx.arange(lo, hi)[None, :]
    todo = (c >= mx.array(rolled)[:, None]) & ((c - mx.array(pad)[:, None]) >= 4)
    b[:, :, lo:hi] = mx.where(todo[:, None, :, None], q, cols)
    assert mx.array_equal(a, b)
