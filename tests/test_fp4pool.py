import os, sys
import mlx.core as mx
import numpy as np
sys.path.insert(0, os.path.expanduser("~/kyojin-mlx/vlm_pkg"))
from kyojin_mlx import fp4pack, fp4pool, rows as R
from kyojin_mlx.fp4pool import PoolSlots, KeySlots, make_packed_class
from kyojin_mlx.serve.backends import arrays_of


def grid(n, D=512, seed=0, block=16, e4=True):
    x = mx.array(np.random.default_rng(seed).standard_normal((1, n, D)).astype(np.float32)).astype(mx.float16)
    c, s = fp4pack.pack(x, block, e4)
    return fp4pack.unpack(c, s, block, e4, mx.float16)


def test_append_and_read_bit_identical():
    s = PoolSlots([None, None])
    parts = [grid(5, seed=1), grid(300, seed=2), grid(1, seed=3)]
    dense = None
    for p in parts:
        out = s.append_rows(0, p)
        dense = p if dense is None else mx.concatenate([dense, p], axis=1)
        assert mx.array_equal(out, dense)
    assert mx.array_equal(s[0], dense) and s[1] is None
    assert sum(a.nbytes for a in arrays_of(s)) < dense.nbytes * 0.32


def test_index_keys_block32_pow2():
    k = grid(70, D=128, seed=4, block=32, e4=False)
    s = KeySlots([None])
    s.append_rows(0, k[:, :40]); s.append_rows(0, k[:, 40:])
    assert mx.array_equal(s[0], k) and len(arrays_of(s)) == 2


def test_off_grid_stays_dense_and_exact():
    s = PoolSlots([None])
    g = grid(8, seed=5)
    s.append_rows(0, g)
    raw = mx.array(np.random.default_rng(6).standard_normal((1, 3, 512)).astype(np.float32)).astype(mx.float16)
    out = s.append_rows(0, raw)
    assert mx.array_equal(out, mx.concatenate([g, raw], axis=1))
    s[0] = raw
    assert mx.array_equal(s[0], raw)


def test_copy_is_independent_and_fork_and_iter():
    s = PoolSlots([grid(10, seed=7), None])
    c = R.fork([s])[0]
    assert isinstance(c, PoolSlots)
    c.append_rows(0, grid(4, seed=8))
    assert s[0].shape[1] == 10 and c[0].shape[1] == 14
    assert [x is None for x in s] == [False, True]
    s[:] = [grid(3, seed=9), grid(2, seed=10)]
    assert s[1].shape[1] == 2


def test_vendor_cache_roundtrip():
    from mlx_vlm.models.deepseek_v41.language import DeepseekV41Cache
    P = make_packed_class(DeepseekV41Cache)
    d, p = DeepseekV41Cache(2), P(2)
    for c in (d, p):
        c.append_compressed(0, grid(20, seed=11)); c.append_compressed(0, grid(4, seed=12)); c.append_compressed(1, grid(7, seed=13))
        c.append_index_keys(1, grid(9, D=128, seed=14, block=32, e4=False))
        c.window[0] = mx.ones((1, 8, 4), dtype=mx.float16); c.advance(5)
    ds, ps = d.state, p.state
    assert len(ds) == len(ps) and all(mx.array_equal(a, b) for a, b in zip(ds, ps))
    q = P.from_state(ps, p.meta_state)
    assert isinstance(q.compress, PoolSlots) and all(mx.array_equal(a, b) for a, b in zip(q.state, ds))
    e = p.extract(0)
    assert isinstance(e, P) and mx.array_equal(e.compress[0], d.compress[0])
    assert sum(a.nbytes for a in arrays_of(p.compress)) < sum(a.nbytes for a in d.compress if a is not None) * 0.33
    f = R.fork(p)
    f.append_compressed(1, grid(2, seed=15))
    assert p.compress[1].shape[1] == 7 and f.compress[1].shape[1] == 9


def _extract_cache(n=300):
    from mlx_vlm.models.deepseek_v41.language import DeepseekV41Cache
    c = make_packed_class(DeepseekV41Cache)(3)
    c.offset = n
    c.compress.append_rows(0, grid(n, seed=21))
    c.keys.append_rows(0, grid(n, D=128, seed=22, block=32, e4=False))
    c.compress[1] = mx.array(np.random.default_rng(23).standard_normal((1, 7, 512)).astype(np.float32)).astype(mx.float16)   # off grid: stays dense
    c.window[0] = grid(4, seed=24)
    c.engram = mx.zeros((1, 2), dtype=mx.int32)
    return c


def _same(a, b):
    assert a.offset == b.offset
    for x, y in zip(a.state, b.state):
        assert (x is None and y is None) or mx.array_equal(x, y)


def test_extract_raw_equals_vendor_extract(monkeypatch):
    c = _extract_cache()
    monkeypatch.setenv("KYOJIN_EXTRACT_RAW", "0")
    ref = c.extract(0)
    monkeypatch.setenv("KYOJIN_EXTRACT_RAW", "1")
    calls = []
    orig = fp4pool.fp4pack.pack
    monkeypatch.setattr(fp4pool.fp4pack, "pack", lambda *a, **k: (calls.append(1), orig(*a, **k))[1])
    got = c.extract(0)
    assert not calls, "the layers were packed again"
    assert isinstance(got.compress, PoolSlots) and isinstance(got.keys, KeySlots)
    assert got.compress.raw(0) is not c.compress.raw(0)
    _same(got, ref)


def test_extract_raw_copy_is_independent(monkeypatch):
    monkeypatch.setenv("KYOJIN_EXTRACT_RAW", "1")
    c = _extract_cache()
    got = c.extract(0)
    before = mx.array(c.compress[0])
    extra = grid(16, seed=25)
    got.compress.append_rows(0, extra)
    mx.eval(got.compress[0])
    assert mx.array_equal(c.compress[0], before), "an append to the extract moved the source"
    assert mx.array_equal(got.compress[0], mx.concatenate([before, extra], axis=1))
    c.compress.append_rows(0, grid(16, seed=26))
    assert mx.array_equal(got.compress[0], mx.concatenate([before, extra], axis=1)), "an append to the source moved the extract"


def test_extract_raw_keeps_batched_entries_on_the_vendor_path(monkeypatch):
    monkeypatch.setenv("KYOJIN_EXTRACT_RAW", "1")
    c = _extract_cache()
    c.compress[2] = mx.concatenate([grid(8, seed=27), grid(8, seed=28)], axis=0)
    got = c.extract(0)
    assert got.compress[2].shape[0] == 1 and mx.array_equal(got.compress[2], c.compress[2][:1])
