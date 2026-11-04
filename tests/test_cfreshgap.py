"""cfreshgap: pf_keys_lazy (fp4pool.LAZY_KEYS) = the deferred on-grid check of an index-key append (the pool already has it as pf_pool_lazy). CPU only; nothing is timed.
Default off: the checked state is untouched. On: same reads, same codes, same off-grid behaviour, same logits of a tiny DS4.1 chunk chain."""
import os
import sys

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from kyojin_mlx import depthflat as DF, fp4pack, fp4pool  # noqa: E402


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(DF, "ENABLED", True)
    monkeypatch.setattr(DF, "MIRROR_BYTES", 1 << 30)
    yield
    mx.set_default_device(dev)


def eq(a, b):
    v = lambda x: x.view({2: mx.uint16, 4: mx.uint32}[x.dtype.size]) if x.dtype.size in (2, 4) else x
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(v(a), v(b)).item())


def grid(n, seed, D=128):
    x = mx.array(np.random.default_rng(seed).standard_normal((1, n, D)).astype(np.float32)).astype(mx.float16)
    c, s = fp4pack.pack(x, 32, False)
    x = fp4pack.unpack(c, s, 32, False, mx.float16)
    return mx.where(x == 0, mx.array(-0.0, dtype=mx.float16), x)


def off_grid(D=128):
    return mx.array(np.random.default_rng(99).standard_normal((1, 5, D)).astype(np.float32)).astype(mx.float16)


def run(lazy, parts, monkeypatch):
    monkeypatch.setattr(fp4pool, "LAZY_KEYS", lazy)
    for k in fp4pool.HITS:
        fp4pool.HITS[k] = 0
    s = fp4pool.KeySlots([None])
    reads = [mx.array(s.append_rows(0, p)) for p in parts]
    return s, reads, dict(fp4pool.HITS)


def test_flag_default_is_off():
    from kyojin_mlx import flags
    assert flags.FLAGS["pf_keys_lazy"].default == "0" and fp4pool.LAZY_KEYS is False
    assert not fp4pool.KeySlots().lazy_check


def test_off_takes_no_deferred_check(monkeypatch):
    s, reads, h = run(False, [grid(100, 1), grid(30, 2), grid(300, 3)], monkeypatch)
    assert h["lazy"] == h["settle"] == 0 and s.raw(0).ok is None and len(s.raw_arrays()) == 2


def test_on_reads_and_codes_equal_the_checked_append(monkeypatch):
    parts = [grid(300, 1), grid(40, 2), grid(1, 3), grid(70, 4), grid(256, 5), grid(9, 6)]
    s0, r0, _ = run(False, parts, monkeypatch)
    s1, r1, h = run(True, parts, monkeypatch)
    assert h["lazy"] == len(parts) - 1 and h["settle_dense"] == 0
    assert all(eq(a, b) for a, b in zip(r0, r1)) and eq(s0[0], s1[0])
    e0, e1 = s0.raw(0), s1.raw(0)
    assert e0.n == e1.n and eq(e0.codes[:, :e0.n], e1.codes[:, :e1.n]) and eq(e0.scales[:, :e0.n], e1.scales[:, :e1.n])


@pytest.mark.parametrize("where", [1, 2, 3])
def test_on_off_grid_append_goes_dense_like_the_checked_append(where, monkeypatch):
    parts = [grid(100, 1), grid(30, 2), grid(20, 3), grid(10, 4)]
    parts[where] = off_grid()
    s0, r0, _ = run(False, parts, monkeypatch)
    s1, r1, _ = run(True, parts, monkeypatch)
    assert all(eq(a, b) for a, b in zip(r0, r1)) and eq(s0[0], s1[0])
    assert not isinstance(s0.raw(0), fp4pool._Pk) and not isinstance(s1.raw(0), fp4pool._Pk)
    assert fp4pool.HITS["settle_dense"] >= 1
    s0.append_rows(0, grid(4, 8)); s1.append_rows(0, grid(4, 8))
    assert eq(s0[0], s1[0])


def test_on_segment_snapshot_settles_first(monkeypatch):
    monkeypatch.setattr(fp4pool, "SEG", 32)
    s, _, h = run(True, [grid(100, 1), grid(30, 2), grid(300, 3)], monkeypatch)
    live = s[0]
    snap = s.segment_snapshot()
    assert fp4pool.HITS["seal"] > 0 and eq(mx.array(snap[0]), live) and eq(snap.copy()[0], live)


def test_tiny_model_chunk_chain_logits_equal(monkeypatch):
    pytest.importorskip("mlx_vlm.models.deepseek_v41")
    import mini_ds41 as M
    for dt in (mx.float16, mx.float32):
        lm, _ = M.make(dtype=dt)
        ids = M.tokens(300, seed=5)
        res = {}
        for on in (False, True):
            monkeypatch.setattr(fp4pool, "LAZY_KEYS", on)
            cache = fp4pool.wrap_cache(lm.make_cache())
            for k in fp4pool.HITS:
                fp4pool.HITS[k] = 0
            outs, pos = [], 0
            for T in (70, 1, 33, 64, 5, 90, 37):
                outs.append(lm(mx.array([ids[pos:pos + T]]), cache=cache).logits)
                pos += T
            mx.eval(outs)
            res[on] = (outs, [mx.array(a) for a in cache[0].keys if a is not None], fp4pool.HITS["lazy"])
        assert all(eq(a, b) for a, b in zip(res[False][0], res[True][0]))
        assert all(eq(a, b) for a, b in zip(res[False][1], res[True][1]))
        assert res[False][2] == 0
