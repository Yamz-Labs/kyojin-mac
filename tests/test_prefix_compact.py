"""Flag prefix_compact: a prefix snapshot (and its drafter ring) in buffers of exactly their size. Same bits, same hit; the pinned parent buffers go.
CPU only (the DS4.1 vendor cache class + the packed pool); what Metal does with a slice is measured by tools/prefix_mem_probe.py."""
import gc
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

pytest.importorskip("mlx_vlm.models.deepseek_v41")
from kyojin_mlx import fp4pack, fp4pool, rows as R, flags  # noqa: E402
from kyojin_mlx.fp4pool import KeySlots, PoolSlots, make_packed_class, own  # noqa: E402

MB = 2 ** 20


def grid(n, D, seed, block, e4):
    x = mx.array(np.random.default_rng(seed).standard_normal((1, n, D)).astype(np.float32)).astype(mx.float16)
    c, s = fp4pack.pack(x, block, e4)
    return fp4pack.unpack(c, s, block, e4, mx.float16)


def make_live(rows=3000, layers=3, steps=(1000, 700, 1300)):
    from mlx_vlm.models.deepseek_v41.language import DeepseekV41Cache
    c = make_packed_class(DeepseekV41Cache)(layers)
    for k, n in enumerate(steps):
        for l in range(layers):
            c.append_compressed(l, grid(n, 512, 10 * k + l, 16, True))
            c.append_index_keys(l, grid(n, 128, 100 + 10 * k + l, 32, False))
    big = mx.random.normal((1, 5000, 64)).astype(mx.float16)
    for l in range(layers):
        c.window[l] = big[:, l * 100:l * 100 + 128]                       # a slice of a bigger evaluated array
        c.kv_state[l] = mx.random.normal((1, 8, 64)).astype(mx.float32)
        c.score_state[l] = mx.random.normal((1, 8, 64)).astype(mx.float32)
    c.engram = mx.arange(sum(steps), dtype=mx.int64)[None]
    c.offset = sum(steps)
    c.compress_kv = c.compress[0]
    mx.eval(*[a for a in c.state])
    return c


def bits(a):
    return np.asarray(a.view(mx.uint8) if a.dtype != mx.uint8 else a)


def test_compact_snapshot_has_the_same_bits_and_the_same_hit(monkeypatch):
    monkeypatch.setattr(fp4pool, "SEG", 0)                                # the flat path; compact + kvseg is tests/test_kvseg.py
    c = make_live()
    plain = R.snapshot([c], compacted=False)[0]
    comp = R.snapshot([c], compacted=True)[0]
    for l in range(3):
        assert comp.compress.raw(l).codes.shape[1] == comp.compress.raw(l).n == 3000          # exactly the valid rows, no slack
        assert comp.compress.raw(l).mirror is None
    a, b = plain.state, comp.state                                   # (a dense read rebuilds a mirror: structure first)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.shape == y.shape and x.dtype == y.dtype and np.array_equal(bits(x), bits(y))
    assert comp._compress_buffers == [None] * 3 and comp.offset == plain.offset and comp.compress_kv is None
    from kyojin_mlx.serve.backends import arrays_of
    assert sum(x.nbytes for x in arrays_of(plain)) == sum(x.nbytes for x in arrays_of(comp))   # the accounted size does not move
    # a hit: fork, append, read -> identical to the plain snapshot's hit
    h1, h2 = R.fork([plain])[0], R.fork([comp])[0]
    new = grid(37, 512, 77, 16, True)
    k2 = grid(37, 128, 78, 32, False)
    for h in (h1, h2):
        for l in range(3):
            h.append_compressed(l, new)
            h.append_index_keys(l, k2)
    for l in range(3):
        assert np.array_equal(bits(h1.compress[l]), bits(h2.compress[l])) and np.array_equal(bits(h1.keys[l]), bits(h2.keys[l]))
        assert h2.compress[l].shape[1] == 3037
    # the snapshot does not move when the session goes on
    c.append_compressed(0, new)
    assert np.array_equal(bits(comp.compress[0]), bits(plain.compress[0]))


def held(f):
    gc.collect()
    mx.eval(mx.zeros(1))
    b0 = mx.get_active_memory()
    out = f()
    gc.collect()
    return (mx.get_active_memory() - b0) / MB, out


def test_compact_snapshot_releases_the_capacity_slack():
    """Held bytes of a snapshot after its live session is gone: counted bytes with the flag, more without (the buffers behind the slices)."""
    def build(compacted):
        c = make_live()
        s = R.snapshot([c], compacted=compacted)[0]
        del c
        return s
    from kyojin_mlx.serve.backends import arrays_of
    h0, s0 = held(lambda: build(False))
    h1, s1 = held(lambda: build(True))
    counted = sum(x.nbytes for x in arrays_of(s1)) / MB
    print(f"held plain {h0:.2f} MB, compact {h1:.2f} MB, counted {counted:.2f} MB")
    assert h1 <= counted * 1.02 + 0.05
    assert h0 > h1 + 0.5                                             # the slack the flag releases (capacity > rows on this fixture)


def test_own_is_bit_exact_and_not_a_view():
    a = mx.array([[[-0.0, float("nan"), float("inf"), 1.5], [2.0, -2.0, 3.0, 4.0]]], dtype=mx.float16)
    assert np.array_equal(bits(own(a)), bits(a))
    gc.collect()
    b0 = mx.get_active_memory()
    big = mx.random.normal((1, 40960, 512)).astype(mx.float16)
    mx.eval(big)
    s = big[:, -128:]
    mx.eval(s)
    del big
    gc.collect()
    pinned = (mx.get_active_memory() - b0) / MB
    o = own(s)
    mx.eval(o)
    del s
    gc.collect()
    kept = (mx.get_active_memory() - b0) / MB
    print(f"slice pins {pinned:.1f} MB for {o.nbytes / MB:.3f} MB of rows; own() keeps {kept:.3f} MB")
    assert pinned > 30 and kept < 0.5
    assert own(None) is None and own(mx.zeros((0, 3))).size == 0


def test_draft_snapshot_copies_the_ring_only_with_the_flag(monkeypatch):
    from kyojin_mlx.serve.backends import VlmBackend
    be = object.__new__(VlmBackend)
    be.R = R
    be._flush_lazy = lambda ds: None
    parent = mx.random.normal((1, 4096 + 128, 64)).astype(mx.float16)
    mx.eval(parent)
    d = SimpleNamespace(rows=[parent[:, -128:], None])
    monkeypatch.setattr(R, "PREFIX_COMPACT", False)
    off = be.draft_snapshot(d)
    assert off[0] is d.rows[0] or np.array_equal(bits(off[0]), bits(d.rows[0]))
    monkeypatch.setattr(R, "PREFIX_COMPACT", True)
    on = be.draft_snapshot(d)
    assert on[1] is None and np.array_equal(bits(on[0]), bits(d.rows[0])) and on[0].shape == (1, 128, 64)
    assert be.draft_nbytes(SimpleNamespace(rows=on)) == be.draft_nbytes(d)


def test_flag_registered_default_off_and_switches_the_module():
    f = flags.FLAGS["prefix_compact"]
    assert f.default == "0" and f.runtime
    prev = R.PREFIX_COMPACT
    try:
        flags.set("prefix_compact", "1")
        assert R.PREFIX_COMPACT is True
        flags.set("prefix_compact", "0")
        assert R.PREFIX_COMPACT is False
    finally:
        flags.set("prefix_compact", "1" if prev else "0")
