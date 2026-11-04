"""cpfbase: two exact levers of the non-index prefill part, CPU only (no Metal kernel in either).
pf_win_slice (pf_ring): window ring read / write by slices against the vendor Gather / Scatter, then through the two kyojin `_window_part`s and a chain of chunks.
pf_pool_lazy (fp4pool): deferred on-grid check of a pool append against the checked append: reads, raw arrays, copies, an off-grid write, a whole tiny model."""
import os
import sys

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
pytest.importorskip("mlx_vlm.models.deepseek_v41")
import mini_ds41 as M  # noqa: E402
from kyojin_mlx import depthflat as DF, fp4pack, fp4pool, pf_ring as R  # noqa: E402


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(DF, "ENABLED", True)
    monkeypatch.setattr(DF, "MIRROR_BYTES", 1 << 30)
    yield
    mx.set_default_device(dev)


def bits(a):
    return a.view({2: mx.uint16, 4: mx.uint32}[a.dtype.size]) if a.dtype.size in (2, 4) else a


def eq(a, b):
    """Bit for bit (-0.0 differs from +0.0)."""
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(bits(a), bits(b)).item())


# ------------------------------------------------------------------ pf_win_slice
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_ring_read_write_equal_the_gather_and_the_scatter(dtype):
    rng = np.random.default_rng(0)
    n = 0
    for win in (1, 2, 5, 16):
        for B, batch in ((1, 1), (3, 2)):
            for start in range(0, 3 * win + 3):
                for T in (1, 2, 3, win, win + 1, 2 * win + 3):
                    buf = mx.array(rng.standard_normal((B, win, 4)).astype(dtype))
                    kv = mx.array(rng.standard_normal((batch, T, 4)).astype(dtype))
                    base, end, r = max(0, start - win + 1), start + T, min(T, win)
                    if start:
                        assert eq(R.read(buf, batch, base, start, win), mx.take(buf[:batch], mx.arange(base, start) % win, axis=1))
                    ref = mx.array(buf)
                    ref[:batch, mx.arange(end - r, end) % win] = kv[:, -r:]
                    assert eq(R.write(buf, batch, kv[:, -r:], end, win), ref), (win, B, batch, start, T)
                    n += 1
    assert n > 500


def test_ring_write_casts_the_tail_to_the_ring_dtype_like_the_assignment():
    buf = mx.zeros((1, 8, 4), dtype=mx.float16)
    kv = mx.random.normal((1, 5, 4))
    ref = mx.array(buf)
    ref[:1, mx.arange(3, 8) % 8] = kv
    assert eq(R.write(buf, 1, kv, 8, 8), ref)


def _attn():
    lm, cfg = M.make(dtype=mx.float16)
    return lm.layers[2].attn, lm


@pytest.mark.parametrize("which", ["attn_prefill", "fused_group"])
def test_window_part_chain_is_bit_equal_with_the_flag(which, monkeypatch):
    """Chunks of odd lengths over several window wraps (sliding_window = 16), fp16: window kv, mask, and the ring after every chunk."""
    from kyojin_mlx import attn_prefill as AP, fused_group as FG
    import mlx_vlm.models.deepseek_v41.language as L
    attn, lm = _attn()
    cls = type(attn)
    orig = cls._window_part
    if which == "fused_group":
        assert FG.share_window_math()
        fn = lambda a, x, s, c: cls._window_part(a, x, s, c)      # noqa: E731
    else:
        fn = lambda a, x, s, c: AP.window_part(a, x, s, c, need_mask=True)    # noqa: E731
    try:
        outs = {}
        for flag in (False, True):
            monkeypatch.setattr(R, "ENABLED", flag)
            cache = L.DeepseekV41Cache(8)
            pos, res = 0, []
            for T in (40, 1, 17, 16, 15, 3, 70, 2, 33):
                x = mx.random.normal((1, T, cfg_hidden(lm)), key=mx.random.key(pos)).astype(mx.float16)
                part, mask = fn(attn, x, pos, cache)
                res.append((part, mask, mx.array(cache.window[attn.layer_idx])))
                pos += T
            outs[flag] = res
        for (p0, m0, w0), (p1, m1, w1) in zip(outs[False], outs[True]):
            assert eq(p0, p1) and eq(w0, w1) and eq(m0, m1)
        assert R.HITS["write"] > 0
    finally:
        cls._window_part = orig


def cfg_hidden(lm):
    return lm.layers[0].attn.wq_a.weight.shape[-1]


# ------------------------------------------------------------------ pf_pool_lazy
def grid(n, D=64, seed=0, block=16, e4=True):
    x = mx.array(np.random.default_rng(seed).standard_normal((1, n, D)).astype(np.float32)).astype(mx.float16)
    c, sc = fp4pack.pack(x, block, e4)
    x = fp4pack.unpack(c, sc, block, e4, mx.float16)
    return mx.where(x == 0, mx.array(-0.0, dtype=mx.float16), x)                 # negative zeros on purpose: on the grid (-0.0 == +0.0) but the codes read back as +0.0


def raw_rows(D=64, seed=99):
    return mx.array(np.random.default_rng(seed).standard_normal((1, 5, D)).astype(np.float32)).astype(mx.float16)     # off the grid


def run_slots(lazy, parts, monkeypatch):
    monkeypatch.setattr(fp4pool, "LAZY_POOL", lazy)
    for k in fp4pool.HITS:
        fp4pool.HITS[k] = 0
    s = fp4pool.PoolSlots([None])
    reads = []
    for p in parts:
        reads.append(mx.array(s.append_rows(0, p)))
    return s, reads, dict(fp4pool.HITS)


def test_lazy_pool_append_reads_the_same_rows_and_keeps_the_codes(monkeypatch):
    parts = [grid(300, seed=1), grid(40, seed=2), grid(1, seed=3), grid(70, seed=4), grid(256, seed=5), grid(9, seed=6)]
    s0, r0, _ = run_slots(False, parts, monkeypatch)
    s1, r1, h = run_slots(True, parts, monkeypatch)
    assert h["lazy"] == len(parts) - 1 and h["settle_dense"] == 0               # the first append has no mirror yet: checked
    for a, b in zip(r0, r1):
        assert eq(a, b)
    assert eq(s0[0], s1[0])
    e0, e1 = s0.raw(0), s1.raw(0)                                               # raw() settles
    assert e0.n == e1.n and eq(e0.codes[:, :e0.n], e1.codes[:, :e1.n]) and eq(e0.scales[:, :e0.n], e1.scales[:, :e1.n])
    assert fp4pool.HITS["settle"] >= 1


def test_lazy_pool_raw_arrays_carry_the_verdict_and_copy_settles(monkeypatch):
    parts = [grid(100, seed=1), grid(30, seed=2), grid(30, seed=3)]
    s, _, _ = run_slots(True, parts, monkeypatch)
    arrs = s.raw_arrays()
    assert len(arrs) == 3                                                       # codes, scales, deferred verdict
    mx.eval(*arrs)
    c = s.copy()
    assert fp4pool.HITS["settle"] >= 1 and eq(c[0], s[0])


@pytest.mark.parametrize("where", [1, 2, 3])
def test_lazy_pool_off_grid_append_goes_dense_exactly_like_the_checked_append(where, monkeypatch):
    parts = [grid(100, seed=1), grid(30, seed=2), grid(20, seed=3), grid(10, seed=4)]
    parts[where] = raw_rows()
    s0, r0, _ = run_slots(False, parts, monkeypatch)
    s1, r1, h = run_slots(True, parts, monkeypatch)
    for a, b in zip(r0, r1):
        assert eq(a, b)
    assert eq(s0[0], s1[0])
    assert isinstance(s1.raw(0), fp4pool._Pk) == isinstance(s0.raw(0), fp4pool._Pk) is False       # both layers went dense
    assert fp4pool.HITS["settle_dense"] >= 1
    s1.append_rows(0, grid(4, seed=8))
    s0.append_rows(0, grid(4, seed=8))
    assert eq(s0[0], s1[0])


def test_lazy_pool_is_pool_only(monkeypatch):
    for kk in fp4pool.HITS:
        fp4pool.HITS[kk] = 0
    monkeypatch.setattr(fp4pool, "LAZY_POOL", True)
    k = fp4pool.KeySlots([None])
    k.append_rows(0, grid(100, 128, 1, 32, False))
    k.append_rows(0, grid(10, 128, 2, 32, False))
    assert fp4pool.HITS["lazy"] == 0 and not k.lazy_check


def test_whole_tiny_model_chunk_chain_logits_equal_with_both_flags(monkeypatch):
    """fp16 and fp32 tiny DS4.1, packed pool cache, unequal chunks over several window wraps: logits and the dense cache are bit-equal with both flags on."""
    from kyojin_mlx import attn_prefill as AP, fused_group as FG
    import mlx_vlm.models.deepseek_v41.language as L
    for dt in (mx.float16, mx.float32):
        lm, cfg = M.make(dtype=dt)
        base = type(lm.layers[2].attn)
        orig = base._window_part
        FG.share_window_math()
        ids = M.tokens(300, seed=5)
        res = {}
        try:
            for flags in ((False, False), (True, True)):
                monkeypatch.setattr(R, "ENABLED", flags[0])
                monkeypatch.setattr(fp4pool, "LAZY_POOL", flags[1])
                cache = fp4pool.wrap_cache(lm.make_cache())
                assert isinstance(cache[0].compress, fp4pool.PoolSlots)
                for d in (R.HITS, fp4pool.HITS):
                    for kk in d:
                        d[kk] = 0
                outs, pos = [], 0
                for T in (70, 1, 33, 64, 5, 90, 37):
                    outs.append(lm(mx.array([ids[pos:pos + T]]), cache=cache).logits)
                    pos += T
                mx.eval(outs)
                state = [mx.array(a) for a in cache[0].compress if a is not None]
                res[flags] = (outs, state)
                if flags[0]:
                    assert R.HITS["write"] > 0 and R.HITS["read"] > 0 and fp4pool.HITS["lazy"] > 0 and fp4pool.HITS["settle_dense"] == 0, (R.HITS, fp4pool.HITS)
        finally:
            base._window_part = orig
        (o0, s0), (o1, s1) = res[(False, False)], res[(True, True)]
        assert all(eq(a, b) for a, b in zip(o0, o1)) and len(s0) == len(s1) and all(eq(a, b) for a, b in zip(s0, s1))


# ------------------------------------------------------------------ pf_pool_lazy x KYOJIN_KVSEG (the int1 lever)
def test_pool_flag_off_leaves_the_checked_state(monkeypatch):
    """Flag off: no deferred check is taken (the codes, the state arrays and every read stay the checked append's), with the segmented snapshots on too."""
    monkeypatch.setattr(fp4pool, "SEG", 32)
    s, reads, h = run_slots(False, [grid(100, seed=1), grid(30, seed=2), grid(300, seed=3)], monkeypatch)
    assert h["lazy"] == h["settle"] == h["settle_dense"] == 0
    e = s.raw(0)
    assert e.ok is None and len(s.raw_arrays()) == 2                       # codes, scales: nothing pending
    assert eq(reads[-1], fp4pack.unpack(e.codes[:, :e.n], e.scales[:, :e.n], 16, True, mx.float16))
    snap = s.segment_snapshot()
    assert fp4pool.HITS["seal"] > 0 and eq(mx.array(snap[0]), s[0])


def test_lazy_pending_layer_is_settled_by_a_segmented_snapshot(monkeypatch):
    """pf_pool_lazy + KYOJIN_KVSEG: a snapshot resolves the deferred checks first, so its segments and tail hold exactly the rows the live layer read; the thawed copy and
    the copy of the snapshot too (a pending layer's codes are the round trip's bits, not necessarily its rows)."""
    monkeypatch.setattr(fp4pool, "SEG", 32)
    s, _, h = run_slots(True, [grid(100, seed=1), grid(30, seed=2), grid(300, seed=3)], monkeypatch)
    assert h["lazy"] == 2 and h["settle"] == 0                              # two appends deferred, nothing read raw yet
    live = s[0]                                                            # a pending layer reads through its mirror
    assert isinstance(s.raw(0), fp4pool._Pk)                                # (this settles both checks; the codes were on the grid)
    snap = s.segment_snapshot()
    assert fp4pool.HITS["settle"] == 1 and fp4pool.HITS["seal"] > 0 and fp4pool.HITS["settle_dense"] == 0
    assert eq(mx.array(snap[0]), live) and eq(snap.copy()[0], live)
