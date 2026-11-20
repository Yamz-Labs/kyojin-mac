# SPDX-License-Identifier: Apache-2.0
"""KYOJIN_HY4_FIXED_KV: the MLA attention on fixed_kv.FixedKVCache equals the same attention on the concatenating KVCache (SWA window, kpool, plain), across a
chunked prefill, a buffer growth and decode steps, eager and as one mx.compile program. CPU only (the Metal part is behind KYOJIN_METAL_TESTS=1)."""
import os
import sys

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kyojin_mlx import fixed_kv as FK  # noqa: E402
from kyojin_mlx import foreign_skeleton_mla as MLA  # noqa: E402
from test_hy4_attn_slice import _fill, _kp_model, _swa_model, _chunks  # noqa: E402

TOL = 2e-6
METAL = os.environ.get("KYOJIN_METAL_TESTS") == "1"


def _ref(m, xs, window=0):
    from mlx_lm.models.base import create_causal_mask
    from mlx_lm.models.cache import KVCache
    c, outs = KVCache(), []
    for x in xs:
        L = x.shape[1]
        mask = create_causal_mask(L, c.offset, window_size=window) if window else ("causal" if L > 1 else None)
        outs.append(m(x, mask=mask, cache=c))
    mx.eval(outs)
    return outs


def _fixed(m, xs, chunk, compiled, pk=False, trims=None, ring=False):
    """trims: {step index: rows} -> the cache is rewound by that many rows before the step (speculation rollback), the same rows are then re-fed by the caller's next xs."""
    at = m
    kp = at.kp
    c = FK.FixedKVCache(at.kv + at.r + (2 * kp["dx"] if kp else 0), kp["kp"] if kp else 1, at.window if at.swa else 0, chunk, pk_dx=kp["dx"] if kp and pk else 0, ring=ring)
    outs, fn = [], None
    for i, x in enumerate(xs):
        L = x.shape[1]
        if trims and i in trims:
            c.trim(trims[i])
        c.ensure(c._n + L, x.dtype, L)
        if compiled and L == 1:
            if fn is None:
                def fn(x_, *st):
                    sh = FK.FixedKVCache.__new__(FK.FixedKVCache)
                    sh.__dict__.update(c.__dict__)
                    sh.cache = list(st)
                    o = m(x_, mask=None, cache=sh)
                    return (o, *sh.cache)
                fn = mx.compile(fn)
            o, *st = fn(x, *c.cache)
            c.cache[:] = st
        else:
            o = m(x, mask=None, cache=c)
        c._n += L
        outs.append(o)
    mx.eval(outs, c.cache)
    assert int(c.cache[1].item()) == c._n
    return outs


def _diff(a, b):
    mag = max(1.0, max(float(mx.max(mx.abs(x)).item()) for x in a))
    return max(float(mx.max(mx.abs(x - y)).item()) for x, y in zip(a, b)) / mag


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("kind", ["plain", "swa", "kpool"])
def test_fixed_equals_concat(kind, compiled):
    rng = np.random.default_rng(3)
    d = 64
    if kind == "swa":
        m = _swa_model(rng, d, 4, 32, 24, 8, 16, 40)
    elif kind == "kpool":
        m = _kp_model(rng, d, 4, 32, 24, 8, 16, 2, 16, top_k=32)
        m.swa, m.window = False, 0
    else:
        m = _fill(MLA.MlaAttention(d, 4, 48, 32, 24, 8, 16, sinks=True, gate=True), rng)
        m.swa, m.window = False, 0
    xs = _chunks(rng, d, 150, 6)                      # 150 rows: two prefill chunks of 512? no: one chunk of 150, then 6 decode rows
    xs = [xs[0][:, :90], xs[0][:, 90:]] + xs[1:]       # a ragged two-chunk prefill; chunk 64 -> the buffer grows twice
    ref = _ref(m, xs, m.window)
    new = _fixed(m, xs, 64, compiled)
    dd = _diff(ref, new)
    print(f"fixed_kv {kind} compiled={compiled}: max rel diff {dd:.3g}")
    assert dd <= TOL


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("split", [(90, 60), (30, 100, 20)])
def test_ring_equals_concat(split, compiled, monkeypatch):
    """KYOJIN_HY4_SWA_RING: the windowed layer on a ring (slot = position % rows; ring rows 8-aligned here so the 60 decode rows wrap it, and a longer second chunk rebuilds it)
    equals the concatenating cache."""
    monkeypatch.setattr(FK, "RING_ALIGN", 8)
    monkeypatch.setattr(FK, "RING_ROWS", 8)
    rng = np.random.default_rng(11)
    d = 64
    m = _swa_model(rng, d, 4, 32, 24, 8, 16, 40)
    pre, *dec = _chunks(rng, d, sum(split), 60)
    xs, a = [], 0
    for n in split:
        xs.append(pre[:, a:a + n]); a += n
    xs += dec
    ref = _ref(m, xs, m.window)
    new = _fixed(m, xs, 64, compiled, ring=True)
    dd = _diff(ref, new)
    print(f"ring_kv split={split} compiled={compiled}: max rel diff {dd:.3g}")
    assert dd <= TOL


@pytest.mark.parametrize("compiled", [False, True])
def test_ring_with_verify_blocks_and_rollback_equals_plain_fixed(compiled, monkeypatch):
    """The MTP shape on a wrapping ring: 6-row verify blocks, a rollback of 2-4 rows before some of them, single decode rows between: equal to the non-ring fixed cache."""
    monkeypatch.setattr(FK, "RING_ALIGN", 8)
    monkeypatch.setattr(FK, "RING_ROWS", 8)
    rng = np.random.default_rng(13)
    d = 64
    m = _swa_model(rng, d, 4, 32, 24, 8, 16, 40)
    xs = _chunks(rng, d, 100, 0)
    trims = {}
    for i in range(30):
        n = 6 if i % 3 else 1
        xs.append(mx.array(rng.standard_normal((1, n, d)).astype(np.float32)))
        if i % 4 == 2:
            trims[len(xs) - 1] = 2 + i % 3
    ref = _fixed(m, xs, 64, compiled, trims=trims)
    new = _fixed(m, xs, 64, compiled, trims=trims, ring=True)
    dd = _diff(ref, new)
    print(f"ring_kv verify blocks + rollback compiled={compiled}: max rel diff {dd:.3g}")
    assert dd <= TOL


def test_ring_rows_stay_small(monkeypatch):
    monkeypatch.setattr(FK, "RING_ALIGN", 8)
    monkeypatch.setattr(FK, "RING_ROWS", 8)
    rng = np.random.default_rng(12)
    m = _swa_model(rng, 64, 4, 32, 24, 8, 16, 40)
    xs = _chunks(rng, 64, 150, 60)
    c = FK.FixedKVCache(m.kv + m.r, 1, m.window, 64, ring=True)
    for x in xs:
        c.ensure(c._n + x.shape[1], x.dtype)
        m(x, mask=None, cache=c)
        c._n += x.shape[1]
    assert c.cache[0].shape[2] <= 208 and c._n >= 210, c.cache[0].shape


@pytest.mark.parametrize("kind", ["plain", "swa", "kpool"])
def test_fixed_prefill_query_blocks_equal_concat(kind, monkeypatch):
    """Prefill chunks (> 16 rows) on the fixed-state cache take the query-block attention (KYOJIN_HY4_ATTN_QBLK / _SWA) and the fused chain: equal to the stock concatenating path."""
    rng = np.random.default_rng(5)
    d = 64
    if kind == "swa":
        m = _swa_model(rng, d, 4, 32, 24, 8, 16, 40)
    elif kind == "kpool":
        m = _kp_model(rng, d, 4, 32, 24, 8, 16, 2, 16, top_k=32)
        m.swa, m.window = False, 0
    else:
        m = _fill(MLA.MlaAttention(d, 4, 48, 32, 24, 8, 16, sinks=True, gate=True), rng)
        m.swa, m.window = False, 0
    xs = _chunks(rng, d, 150, 6)
    xs = [xs[0][:, :90], xs[0][:, 90:]] + xs[1:]
    ref = _ref(m, xs, m.window)
    monkeypatch.setenv("KYOJIN_HY4_ATTN_QBLK", "32")
    monkeypatch.setenv("KYOJIN_HY4_ATTN_QBLK_SWA", "16")
    monkeypatch.setenv("KYOJIN_HY4_ATTN_FUSE", "1")
    monkeypatch.setenv("KYOJIN_HY4_SWA_SLICE", "1")
    n0 = MLA.MlaAttention.PATHS["qblocks"]
    new = _fixed(m, xs, 64, False)
    assert MLA.MlaAttention.PATHS["qblocks"] > n0, "the query-block path did not run on the fixed-state cache"
    dd = _diff(ref, new)
    print(f"fixed_kv qblocks {kind}: max rel diff {dd:.3g}")
    assert dd <= 1e-5
    if kind == "swa":                                                 # the same prefill on the ring (served QBLK_SWA / SWA_SLICE / ATTN_FUSE paths)
        monkeypatch.setattr(FK, "RING_ALIGN", 8)
        monkeypatch.setattr(FK, "RING_ROWS", 8)
        dd = _diff(ref, _fixed(m, xs, 64, False, ring=True))
        print(f"ring qblocks swa: max rel diff {dd:.3g}")
        assert dd <= 1e-5


def test_fork_is_independent_and_continues_the_same():
    """FixedKVCache.fork (prefix snapshot): the copy and the source decode the same rows to the same outputs, and writes to one never show in the other."""
    rng = np.random.default_rng(9)
    d = 64
    m = _fill(MLA.MlaAttention(d, 4, 48, 32, 24, 8, 16, sinks=True, gate=True), rng)
    m.swa, m.window = False, 0
    xs = _chunks(rng, d, 70, 3)
    c = FK.FixedKVCache(m.kv + m.r, 1, 0, 64)
    c.ensure(c._n + 70, xs[0].dtype)
    m(xs[0], mask=None, cache=c)
    c._n += 70
    mx.eval(c.cache)
    f = c.fork()
    assert f._n == c._n and f is not c and f.cache[0] is not c.cache[0]
    outs = []
    for cc in (c, f):
        cc.ensure(cc._n + 1, xs[0].dtype)
        o = m(xs[1][:, :1], mask=None, cache=cc)
        cc._n += 1
        outs.append(o)
    mx.eval(outs, c.cache, f.cache)
    assert _diff([outs[0]], [outs[1]]) == 0.0
    g = f.fork()                                                              # a write on f after the fork leaves g's rows alone
    f.ensure(f._n + 1, xs[0].dtype)
    m(xs[2][:, :1], mask=None, cache=f)
    f._n += 1
    mx.eval(f.cache, g.cache)
    assert g._n == f._n - 1 and int(g.cache[1].item()) == g._n


@pytest.mark.parametrize("nosort", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_fixed_gather_attend_equals_concat(compiled, nosort, monkeypatch):
    """KYOJIN_HY4_FIXED_GATHER=1 / _DEC=1 (kpool layer, fixed-state cache): the attend over only the indexer's kept keys equals the masked attend of the concatenating cache, on the prefill
    chunks and on the decode rows (compiled or not)."""
    rng = np.random.default_rng(11)
    d = 64
    m = _kp_model(rng, d, 4, 32, 24, 8, 16, 2, 16, top_k=32)
    m.swa, m.window = False, 0
    xs = _chunks(rng, d, 150, 6)
    xs = [xs[0][:, :90], xs[0][:, 90:]] + xs[1:]
    ref = _ref(m, xs, 0)
    monkeypatch.setenv("KYOJIN_HY4_FIXED_GATHER", "1")
    monkeypatch.setenv("KYOJIN_HY4_FIXED_GATHER_DEC", "1")
    monkeypatch.setenv("KYOJIN_HY4_GATHER_NOSORT", "1" if nosort else "0")          # decode rows keep the kept slots unsorted (the attend masks by ok)
    n0 = MLA.MlaAttention.PATHS.get("fixed_gather", 0)
    new = _fixed(m, xs, 64, compiled)
    assert MLA.MlaAttention.PATHS.get("fixed_gather", 0) > n0, "the gather path did not run on the fixed-state cache"
    dd = _diff(ref, new)
    print(f"fixed_kv gather compiled={compiled}: max rel diff {dd:.3g}")
    assert dd <= 1e-5


@pytest.mark.parametrize("gather,gather_dec,want", [("0", "0", {"hy4_gather.fb.off"}), ("1000000", "1000000", {"hy4_gather.fb.below_min"}), ("1", "1", {"hy4_gather.fire"})])
def test_fixed_gather_fires_counters(gather, gather_dec, want, monkeypatch):
    """Fires log: hy4_gather counts fire / off / below_min on the fixed-state cache (CPU, eager), hy4_pkcache counts the carried pooled keys."""
    from kyojin_mlx import fired as FD
    monkeypatch.setattr(FD, "AUDIT", True)
    monkeypatch.setattr(FD, "LOG", False)
    FD.reset()
    rng = np.random.default_rng(11)
    d = 64
    m = _kp_model(rng, d, 4, 32, 24, 8, 16, 2, 16, top_k=32)
    m.swa, m.window = False, 0
    xs = _chunks(rng, d, 150, 4)
    xs = [xs[0][:, :90], xs[0][:, 90:]] + xs[1:]
    monkeypatch.setenv("KYOJIN_HY4_FIXED_GATHER", gather)
    monkeypatch.setenv("KYOJIN_HY4_FIXED_GATHER_DEC", gather_dec)
    _fixed(m, xs, 64, False, pk=True)
    assert {k for k in FD.STATS if k.startswith("hy4_gather")} == want, FD.STATS
    assert FD.STATS.get("hy4_pkcache.fire", 0) > 0 and "hy4_pkcache.fb.no_cache" not in FD.STATS
    FD.reset()
    _fixed(m, xs, 64, False, pk=False)
    assert FD.STATS.get("hy4_pkcache.fb.no_cache", 0) > 0 and "hy4_pkcache.fire" not in FD.STATS
    FD.reset()


@pytest.mark.parametrize("compiled", [False, True])
def test_fixed_pooled_key_cache_equals_recompute(compiled):
    """KYOJIN_HY4_FIXED_PKCACHE (pooled keys carried in cache[2], updated by the groups a call completes) equals the per-call recompute: ragged prefill chunks, a rewind into the middle of a
    group (speculation rollback), decode rows; the cached pools are bit-equal to pool_keys."""
    rng = np.random.default_rng(13)
    d = 64
    m = _kp_model(rng, d, 4, 32, 24, 8, 16, 2, 16, top_k=32)
    m.swa, m.window = False, 0
    xs = _chunks(rng, d, 150, 7)
    xs = [xs[0][:, :90], xs[0][:, 90:]] + xs[1:]
    ref = _fixed(m, xs, 64, compiled)
    new = _fixed(m, xs, 64, compiled, pk=True)
    dd = _diff(ref, new)
    print(f"fixed_kv pk cache compiled={compiled}: max rel diff {dd:.3g}")
    assert dd <= 1e-6
    # the rewind: trim 3 rows before decode step 4 (inside a group of 4), refeed rows with the same inputs
    xs2 = xs[:-3] + [xs[-4], xs[-3], xs[-2], xs[-1]]
    a = _fixed(m, xs2, 64, compiled, trims={len(xs) - 3: 3})
    b = _fixed(m, xs2, 64, compiled, pk=True, trims={len(xs) - 3: 3})
    assert _diff(a, b) <= 1e-6


@pytest.mark.parametrize("compiled", [False, True])
def test_fixed_pooled_key_cache_fp16(compiled, monkeypatch):
    """KYOJIN_HY4_FIXED_PKCACHE_F16: the pooled keys carried in fp16 (the scorer widens on load) stay inside fp16 rounding of the fp32 cache's attend output."""
    rng = np.random.default_rng(13)
    d = 64
    m = _kp_model(rng, d, 4, 32, 24, 8, 16, 2, 16, top_k=32)
    m.swa, m.window = False, 0
    xs = _chunks(rng, d, 150, 7)
    xs = [xs[0][:, :90], xs[0][:, 90:]] + xs[1:]
    ref = _fixed(m, xs, 64, compiled, pk=True)
    monkeypatch.setattr(FK, "PK_DTYPE", mx.float16)
    new = _fixed(m, xs, 64, compiled, pk=True)
    dd = _diff(ref, new)
    print(f"fixed_kv pk fp16 compiled={compiled}: max rel diff {dd:.3g}")
    assert dd <= 2e-2


def test_pool_rows_equal_pool_keys():
    from kyojin_mlx import foreign_skeleton_kpool as KP
    rng = np.random.default_rng(3)
    kp, d, S = 4, 16, 40
    val = mx.array(rng.standard_normal((1, S, d)).astype(np.float32)); gate = mx.array(rng.standard_normal((1, S, d)).astype(np.float32))
    sel = KP.KpoolSelect(mx.array(rng.standard_normal((kp, d)).astype(np.float32)), mx.array(rng.standard_normal(d).astype(np.float32)), 1e-6, kp, 16, True, 8, 10000.0, True)
    pk, _ = KP.pool_keys(val, gate, sel.ape, sel.norm_w, sel.eps, mx.zeros((1,), dtype=mx.int32), mx.array([S]), kp, sel.rope, J=S // kp)
    a = KP.pool_rows(val[0, 12:32], gate[0, 12:32], sel.ape, sel.norm_w, sel.eps, kp, sel.rope, mx.array(3, dtype=mx.int32))
    assert float(mx.max(mx.abs(a[0] - pk[0, 3:8])).item()) <= 1e-6 * max(1.0, float(mx.max(mx.abs(pk)).item()))      # the same ops on a smaller launch: equal to the last bits


def test_growth_keeps_rows():
    c = FK.FixedKVCache(8, 4, 0, 16)
    c.ensure(10, mx.float32)
    c.cache[0] = c.cache[0] + mx.arange(c.cache[0].shape[2]).reshape(1, 1, -1, 1)
    c._n = 10
    c.ensure(40, mx.float32)
    b = c.cache[0]
    assert b.shape[2] == 48 and float(b[0, 0, 9, 0].item()) == 9.0 and float(b[0, 0, 10, 0].item()) == 0.0


@pytest.mark.skipif(not METAL, reason="Metal run: KYOJIN_METAL_TESTS=1")
def test_fixed_equals_concat_metal():
    mx.set_default_device(mx.gpu)
    rng = np.random.default_rng(4)
    m = _swa_model(rng, 128, 64, 512, 192, 64, 256, 512)
    xs = _chunks(rng, 128, 1100, 4)
    dd = _diff(_ref(m, xs, 512), _fixed(m, xs, 1024, True))
    print(f"fixed_kv real swa metal: {dd:.3g}")
    assert dd <= 1e-4


@pytest.mark.parametrize("kind", ["swa", "kpool"])
def test_trim_rollback_equals_concat(kind):
    """speculation: a 4-row verify block, a rollback of 2 rows, then different rows: the fixed cache continues exactly like KVCache.trim."""
    from mlx_lm.models.base import create_causal_mask
    from mlx_lm.models.cache import KVCache
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)                                  # the Metal test above leaves the GPU as default; this check is exact on the CPU
    try:
        _trim_body(kind, create_causal_mask, KVCache)
    finally:
        mx.set_default_device(prev)


def _trim_body(kind, create_causal_mask, KVCache):
    rng = np.random.default_rng(7)
    d = 64
    if kind == "swa":
        m = _swa_model(rng, d, 4, 32, 24, 8, 16, 40)
    else:
        m = _kp_model(rng, d, 4, 32, 24, 8, 16, 2, 16, top_k=32)
        m.swa, m.window = False, 0
    xs = _chunks(rng, d, 150, 12)
    pre, dec = [xs[0][:, :90], xs[0][:, 90:]], xs[1:]
    blk = mx.concatenate(dec[:4], axis=1)                         # a 4-row verify block
    tail = dec[4:]
    ref_c, outs_ref = KVCache(), []
    at = m
    kp = at.kp
    fc = FK.FixedKVCache(at.kv + at.r + (2 * kp["dx"] if kp else 0), kp["kp"] if kp else 1, at.window if at.swa else 0, 64)
    outs_new = []
    def step(x, rc, fcache):
        L = x.shape[1]
        mask = create_causal_mask(L, rc.offset, window_size=m.window) if m.window else ("causal" if L > 1 else None)
        outs_ref.append(m(x, mask=mask, cache=rc))
        fcache.ensure(fcache._n + L, x.dtype)
        outs_new.append(m(x, mask=None, cache=fcache))
        fcache._n += L
    for x in pre:
        step(x, ref_c, fc)
    step(blk, ref_c, fc)
    ref_c.trim(2); assert fc.trim(2) == 2
    assert fc.offset == ref_c.offset and int(fc.cache[1].item()) == fc.offset
    for x in tail:
        step(x, ref_c, fc)
    mx.eval(outs_ref, outs_new)
    assert _diff(outs_ref, outs_new) <= TOL


def _ladder(c, dtype, prefill, decode, rows=1, chunk=20):
    """Drive the cache's growth like Model.__call__: prompt chunks of `chunk` rows (> the 16-row limit), then `decode` calls of `rows` rows; -> shapes seen during decode."""
    n = 0
    while n < prefill:
        L = min(chunk, prefill - n)
        c.ensure(c._n + L, dtype, L)
        c._n += L
        n += L
    shapes = []
    for _ in range(decode):
        c.ensure(c._n + rows, dtype, rows)
        c._n += rows
        shapes.append(c.cache[0].shape[2])
    return shapes


def test_headroom_decode_never_grows(monkeypatch):
    """KYOJIN_HY4_FIXED_KV_HEADROOM=N: after any prompt length the buffer holds N free rows, so N decode rows never grow it (fires counters: no decode growth); off = the stock growth lands mid-decode."""
    monkeypatch.setenv("KYOJIN_HY4_FIXED_KV_HEADROOM", "0")
    for prefill in (47, 48, 60, 63, 64, 100):                      # prompt ends 0..63 rows below a 64-row chunk boundary: some leave < 32 rows of room
        c = FK.FixedKVCache(8, 4, 0, 64, pk_dx=2)
        sh = _ladder(c, mx.float32, prefill, 32)
        grew_off = len(set(sh)) > 1
        monkeypatch.setenv("KYOJIN_HY4_FIXED_KV_HEADROOM", "32")
        before = dict(FK.STATS)
        c2 = FK.FixedKVCache(8, 4, 0, 64, pk_dx=2)
        sh2 = _ladder(c2, mx.float32, prefill, 32)
        monkeypatch.setenv("KYOJIN_HY4_FIXED_KV_HEADROOM", "0")
        assert len(set(sh2)) == 1, (prefill, sh2)
        assert FK.STATS["grow_decode"] == before["grow_decode"]
        assert sh2[0] >= prefill + 32 and sh2[0] % 64 == 0
        if prefill in (47, 48, 60):
            assert grew_off                                         # the control: without the flag these prompts grow inside the decode window
    c = FK.FixedKVCache(8, 4, 0, 64, pk_dx=2)                       # decode longer than the headroom: one growth by headroom, not by one chunk per 64 rows
    monkeypatch.setenv("KYOJIN_HY4_FIXED_KV_HEADROOM", "128")
    sh = _ladder(c, mx.float32, 100, 300)
    monkeypatch.setenv("KYOJIN_HY4_FIXED_KV_HEADROOM", "0")
    sh0 = _ladder(FK.FixedKVCache(8, 4, 0, 64, pk_dx=2), mx.float32, 100, 300)
    assert len(set(sh)) == 2 and sh[-1] >= 154 + 3 + 128 and len(set(sh0)) == 6


def test_headroom_counters_fire(monkeypatch):
    from kyojin_mlx.foreign_skeleton_mla import MlaAttention
    monkeypatch.setenv("KYOJIN_HY4_FIXED_KV_HEADROOM", "0")
    b = dict(FK.STATS); p = dict(MlaAttention.PATHS)
    c = FK.FixedKVCache(8, 4, 0, 64)
    _ladder(c, mx.float32, 100, 40)                                 # one prefill growth (64 -> 128), then one decode growth (140 > 128)
    assert FK.STATS["grow_decode"] == b["grow_decode"] + 1 and FK.STATS["grow_prefill"] > b["grow_prefill"]
    assert MlaAttention.PATHS["fixed_kv_grow_decode"] == p.get("fixed_kv_grow_decode", 0) + 1
    assert MlaAttention.PATHS.get("fixed_kv_headroom", 0) == p.get("fixed_kv_headroom", 0)      # flag off: the headroom counter never moves
    monkeypatch.setenv("KYOJIN_HY4_FIXED_KV_HEADROOM", "32")
    h0 = MlaAttention.PATHS.get("fixed_kv_headroom", 0)
    _ladder(FK.FixedKVCache(8, 4, 0, 64), mx.float32, 100, 40)
    assert MlaAttention.PATHS["fixed_kv_headroom"] > h0               # fires check: the headroom counter moves with the flag on
    monkeypatch.setenv("KYOJIN_HY4_FIXED_KV_HEADROOM", "0")


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("kind", ["swa", "kpool"])
def test_headroom_outputs_equal_stock(kind, compiled, monkeypatch):
    """The extra masked rows change nothing: the attention outputs with the headroom equal the stock growth's (and the concatenating cache's) across a prefill, a growth and decode rows."""
    rng = np.random.default_rng(5)
    d = 64
    if kind == "swa":
        m = _swa_model(rng, d, 4, 32, 24, 8, 16, 40)
    else:
        m = _kp_model(rng, d, 4, 32, 24, 8, 16, 2, 16, top_k=32)
        m.swa, m.window = False, 0
    xs = _chunks(rng, d, 150, 6)
    xs = [xs[0][:, :90], xs[0][:, 90:]] + xs[1:]
    monkeypatch.setenv("KYOJIN_HY4_FIXED_KV_HEADROOM", "0")
    a = _fixed(m, xs, 64, compiled, pk=(kind == "kpool"))
    monkeypatch.setenv("KYOJIN_HY4_FIXED_KV_HEADROOM", "40")
    b = _fixed(m, xs, 64, compiled, pk=(kind == "kpool"))
    monkeypatch.setenv("KYOJIN_HY4_FIXED_KV_HEADROOM", "0")
    dd = _diff(a, b)
    print(f"fixed_kv headroom {kind} compiled={compiled}: max rel diff {dd:.3g}")
    assert dd <= TOL
    assert _diff(_ref(m, xs, m.window), b) <= TOL
