"""pf_no_mirror: the prompt attention gathers the pooled rows from their fp4 codes, so a prefill chunk builds no dense pool mirror.

Metal: the K1 / K3 gathers (and the pf_attn_online pair) of a packed pool against the same call on the dense rows, equal bit for bit (not a tolerance: the
decode is `half(LUT[nibble] * fp32(e4m3 scale))`, the one fp32 multiply and one rounding `fp4pack.unpack` performs). CPU glue: the layer call hands back a
`PackedView` and drops the mirror, an append takes no read-back, a layer that is not packed keeps the dense path, and the admin flag switches the lever at
runtime.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

from kyojin_mlx import attn_prefill as AP, depthflat as DF, flags as FL, fp4pack, fp4pool, metal_nax as NX, pfnomirror as PNF  # noqa: E402
from kyojin_mlx.fp4pool import PoolSlots  # noqa: E402

if not NX.nax_available():
    pytest.skip("no neural accelerators on this GPU", allow_module_level=True)

D = 256


def make(T, hist, Np, topk, H=32, D=D, seed=0, ratio=2):
    rng = np.random.default_rng(seed)
    q = mx.array((rng.standard_normal((1, H, T, D)) * 0.6).astype(np.float16))
    window = mx.array((rng.standard_normal((1, hist + T, D)) * 0.6).astype(np.float16))
    pool = mx.array((rng.standard_normal((1, Np, D)) * 0.6).astype(np.float16))
    idx = np.full((1, T, topk), -1, np.int32)
    for t in range(T):
        n = min(Np, (hist + t + 1) // ratio)
        k = min(topk, n)
        if k:
            idx[0, t, :k] = np.sort(rng.choice(n, k, replace=False))
    return q, window, pool, mx.array(idx), mx.array(rng.standard_normal(H).astype(np.float32))


def on_grid(pool):
    """The rows the vendor hands the cache: already on the fp4 grid of (block 16, e4m3 scale)."""
    return fp4pack.unpack(*fp4pack.pack(pool, 16, True), 16, True)


def canon(pool):
    """(the dense rows a flag 0 mirror would hold, the PackedView of the same layer): both derived from ONE code set, as the cache does. The entry's codes
    are what `fp4pool._try_pack` stores, and its mirror is `unpack(codes)` (a re-pack can turn a -0.0 the vendor wrote into +0.0, which is the same value)."""
    c, s = fp4pack.pack(pool, 16, True)
    return fp4pack.unpack(c, s, 16, True), fp4pool.PackedView(c, s, pool.shape[1], mx.float16, None, 16, True)


def entry_rows(e, n=None):
    """The dense rows of a packed entry: exactly what `depthflat.mirror_read` builds (and what the packed kernels decode)."""
    n = e.n if n is None else n
    return fp4pack.unpack(e.codes[:, :n], e.scales[:, :n], 16, True)


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.setattr(PNF, "ENABLED", True)
    monkeypatch.setattr(DF, "ENABLED", True)
    monkeypatch.setattr(DF, "MIRROR_BYTES", 1 << 30)
    monkeypatch.setattr(fp4pool, "MEMO_BYTES", 0)
    for k in list(DF.HITS):
        DF.HITS[k] = 0
    for k in list(PNF.HITS):
        PNF.HITS[k] = 0
    yield


def bits(a):
    return np.array(a.view(mx.uint16))


@pytest.mark.parametrize("T,hist,Np,topk", [(70, 0, 40, 48), (96, 127, 80, 64), (130, 127, 150, 100)])
@pytest.mark.parametrize("H,D", [(32, 256), (64, 512)])
@pytest.mark.parametrize("online", [True, False])
def test_packed_pool_prefill_equals_dense_bit_for_bit(T, hist, Np, topk, H, D, online, monkeypatch):
    monkeypatch.setattr(AP, "ONLINE", online)
    q, window, pool, idxs, sinks = make(T, hist, Np, topk, H=H, D=D, seed=T)
    pool, view = canon(on_grid(pool))
    scale = D ** -0.5
    ref = AP.sparse_attention(q, window, pool, idxs, scale, sinks, 128)
    assert PNF.HITS["attention"] == 0
    got = AP.sparse_attention(q, window, view, idxs, scale, sinks, 128)
    assert PNF.HITS["attention"] == 1
    assert got.shape == ref.shape and got.dtype == ref.dtype == mx.float16
    assert np.array_equal(bits(got), bits(ref)), "packed pool differs from the dense rows"


def test_a_dense_pool_and_the_flag_off_build_no_view(monkeypatch):
    q, window, pool, idxs, sinks = make(70, 0, 40, 48)
    pool, view = canon(on_grid(pool))
    monkeypatch.setattr(PNF, "ENABLED", False)
    got = AP.sparse_attention(q, window, view, idxs, 256 ** -0.5, sinks, 128)               # flag off: the view is unpacked, no codes read
    assert PNF.HITS["attention"] == 0
    assert np.array_equal(bits(got), bits(AP.sparse_attention(q, window, pool, idxs, 256 ** -0.5, sinks, 128)))


def _layer_cache(n, D=D, seed=0):
    rng = np.random.default_rng(seed)
    rows = on_grid(mx.array((rng.standard_normal((1, n, D)) * 0.7).astype(np.float16)))
    slots = PoolSlots([rows])
    e = slots.raw(0)
    assert isinstance(e, fp4pool._Pk) and e.mirror is None
    return SimpleNamespace(compress=slots, compress_kv=None), rows, e


def test_the_layer_call_hands_back_the_codes_and_drops_the_mirror(monkeypatch):
    """`AP._compress_part_packed` around the vendor `_compress_part` (append, publish, read the pool): the layer comes back as its codes, no mirror is built."""
    q, window, pool, idxs, sinks = make(70, 0, 40, 48)
    cache, rows, e = _layer_cache(40)
    e.mirror, e.mn = mx.zeros((1, 64, D), dtype=mx.float16), 40                       # a mirror an earlier chunk (flag off) built
    freed = int(e.mirror.nbytes)
    rng = np.random.default_rng(2)
    new = on_grid(mx.array((rng.standard_normal((1, 4, D)) * 0.7).astype(np.float16)))

    seen = {}

    class Attn:
        layer_idx, compress_ratio = 0, 2

        def _compress_part(self, x, qr, start_pos, cache):                            # the vendor body of one kv-source layer
            seen["lazy"], seen["defer"] = cache.compress.lazy, cache.compress.defer_dense
            cache.compress.append_rows(0, new)                                        # append_compressed
            cache.compress_kv = cache.compress[self.layer_idx]
            return cache.compress_kv[:1, :44], idxs

    attn = Attn()
    got_pool, _ = AP._compress_part_packed(attn, None, None, 0, cache)
    assert seen == {"lazy": True, "defer": True}
    assert isinstance(got_pool, fp4pool.PackedView) and got_pool.n == 44
    assert cache.compress.raw(0).mirror is None and DF.HITS["build"] == 0             # nothing built the dense form of the layer
    assert PNF.HITS["layers"] == 1 and PNF.HITS["drop_bytes"] == freed and PNF.HITS["views"] == 1
    assert not cache.compress.lazy and not cache.compress.defer_dense                 # both restored on the way out
    assert np.array_equal(bits(cache.compress[0]), bits(entry_rows(cache.compress.raw(0))))   # a reader after the call still gets the rows
    assert DF.HITS["build"] == 1                                                      # and only that explicit dense read builds a mirror


def test_an_append_under_the_flag_takes_no_read_back(monkeypatch):
    cache, rows, e = _layer_cache(24)
    rng = np.random.default_rng(1)
    new = on_grid(mx.array((rng.standard_normal((1, 6, D)) * 0.7).astype(np.float16)))
    with PNF.prefill_pool(cache, 0):
        assert cache.compress.append_rows(0, new) is None                            # defer_dense: no dense layer, no mirror
    e = cache.compress.raw(0)
    assert e.n == 30 and e.mirror is None and DF.HITS["build"] == 0
    assert np.array_equal(bits(cache.compress[0]), bits(entry_rows(e)))              # same rows after the append


def test_a_layer_that_is_not_packed_keeps_the_dense_path(monkeypatch):
    cache = SimpleNamespace(compress=PoolSlots([None]), compress_kv=None)
    with PNF.prefill_pool(cache, 0) as packed:
        assert not packed
    assert PNF.HITS["skip"] == 1 and PNF.HITS["layers"] == 0


def test_the_admin_flag_switches_the_lever_at_runtime():
    f = FL.FLAGS["pf_no_mirror"]
    assert f.env == "KYOJIN_PF_NO_MIRROR" and f.default == "0" and f.runtime
    try:
        FL.set("pf_no_mirror", "1")
        assert PNF.ENABLED and os.environ["KYOJIN_PF_NO_MIRROR"] == "1"
        FL.set("pf_no_mirror", "0")
        assert not PNF.ENABLED
    finally:
        FL.set("pf_no_mirror", "0")
