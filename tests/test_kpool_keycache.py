# SPDX-License-Identifier: Apache-2.0
"""KYOJIN_KPOOL_KEYCACHE: the per-cache pooled keys (KpoolSelect.cached_keys / keep_cached) vs the stock recompute (pool_keys on all rows).
CPU: prefill chunks + single-token decode + a rewind + a forked (new) cache, bit-equal pooled keys and keep masks. Metal part (fused select on cached keys)
behind KYOJIN_METAL_TESTS=1."""
import os
import numpy as np
import mlx.core as mx
import pytest

from kyojin_mlx import foreign_skeleton_kpool as KP

D, NROT, H, KPOOL, TOPK, BASE, EPS = 16, 8, 3, 4, 12, 10000.0, 1e-6


class Cache:
    def __init__(self):
        self.keys = None


def _sel(rng):
    ape = mx.array(rng.standard_normal((KPOOL, D)).astype(np.float32))
    nw = mx.array(rng.standard_normal(D).astype(np.float32) * .3 + 1)
    return KP.KpoolSelect(ape, nw, EPS, KPOOL, TOPK, True, NROT, BASE, True)


def _stock(sel, val, gate, off, L):
    n = val.shape[1]
    pad, offv = KP.row_pad(off, L, n, 1)
    return KP.pool_keys(val, gate, sel.ape, sel.norm_w, sel.eps, pad, offv + L, KPOOL, sel.rope)


def _run(steps, rewind_to=None, fused=False):
    rng = np.random.default_rng(3)
    sel = _sel(rng)
    V = mx.array(rng.standard_normal((1, 80, D)).astype(np.float32)).astype(mx.float16)
    G = mx.array(rng.standard_normal((1, 80, D)).astype(np.float32)).astype(mx.float16)
    c, off = Cache(), 0
    c.keys = mx.zeros((1,))
    for L in steps:
        n = off + L
        r = sel.cached_keys(c, V[:, :n], G[:, :n], off, L)
        if n // KPOOL < 1:
            assert r is None
        else:
            pk, valid, _ = r
            spk, svalid = _stock(sel, V[:, :n], G[:, :n], off, L)
            assert pk.shape == spk.shape and bool(mx.all(valid == svalid).item())
            assert bool(mx.array_equal(pk, spk).item()), (off, L)
            iq = mx.array(rng.standard_normal((1, H, L, D)).astype(np.float32)); w = mx.array(rng.standard_normal((1, L, H)).astype(np.float32))
            k1 = sel.keep_cached(iq, w, c, V[:, :n], G[:, :n], off, False)
            k0 = sel.keep(iq, w, V[:, :n], G[:, :n], off)
            assert bool(mx.array_equal(k0, k1).item())
        off = n
    return sel, V, G, c, off


def test_prefill_then_decode_bit_equal():
    _run([5, 8, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1])


def test_decode_from_empty():
    _run([1] * 30)


def test_hits_do_not_recompute():
    KP.PATHS.update(keycache_hit=0, keycache_new=0)
    _run([8] + [1] * 3)               # n = 8 (new), 9, 10, 11 (hits: no new group)
    assert KP.PATHS["keycache_hit"] == 6 and KP.PATHS["keycache_new"] == 2      # keep_cached repeats the call at the same offset: a rewind to 0


def test_rewind_and_new_cache():
    sel, V, G, c, off = _run([9, 1, 1, 1, 1, 1])             # 15 rows
    for back in (15, 11, 6):                                  # rewind to `back`, then 1 row: pools past `back` are dropped and rebuilt
        r = sel.cached_keys(c, V[:, :back + 1], G[:, :back + 1], back, 1)
        spk, _ = _stock(sel, V[:, :back + 1], G[:, :back + 1], back, 1)
        assert bool(mx.array_equal(r[0], spk).item())
    c2 = Cache(); c2.keys = mx.zeros((1,))                    # a copy is a new object: full recompute
    r = sel.cached_keys(c2, V[:, :13], G[:, :13], 12, 1)
    assert bool(mx.array_equal(r[0], _stock(sel, V[:, :13], G[:, :13], 12, 1)[0]).item())


def test_keys_buffer_change_misses():
    sel, V, G, c, off = _run([9, 1])
    V2 = V + 1                                                # same offsets, other rows, other buffer object (a restored cache)
    c.keys = mx.zeros((1,))
    r = sel.cached_keys(c, V2[:, :11], G[:, :11], 10, 1)
    assert bool(mx.array_equal(r[0], _stock(sel, V2[:, :11], G[:, :11], 10, 1)[0]).item())


def test_not_applicable_falls_back():
    sel, V, G, c, off = _run([9])
    assert sel.cached_keys(c, V[:, :9], G[:, :9], mx.array([8]), 1) is None        # array offset
    assert sel.cached_keys(None, V[:, :9], G[:, :9], 8, 1) is None
    assert sel.cached_keys(c, V[:, :9], G[:, :9], 7, 1) is None                    # rows not right-aligned from 0 (S != off + L)


@pytest.mark.skipif(os.environ.get("KYOJIN_METAL_TESTS") != "1", reason="Metal")
def test_fused_on_cached_keys_metal(monkeypatch):
    from kyojin_mlx import kpool_fuse as KF
    mx.set_default_device(mx.gpu)
    monkeypatch.setattr(KF, "ON", True)                                                # restored at teardown (no leak into test_kpool_fuse)
    rng = np.random.default_rng(5)
    sel = _sel(rng)
    V = mx.array(rng.standard_normal((1, 80, D)).astype(np.float32)).astype(mx.float16)
    G = mx.array(rng.standard_normal((1, 80, D)).astype(np.float32)).astype(mx.float16)
    c = Cache(); c.keys = mx.zeros((1,))
    off = 0
    for L in [40, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]:
        n = off + L
        iq = mx.array(rng.standard_normal((1, H, L, D)).astype(np.float32)); w = mx.array(rng.standard_normal((1, L, H)).astype(np.float32))
        if L == 1:
            a = sel.keep_cached(iq, w, c, V[:, :n], G[:, :n], off, True)
            b = sel.keep(iq, w, V[:, :n], G[:, :n], off)
            assert bool(mx.array_equal(a, b).item())
        else:
            sel.keep_cached(iq, w, c, V[:, :n], G[:, :n], off, False)
        off = n
