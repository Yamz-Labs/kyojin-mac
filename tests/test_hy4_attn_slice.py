# SPDX-License-Identifier: Apache-2.0
"""KYOJIN_HY4_SWA_SLICE and KYOJIN_HY4_KPOOL_GATHER: the sliced / gathered attention computes the same function as today's mask path (CPU, no Metal).

Each test runs the SAME inputs through two caches, flag off (masked full S) and flag on, chunk by chunk (512-row prefill chunks over the cache, then decode
rows), and asserts (1) the flag path really ran (MlaAttention.PATHS counters; before the flags existed this attribute is missing, so the tests fail),
(2) max rel diff <= TOL. The selection is one shared function (KP.select_pools): the gather test also checks gather_slots == keep_mask set-wise.
"""
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from kyojin_mlx import foreign_skeleton_kpool as KP
from kyojin_mlx import foreign_skeleton_mla as MLA

mx.set_default_device(mx.cpu)

TOL = 2e-6          # relative to max(1, max|out|), fp32; the measured diffs are printed with -s


def _fill(mod, rng, scale=0.2):
    flat = {}
    for k, a in nn.utils.tree_flatten(mod.parameters()):
        s = rng.standard_normal(a.shape).astype(np.float32)
        flat[k] = mx.array(1.0 + 0.1 * s if ("norm" in k and a.ndim == 1) else scale * s)
    mod.update(nn.utils.tree_unflatten(list(flat.items())))
    return mod


def _cache(B=1):
    from mlx_lm.models.cache import KVCache
    return KVCache()


def _run(m, xs, flag, val, window=0, swa=False):
    """Feed chunks xs [1, L, d] through m with the env flag = val, a fresh cache; -> list of outputs."""
    from mlx_lm.models.base import create_causal_mask
    old = os.environ.get(flag)
    os.environ[flag] = val
    try:
        c, outs = _cache(), []
        for x in xs:
            L = x.shape[1]
            if swa:
                mask = create_causal_mask(L, c.offset, window_size=window)
            else:
                mask = "causal" if L > 1 else None
            outs.append(m(x, mask=mask, cache=c))
        mx.eval(outs)
        return outs
    finally:
        if old is None:
            os.environ.pop(flag, None)
        else:
            os.environ[flag] = old


def _chunks(rng, d, total, dec):
    xs, left = [], total
    while left > 0:
        n = min(512, left)
        xs.append(mx.array(rng.standard_normal((1, n, d)).astype(np.float32)))
        left -= n
    xs += [mx.array(rng.standard_normal((1, 1, d)).astype(np.float32)) for _ in range(dec)]
    return xs


def _maxdiff(a, b):
    """max rel diff relative to max(1, max |ref|): the real-shape outputs are O(100) with the random weights."""
    mag = max(1.0, max(float(mx.max(mx.abs(x)).item()) for x in a))
    return max(float(mx.max(mx.abs(x - y)).item()) for x, y in zip(a, b)) / mag


def _swa_model(rng, d, nh, kv, nope, rope, v, window):
    m = _fill(MLA.MlaAttention(d, nh, 48, kv, nope, rope, v, sinks=True, gate=True), rng)
    m.swa, m.window = True, window
    return m


@pytest.mark.parametrize("shape", ["small", "real"])
def test_swa_slice_equals_mask(shape):
    rng = np.random.default_rng(1)
    if shape == "small":
        m = _swa_model(rng, 64, 4, 32, 24, 8, 16, 512)
    else:   # Hy4 SWA layer: nh 64, latent 512 + rope 64, nope 192, v 256, window 512
        m = _swa_model(rng, 128, 64, 512, 192, 64, 256, 512)
    xs = _chunks(rng, m.q_a_proj.weight.shape[1], 1536 + 100, 3)       # S > 1024 > window, a ragged chunk, then decode
    n0 = MLA.MlaAttention.PATHS["swa_slice"]
    ref = _run(m, xs, "KYOJIN_HY4_SWA_SLICE", "0", 512, True)
    assert MLA.MlaAttention.PATHS["swa_slice"] == n0, "default path must not slice"
    new = _run(m, xs, "KYOJIN_HY4_SWA_SLICE", "1", 512, True)
    assert MLA.MlaAttention.PATHS["swa_slice"] > n0, "the slice path did not run"
    d = _maxdiff(ref, new)
    print(f"swa_slice {shape}: max rel diff {d:.3g} over {len(xs)} calls")
    assert d <= TOL


def _kp_model(rng, d, nh, kv, nope, rope, v, H, dx, top_k=2048):
    kpool = dict(H=H, dx=dx, kp=4, top_k=top_k, select_tail=True, n_rot=min(rope, dx), base=10000.0)
    m = _fill(MLA.MlaAttention(d, nh, 48, kv, nope, rope, v, sinks=True, gate=True, kpool=kpool), rng)
    m.kpool.ape = mx.array(0.2 * rng.standard_normal(m.kpool.ape.shape).astype(np.float32))
    return m


@pytest.mark.parametrize("shape", ["small", "real"])
def test_kpool_gather_equals_mask(shape):
    rng = np.random.default_rng(2)
    if shape == "small":
        m = _kp_model(rng, 64, 4, 32, 24, 8, 16, 2, 16)
    else:   # Hy4 global indexer layer: nh 32, latent 512 + rope 64, nope 192, v 256, indexer 16 heads x 128
        m = _kp_model(rng, 128, 32, 512, 192, 64, 256, 16, 128)
    total = 2560 + 37                       # S > 2048: 649 pools > 512, chunk edges at 512 multiples, a ragged last chunk, then 3 decode rows
    xs = _chunks(rng, 64 if shape == "small" else 128, total, 3)
    n0 = MLA.MlaAttention.PATHS["kpool_gather"]
    ref = _run(m, xs, "KYOJIN_HY4_KPOOL_GATHER", "0")
    assert MLA.MlaAttention.PATHS["kpool_gather"] == n0, "default path must not gather"
    new = _run(m, xs, "KYOJIN_HY4_KPOOL_GATHER", "1")
    assert MLA.MlaAttention.PATHS["kpool_gather"] > n0, "the gather path did not run"
    d = _maxdiff(ref, new)
    # rows past 2048 positions are the ones where the selection bites: compare them alone too
    late = _maxdiff(ref[4:], new[4:])
    print(f"kpool_gather {shape}: max rel diff {d:.3g} (chunks past 2048: {late:.3g}) over {len(xs)} calls")
    assert d <= TOL


def test_gather_slots_are_the_keep_set():
    """The two consumers of KP.select_pools agree set-wise, incl. the forced tail group, a left-padded row and an unfilled pool."""
    rng = np.random.default_rng(3)
    B, H, L, dx, kp, S = 2, 2, 6, 8, 4, 3000
    J = S // kp
    iq = mx.array(rng.standard_normal((B, H, L, dx)).astype(np.float32))
    w = mx.array(rng.standard_normal((B, L, H)).astype(np.float32))
    pk = mx.array(rng.standard_normal((B, J, dx)).astype(np.float32))
    off = mx.array([2990 - L, 2900 - L])    # row 1 is left padded by 90 slots
    valid = mx.ones((B, J), dtype=mx.bool_)
    s = KP.select_pools(iq, w, pk, valid, off, S, kp, 2048, True)
    keep = np.array(KP.keep_mask(s))[:, 0]                        # [B, L, S]
    slots, ok = KP.gather_slots(s)
    slots, ok = np.array(slots), np.array(ok)
    assert slots.shape[-1] <= 2048
    for b in range(B):
        for l in range(L):
            got = np.zeros(S, bool)
            got[slots[b, l][ok[b, l]]] = True
            assert (got == keep[b, l]).all()
            assert (np.diff(slots[b, l][ok[b, l]]) > 0).all()    # ascending, no duplicates
