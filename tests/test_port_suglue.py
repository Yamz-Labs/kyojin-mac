# SPDX-License-Identifier: Apache-2.0
"""suglue stages A / B / E against the separate launches they replace (gluefuse.mixvec, glue_fuse.mix_norm, fuse.mhc_mix_pre, fuse.mhc_expand):
same bits, for 1, 3 and 8 rows, fp16 and bf16."""
import types

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn

from kyojin_mlx import fuse as F
from kyojin_mlx import glue_fuse as GN
from kyojin_mlx import gluefuse as GF
from kyojin_mlx import suglue as SG


def same(a, b):
    a, b = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def make_blk(rng, D, dt):
    blk = types.SimpleNamespace(hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6, norm_eps=1e-20)
    for nm in ("attn", "ffn"):
        n = nn.RMSNorm(D, eps=1e-6)
        n.weight = mx.array((rng.standard_normal(D) * 0.5).astype(np.float32)).astype(dt)
        setattr(blk, nm + "_norm", n)
        setattr(blk, f"hc_{nm}_fn", mx.array(rng.standard_normal((24, 4 * D)).astype(np.float32) * 0.01))
        setattr(blk, f"hc_{nm}_base", mx.array(rng.standard_normal(24).astype(np.float32)))
        setattr(blk, f"hc_{nm}_scale", mx.array([1.0, 0.5, 2.0], dtype=mx.float32))
    return blk


@pytest.mark.parametrize("dt", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("D", [4096, 5120])
@pytest.mark.parametrize("B", [1, 3, 8])
def test_stages_bit_identical(D, B, dt):
    rng = np.random.default_rng(D + B)
    blk = make_blk(rng, D, dt)
    h = mx.array((rng.standard_normal((B, 1, 4, D)) * 0.05).astype(np.float32)).astype(dt)
    pre = mx.array(rng.standard_normal((B, 1, 4)).astype(np.float32))
    ao = mx.array((rng.standard_normal((B, 1, D)) * 0.05).astype(np.float32)).astype(dt)
    fo = mx.array((rng.standard_normal((B, 1, D)) * 0.05).astype(np.float32)).astype(dt)
    it, he = blk.hc_sinkhorn_iters, blk.hc_eps
    # reference: the separate launches
    z = GF.mixvec(h.reshape(B, 4 * D), blk.hc_attn_fn, blk.norm_eps).reshape(B, 1, 24)
    rx, rpre, rpost, rcomb = GN.mix_norm(z, h, pre, blk.attn_norm.weight, blk.attn_norm.eps, blk.hc_attn_scale, blk.hc_attn_base, it, he)
    rx2, _ = F.mhc_expand(ao, h, rpost, rcomb)
    zf = GF.mixvec(rx2.reshape(B, 4 * D), blk.hc_ffn_fn, blk.norm_eps).reshape(B, 1, 24)
    ry, rfpre, rfpost, rfcomb = GN.mix_norm(zf, rx2, rpre, blk.ffn_norm.weight, blk.ffn_norm.eps, blk.hc_ffn_scale, blk.hc_ffn_base, it, he)
    rh2, rh232 = F.mhc_expand(fo, rx2, rfpost, rfcomb)
    # stages
    x, za = SG.stage_a(blk, h, pre)
    yn, x2, zfs = SG.stage_b(blk, ao, h, za)
    h2, h232, pre2 = SG.stage_e(blk, fo, x2, zfs)
    mx.eval(x, za, yn, x2, zfs, h2, h232, pre2, rx, z, ry, rx2, rh2, rh232, rfpre)
    assert same(za, z), "z_attn"
    assert same(x, rx), "attn input"
    assert same(x2, rx2), "expanded stream"
    assert same(zfs, zf), "z_ffn"
    assert same(yn, ry), "ffn input"
    assert same(h2, rh2) and same(h232, rh232), "h'"
    assert same(pre2, rfpre), "pre'"


def test_row_independence():
    rng = np.random.default_rng(3)
    D, B = 5120, 3
    blk = make_blk(rng, D, mx.float16)
    h = mx.array((rng.standard_normal((B, 1, 4, D)) * 0.05).astype(np.float16))
    pre = mx.array(rng.standard_normal((B, 1, 4)).astype(np.float32))
    ao = mx.array((rng.standard_normal((B, 1, D)) * 0.05).astype(np.float16))
    x, za = SG.stage_a(blk, h, pre)
    yn, x2, zf = SG.stage_b(blk, ao, h, za)
    for r in range(B):
        x1, z1 = SG.stage_a(blk, h[r:r + 1], pre[r:r + 1])
        y1, x21, zf1 = SG.stage_b(blk, ao[r:r + 1], h[r:r + 1], z1)
        mx.eval(x1, z1, y1, x21, zf1)
        assert same(x1, x[r:r + 1]) and same(z1, za[r:r + 1]) and same(y1, yn[r:r + 1]) and same(x21, x2[r:r + 1]) and same(zf1, zf[r:r + 1])


@pytest.mark.parametrize("dt", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("rows", [1, 3])
def test_qk_norm_bit_identical(dt, rows):
    rng = np.random.default_rng(rows)
    attn = types.SimpleNamespace()
    for nm, n in (("q_norm", 1280), ("kv_norm", 512)):
        m = nn.RMSNorm(n, eps=1e-6)
        m.weight = mx.array((rng.standard_normal(n) * 0.5).astype(np.float32)).astype(dt)
        setattr(attn, nm, m)
    q = mx.array((rng.standard_normal((rows, 1, 1280)) * 3).astype(np.float32)).astype(dt)
    kv = mx.array((rng.standard_normal((rows, 1, 512)) * 0.1).astype(np.float32)).astype(dt)
    old = SG.ENABLED
    SG.ENABLED = True
    try:
        assert SG.qk_norm_ok(attn, q, kv)
        qo, kvo = SG.qk_norm(attn, q, kv)
    finally:
        SG.ENABLED = old
    mx.eval(qo, kvo)
    assert same(qo, attn.q_norm(q)) and same(kvo, attn.kv_norm(kv))


def test_model_three_rows_bit_identical():
    """3 user rows through the synthetic DS4.1-dims stack (layer 0 is a ratio-0 local layer): SG.block path == stock."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    import ds41_glue_bench as B
    m, _ = B.build(2, rows=True, seed=0)
    outs, old = {}, SG.ENABLED
    try:
        for on in (False, True):
            SG.ENABLED = on
            c = m.make_cache()
            n0 = SG.STATS["A"]
            r = []
            for t in range(3):
                o = m(mx.array([[5 + t], [9 + t], [20 + t]]), cache=c).logits
                mx.eval(o)
                r.append(o)
            outs[on] = r
            assert (SG.STATS["A"] - n0 == 6) == on
    finally:
        SG.ENABLED = old
    assert all(same(a, b) for a, b in zip(outs[False], outs[True]))
