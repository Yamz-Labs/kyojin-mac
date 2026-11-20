"""G3b: fused mHC decode glue vs the vendor HyperConnection.apply_branch chain (random weights, real shapes)."""
import os, sys, types
os.environ["KYOJIN_GLM_HC_FUSE"] = "1"
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.deepseek_v4.hyper_connection import HyperConnection
from kyojin_mlx import glm_hc as G


def mk(H, dt, seed):
    mx.random.seed(seed)
    cfg = types.SimpleNamespace(hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6, rms_norm_eps=1e-5, hidden_size=H)
    hc = HyperConnection(cfg)
    hc.fn = (mx.random.normal((24, 4 * H)) * 0.02)
    hc.base = mx.random.normal((24,)) * 0.5
    hc.scale = mx.array([0.3, 0.2, 0.4], dtype=mx.float32)
    nrm = nn.RMSNorm(H, eps=1e-5)
    nrm.weight = (1 + 0.3 * mx.random.normal((H,))).astype(dt)
    W = (mx.random.normal((H, H)) * (H ** -0.5)).astype(dt)
    return hc, nrm, W


def run(dt, T, nbr=8, H=4096, fn16=False):
    hcs = [mk(H, dt, 10 + i) for i in range(nbr)]
    if fn16:                                   # the served loader stores the mix weight in fp16 (both paths then read the same values)
        for hc, _, _ in hcs:
            hc.fn = hc.fn.astype(mx.float16)
    x0 = (mx.random.normal((T, 1, 4, H)) * 1.5).astype(dt)
    # stock chain
    x = x0
    for hc, nrm, W in hcs:
        x = hc.apply_branch(x, nrm, lambda z, W=W: (z @ W.T) * 0.7)
    ref = x
    # fused chain
    x = x0
    part = G.partials(x, hcs[0][0].fn)
    for i, (hc, nrm, W) in enumerate(hcs):
        xo, pc = G.pre(x, part, hc, nrm.weight, 1e-5)
        o = (xo @ W.T) * 0.7
        nf = hcs[i + 1][0].fn if i + 1 < nbr else None
        x, part = G.expand(o, x, pc, nf)
    mx.eval(ref, x)
    a, b = np.asarray(ref.astype(mx.float32)), np.asarray(x.astype(mx.float32))
    err = np.abs(a - b)
    return err.max(), err.mean(), np.abs(a).mean(), (a != b).mean()


import pytest


@pytest.mark.parametrize("fn16", [False, True])
@pytest.mark.parametrize("dt,tol", [(mx.float16, 0.05), (mx.bfloat16, 0.2)])
@pytest.mark.parametrize("T", [1, 2, 4, 8])
def test_chain_matches_stock(dt, tol, T, fn16):
    mx.random.seed(0)
    mx_, mean, _, _ = run(dt, T, fn16=fn16)
    assert mx_ < tol and mean < tol / 4


def test_ok_shape_accepts_served_fp16_weight():
    hc, _, _ = mk(4096, mx.float16, 1)
    x = mx.zeros((1, 1, 4, 4096), dtype=mx.float16)
    assert G.ok_shape(x, hc)
    hc.fn = hc.fn.astype(mx.float16)
    assert G.ok_shape(x, hc)
    assert not G.ok_shape(mx.zeros((1, 32, 4, 4096), dtype=mx.float16), hc)       # prompt rows stay stock


if __name__ == "__main__":
    for fn16 in (False, True):
        for dt in (mx.float16, mx.bfloat16):
            for T in (1, 2, 4, 8):
                print(dt, "T", T, "fn16", fn16, "maxerr %.4g meanerr %.4g meanabs %.4g frac_diff %.3f" % run(dt, T, fn16=fn16), flush=True)
