# SPDX-License-Identifier: Apache-2.0
"""KYOJIN_HY4_HC_FUSE (hc_fuse_pf): the two-launch hyper-connection glue of a prompt chunk equals the stock chain (identity form). CPU: the gate (`ok`) only.
Metal half (KYOJIN_METAL_TESTS=1): HcStream flag on vs off on random streams, ragged row counts (not a multiple of 4), fp32 and fp16 streams, Hy4 sizes (hc 4, D 3072)."""
import os

import mlx.core as mx
import numpy as np
import pytest

from kyojin_mlx import foreign_skeleton_hc as HC, hc_fuse_pf as HF

METAL = mx.metal.is_available() and os.environ.get("KYOJIN_METAL_TESTS", "0") not in ("0", "")


def _hc(rng, n, D, post_eps=False):
    fn = mx.array((0.02 * rng.standard_normal((2 * n, n * D))).astype(np.float32))
    base = mx.array(rng.standard_normal(2 * n).astype(np.float32))
    scale = mx.array(np.array([0.7, 1.3], np.float32))
    return HC.HcStream(fn, base, scale, n, 1e-6, 2.0, 1e-6, 0, post_eps)


def test_gate():
    rng = np.random.default_rng(1)
    hc = _hc(rng, 4, 64)
    assert HF.ok(hc, mx.zeros((1, 128, 4, 64)))
    assert not HF.ok(hc, mx.zeros((1, 8, 4, 64))), "decode rows stay on the stock path"
    assert not HF.ok(hc, mx.zeros((1, 128, 4, 60))) and not HF.ok(hc, mx.zeros((1, 128, 3, 64)))


@pytest.mark.skipif(not METAL, reason="Metal half: KYOJIN_METAL_TESTS=1")
@pytest.mark.parametrize("dt,rows,D,post_eps", [(mx.float32, 203, 3072, False), (mx.float32, 64, 256, True), (mx.float16, 131, 512, False)])
def test_fused_equals_stock(dt, rows, D, post_eps):
    mx.set_default_device(mx.gpu)
    try:
        rng = np.random.default_rng(3)
        hc = _hc(rng, 4, D, post_eps)
        h = mx.array(rng.standard_normal((1, rows, 4, D)).astype(np.float32)).astype(dt)
        w = mx.array(0.1 * rng.standard_normal((D, D)).astype(np.float32))
        sub = lambda x: (x.astype(mx.float32) @ w).astype(mx.float16)
        outs = []
        for flag in ("0", "1"):
            os.environ["KYOJIN_HY4_HC_FUSE"] = flag
            outs.append(hc(h, sub)); mx.eval(outs[-1])
        os.environ.pop("KYOJIN_HY4_HC_FUSE", None)
        a, b = outs[0].astype(mx.float32), outs[1].astype(mx.float32)
        d = float(mx.max(mx.abs(a - b)).item()) / max(1.0, float(mx.max(mx.abs(a)).item()))
        print(f"hc_fuse {dt} rows {rows} D {D}: max rel diff {d:.3g}")
        assert outs[1].dtype == outs[0].dtype and d < 3e-3     # the sub-layer output is rounded to fp16: a 1e-6 difference of x flips fp16 roundings (1e-3)
    finally:
        os.environ.pop("KYOJIN_HY4_HC_FUSE", None)
        mx.set_default_device(mx.cpu)
