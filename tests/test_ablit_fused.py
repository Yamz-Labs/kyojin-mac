# SPDX-License-Identifier: Apache-2.0
"""ablit_fused: the one-launch decensor projection against the exact projection (float64), and against the stock path where the
stock path is exact (one row: a GEMV). With several rows the stock path's fp32 matmul runs at reduced precision on Apple GPUs
(errors up to ~4e-3 measured 05/11), the fused kernel does not (~5e-7), and it computes every row the same way whatever the row count."""
import mlx.core as mx
import numpy as np
import pytest

from kyojin_mlx import ablit


def _run(fused, y, w, R, mu):
    old = ablit.FUSED
    ablit.FUSED = fused
    try:
        return ablit.project(y, w, R, mu)
    finally:
        ablit.FUSED = old


def _case(shape, K, with_mu, dtype, seed=0):
    H = shape[-1]
    rng = np.random.default_rng(seed + K * 7 + H)
    yn = (rng.standard_normal(shape) * 3.0).astype(np.float32)
    q, _ = np.linalg.qr(rng.standard_normal((H, K)).astype(np.float32))       # orthonormal rows, as the spec's R
    Rn = q.T[:K].astype(np.float32)
    mun = rng.standard_normal(K).astype(np.float32) if with_mu else None
    y = mx.array(yn).astype(dtype)
    y64 = np.array(y.astype(mx.float32)).astype(np.float64)                   # the input as the kernel sees it
    d = y64 @ Rn.T.astype(np.float64)
    if mun is not None:
        d = d - mun
    exact = y64 - 2.2 * (d @ Rn.astype(np.float64))
    return y, mx.array(Rn), (mx.array(mun) if mun is not None else None), exact


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
@pytest.mark.parametrize("K", [1, 3])
@pytest.mark.parametrize("with_mu", [False, True])
@pytest.mark.parametrize("shape", [(1, 1, 5120), (6, 5120), (2, 3, 640)])
def test_fused_is_exact(dtype, K, with_mu, shape):
    y, R, mu, exact = _case(shape, K, with_mu, dtype)
    b = _run(True, y, 2.2, R, mu)
    assert b.shape == y.shape and b.dtype == y.dtype
    bv = np.array(b.astype(mx.float32)).astype(np.float64)
    if dtype == mx.float16:                                                   # the exact value rounded once to fp16: within one ulp
        ulp = np.maximum(np.abs(exact) * 2.0 ** -10, 2.0 ** -24)
        assert (np.abs(bv - exact) <= ulp * 1.0001).all(), np.abs(bv - exact).max()
    else:
        assert np.abs(bv - exact).max() <= 2e-6, np.abs(bv - exact).max()


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
def test_one_row_matches_stock(dtype):
    y, R, mu, _ = _case((1, 1, 5120), 1, True, dtype, seed=3)
    a = np.array(_run(False, y, 2.2, R, mu).astype(mx.float32))
    b = np.array(_run(True, y, 2.2, R, mu).astype(mx.float32))
    if dtype == mx.float16:
        assert (np.abs(a - b) <= np.maximum(np.abs(a) * 2.0 ** -10, 2.0 ** -24) * 1.0001).all()
        assert (a != b).mean() < 0.01                                        # equal in > 99 % of the columns
    else:
        assert np.abs(a - b).max() <= 2e-6


def test_rows_are_row_count_invariant():
    y, R, mu, _ = _case((6, 5120), 1, True, mx.float16, seed=5)
    full = np.array(_run(True, y, 2.2, R, mu))
    one = np.concatenate([np.array(_run(True, y[i:i + 1], 2.2, R, mu)) for i in range(6)])
    assert np.array_equal(full, one)


def test_fused_saturates_fp16_inputs_like_stock():
    H = 5120
    yn = np.ones((1, H), dtype=np.float32)
    yn[0, 7], yn[0, 9] = np.inf, -np.inf
    y = mx.array(yn).astype(mx.float16)
    _, R, _, _ = _case((1, H), 1, False, mx.float16, seed=9)
    a = np.array(_run(False, y, 2.0, R, None).astype(mx.float32))
    b = np.array(_run(True, y, 2.0, R, None).astype(mx.float32))
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert np.allclose(a, b, atol=0.07, rtol=1e-3)


def test_flag_off_is_stock_path():
    H = 640
    y = mx.array(np.random.default_rng(1).standard_normal((2, H)).astype(np.float16))
    _, R, _, _ = _case((2, H), 1, False, mx.float16, seed=11)
    a = _run(False, y, 1.5, R, None)
    b = ablit._project_c(mx.clip(y, -65504, 65504), mx.array(1.5, dtype=mx.float32), R)
    assert np.array_equal(np.array(a), np.array(b))
