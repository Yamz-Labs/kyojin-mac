"""fmm8 (flags foreign_fmm8 / foreign_fmm8_max / head_fmm8): the few-row simdgroup-matrix Q8_0 product against a float64 numpy product of the stored bytes and against gemv8.

CPU part: tiling, route predicates, flag defaults. Metal part (light, one small launch per case; run as one file, never beside a timed cell): rows 1..16, n with tails (not a
multiple of 8 NT), k with fewer units than simdgroups, float32 output, the head pack, DevPack.apply routing, fmm8 vs gemv8 (same products, other sum order)."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from kyojin_mlx import fmm8 as F8, foreign_linear as FL, foreign_pack as FP, flags  # noqa: E402

needs_metal = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")


def _pack(n, k, rng):
    q = rng.integers(-128, 128, (n, k), dtype=np.int8)
    return FP.Pack("q8_0", n, k, q.view(np.uint8), rng.uniform(1e-3, 3e-3, (n, k // 32)).astype(np.float16), np.ones(n, np.float32))


def test_tiling_and_defaults():
    assert F8.tiling(4, 129280, 3072) == (8, 4, 1)
    assert F8.tiling(4, 3072, 4096) == (16, 2, 1)
    assert F8.tiling(4, 1536, 4096) == (16, 1, 1)
    assert F8.tiling(12, 129280, 3072)[2] == 2
    assert F8.tiling(4, 4096, 64)[0] == 2                              # two units: two simdgroups
    assert F8.MIN_ROWS == 0 and F8.MAX_ROWS == 8                       # default off
    assert flags.FLAGS["foreign_fmm8"].default == "0" if hasattr(flags.FLAGS["foreign_fmm8"], "default") else True


@needs_metal
@pytest.mark.parametrize("R", [1, 2, 3, 5, 8, 9, 13, 16])
@pytest.mark.parametrize("n,k", [(264, 256), (136, 96), (520, 1024), (24, 32)])
def test_run_vs_numpy(R, n, k):
    rng = np.random.default_rng(R * 1000 + n + k)
    d = FL.DevPack(_pack(n, k, rng))
    x = mx.array(rng.standard_normal((R, k)).astype(np.float16) * 3.0)
    ref = F8.np_reference(x, d.qs, d.meta, d.gs)
    y = F8.run(x, d.qs, d.meta, d.gs, n, k, mx.float32)
    np.testing.assert_allclose(np.array(y), ref, rtol=2e-5, atol=1e-3)
    yh = F8.run(x, d.qs, d.meta, d.gs, n, k, mx.float16)
    assert yh.dtype == mx.float16
    np.testing.assert_allclose(np.array(yh).astype(np.float64), ref, rtol=2e-3, atol=2e-2)


@needs_metal
@pytest.mark.parametrize("R", [2, 4, 6, 8])
def test_vs_gemv8_and_routing(R):
    rng = np.random.default_rng(R)
    n, k = 512, 2048
    d = FL.DevPack(_pack(n, k, rng))
    x = mx.array(rng.standard_normal((R, k)).astype(np.float16))
    old = (F8.MIN_ROWS, F8.MAX_ROWS)
    try:
        F8.MIN_ROWS, F8.MAX_ROWS = 0, 8
        assert not d.fmm8_ok(R)
        y0 = d.apply(x)
        F8.MIN_ROWS = 2
        assert d.fmm8_ok(R) and d.cf_ok(R) and d.wide_stored(R)
        assert not d.fmm8_ok(1) and not d.fmm8_ok(9)
        y1 = d.apply(x)
    finally:
        F8.MIN_ROWS, F8.MAX_ROWS = old
    np.testing.assert_allclose(np.array(y1).astype(np.float64), np.array(y0).astype(np.float64), rtol=3e-3, atol=3e-3)


@needs_metal
def test_head_pack():
    rng = np.random.default_rng(1)
    w = mx.array(rng.standard_normal((200, 128)).astype(np.float32) * 0.05)
    pk = F8.head_pack(w)
    assert F8.head_pack(w) is pk                                         # built once
    x = mx.array(rng.standard_normal((3, 128)).astype(np.float16))
    y = np.array(pk.apply(x, mx.float32))
    ref = np.array(x).astype(np.float64) @ np.array(w).astype(np.float64).T
    assert np.abs(y - ref).max() < 0.02 * np.abs(ref).max()             # 8-bit rounding of the weights


@needs_metal
@pytest.mark.parametrize("R", [2, 6])
def test_float32_x_cast_free(R):
    rng = np.random.default_rng(R)
    n, k = 200, 512
    d = FL.DevPack(_pack(n, k, rng))
    xf = mx.array(rng.standard_normal((R, k)).astype(np.float32) * 5.0)
    y_cf = F8.run(xf, d.qs, d.meta, d.gs, n, k, mx.float32)                          # x float32 read as stored, rounded to half in the kernel
    y_ref = F8.run(xf.astype(mx.float16), d.qs, d.meta, d.gs, n, k, mx.float32)
    assert np.array_equal(np.array(y_cf), np.array(y_ref))
    old = (F8.MIN_ROWS, F8.MAX_ROWS)
    try:
        F8.MIN_ROWS, F8.MAX_ROWS = 2, 8
        assert d.cf_ok(R)
        y2 = d.gemv8_cf(xf, mx.float32)
    finally:
        F8.MIN_ROWS, F8.MAX_ROWS = old
    assert np.array_equal(np.array(y2), np.array(y_ref))
