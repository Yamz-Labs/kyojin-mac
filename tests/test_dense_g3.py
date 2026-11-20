"""G3 dense GEMV (flag glmdense, KYOJIN_GLM_DENSE_GEMV): fp8 / fp8b linears of 1..4 rows equal dequant + matmul, at the real GLM-5.3 dense shapes."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from kyojin_mlx import foreign_linear as FL, foreign_metal as FM, foreign_pack as FP

needs_gpu = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal GPU")


def _pack(fmt, n, k, rng):
    q = rng.integers(0, 256, (n, k), dtype=np.uint8)
    q[(q & 127) == 127] = 0x38                                     # no NaN codes (the table kernels decode them to 0, G3 to +-480)
    if fmt == "fp8":
        return FP.Pack("fp8", n, k, q, None, rng.uniform(5e-5, 2e-4, n).astype(np.float32))
    g = rng.uniform(64.0, 127.0, (-(-n // 128), -(-k // 128))).astype(np.float32)
    return FP.Pack("fp8b", n, k, q, g[np.arange(n) // 128][:, (np.arange(k // 32) * 32) // 128].astype(np.float16), np.float32(2.0 ** -20))


def _close(y, ref):
    y, ref = np.array(y, np.float32), np.array(ref, np.float32)
    assert np.isfinite(y).all()
    assert np.abs(y - ref).max() <= 2e-3 * np.abs(ref).max() + 1e-3, np.abs(y - ref).max()


@needs_gpu
@pytest.mark.parametrize("fmt", ["fp8", "fp8b"])
@pytest.mark.parametrize("n,k", [(512, 4096), (1536, 4096), (16384, 1536), (4096, 2048), (4096, 12288), (8192, 4096), (4096, 16384), (4104, 2560)])
def test_g3_gemv(fmt, n, k, monkeypatch):
    monkeypatch.setattr(FL, "DENSE_GEMV", True)
    rng = np.random.default_rng(n + k)
    d = FL.DevPack(_pack(fmt, n, k, rng))
    w = d.dequant().astype(mx.float32)
    for R in (1, 2, 3, 4):
        assert d.g3_ok(R)
        x = mx.array(rng.normal(0, 1, (R, k)).astype(np.float16))
        _close(d.apply(x), x.astype(mx.float32) @ w.T)


@needs_gpu
def test_g3_off_and_rows(monkeypatch):
    rng = np.random.default_rng(1)
    d = FL.DevPack(_pack("fp8b", 256, 1024, rng))
    monkeypatch.setattr(FL, "DENSE_GEMV", False)
    assert not d.g3_ok(1)
    monkeypatch.setattr(FL, "DENSE_GEMV", True)
    assert d.g3_ok(4) and not d.g3_ok(5) and not d.g3_ok(0)
    assert not FL.DevPack(_pack("fp8b", 256, 768, rng)).g3_ok(1)           # k % 512


def test_g3_config_fits_threadgroup_memory():
    for r in (1, 2, 3, 4):
        for k in (512, 1536, 4096, 12288, 16384):
            assert r * FM.gemv8g3_kc(r, k) * 2 <= 32768 and FM.gemv8g3_kc(r, k) % 512 == 0


@needs_gpu
@pytest.mark.parametrize("dt", [mx.float32, mx.bfloat16])
def test_g3_staged_cast(dt, monkeypatch):
    """float32 / bfloat16 x: the kernel casts to half while staging = the same result as casting first."""
    monkeypatch.setattr(FL, "DENSE_GEMV", True)
    rng = np.random.default_rng(3)
    d = FL.DevPack(_pack("fp8b", 1024, 4096, rng))
    for R in (1, 2, 4):
        x = mx.array(rng.normal(0, 1, (R, 4096)).astype(np.float32)).astype(dt)
        assert d.g3_ok(R)
        y = d._gemv8g3(x)
        ref = d._gemv8g3(x.astype(mx.float16))
        assert np.array_equal(np.array(y.astype(mx.float32)), np.array(ref.astype(mx.float32)))


@needs_gpu
@pytest.mark.parametrize("k,limit", [(2048, 7.0), (12288, 3.5)])
def test_g3_glu(k, limit, monkeypatch):
    monkeypatch.setattr(FL, "DENSE_GEMV", True)
    rng = np.random.default_rng(k)
    d = FL.DevPack(_pack("fp8b", 4096, k, rng))
    w = d.dequant().astype(mx.float32)
    for R in (1, 3):
        y = mx.array((rng.normal(0, 3, (R, 2 * k))).astype(np.float16))
        gate, up = y[:, :k].astype(mx.float32), y[:, k:].astype(mx.float32)
        h = (mx.minimum(gate, limit) * mx.sigmoid(mx.minimum(gate, limit))) * mx.clip(up, -limit, limit)
        _close(d._gemv8g3(y, limit=limit), h @ w.T)
