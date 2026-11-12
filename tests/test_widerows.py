"""Wide-row stored products (foreign_wide): the register-fed NAX kernel against the float32 decode of the same stored bytes, every format, R 9..64, N off the 32 grid."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from kyojin_mlx import foreign_gguf as GF, foreign_linear as FL, foreign_pack as FP, foreign_synth as SY, foreign_metal as FM

needs_gpu = pytest.mark.skipif(not mx.metal.is_available(), reason="needs a Metal GPU")
FORMATS = ("nvfp4", "mxfp4", "q4_0", "q4_k", "q5_k", "q6_k", "iq4_xs", "q3_k", "q2_k", "iq3_s", "iq3_xxs", "iq2_xxs", "iq2_xs", "iq2_s", "iq1_s", "iq1_m", "q8_0", "fp8", "fp8b")


def _pack(fmt, n, k, seed=0):
    rng = np.random.default_rng(seed)
    if fmt == "nvfp4":
        return FP.Pack("nvfp4", n, k, rng.integers(0, 256, (n, k // 2), dtype=np.uint8), rng.integers(0x28, 0x46, (n, k // 32, 2), dtype=np.uint8).view(np.uint16).reshape(n, k // 32), 1e-4)
    if fmt == "fp8":     # E4M3 bytes without the NaN codes, one scale per row
        w8 = rng.integers(0, 256, (n, k), dtype=np.uint8)
        w8[(w8 & 127) == 127] = 0x38
        return FP.Pack("fp8", n, k, w8, None, rng.uniform(0.5, 2.0, n).astype(np.float32))
    if fmt == "fp8b":    # E4M3 bytes, block scale per 32-unit constant over 128 x 128 blocks, power-of-two row scale (the loader's layout)
        w8 = rng.integers(0, 256, (n, k), dtype=np.uint8)
        w8[(w8 & 127) == 127] = 0x38
        g = rng.uniform(16.0, 120.0, (-(-n // 128), -(-k // 128))).astype(np.float32)
        return FP.Pack("fp8b", n, k, w8, g[np.arange(n) // 128][:, (np.arange(k // 32) * 32) // 128].astype(np.float16), np.float32(2.0 ** -7))
    return GF.to_pack(fmt.upper(), SY.random_blocks(fmt.upper(), n, k, rng), k)


@needs_gpu
@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("R", (9, 16, 17, 33, 48, 49, 64))
def test_wide_matches_decode(fmt, R):
    d = FL.DevPack(_pack(fmt, 100, 1024, seed=R))
    x = (np.random.default_rng(R).standard_normal((R, 1024)) * 0.5).astype(np.float16)
    y = d._wide(mx.array(x), share=False)
    ref = x.astype(np.float32) @ np.array(d.dequant().astype(mx.float32)).T
    y = np.array(y.astype(mx.float32))
    assert y.shape == (R, 100)
    assert np.max(np.abs(y - ref)) <= 3e-3 * max(1.0, np.max(np.abs(ref)))


@needs_gpu
@pytest.mark.parametrize("fmt", ("nvfp4", "q4_k", "q6_k", "fp8", "q8_0", "fp8b"))
def test_wide_route(fmt, monkeypatch):
    """Rows 9..WIDE_MAX go through the wide product, 1..8 and above do not; KYOJIN_FOREIGN_WIDE=0 switches it off."""
    d = FL.DevPack(_pack(fmt, 64, 512))
    calls = []
    orig = d._wide
    monkeypatch.setattr(d, "_wide", lambda x, share=True: (calls.append(x.shape[0]), orig(x, share))[1])
    hi = FL.WIDE_MAX_FMT[fmt]
    for R in (1, 8, 9, hi, hi + 1):
        d.apply(mx.zeros((R, 512), mx.float16))
    assert calls == [9, hi]
    monkeypatch.setenv("KYOJIN_FOREIGN_WIDE", "0")
    d.apply(mx.zeros((12, 512), mx.float16))
    assert calls == [9, hi]


@needs_gpu
@pytest.mark.parametrize("fmt", FORMATS)
def test_apply_every_row_count(fmt):
    """The routed product (whichever kernel serves the row count) matches the decode of the stored bytes for R = 1..64."""
    d = FL.DevPack(_pack(fmt, 96, 1024, seed=3))
    W = np.array(d.dequant().astype(mx.float32))
    for R in (1, 2, 5, 8, 9, 16, 24, 32, 33, 40, 48, 49, 56, 64):
        x = (np.random.default_rng(R).standard_normal((R, 1024)) * 0.5).astype(np.float16)
        y = np.array(d.apply(mx.array(x)).astype(mx.float32))
        ref = x.astype(np.float32) @ W.T
        assert y.shape == (R, 96)
        assert np.max(np.abs(y - ref)) <= 1e-2 * max(1.0, np.max(np.abs(ref))), (fmt, R)
