"""Flag zq: NVFP4 / MXFP4 / FP8 stored bytes read in place by MLX's quantized matmul (no copy), and the input scale that keeps the raw fp16 product finite."""
import mlx.core as mx
import numpy as np
import pytest

from kyojin_mlx import foreign_linear as FL
from kyojin_mlx import foreign_pack as FP

N, K = 96, 1024
RS = [1, 2, 3, 4, 6, 8, 12, 16, 17, 24, 32, 33, 48, 64]


def _nvfp4(gs):
    rng = np.random.default_rng(5)
    return FP.Pack("nvfp4", N, K, rng.integers(0, 256, (N, K // 2), dtype=np.uint8),
                   rng.integers(0x28, 0x46, (N, K // 32, 2), dtype=np.uint8).view(np.uint16).reshape(N, K // 32), gs)


def _mxfp4():
    rng = np.random.default_rng(6)
    return FP.Pack("mxfp4", N, K, rng.integers(0, 256, (N, K // 2), dtype=np.uint8), rng.integers(118, 130, (N, K // 32), dtype=np.uint8), 1.0)


def _fp8(gs):
    rng = np.random.default_rng(7)
    w = rng.integers(0, 256, (N, K), dtype=np.uint8)
    w[(w & 0x7F) == 0x7F] = 0x38                              # E4M3 NaN codes
    return FP.Pack("fp8", N, K, w, None, (rng.uniform(0.5, 2.0, N) * gs).astype(np.float32))


@pytest.fixture(autouse=True)
def _gates(monkeypatch):
    monkeypatch.setattr(FL, "MQ_SMALL", 0)                     # the test matrices are far under the shape gates
    monkeypatch.setattr(FL, "MQ_QMV_N", 0)


@pytest.mark.parametrize("R", RS)
@pytest.mark.parametrize("mk", [lambda: _nvfp4(1.1e-4), lambda: _nvfp4(1.0), _mxfp4, lambda: _fp8(1e-2)], ids=["nvfp4_small_gs", "nvfp4_unit", "mxfp4", "fp8"])
def test_every_row_count_matches_reference(mk, R):
    p = mk()
    dp = FL.DevPack(p)
    x = np.random.default_rng(8).standard_normal((R, K)).astype(np.float16) * 4        # large inputs: the raw product of a 1/gscale scaled weight overflows fp16
    ref = x.astype(np.float32) @ FP.dequant(p).T
    got = np.array(dp.apply(mx.array(x)).astype(mx.float32))
    assert np.isfinite(got).all()
    assert np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-6) < 5e-3


@pytest.mark.parametrize("R", [2, 4, 24, 64])
def test_nvfp4_route_is_quantized_matmul_without_a_copy(R):
    p = _nvfp4(1.1e-4)
    dp = FL.DevPack(p)
    assert dp.mq is None and dp.w16 is None
    assert dp._zq_route(R)
    wq, sc, _, mode, g, bits = FP.mq_arrays(p)
    zq, zs = dp._zq_view()
    assert np.array_equal(np.array(zq), wq) and np.array_equal(np.array(zs), sc)       # MLX's layout is the stored one
    assert sum(a.nbytes for a in dp.arrays()) == dp.nbytes()


def test_fp8_large_rows_use_mxfp8(monkeypatch):
    dp = FL.DevPack(_fp8(1e-2))
    assert dp._zq8_ok(64) and dp._zq8_ok(20)                                        # good shape class (gates patched open)
    monkeypatch.setattr(FL, "MQ_QMV_N", 10 ** 9)
    assert dp._zq8_ok(64) and not dp._zq8_ok(20)
    monkeypatch.setattr(FL, "ZQ", False)
    assert not dp._zq8_ok(64)


@pytest.mark.parametrize("R", [17, 20, 29, 32, 47, 48])
@pytest.mark.parametrize("mk", [lambda: _nvfp4(1.1e-4), lambda: _fp8(1e-2)], ids=["nvfp4", "fp8"])
def test_other_shape_class_pads_rows_and_matches(mk, R, monkeypatch):
    monkeypatch.setattr(FL, "MQ_QMV_N", 10 ** 9)                # no matrix is of the good class: rows 17..47 run padded to 48
    p = mk()
    dp = FL.DevPack(p)
    x = np.random.default_rng(9).standard_normal((R, K)).astype(np.float16) * 4
    ref = x.astype(np.float32) @ FP.dequant(p).T
    got = np.array(dp.apply(mx.array(x)).astype(mx.float32))
    assert got.shape == ref.shape and np.isfinite(got).all()
    assert np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-6) < 5e-3
    assert bool(dp._zq_route(R)) == (p.fmt == "nvfp4") or p.fmt == "fp8"


@pytest.mark.parametrize("t", ["Q4_0", "Q4_K"])
@pytest.mark.parametrize("R", [17, 24, 32, 33])
def test_gguf_4bit_rows_17_to_32_two_launches(t, R):
    from kyojin_mlx import foreign_gguf as GF
    from kyojin_mlx import foreign_synth as SY
    p = GF.to_pack(t, SY.random_blocks(t, 128, 1024, np.random.default_rng(3)), 1024)
    dp = FL.DevPack(p)
    x = np.random.default_rng(4).standard_normal((R, 1024)).astype(np.float16)
    ref = x.astype(np.float32) @ FP.dequant(p).T
    got = np.array(dp.apply(mx.array(x)).astype(mx.float32))
    assert np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-6) < 5e-3
