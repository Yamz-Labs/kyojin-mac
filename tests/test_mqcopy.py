"""Flag mqcopy: the resident MLX-native quantized copy of a 4-bit stored format decodes to the stored weights and serves the same product."""
import mlx.core as mx
import numpy as np
import pytest

from kyojin_mlx import foreign_gguf as GF
from kyojin_mlx import foreign_linear as FL
from kyojin_mlx import foreign_pack as FP
from kyojin_mlx import foreign_synth as SY

N, K = 96, 1024


def _pack(t):
    raw = SY.random_blocks(t, N, K, np.random.default_rng(3))
    return GF.to_pack(t, raw, K)


def _dp(t, monkeypatch):
    monkeypatch.setenv("KYOJIN_FOREIGN_MQCOPY", "1")
    monkeypatch.setattr(FL, "MQ_SMALL", 0)                  # the test matrices are far under the shape gates
    monkeypatch.setattr(FL, "MQ_QMV_N", 0)
    monkeypatch.setattr(FL, "MQ_Q5K", True)                 # the Q5_K copy is off by default (end-to-end loss), its mapping stays tested
    p = _pack(t)
    if p.fmt == "nvfp4":
        p.gscale = 0.37
    return p, FL.DevPack(p)


@pytest.mark.parametrize("t", ["Q4_0", "Q4_K", "Q5_K", "MXFP4", "NVFP4"])
def test_copy_decodes_to_stored_weights(t, monkeypatch):
    p, dp = _dp(t, monkeypatch)
    wq, sc, bi, mode, g, unit, bits = dp.mq
    w = np.array(mx.dequantize(wq, sc, bi, group_size=g, bits=bits, mode=mode).astype(mx.float32))
    ref = FP.dequant(p, g=False)
    assert np.abs(w - ref).max() <= {"Q4_0": 1e-3, "Q4_K": 2e-3, "Q5_K": 2e-3}.get(t, 0) * np.abs(ref).max()   # fp16 arithmetic of the affine decode / of d sc, dmin mn; fp4 modes exact


@pytest.mark.parametrize("t", ["Q4_0", "Q4_K", "Q5_K", "MXFP4", "NVFP4"])
@pytest.mark.parametrize("R", [12, 16, 32, 64])
def test_copy_route_matches_reference(t, R, monkeypatch):
    p, dp = _dp(t, monkeypatch)
    x = np.random.default_rng(4).standard_normal((R, K)).astype(np.float16)
    ref = x.astype(np.float32) @ FP.dequant(p).T
    assert dp.mq_rows_ok(R)
    got = np.array(dp.apply(mx.array(x)).astype(mx.float32))
    assert np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-6) < 5e-3


def test_flag_off_and_experts(monkeypatch):
    p = _pack("Q4_0")
    monkeypatch.setenv("KYOJIN_FOREIGN_MQCOPY", "0")
    assert FL.DevPack(p).mq is None
    monkeypatch.setenv("KYOJIN_FOREIGN_MQCOPY", "1")
    assert FL.DevPack([p, p]).mq is None
