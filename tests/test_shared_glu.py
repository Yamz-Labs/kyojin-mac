"""shared_glu: the shared expert's clamped SwiGLU + fp8 round trip inside the prologue of the down EXL3 GEMV (Mul1Gemv.fused_glu) vs the stock chain
(compiled swiglu launch, fp8 round trip, fused down GEMV): bit-identical, 1-4 rows, with and without the clamp, repeated calls."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

from kyojin_mlx import exl3_linear as EL
from kyojin_mlx import fqfuse as FQF
from kyojin_mlx import kernels as KN
from kyojin_mlx import pf_glue as PG
from kyojin_mlx import flags as FL, fired as FD
from tests.test_linear_fused import bits, trellis

PAIRS = [(512, 1280), (2048, 4096), (2304, 5120), (4096, 1024)]


def gu(rows, k, seed, scale):
    r = np.random.default_rng(seed)
    return (mx.array((r.standard_normal((rows, k)) * scale).astype(np.float16)), mx.array((r.standard_normal((rows, k)) * scale).astype(np.float16)))


@pytest.mark.parametrize("limit", [10.0, 0.0, 3.5])
@pytest.mark.parametrize("K", [2, 3])
@pytest.mark.parametrize("k,nout", PAIRS)
@pytest.mark.parametrize("rows", [1, 2, 3, 4])
def test_fused_glu_bit_identical(limit, K, k, nout, rows):
    tr, su, sv = trellis(k, nout, K, k + K)
    g = KN.Mul1Gemv(tr, su, sv, rows=rows, out_dtype=mx.float16, cb="mcg")
    assert g.can_fuse_glu(rows)
    for scale in (1.0, 12.0, 60.0):                                 # saturates the clamp, subnormal sigmoid tails, fp8 scale edges
        a, b = gu(rows, k, rows + int(scale), scale)
        act = PG._swiglu_h(limit, mx.float16)(a, b)
        want = g.fused(act, fq8=True)
        for _ in range(6):                                          # the counter sets rotate
            assert np.array_equal(bits(g.fused_glu(a, b, limit)), bits(want))


def test_flag_registered_and_off_by_default():
    assert FL.FLAGS["shared_glu"].default == "0"
    assert PG.SHARED_GLU in (False, True)


def test_module_routing_matches_stock(monkeypatch):
    import mlx.nn as nn
    from mlx_vlm.models.deepseek_v41 import language as L
    k, h = 1024, 2048
    rng = np.random.default_rng(3)
    mlp = type("M", (), {})()
    mlp.swiglu_limit = 10.0
    mlp.gate_proj, mlp.up_proj = nn.Linear(k, h, bias=False), nn.Linear(k, h, bias=False)
    for m in (mlp.gate_proj, mlp.up_proj):
        m.weight = (m.weight * 40).astype(mx.float16)
    tr, su, sv = trellis(h, k, 3, 5)
    mlp.down_proj = EL.Exl3Linear(mx.array(tr), mx.array(su), mx.array(sv), cb="mcg")
    stock = PG._impl_shared_mlp(lambda self, x: None)
    monkeypatch.setattr(FD, "ON", True)
    for rows in (1, 2, 4):
        x = mx.array(rng.standard_normal((rows, 1, k)).astype(np.float32)).astype(mx.float16)
        monkeypatch.setattr(PG, "SHARED_GLU", False)
        want = stock(mlp, x)
        monkeypatch.setattr(PG, "SHARED_GLU", True)
        n0 = FD.HITS.get("shared_glu", 0) if hasattr(FD, "HITS") else 0
        got = stock(mlp, x)
        assert np.array_equal(bits(got), bits(want))
        if hasattr(FD, "HITS"):
            assert FD.HITS.get("shared_glu", 0) > n0, "the fused path was not taken"
