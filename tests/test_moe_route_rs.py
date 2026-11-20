"""moe_route_rs: GenericMoE.route with the fused router (glm_moe_rs.rowsel, flag KYOJIN_MOE_ROUTE_RS) against the stock route of the same block (Hy4 shape: sigmoid, selection bias, norm, scale)."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
from kyojin_mlx import foreign_moe_generic as G  # noqa: E402


def block(E=512, K=16, H=3072, seed=0):
    rng = np.random.default_rng(seed)
    spec = G.MoeSpec(n_expert=E, top_k=K, gating="sigmoid", select_bias=True, norm_weights=True, weight_scale=3.998)
    t = {"gate_inp": mx.array((rng.standard_normal((E, H)) * 0.02).astype(np.float32)), "exp_probs_b": mx.array((rng.standard_normal((E,)) * 0.1).astype(np.float32)),
         "up_exps": None, "down_exps": None}
    blk = G.GenericMoE.__new__(G.GenericMoE)
    G.nn.Module.__init__(blk)
    blk.spec_ = spec
    blk.router_w, blk.router_b, blk.sel_b = t["gate_inp"], None, t["exp_probs_b"]
    return blk


@pytest.mark.parametrize("T", [1, 2, 4, 6, 8])
def test_route_matches_stock(T):
    blk = block()
    rng = np.random.default_rng(10 + T)
    for i in range(4):                                       # consecutive launches: the arrival counter resets itself
        x2 = mx.array((rng.standard_normal((T, 3072)) * 0.5).astype(np.float16))
        G.ROUTE_RS = False
        i0, w0 = blk.route(x2)
        G.ROUTE_RS = True
        try:
            i1, w1 = blk.route(x2)
        finally:
            G.ROUTE_RS = False
        assert i1.shape == (T, 16) and w1.shape == (T, 16)
        a, b = np.array(i0), np.array(i1)
        assert (a == b).all(), (a, b)
        np.testing.assert_allclose(np.array(w1), np.array(w0), rtol=2e-5)


def test_ineligible_goes_stock():
    blk = block(E=64, K=4, H=8192)                           # pick_ks(8192, 64) > 1: the fused kernel does not apply
    x2 = mx.array(np.zeros((2, 8192), np.float16))
    G.ROUTE_RS = True
    try:
        assert not blk._rs_ok(x2)
        blk.route(x2)
    finally:
        G.ROUTE_RS = False
