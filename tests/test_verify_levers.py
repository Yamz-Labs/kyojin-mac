"""Hy4 verify levers: batched rowdot (MLA absorb k_b / v_b at 2..8 rows) and the fused router weight normalisation (Metal only)."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(mx.default_device() != mx.gpu, reason="Metal kernels")


@pytest.mark.parametrize("H,L,K,N", [(8, 2, 128, 512), (8, 4, 512, 128), (16, 8, 128, 512), (4, 3, 100, 64)])
def test_absorb_matches_float64(H, L, K, N, monkeypatch):
    from kyojin_mlx import rowinv as RI
    monkeypatch.setattr(RI, "SKINNY_ABSORB", 8)
    mx.random.seed(1)
    x, w = mx.random.normal((1, H, L, K)), mx.random.normal((H, N, K))
    got = RI.absorb_mm(x, w)
    assert got is not None and got.shape == (1, H, L, N)
    ref = np.einsum("hlk,hnk->hln", np.array(x[0], dtype=np.float64), np.array(w, dtype=np.float64))
    assert np.abs(np.array(got[0]) - ref).max() < 1e-3


def test_absorb_off_or_out_of_range(monkeypatch):
    from kyojin_mlx import rowinv as RI
    x, w = mx.zeros((1, 4, 1, 16)), mx.zeros((4, 8, 16))
    monkeypatch.setattr(RI, "SKINNY_ABSORB", 8)
    assert RI.absorb_mm(x, w) is None                       # one row keeps the stock gemv
    monkeypatch.setattr(RI, "SKINNY_ABSORB", 0)
    assert RI.absorb_mm(mx.zeros((1, 4, 3, 16)), w) is None


@pytest.mark.parametrize("eps,scale", [(0.0, 1.0), (1e-6, 2.5)])
def test_route_norm(eps, scale):
    from kyojin_mlx import rowinv as RI
    w = mx.random.uniform(shape=(5, 8))
    tot = w.sum(-1, keepdims=True)
    tot = tot + eps if eps > 0 else mx.maximum(tot, 6.1035e-5)
    ref = w / tot * (scale if scale != 1.0 else 1.0)
    assert float(mx.abs(RI.route_norm(w, eps, 6.1035e-5, scale) - ref).max()) < 1e-6
