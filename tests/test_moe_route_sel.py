"""moe_route_sel: the one-launch router selection (flag KYOJIN_MOE_ROUTE_SEL) against the stock composition on the same logits.

Indices must be EXACT (tie rule: value descending, lowest expert index first = stable argsort), weights within fp32 tolerance.
Light Metal test: one small launch per case. Run as one file, never beside a timed chain cell."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
from kyojin_mlx import foreign_moe_generic as G  # noqa: E402
from kyojin_mlx import moe_route_sel as S  # noqa: E402


def logits(T, E, seed, quant=None):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, E)).astype(np.float32) * 1.5
    if quant:                                   # forced ties: only `quant` distinct values
        x = np.round(x * quant) / quant
    return mx.array(x)


def check(lg, **kw):
    K = kw["K"]
    rb, sb = kw.pop("router_bias", None), kw.pop("sel_bias", None)
    i1, w1 = S.select(lg, rb, sb, **kw)
    i0, w0 = S.route_ref(lg, rb, sb, **kw)
    a, b = np.array(i0), np.array(i1)
    assert i1.shape == (lg.shape[0], K) and w1.shape == (lg.shape[0], K)
    assert (a == b).all(), (a, b)
    np.testing.assert_allclose(np.array(w1), np.array(w0), rtol=3e-5, atol=1e-7)


def vec(E, seed, s=0.1):
    return mx.array((np.random.default_rng(seed).standard_normal((E,)) * s).astype(np.float32))


@pytest.mark.parametrize("T", [1, 2, 4, 8, 16])
def test_hy4_shape(T):                          # sigmoid, selection bias, norm, scale 3.998, E 512, K 16
    for i in range(3):
        check(logits(T, 512, 100 * T + i), K=16, gating="sigmoid", sel_bias=vec(512, 1), norm=True, scale=3.998)


@pytest.mark.parametrize("gating", ["softmax", "sigmoid", "sqrt_softplus", "none", "softmax_weight"])
@pytest.mark.parametrize("norm,eps", [(False, 0.0), (True, 0.0), (True, 1e-20)])
def test_gatings(gating, norm, eps):
    check(logits(5, 128, 7), K=8, gating=gating, norm=norm, eps=eps, scale=2.5, router_bias=vec(128, 2), sel_bias=vec(128, 3) if gating != "none" else None)


def test_no_bias_unit_scale():
    check(logits(3, 64, 11), K=4, gating="softmax", norm=True)


def test_temperature():
    check(logits(3, 96, 12), K=6, gating="softmax", norm=True, temperature=0.7)


@pytest.mark.parametrize("E,K,ng,gu", [(256, 8, 8, 4), (64, 6, 8, 3), (160, 6, 8, 3), (512, 16, 16, 4)])
def test_groups(E, K, ng, gu):
    for gating in ("sigmoid", "softmax"):
        check(logits(4, E, E + ng), K=K, gating=gating, sel_bias=vec(E, 5), norm=True, scale=2.0, n_group=ng, group_used=gu)


@pytest.mark.parametrize("gating", ["sigmoid", "softmax", "none"])
def test_ties_lowest_index(gating):
    # logits quantised to 8 levels: dozens of exact ties around the K-th place; the kernel and the stable argsort must choose the same experts
    lg = logits(8, 512, 21, quant=2)
    check(lg, K=16, gating=gating, norm=True, scale=1.5)
    i, _ = S.select(lg, None, None, K=16, gating=gating, norm=True)
    sc = np.array(lg)
    for r in range(8):                          # the chosen set is the K best by (value desc, index asc)
        want = np.argsort(-sc[r], kind="stable")[:16]
        assert (np.array(i)[r] == want).all()


def test_ties_grouped():
    check(logits(8, 256, 22, quant=2), K=8, gating="sigmoid", sel_bias=None, norm=True, n_group=8, group_used=4)


def test_masked_fewer_than_k():                 # 2 groups of 4 kept of 8 groups of 4 = 8 candidates, K = 8 (all candidates are taken, in rank order)
    check(logits(2, 32, 23), K=8, gating="sigmoid", norm=True, n_group=8, group_used=2)


def test_eligible():
    assert not S.eligible(17, 512, 16)
    assert not S.eligible(1, 2048, 16)
    assert not S.eligible(1, 100, 8, n_group=8, group_used=4)            # 100 % 8 != 0
    assert not S.eligible(1, 64, 33)
    assert S.eligible(16, 512, 16)


def test_route_flag_matches_stock():             # GenericMoE.route with the flag against the same block without it
    E, K, H = 512, 16, 256
    rng = np.random.default_rng(0)
    spec = G.MoeSpec(n_expert=E, top_k=K, gating="sigmoid", select_bias=True, norm_weights=True, weight_scale=3.998)
    blk = G.GenericMoE.__new__(G.GenericMoE)
    G.nn.Module.__init__(blk)
    blk.spec_ = spec
    blk.router_w, blk.router_b, blk.sel_b = mx.array((rng.standard_normal((E, H)) * 0.05).astype(np.float32)), None, vec(E, 4)
    for T in (1, 2, 4, 8, 16):
        x2 = mx.array((rng.standard_normal((T, H)) * 0.5).astype(np.float16))
        G.ROUTE_SEL = False
        i0, w0 = blk.route(x2)
        G.ROUTE_SEL = True
        try:
            i1, w1 = blk.route(x2)
        finally:
            G.ROUTE_SEL = False
        assert (np.array(i0) == np.array(i1)).all()
        np.testing.assert_allclose(np.array(w1), np.array(w0), rtol=3e-5)
