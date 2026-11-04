"""Launch cuts (kyojin_mlx.fuse): every fused kernel is bit-identical to the stock op sequence it replaces. Needs a Mac (Metal)."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn                                                        # noqa: E402
from kyojin_mlx import fuse as F                                           # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")


def bits(a):
    a = np.array(a)
    return a.view(np.uint32 if a.dtype == np.float32 else np.uint16) if a.dtype in (np.float32, np.float16) else a


def stock_router(logits, bias, K, norm, scale):
    scores = mx.sqrt(nn.softplus(logits))
    inds = mx.argpartition(-(scores + bias), kth=K - 1, axis=-1)[..., :K].astype(mx.int32)
    w = mx.take_along_axis(scores, inds, axis=-1)
    if norm and K > 1:
        w = w / (w.sum(axis=-1, keepdims=True) + 1e-20)
    return inds, w * scale


def logits_cases(rng, R, E):
    yield "normal", rng.standard_normal((R, E)) * 3
    yield "wide", rng.standard_normal((R, E)) * 40
    yield "extreme", rng.choice([-300.0, -90.0, -30.0, -1e-3, 0.0, 1e-3, 30.0, 90.0, 300.0], size=(R, E))
    yield "ties", np.round(rng.standard_normal((R, E)) * 2) / 2              # many equal scores: tie break by index
    yield "all_equal", np.full((R, E), 0.25)


@pytest.mark.parametrize("E", [32, 256, 384])
@pytest.mark.parametrize("K,norm,scale", [(6, True, 1.5), (8, True, 2.5), (6, False, 1.0), (1, True, 1.0)])
def test_router_select_bit_identical(E, K, norm, scale):
    rng = np.random.default_rng(E * 31 + K)
    for R in (1, 3, 8):
        for name, lg in logits_cases(rng, R, E):
            for bname, bias in (("zero", np.zeros(E)), ("rand", rng.standard_normal(E) * 0.5), ("big", rng.standard_normal(E) * 20)):
                lgm, bm = mx.array(lg.astype(np.float32)), mx.array(bias.astype(np.float32))
                ri, rw = stock_router(lgm, bm, K, norm, scale)
                fi, fw = F.router_select(lgm, bm, K, norm and K > 1, scale)
                assert np.array_equal(np.array(ri), np.array(fi)), (E, K, R, name, bname)
                assert np.array_equal(bits(rw), bits(fw)), (E, K, R, name, bname)


def test_router_select_row_invariant():
    rng = np.random.default_rng(7)
    lg = mx.array((rng.standard_normal((8, 384)) * 3).astype(np.float32))
    bias = mx.array((rng.standard_normal(384) * 0.5).astype(np.float32))
    ai, aw = F.router_select(lg, bias, 6, True, 1.5)
    for r in range(8):
        bi, bw = F.router_select(lg[r:r + 1], bias, 6, True, 1.5)
        assert np.array_equal(np.array(ai)[r], np.array(bi)[0]) and np.array_equal(bits(aw)[r], bits(bw)[0])


def test_router_select_image_mask():
    rng = np.random.default_rng(11)
    E, K = 384, 6
    lg = mx.array((rng.standard_normal((6, E)) * 3).astype(np.float32))
    b = mx.array((rng.standard_normal(E) * 0.5).astype(np.float32))
    bvl = mx.array((rng.standard_normal(E) * 5).astype(np.float32))
    mask = mx.array([True, False, True, False, False, True])
    scores = mx.sqrt(nn.softplus(lg))
    bias = mx.where(mask[:, None], bvl, b)
    ri = mx.argpartition(-(scores + bias), kth=K - 1, axis=-1)[..., :K].astype(mx.int32)
    rw = mx.take_along_axis(scores, ri, axis=-1)
    rw = (rw / (rw.sum(axis=-1, keepdims=True) + 1e-20)) * 1.5
    fi, fw = F.router_select(lg, b, K, True, 1.5, bvl, mask)
    assert np.array_equal(np.array(ri), np.array(fi)) and np.array_equal(bits(rw), bits(fw))


def _lang():
    return pytest.importorskip("mlx_vlm.models.deepseek_v41.language")


@pytest.mark.parametrize("dt", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("B,D", [(1, 5120), (3, 5120), (8, 4096), (2, 130)])
def test_mhc_expand_bit_identical(dt, B, D):
    L = _lang()
    rng = np.random.default_rng(B * 7 + D)
    for scale in (1.0, 30.0, 0.01):
        x = mx.array((rng.standard_normal((B, 1, D)) * scale).astype(np.float32)).astype(dt)
        res = mx.array((rng.standard_normal((B, 1, 4, D)) * scale).astype(np.float32)).astype(dt)
        post = mx.array((rng.random((B, 1, 4)) * 2).astype(np.float32))
        comb = mx.array((rng.random((B, 1, 4, 4)) * 0.5).astype(np.float32))
        ref = L.hc_expand(x, res, post, comb)
        y, y32 = F.mhc_expand(x, res, post, comb)
        assert y.dtype == dt and y.shape == ref.shape
        assert np.array_equal(np.array(ref.astype(mx.float32)), np.array(y.astype(mx.float32)))
        assert np.array_equal(np.array(y32), np.array(y.astype(mx.float32)))


@pytest.mark.parametrize("dt", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("B,D", [(1, 5120), (5, 5120), (2, 130)])
def test_mhc_pre_bit_identical(dt, B, D):
    L = _lang()
    rng = np.random.default_rng(B * 5 + D)
    for scale in (1.0, 30.0):
        x = mx.array((rng.standard_normal((B, 1, 4, D)) * scale).astype(np.float32)).astype(dt)
        pre = mx.array((rng.random((B, 1, 4))).astype(np.float32))
        ref = L.DeepseekV41Block.hc_pre(x, pre)
        y = F.mhc_pre(x, pre)
        assert y.dtype == dt and np.array_equal(np.array(ref.astype(mx.float32)), np.array(y.astype(mx.float32)))
