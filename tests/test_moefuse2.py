"""OC_moefuse2: the KYOJIN_MOEFUSE=1 variant (election sharing via the union path) is
bit-identical to the stock gather path and to the per-expert reference at small
random dims, R=1..4, distinct/shared routing. Needs a Mac (Metal)."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

from kyojin_mlx import moe as M  # noqa: E402
from kyojin_mlx import moe_fused as MF  # noqa: E402
from test_moe import E, KS, TOPK, _route, _x, make_stack  # noqa: E402
from test_moe_union import route  # noqa: E402


@pytest.fixture(autouse=True)
def stock_decode(monkeypatch):
    """The references are the stock decode: KYOJIN_MOE_FASTDEC (mul1fd, not bit-identical, gated in test_moe_fastdec) stays off here."""
    monkeypatch.setattr(M, "FASTDEC", False)


@pytest.fixture(scope="module", params=["mul1", "mcg"])
def layer(request):
    cb = request.param
    kin, ninter = 256, 384
    g, _ = make_stack(kin, ninter, KS, cb, 1)
    u, _ = make_stack(kin, ninter, KS, cb, 2)
    d, _ = make_stack(ninter, kin, KS, cb, 3)
    return g, u, d


def _check_fused_bitwise(g, u, d, rows, ids, w, x, **kw):
    stock = M.MoeGather(g, u, d, TOPK, **kw)
    stock._is_fused_inner = True  # hold the reference on the stock path
    old = MF.MOEFUSE
    MF.MOEFUSE = True
    try:
        m = M.MoeGather(g, u, d, TOPK, **kw)
        yf, ys = m(x, ids, w), stock(x, ids, w)
        mx.eval(yf, ys)
    finally:
        MF.MOEFUSE = old
    assert np.array_equal(np.array(yf).view(np.uint32), np.array(ys).view(np.uint32)), (rows, kw)


@pytest.mark.parametrize("rows", [1, 2, 3, 4])
@pytest.mark.parametrize("mode", ["distinct", "identical", "partial"])
def test_flag_path_matches_stock(layer, rows, mode):
    """KYOJIN_MOEFUSE=1 == stock gather, bitwise, production flags (fq + wpre)."""
    g, u, d = layer
    ids, w = route(rows, mode, 7)
    _check_fused_bitwise(g, u, d, rows, ids, w, _x(rows, g.kin, 8),
                         limit=10.0, wpre=True, fq=True)


@pytest.mark.parametrize("rows", [1, 2, 3, 4])
@pytest.mark.parametrize("wpre", [False, True])
def test_flag_path_matches_stock_nofq(layer, rows, wpre):
    g, u, d = layer
    ids, w = _route(rows, 9)
    _check_fused_bitwise(g, u, d, rows, ids, w, _x(rows, g.kin, 10), limit=10.0, wpre=wpre)


@pytest.mark.parametrize("rows", [1, 4])
def test_flag_path_matches_unfused(layer, rows):
    """Flag path vs the per-expert reference with matched k-splits: bitwise."""
    g, u, d = layer
    for tgk in (1, 2):
        kw = dict(limit=10.0, wpre=True, fq=False, tgk_gu=tgk, tgk_dn=tgk)
        ids, w = _route(rows, 11)
        x = _x(rows, g.kin, 12)
        old = MF.MOEFUSE
        MF.MOEFUSE = True
        try:
            yf = M.MoeGather(g, u, d, TOPK, **kw)(x, ids, w)
            mx.eval(yf)
        finally:
            MF.MOEFUSE = old
        yr = M.MoeUnfused(g, u, d, TOPK, **kw)(x, ids, w)
        mx.eval(yr)
        got, ref = np.array(yf), np.array(yr)
        assert np.array_equal(got.view(np.uint32), ref.view(np.uint32)), (rows, tgk)
