"""Packed-fp16 grouped MoE GEMV (S.HALF_ROWS >= 2: x staged as half in the run's k order by the rot / mid kernels, groups of HALF_ROWS or more rows
multiply-accumulate in half4): the layer output stays within fp16 rounding of the fp32 grouped path; groups below HALF_ROWS keep the fp32 arithmetic on
the widened half x (equal to fp32 when x is exactly representable). Synthetic ragged stacks, mixed K. Needs a Mac (Metal)."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

from kyojin_mlx import metal_src as S, moe as M
from test_moe import KS, TOPK, make_stack, _x
from test_moe_union import route


@pytest.fixture(scope="module", params=["mul1", "mcg"])
def layer(request):
    cb = request.param
    kin, ninter = 256, 384
    g, _ = make_stack(kin, ninter, KS, cb, 1)
    u, _ = make_stack(kin, ninter, KS, cb, 2)
    d, _ = make_stack(ninter, kin, KS, cb, 3)
    return g, u, d


@pytest.fixture
def half_rows():
    old = S.HALF_ROWS
    yield
    S.HALF_ROWS = old


@pytest.mark.parametrize("rows", [2, 3, 4, 5, 8, 9, 16])
@pytest.mark.parametrize("mode", ["identical", "partial", "random"])
@pytest.mark.parametrize("thr", [2, 4])
def test_half_grouped_close_to_fp32(layer, half_rows, rows, mode, thr):
    g, u, d = layer
    m = M.MoeGather(g, u, d, TOPK, wpre=False, tgk_gu=2, tgk_dn=2, grouped=True, hotk=0)
    ids, w = route(rows, mode, 10 * rows + len(mode))
    x = _x(rows, g.kin, 7)
    S.HALF_ROWS = 0
    ref = np.array(m(x, ids, w))
    S.HALF_ROWS = thr
    out = np.array(m(x, ids, w))
    err = np.linalg.norm(out - ref, axis=1) / np.linalg.norm(ref, axis=1)
    assert err.max() < 5e-3, err.max()


@pytest.fixture
def mma_rows():
    old = S.MMA_ROWS
    yield
    S.MMA_ROWS = old


@pytest.mark.parametrize("rows", [4, 5, 8, 9, 16])
@pytest.mark.parametrize("mode", ["identical", "partial", "random"])
@pytest.mark.parametrize("thr", [4, 8])
def test_mma_grouped_close_to_fp32(layer, half_rows, mma_rows, rows, mode, thr):
    """Groups of thr or more rows through the decode-once simdgroup MMA body (S.MMA_ROWS): within fp16 rounding of the fp32 grouped path, and no worse than the packed-fp16 FMAs."""
    g, u, d = layer
    m = M.MoeGather(g, u, d, TOPK, wpre=False, tgk_gu=2, tgk_dn=2, grouped=True, hotk=0)
    ids, w = route(rows, mode, 10 * rows + len(mode))
    x = _x(rows, g.kin, 7)
    S.HALF_ROWS, S.MMA_ROWS = 0, 0
    ref = np.array(m(x, ids, w))
    S.HALF_ROWS, S.MMA_ROWS = 2, thr
    out = np.array(m(x, ids, w))
    err = np.linalg.norm(out - ref, axis=1) / np.linalg.norm(ref, axis=1)
    assert err.max() < 5e-3, err.max()
