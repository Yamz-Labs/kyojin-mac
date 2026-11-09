"""KYOJIN_GATHER_ROOF: the rows = 1 gate/up gather GEMV as 128-thread threadgroups (one column half each). Outputs must be bit-identical to the 256-thread launch,
on the plain path (gemv_gu partials), the moe_chain path (partials + mid + combine) and with the routed_fx prologue left alone; rows >= 2 keeps the old launch."""
import os
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

sys.path.insert(0, os.path.dirname(__file__))
from kyojin_mlx import moe as M, metal_src as S, fired as FD  # noqa: E402
from test_moe import make_stack, _route, _x, _bits, KS, TOPK  # noqa: E402


@pytest.fixture(autouse=True)
def restore():
    f, c, o = M.GATHER_ROOF, M.CHAIN, FD.ON
    yield
    M.GATHER_ROOF, M.CHAIN, FD.ON = f, c, o


def _mk(kin, ninter, cb, wpre, fq):
    g, _ = make_stack(kin, ninter, KS, cb, 1)
    u, _ = make_stack(kin, ninter, KS, cb, 2)
    d, _ = make_stack(ninter, kin, KS, cb, 3)
    return M.MoeGather(g, u, d, TOPK, wpre=wpre, fq=fq)


@pytest.mark.parametrize("cb", ["mul1", "mcg"])
@pytest.mark.parametrize("chain", [True, False])
@pytest.mark.parametrize("wpre,fq", [(False, False), (True, True)])
@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("kin,ninter", [(384, 640), (2048, 512), (5120, 640)])
def test_bit_identical(cb, chain, wpre, fq, rows, kin, ninter):
    m = _mk(kin, ninter, cb, wpre, fq)
    ids, w = _route(rows, 7)
    M.CHAIN = chain
    for seed in range(3):
        x = _x(rows, kin, 10 + seed)
        M.GATHER_ROOF = False
        ref = np.array(m(x, ids, w, fq8=fq))
        M.GATHER_ROOF = True
        for _ in range(3):                                      # repeated: the arrival counters are zero again after each launch
            assert np.array_equal(_bits(np.array(m(x, ids, w, fq8=fq))), _bits(ref))


@pytest.mark.parametrize("kin,ninter", [(384, 640), (5120, 640)])
def test_partials_bit_identical(kin, ninter):
    m = _mk(kin, ninter, "mul1", True, True)
    ids, _ = _route(1, 5)
    ids = ids.reshape(-1).astype(mx.int32)
    xh = m.rot_gu(_x(1, kin, 3), ids, True)
    M.GATHER_ROOF = False
    ref = np.array(m.gemv_gu(xh, ids, 1))
    M.GATHER_ROOF = True
    assert np.array_equal(_bits(np.array(m.gemv_gu(xh, ids, 1))), _bits(ref))


def test_fires_rows1_only():
    FD.ON = True
    m = _mk(2048, 512, "mul1", True, True)
    for rows, fires in ((1, True), (2, False)):
        M.GATHER_ROOF = True
        FD.reset()
        mx.eval(m(_x(rows, 2048, 1), *_route(rows, 7)))
        assert (FD.HITS.get("gather_roof", 0) > 0) == fires
        M.GATHER_ROOF = False
        FD.reset()
        mx.eval(m(_x(rows, 2048, 1), *_route(rows, 7)))
        assert FD.HITS.get("gather_roof", 0) == 0               # flag off: old path intact


def test_source_shape():
    base = S.gather_gemv_source((2, 3), 2, "mul1r1", True)
    half = S.gather_gemv_source((2, 3), 2, "mul1r1", True, False, False, True)
    assert base != half and "threadgroup_position_in_grid.x >> 1" in half and "idx < 64" in half
    assert S.gather_gemv_source((2, 3), 2, "mul1r1", True, False, False, False) == base
