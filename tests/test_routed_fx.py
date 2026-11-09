"""KYOJIN_ROUTED_FX: the rot folded into the prologue of the gate/up gather GEMV (2 launches per routed layer). Outputs must be bit-identical to the
3-launch chain; shapes whose k slice does not start on a 128-block keep the 3-launch chain."""
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
    f, c, o = M.ROUTED_FX, M.CHAIN, FD.ON
    yield
    M.ROUTED_FX, M.CHAIN, FD.ON = f, c, o


@pytest.mark.parametrize("cb", ["mul1", "mcg"])
@pytest.mark.parametrize("wpre,fq", [(False, False), (True, True)])
@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("kin,ninter", [(384, 640), (2048, 512), (5120, 512)])
def test_bit_identical(cb, wpre, fq, rows, kin, ninter):
    g, _ = make_stack(kin, ninter, KS, cb, 1)
    u, _ = make_stack(kin, ninter, KS, cb, 2)
    d, _ = make_stack(ninter, kin, KS, cb, 3)
    m = M.MoeGather(g, u, d, TOPK, wpre=wpre, fq=fq)
    ids, w = _route(rows, 7)
    for seed in range(3):
        x = _x(rows, kin, 10 + seed)
        M.ROUTED_FX = False
        ref = np.array(m(x, ids, w, fq8=fq))
        M.ROUTED_FX = True
        for _ in range(3):                                      # repeated: the arrival counters are zero again after each launch
            assert np.array_equal(_bits(np.array(m(x, ids, w, fq8=fq))), _bits(ref))


def test_fires_and_fallback():
    FD.ON = True
    for kin, ninter, fires in ((5120, 512, True), (384, 640, None)):
        g, _ = make_stack(kin, ninter, KS, "mul1", 1)
        u, _ = make_stack(kin, ninter, KS, "mul1", 2)
        d, _ = make_stack(ninter, kin, KS, "mul1", 3)
        m = M.MoeGather(g, u, d, TOPK)
        M.ROUTED_FX = True
        FD.reset()
        mx.eval(m(_x(1, kin, 1), *_route(1, 7)))
        assert (FD.HITS.get("routed_fx", 0) > 0) == S.fx_ok(g.kt, m._params(1)["tgu"])
        if fires:
            assert FD.HITS.get("routed_fx", 0) > 0
        M.ROUTED_FX = False
        FD.reset()
        mx.eval(m(_x(1, kin, 1), *_route(1, 7)))
        assert FD.HITS.get("routed_fx", 0) == 0                 # flag off: old path intact


def test_fx_ok_shapes():
    assert S.fx_ok(320, 4) and S.fx_ok(320, 2)
    assert not S.fx_ok(20, 1)                                   # odd chunk: the slice would not start on a 128-block
