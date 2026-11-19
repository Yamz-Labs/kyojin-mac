"""M8: activation kind "situ" of the EXL3 expert kernels vs stock MLX ops (Metal micro tests on random tensors)."""
import os
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

sys.path.insert(0, os.path.dirname(__file__))
from kyojin_mlx import moe as M, metal_src as S  # noqa: E402
from test_moe import _x, _bits  # noqa: E402
from test_op_moe_topk import _route_k, _stacks, _ref_input  # noqa: E402


@pytest.fixture(autouse=True)
def restore():
    c = M.CHAIN
    yield
    M.CHAIN = c


SITU_B, SITU_LB = 4.0, 25.0


@pytest.mark.parametrize("wpre,fq", [(False, False), (True, True)])
@pytest.mark.parametrize("topk,rows", [(6, 1), (6, 3), (10, 2)])
def test_situ_vs_mlx_ops(topk, rows, wpre, fq):
    """act="situ": chain (3 launches) == 5-launch path bit for bit; both vs the per-expert path whose activation is stock MLX ops (max abs diff reported)."""
    kin, ninter = 384, 640
    g, u, d = _stacks(kin, ninter)
    kw = dict(wpre=wpre, fq=fq, alpha=SITU_B, beta=SITU_LB, act="situ")
    m = M.MoeGather(g, u, d, topk, **kw)
    ids, w = _route_k(rows, topk, 5)
    x = _x(rows, kin, 21)
    M.CHAIN = False
    ref5 = np.array(m(x, ids, w, fq8=fq))
    M.CHAIN = True
    got = np.array(m(x, ids, w, fq8=fq))
    assert np.array_equal(_bits(got), _bits(ref5))
    un = np.array(M.MoeUnfused(g, u, d, topk, **kw)(_ref_input(x, fq), ids, w))
    scale = float(np.abs(un).max())
    diff = float(np.abs(un - got).max())
    sw = np.array(M.MoeGather(g, u, d, topk, wpre=wpre, fq=fq)(x, ids, w, fq8=fq))
    assert not np.allclose(sw, got), "situ must differ from swiglu"
    print(f"situ topk {topk} rows {rows} wpre {wpre}: max abs diff vs mlx-op activation {diff:.3g} (output max {scale:.3g}, rel {diff / scale:.2g})")
    assert diff / scale < 2e-4                                    # fp32 tanh / exp of the kernel vs mlx; the fp8 fake quant (fq) may flip one fp8 step of an element


def test_situ_source_and_default_kind():
    a = S.mid_source(10.0, False, False)
    assert a == S.mid_source(10.0, False, False, 1.0, 0.0, "swiglu") and "min(gu[0]" in a
    b = S.mid_source(10.0, False, False, 4.0, 25.0, "situ")
    assert "tanh" in b and "min(gu[0]" not in b and "LIMIT" not in b
    with pytest.raises(AssertionError):
        S.mid_source(10.0, False, False, 1.0, 0.0, "relu")
