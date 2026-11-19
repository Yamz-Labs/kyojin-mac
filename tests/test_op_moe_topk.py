"""M1: MoE routed-expert chain at top-k above 8 (combine tail strided over the slots).
Micro tests on random tensors (Metal). Existing shapes stay byte-identical in generated source (checked outside the repo by hashing every source before / after)."""
import os
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

sys.path.insert(0, os.path.dirname(__file__))
from kyojin_mlx import moe as M, metal_src as S  # noqa: E402
from test_moe import make_stack, _x, _bits  # noqa: E402

NE = 24
KS24 = [2, 3, 3, 2, 2, 3, 2, 2, 3, 2, 3, 3] * 2


@pytest.fixture(autouse=True)
def restore():
    c = M.CHAIN
    yield
    M.CHAIN = c


def _route_k(rows, topk, seed):
    rng = np.random.default_rng(seed)
    ids = np.stack([rng.choice(NE, topk, replace=False) for _ in range(rows)]).astype(np.int32)
    w = rng.uniform(0.1, 1.0, (rows, topk)).astype(np.float32)
    w = (w / w.sum(-1, keepdims=True) * 1.5).astype(np.float32)
    return mx.array(ids), mx.array(w)


def _ref_input(x, fq):
    """fq8=True fake-quants the experts' input inside the fused rot kernel; the per-expert reference takes the input already quantized."""
    if not fq:
        return x
    pytest.importorskip("mlx_vlm.models.deepseek_v41.fakequant")
    from kyojin_mlx import archs as AR
    return AR.fq(x)


def _stacks(kin, ninter, cb="mul1"):
    g, _ = make_stack(kin, ninter, KS24, cb, 1)
    u, _ = make_stack(kin, ninter, KS24, cb, 2)
    d, _ = make_stack(ninter, kin, KS24, cb, 3)
    return g, u, d


@pytest.mark.parametrize("topk", [9, 10, 12, 16])
@pytest.mark.parametrize("wpre,fq", [(False, False), (True, True)])
@pytest.mark.parametrize("rows", [1, 2, 3])
def test_chain_topk_above_8(topk, wpre, fq, rows):
    """top-k > 8: the 3-launch chain is armed and bit-identical to the 5-launch path and to the per-expert path (max abs diff reported)."""
    kin, ninter = 384, 640
    g, u, d = _stacks(kin, ninter)
    m = M.MoeGather(g, u, d, topk, wpre=wpre, fq=fq)
    M.CHAIN = True
    assert m.chain_ok()
    ids, w = _route_k(rows, topk, 7)
    x = _x(rows, kin, 10)
    M.CHAIN = False
    ref = np.array(m(x, ids, w, fq8=fq))
    M.CHAIN = True
    outs = [np.array(m(x, ids, w, fq8=fq)) for _ in range(3)]       # repeated: the arrival counters are zero again after each launch
    for o in outs:
        assert np.array_equal(_bits(o), _bits(ref))
    assert not np.any(np.array(m._chain_ctr(rows)[0])) and not np.any(np.array(m._chain_ctr(rows)[1]))
    un = np.array(M.MoeUnfused(g, u, d, topk, wpre=wpre, fq=fq)(_ref_input(x, fq), ids, w))
    assert np.array_equal(_bits(un), _bits(ref)) or float(np.abs(un - ref).max() / np.abs(ref).max()) < 1e-6
    print(f"topk {topk} rows {rows}: chain vs 5-launch bit-identical; vs per-expert max abs diff {float(np.abs(un - ref).max()):.3g}")


def test_chain_topk_limit_is_a_parameter():
    kin, ninter = 256, 384
    g, u, d = _stacks(kin, ninter)
    M.CHAIN = True
    assert M.MoeGather(g, u, d, S.CHAIN_TOPK_MAX).chain_ok()
    assert not M.MoeGather(g, u, d, S.CHAIN_TOPK_MAX + 1).chain_ok()
    with pytest.raises(AssertionError):
        S.chain_dn_source((2, 3), "mul1", True, False, 2, S.CHAIN_TOPK_MAX + 1, False)


def test_chain_tail_unchanged_up_to_8():
    for k in range(1, 9):
        assert "if (warp < %d) {" % k in S.chain_dn_source((2, 3), "mul1", True, False, 2, k, False)
    assert "for (uint j = warp; j < 10; j += 8u)" in S.chain_dn_source((2, 3), "mul1", True, False, 2, 10, False)
