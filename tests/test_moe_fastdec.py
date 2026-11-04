"""KYOJIN_MOE_FASTDEC (mul1fd decode: byte sum as denormal-half lane sums, affine folded) vs the stock decode.
Quality gate: rel L2 of the layer output < 1e-3 with the fp8 fake quant off (pure decode difference; the weight is no longer rounded to half),
fake-quant on stays within the fp8 flip class (< 1e-2). Grouped rows (>= 2) are untouched: bit-identical. Dense Mul1Gemv(cb="mul1fd") < 1e-3."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

from kyojin_mlx import kernels as KN, moe as M
from test_moe import E, KS, TOPK, make_stack, _route, _x


def rel(a, b):
    a, b = np.array(a, np.float64), np.array(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.fixture(scope="module")
def stacks():
    kin, ninter = 256, 384
    return (make_stack(kin, ninter, KS, "mul1", 1)[0], make_stack(kin, ninter, KS, "mul1", 2)[0], make_stack(ninter, kin, KS, "mul1", 3)[0])


def layer(stacks, monkeypatch, on, **kw):
    m = M.MoeGather(*stacks, TOPK, limit=10.0, **kw)
    m.fastdec = on                                                  # pinned: the module switch is read at call time
    return m


@pytest.mark.parametrize("rows,grouped", [(1, True), (2, False), (3, False)])
@pytest.mark.parametrize("wpre", [False, True])
def test_gather_quality(stacks, monkeypatch, rows, grouped, wpre):
    ref = layer(stacks, monkeypatch, False, wpre=wpre, grouped=grouped)
    fast = layer(stacks, monkeypatch, True, wpre=wpre, grouped=grouped)
    assert (ref.cbf, fast.cbf) == (M._roof("mul1"), M._roof("mul1fd"))
    ids, w = _route(rows, 40 + rows)
    x = _x(rows, stacks[0].kin, 9)
    assert rel(fast(x, ids, w), ref(x, ids, w)) < 1e-3


def test_gather_fq_flip_class(stacks, monkeypatch):
    ref = layer(stacks, monkeypatch, False, wpre=True, fq=True)
    fast = layer(stacks, monkeypatch, True, wpre=True, fq=True)
    ids, w = _route(1, 77)
    x = _x(1, stacks[0].kin, 5)
    assert rel(fast(x, ids, w, fq8=True), ref(x, ids, w, fq8=True)) < 1e-2


@pytest.mark.parametrize("rows", [2, 4])
def test_grouped_rows_untouched(stacks, monkeypatch, rows):
    ref = layer(stacks, monkeypatch, False, wpre=True)
    fast = layer(stacks, monkeypatch, True, wpre=True)
    ids, w = _route(rows, 90 + rows)
    x = _x(rows, stacks[0].kin, 3)
    assert np.array_equal(np.array(fast(x, ids, w)), np.array(ref(x, ids, w)))


@pytest.mark.parametrize("K", [2, 3, 4])
def test_dense_gemv_one_row(K):
    rng = np.random.default_rng(K)
    kin, nout = 512, 384
    tr = rng.integers(-32768, 32768, (kin // 16, nout // 16, 16 * K), dtype=np.int16)
    suh = rng.uniform(0.5, 1.5, kin).astype(np.float16)
    svh = rng.uniform(0.5, 1.5, nout).astype(np.float16)
    x = _x(1, kin, K)
    assert rel(KN.Mul1Gemv(tr, suh, svh, rows=1, cb="mul1fd")(x), KN.Mul1Gemv(tr, suh, svh, rows=1)(x)) < 1e-3
