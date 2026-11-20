"""N2-ROWDOT: the fbg_a row-dot folded into the 6-bit qkv GEMV launch is bit-identical to the two stock launches (random weights, real GLM-5.3 shapes)."""
import types

import mlx.core as mx
import pytest

from kyojin_mlx import glm_dense6 as G6, glm_rowfold as RF, rowinv as RI


@pytest.fixture(params=[True, False], ids=["front", "tail"], autouse=True)
def _front(request):
    old = RF.FRONT
    RF.FRONT = request.param
    yield
    RF.FRONT = old


def _q6(n, k, seed):
    mx.random.seed(seed)
    w = (mx.random.normal((n, k)) * 0.02).astype(mx.float16)
    qs, mt = G6.quantize_mx(w)
    mx.eval(qs, mt)
    return qs, mt


def _same(a, b):
    return bool(mx.all(a == b).item()) and a.dtype == b.dtype and a.shape == b.shape


@pytest.mark.parametrize("n,k,nx", [(24576, 4096, 320), (4096, 4096, 320), (4096, 2048, 320), (8192, 4096, 192), (2048, 4096, 256)])
@pytest.mark.parametrize("r", [1, 2, 3, 4])
def test_bit_identical(n, k, nx, r):
    qs, mt = _q6(n, k, 1)
    mx.random.seed(2)
    x = (mx.random.normal((r, k)) * 2).astype(mx.float16)
    w2 = (mx.random.normal((nx, k)) * 0.05).astype(mx.float16)
    ones = mx.ones((n,), mx.float32)
    y, y2 = RF.gemv_fold(x, qs, mt, ones, n, k, w2)
    ref = G6.gemv(x, qs, mt, ones, n, k)
    ref2 = RI.rowdot(x, w2)
    mx.eval(y, y2, ref, ref2)
    assert _same(y, ref)
    assert _same(y2, ref2)


def test_row_invariance():
    """Row i of an R-row launch equals the 1-row launch of row i (the row-dot k order does not depend on R)."""
    n, k, nx = 24576, 4096, 320
    qs, mt = _q6(n, k, 3)
    mx.random.seed(4)
    x = (mx.random.normal((4, k)) * 2).astype(mx.float16)
    w2 = (mx.random.normal((nx, k)) * 0.05).astype(mx.float16)
    ones = mx.ones((n,), mx.float32)
    _, y24 = RF.gemv_fold(x, qs, mt, ones, n, k, w2)
    for i in range(4):
        _, y21 = RF.gemv_fold(x[i:i + 1], qs, mt, ones, n, k, w2)
        assert _same(y24[i:i + 1], y21)


class _Dev:
    def __init__(self, n, k, seed):
        self.n, self.k = n, k
        self.g6 = _q6(n, k, seed)
        self._o = mx.ones((n,), mx.float32)

    def g3_ok(self, r):
        return 1 <= r <= 4

    def _ones(self):
        return self._o


class _Qkv:
    def __init__(self, n, k, seed=5, bias=None, perm=None):
        self.dev = _Dev(n, k, seed)
        self._in_perm, self._bias, self.out_dtype = perm, bias, mx.float16


def test_wrapper_matches_stock_and_guards():
    from kyojin_mlx.rowinv import RowLinear
    import mlx.nn as nn
    n, k, nx = 24576, 4096, 320
    mx.random.seed(6)
    lin = nn.Linear(k, nx, bias=False)
    lin.weight = (mx.random.normal((nx, k)) * 0.05).astype(mx.float16)
    fbg = RowLinear(lin)
    q = _Qkv(n, k)
    RF.ENABLED = True
    x = (mx.random.normal((4, 1, k)) * 2).astype(mx.float16)
    out = RF.qkv_fbg(q, fbg, x)
    assert out is not None
    y, y2 = out
    ref = G6.gemv(x.reshape(4, k), q.dev.g6[0], q.dev.g6[1], q.dev._ones(), n, k).reshape(4, 1, n)
    ref2 = fbg(x)
    mx.eval(y, y2, ref, ref2)
    assert _same(y, ref) and _same(y2, ref2)
    # guards: verify rows (L > 1), bf16 x, a bias, a permuted input, flag off -> the stock two launches
    assert RF.qkv_fbg(q, fbg, mx.zeros((1, 2, k), mx.float16)) is None
    assert RF.qkv_fbg(q, fbg, x.astype(mx.bfloat16)) is None
    assert RF.qkv_fbg(_Qkv(n, k, bias=mx.zeros((n,))), fbg, x) is None
    assert RF.qkv_fbg(_Qkv(n, k, perm=mx.arange(k)), fbg, x) is None
    assert RF.qkv_fbg(q, fbg, mx.zeros((5, 1, k), mx.float16)) is None          # 5 rows > DENSE_GEMV_ROWS
    RF.ENABLED = False
    assert RF.qkv_fbg(q, fbg, x) is None
