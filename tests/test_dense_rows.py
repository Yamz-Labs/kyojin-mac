"""G3e2 (flag glm_dense_rows): the verify rows of one session through mlx_vlm's `linear()` are ONE G3 launch per ForeignLinear, equal to the tokenwise path bit for bit."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from kyojin_mlx import foreign_linear as FL, glm_dense_rows as GDR, patch as P, foreign_pack as FP
from test_dense_g3 import _pack

needs_gpu = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal GPU")


@pytest.fixture
def lin_fn(monkeypatch):
    P._install_linear_hook()
    import importlib
    return importlib.import_module("mlx_vlm.models.linear").linear


class Counting(FL.ForeignLinear):
    calls = 0

    def __call__(self, x, permuted=False):
        Counting.calls += 1
        return super().__call__(x, permuted)


def _mk(fmt, n, k, seed, cls=FL.ForeignLinear):
    return cls(_pack(fmt, n, k, np.random.default_rng(seed)))


@needs_gpu
@pytest.mark.parametrize("fmt", ["fp8", "fp8b"])
@pytest.mark.parametrize("n,k", [(512, 4096), (4096, 2048), (16384, 1536), (8192, 4096), (4096, 12288)])
def test_rows_equal_row1_bitwise(fmt, n, k, monkeypatch):
    """Row r of an R-row G3 call == the R = 1 call on that row, for every R and both x dtypes (the contract the route relies on)."""
    monkeypatch.setattr(FL, "DENSE_GEMV", True)
    d = FL.DevPack(_pack(fmt, n, k, np.random.default_rng(n + k)))
    rng = np.random.default_rng(7)
    for dt in (mx.float16, mx.bfloat16):
        x = mx.array(rng.normal(0, 1, (4, k)).astype(np.float32)).astype(dt)
        ref = [np.array(d._gemv8g3(x[i:i + 1])) for i in range(4)]
        for R in (2, 3, 4):
            y = np.array(d._gemv8g3(x[:R]))
            for i in range(R):
                assert np.array_equal(y[i], ref[i][0]), (fmt, n, k, R, i)


@needs_gpu
@pytest.mark.parametrize("L", [2, 3, 4])
def test_linear_route_matches_tokenwise_and_is_one_call(L, lin_fn, monkeypatch):
    monkeypatch.setattr(FL, "DENSE_GEMV", True)
    m = _mk("fp8b", 4096, 2048, 5, Counting)
    m.eval()                                                                # `linear()` skips modules in training mode (the served model is eval())
    x = mx.array(np.random.default_rng(L).normal(0, 1, (1, L, 2048)).astype(np.float32)).astype(mx.bfloat16)
    monkeypatch.setattr(GDR, "ENABLED", False)
    Counting.calls = 0
    ref = np.array(lin_fn(m, x).astype(mx.float32))
    assert Counting.calls == L                                              # stock: tokenwise, one call (launch) per row
    monkeypatch.setattr(GDR, "ENABLED", True)
    Counting.calls = 0
    y = np.array(lin_fn(m, x).astype(mx.float32))
    assert Counting.calls == 1
    assert y.shape == ref.shape == (1, L, 4096)
    assert np.array_equal(y, ref)


def test_route_guards(monkeypatch):
    """CPU: which modules / shapes the route takes."""
    monkeypatch.setattr(GDR, "ENABLED", True)
    monkeypatch.setattr(FL, "DENSE_GEMV", True)
    m = _mk("fp8b", 256, 1024, 1)
    x = lambda s: mx.zeros(s, dtype=mx.float16)
    assert GDR.route(m, x((1, 2, 1024))) and GDR.route(m, x((1, 4, 1024)))
    assert not GDR.route(m, x((1, 5, 1024)))                                # above DENSE_GEMV_ROWS
    assert not GDR.route(m, x((2, 2, 1024)))                                # several sessions: not this path
    assert not GDR.route(_mk("fp8b", 256, 768, 2), x((1, 2, 768)))          # k % 512
    assert not GDR.route(object(), x((1, 2, 1024)))
    monkeypatch.setattr(FL, "DENSE_GEMV", False)
    assert not GDR.route(m, x((1, 2, 1024)))
    monkeypatch.setattr(FL, "DENSE_GEMV", True)
    monkeypatch.setattr(GDR, "ENABLED", False)
    assert not GDR.route(m, x((1, 2, 1024)))                                # flag off = stock


def test_flag_registered_default_off():
    from kyojin_mlx import flags as F
    f = F.FLAGS["glm_dense_rows"]
    assert f.env == "KYOJIN_GLM_DENSE_ROWS" and f.default == "0" and f.runtime
