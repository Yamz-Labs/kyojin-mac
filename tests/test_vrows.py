"""N1-VROWS (flag glm_vrows, KYOJIN_GLM_VROWS): the verify rows [B, L, H] (B * L <= 8) of a dense ForeignLinear are ONE GEMV launch, each row equal to the per-row path bit for bit."""
import importlib

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from kyojin_mlx import foreign_linear as FL, glm_dense_rows as GDR, patch as P
from test_dense_g3 import _pack

needs_gpu = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal GPU")


@pytest.fixture
def lin_fn():
    P._install_linear_hook()
    return importlib.import_module("mlx_vlm.models.linear").linear


class Counting(FL.ForeignLinear):
    calls = 0

    def __call__(self, x, permuted=False):
        Counting.calls += 1
        return super().__call__(x, permuted)


def _mk(fmt, n, k, seed, q6, monkeypatch, cls=FL.ForeignLinear):
    monkeypatch.setattr(FL, "DENSE_GEMV", True)
    monkeypatch.setattr(FL, "DENSE_Q", q6)
    m = cls(_pack(fmt, n, k, np.random.default_rng(seed)))
    assert (m.dev.g6 is not None) == q6
    m.eval()                                                               # `linear()` skips modules in training mode
    return m


@needs_gpu
@pytest.mark.parametrize("q6", [True, False])
@pytest.mark.parametrize("n,k", [(512, 4096), (4096, 2048), (16384, 1536), (8192, 4096), (4096, 12288)])
def test_rows_1_to_8_equal_row1_bitwise(q6, n, k, monkeypatch):
    """Row r of an R-row call (R up to 8) == the R = 1 call on that row, both the 6-bit and the fp8 kernel, fp16 and bf16 x."""
    monkeypatch.setattr(FL, "DENSE_GEMV", True)
    monkeypatch.setattr(FL, "DENSE_Q", q6)
    d = FL.DevPack(_pack("fp8b", n, k, np.random.default_rng(n + k)))
    rng = np.random.default_rng(7)
    for dt in (mx.float16, mx.bfloat16):
        x = mx.array(rng.normal(0, 1, (8, k)).astype(np.float32)).astype(dt)
        ref = [np.array(d._gemv8g3(x[i:i + 1])) for i in range(8)]
        for R in range(2, 9):
            y = np.array(d._gemv8g3(x[:R]))
            for i in range(R):
                assert np.array_equal(y[i], ref[i][0]), (q6, n, k, R, i)


@needs_gpu
@pytest.mark.parametrize("q6", [True, False])
@pytest.mark.parametrize("B,L", [(1, 2), (1, 4), (1, 8), (2, 2), (3, 2), (4, 2), (2, 4)])
def test_linear_route_one_call_bitwise(B, L, q6, lin_fn, monkeypatch):
    m = _mk("fp8b", 4096, 2048, 5, q6, monkeypatch, Counting)
    x = mx.array(np.random.default_rng(B * 10 + L).normal(0, 1, (B, L, 2048)).astype(np.float32)).astype(mx.bfloat16)
    perrow = np.array(mx.concatenate([m(x[b:b + 1, l:l + 1]) for b in range(B) for l in range(L)], axis=0).astype(mx.float32)).reshape(B, L, 4096)   # the R = 1 path, one row at a time
    monkeypatch.setattr(GDR, "VROWS", False)
    monkeypatch.setattr(GDR, "ENABLED", False)
    Counting.calls = 0
    ref = np.array(lin_fn(m, x).astype(mx.float32)) if L > 1 else None
    if L > 1:
        assert Counting.calls == L                                         # stock: tokenwise, L calls of B rows
        assert np.array_equal(ref, perrow)
    monkeypatch.setattr(GDR, "VROWS", True)
    Counting.calls = 0
    y = np.array(lin_fn(m, x).astype(mx.float32))
    assert Counting.calls == 1
    assert y.shape == (B, L, 4096)
    assert np.array_equal(y, perrow)
    assert FL.ROWS_CAP[0] == 0                                             # the cap never leaks out of the call


@needs_gpu
def test_over_cap_and_other_shapes_keep_stock(lin_fn, monkeypatch):
    m = _mk("fp8b", 4096, 2048, 6, True, monkeypatch, Counting)
    monkeypatch.setattr(GDR, "VROWS", True)
    x = mx.array(np.random.default_rng(1).normal(0, 1, (3, 4, 2048)).astype(np.float32)).astype(mx.bfloat16)      # 12 rows > 8
    Counting.calls = 0
    y = lin_fn(m, x)
    assert Counting.calls == 4 and y.shape == (3, 4, 4096)
    monkeypatch.setattr(GDR, "VROWS", False)
    monkeypatch.setattr(GDR, "ENABLED", True)                              # G3e2 flag: one session only, <= DENSE_GEMV_ROWS
    for shp, calls in (((1, 3, 2048), 1), ((2, 2, 2048), 2), ((1, 6, 2048), 6)):
        Counting.calls = 0
        lin_fn(m, mx.zeros(shp, dtype=mx.bfloat16))
        assert Counting.calls == calls, shp


def test_route_guards(monkeypatch):
    """CPU: which modules / shapes the route takes."""
    monkeypatch.setattr(FL, "DENSE_GEMV", True)
    monkeypatch.setattr(GDR, "ENABLED", False)
    monkeypatch.setattr(GDR, "VROWS", True)
    m = _mk("fp8b", 256, 1024, 1, False, monkeypatch)
    x = lambda s: mx.zeros(s, dtype=mx.float16)
    for s in ((1, 2, 1024), (4, 2, 1024), (2, 4, 1024), (1, 8, 1024)):
        assert GDR.route(m, x(s)), s
    for s in ((1, 9, 1024), (5, 2, 1024), (3, 3, 1024)):                    # above the 8-row cap
        assert not GDR.route(m, x(s)), s
    assert FL.ROWS_CAP[0] == 0
    assert not GDR.route(_mk("fp8b", 256, 768, 2, False, monkeypatch), x((1, 2, 768)))   # k % 512
    assert not GDR.route(object(), x((1, 2, 1024)))
    assert not GDR.route(m, mx.zeros((2, 1024), mx.float16))
    monkeypatch.setattr(FL, "DENSE_GEMV", False)
    assert not GDR.route(m, x((1, 2, 1024)))                                # needs DENSE_GEMV
    monkeypatch.setattr(FL, "DENSE_GEMV", True)
    monkeypatch.setattr(GDR, "VROWS", False)
    assert not GDR.route(m, x((1, 2, 1024)))                                # flag off = stock
    monkeypatch.setattr(GDR, "VROWS_MAX", 4)
    monkeypatch.setattr(GDR, "VROWS", True)
    assert GDR.route(m, x((2, 2, 1024))) and not GDR.route(m, x((1, 5, 1024)))


def test_g3_cap_default_unchanged(monkeypatch):
    """Outside a routed call the GEMV row cap is DENSE_GEMV_ROWS (4) as before."""
    monkeypatch.setattr(FL, "DENSE_GEMV", True)
    m = _mk("fp8b", 256, 1024, 1, False, monkeypatch)
    assert m.dev.g3_ok(4) and not m.dev.g3_ok(5) and not m.dev.g3_ok(8)
    FL.ROWS_CAP[0] = 8
    try:
        assert m.dev.g3_ok(8) and not m.dev.g3_ok(9)
    finally:
        FL.ROWS_CAP[0] = 0


def test_flag_registered_default_off():
    from kyojin_mlx import flags as F
    f = F.FLAGS["glm_vrows"]
    assert f.env == "KYOJIN_GLM_VROWS" and f.default == "0" and f.runtime


def test_glm_serve_path_installs_linear_hook():
    """The served GLM-5.3 load (load_foreign, not patch_model) must install the mlx_vlm linear hook, else tokenwise runs and the verify-row flags never fire (found on the cut: hook False)."""
    import inspect
    from kyojin_mlx.serve import backends
    src = inspect.getsource(backends)
    i = src.index("GLM-5.3 prompt hooks")
    assert "_install_linear_hook()" in src[i:i + 600]
