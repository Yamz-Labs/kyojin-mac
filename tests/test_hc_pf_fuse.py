"""hc_pf_fuse.collapse_norm == the stock chain (hc_sinkhorn_collapse kernel, then mx.fast.rms_norm) bit for bit, f16 weight and f32 weight."""
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("needs the GPU", allow_module_level=True)

from kyojin_mlx import hc_pf_fuse as HF  # noqa: E402


def _inputs(S, seed=0):
    mx.random.seed(seed)
    x = (mx.random.normal((1, S, 4, 4096)) * 3).astype(mx.float16)
    mixes = mx.random.normal((1, S, 24)) * 2
    scale = mx.array([0.7, 0.9, 1.3], dtype=mx.float32)
    base = mx.random.normal((24,)) * 0.3
    mx.eval(x, mixes, scale, base)
    return x, mixes, scale, base


@pytest.mark.parametrize("wdt", [mx.float16, mx.float32])
@pytest.mark.parametrize("S", [17, 256, 1000])
def test_collapse_norm_bit_identical(S, wdt):
    from mlx_vlm.models.deepseek_v4 import hyper_connection as hc
    x, mixes, scale, base = _inputs(S)
    w = (mx.random.normal((4096,)) * 0.5 + 1).astype(wdt)
    eps = 1e-5
    got, post, comb = HF.collapse_norm(x, mixes, scale, base, w, eps, 1e-6, 20)
    col, p2, c2 = hc._hc_kernel(x, None, mixes, scale, base, 4, 20, 1e-6)
    ref = mx.fast.rms_norm(col.astype(mx.float32) if wdt == mx.float32 else col, w, eps)
    mx.eval(got, post, comb, ref, p2, c2)
    assert got.dtype == ref.dtype and got.shape == ref.shape
    assert mx.array_equal(post, p2).item() and mx.array_equal(comb, c2).item()
    assert mx.array_equal(got, ref).item(), float(mx.max(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32))).item())


def test_twin_is_the_fp16_cast_of_the_f32_output(monkeypatch):
    x, mixes, scale, base = _inputs(300)
    w = (mx.random.normal((4096,)) * 0.5 + 1).astype(mx.float32)
    monkeypatch.setattr(HF, "TWIN", True)
    got, post, comb = HF.collapse_norm(x, mixes, scale, base, w, 1e-5, 1e-6, 20)
    th = HF.twin_of(got)
    mx.eval(got, th)
    assert th is not None and th.dtype == mx.float16 and mx.array_equal(th, got.astype(mx.float16)).item()
    assert HF.twin_of(got + 0) is None
