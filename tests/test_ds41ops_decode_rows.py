"""Decode rows (R, 1, D) of the DS4.1 block: the MoE combine runs as one pass and the mHC wrappers do not clip twice, both bit-identical to the stock chain."""
import mlx.core as mx
import numpy as np
import pytest

from kyojin_mlx import ds_pf_mhc, fuse, pf_glue as PG

D = 1024


def f32(a):
    return np.array(a.astype(mx.float32))


class _Fake:
    """Stands in for DeepseekV41MoE: gate, switch_mlp (fp32 [R, S, 1, D] like the patched routed MoE) and shared_experts."""
    def __init__(self, y, sh):
        self.y, self.sh, self.calls = y, sh, 0
        self.gate = lambda x, m=None: (None, None)
        self.switch_mlp = lambda x, i, s: self.y
        self.shared_experts = lambda x: self.sh


def _stock(self, x, image_mask=None):
    return (self.y.astype(mx.float32).sum(-2) + self.sh.astype(mx.float32)).astype(x.dtype)


@pytest.mark.parametrize("dt", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("R", [1, 2, 6])
def test_moe_combine_decode_rows(dt, R, monkeypatch):
    from kyojin_mlx import archs as A
    rng = np.random.default_rng(R)
    x = mx.array(rng.standard_normal((R, 1, D)).astype(np.float32) * 3).astype(dt)
    sh = mx.array(rng.standard_normal((R, 1, D)).astype(np.float32) * 3).astype(dt)
    y = mx.array(rng.standard_normal((R, 1, 1, D)).astype(np.float32) * 3)
    m = _Fake(y, sh)
    L = A.lang()
    monkeypatch.setattr(L, "fake_quant_fp8_ue8m0", lambda v: v)
    fn = PG._impl_moe(_stock)
    out = fn(m, x)
    assert out.dtype == dt and out.shape == (R, 1, D)
    assert np.array_equal(f32(out), f32(_stock(m, x)))
    monkeypatch.setattr(PG, "DEC_ROWS", False)
    assert np.array_equal(f32(PG._impl_moe(_stock)(m, x)), f32(_stock(m, x)))


def _mark(f):
    f.saturates = True
    return f


def test_mhc_wrappers_skip_second_clip():
    z = mx.array(np.array([[[np.inf, -np.inf, 1.0, 2.0] * (D // 4)]], dtype=np.float32)).astype(mx.float16)
    x, p = mx.zeros((1, 1, 4, D), mx.float16), mx.zeros((1, 1, 4))
    assert float(mx.max(ds_pf_mhc._wrap_pre(lambda x, p: z)(x, p)).item()) == 65504.0          # stock orig: still clipped
    assert not np.isfinite(f32(ds_pf_mhc._wrap_pre(_mark(lambda x, p: z))(x, p))).all()        # saturating orig: passed through untouched


def test_fuse_impls_are_marked():
    assert fuse._impl_mhc_expand(lambda *a: None).saturates and fuse._impl_mhc_pre(lambda *a: None).saturates


@pytest.mark.parametrize("dt", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("R", [1, 4, 70])
def test_shared_mlp_swiglu_cast_is_bit_identical(dt, R):
    from mlx_vlm.models.deepseek_v41 import language as L
    rng = np.random.default_rng(R)
    g = mx.array(rng.standard_normal((R, 1, 2304)).astype(np.float32) * 12).astype(dt)
    u = mx.array(rng.standard_normal((R, 1, 2304)).astype(np.float32) * 12).astype(dt)
    for limit in (10.0, 0.0):
        ref = L._swiglu(g, u, limit).astype(dt)
        assert np.array_equal(f32(PG._swiglu_h(limit, dt)(g, u)), f32(ref))


def test_shared_mlp_impl_matches_stock():
    import mlx.nn as nn
    from mlx_vlm.models.deepseek_v41 import language as L
    rng = np.random.default_rng(1)
    mlp = type("M", (), {})()
    mlp.swiglu_limit = 10.0
    mlp.gate_proj, mlp.up_proj, mlp.down_proj = [nn.Linear(256, 512, bias=False), nn.Linear(256, 512, bias=False), nn.Linear(512, 256, bias=False)]
    for m in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        m.weight = (m.weight * 8).astype(mx.float16)
    x = mx.array(rng.standard_normal((3, 1, 256)).astype(np.float32)).astype(mx.float16)
    stock = lambda self, x: self.down_proj(L.fake_quant_fp8_ue8m0(L._swiglu(self.gate_proj(L.fake_quant_fp8_ue8m0(x)), self.up_proj(L.fake_quant_fp8_ue8m0(x)), self.swiglu_limit).astype(x.dtype)))
    assert np.array_equal(f32(PG._impl_shared_mlp(stock)(mlp, x)), f32(stock(mlp, x)))
