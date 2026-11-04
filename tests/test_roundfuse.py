"""roundfuse: the decfuse classes for any row count (decode, verify, multi-user rounds) and for foreign packs.
  - residual add + RMSNorm: bit-equal to the stock pair at 1 / 6 / 16 / 48 rows, saturating on fp16 overflow (a stream value past 65504 must not become inf / NaN);
  - the add + norm forward installed on a hybrid model (decrows.enable_foreign): logits bit-equal to the stock forward at every row count;
  - the fused gated-delta rows path of the capture call (gdn_pf.forward_rows): logits within fp16 rounding of the stock captured call, and the rollback
    (qspec.rollback_gdn) to a shorter prefix equal (to rounding) to a pass over that prefix alone.
Tiny random hybrid fp16 model, head dim 128 so the fused gated-delta path is taken."""
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_map
from mlx_lm.models import qwen3_5
from mlx_lm.models.cache import make_prompt_cache

from kyojin_mlx import decfuse as DF, decrows as DR, gdn_pf, qspec as QS

pytestmark = pytest.mark.skipif(mx.default_device() != mx.gpu, reason="Metal kernels")
ROWS = (1, 6, 16, 48)


def _model(seed=5):
    tc = dict(model_type="qwen3_5", hidden_size=256, num_hidden_layers=4, intermediate_size=512, num_attention_heads=2, num_key_value_heads=1, head_dim=64,
              linear_num_value_heads=2, linear_num_key_heads=1, linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
              vocab_size=256, rms_norm_eps=1e-6, rope_theta=1000.0, partial_rotary_factor=0.5, max_position_embeddings=512,
              full_attention_interval=2, tie_word_embeddings=False, rope_parameters={"type": "default", "rope_theta": 1000.0, "partial_rotary_factor": 0.5})
    m = qwen3_5.Model(qwen3_5.ModelArgs.from_dict(dict(model_type="qwen3_5", text_config=tc)))
    mx.random.seed(seed)
    m.update(tree_map(lambda a: (mx.random.normal(a.shape) * 0.05).astype(mx.float16) if a.dtype == mx.float32 else a, m.parameters()))
    for l in m.language_model.model.layers:
        for n in (l.input_layernorm, l.post_attention_layernorm):
            n.weight = (1 + 0.1 * mx.random.normal(n.weight.shape)).astype(mx.float16)
        if l.is_linear:
            a = l.linear_attn
            a.A_log = mx.random.normal(a.A_log.shape).astype(mx.float32) * 0.5
            a.dt_bias = a.dt_bias.astype(mx.float16)
            a.norm.weight = (1 + 0.1 * mx.random.normal(a.norm.weight.shape)).astype(mx.float16)
    m.eval()
    mx.eval(m.parameters())
    return m


def _prefilled(m, n=10):
    cache = make_prompt_cache(m)
    mx.eval(m(mx.array(np.arange(1, n + 1) % 250)[None], cache=cache))
    return cache


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("dt", [mx.float16, mx.bfloat16])
def test_add_rms_rows_bit_equal(rows, dt):
    for H in (256, 1024, 5120):
        mx.random.seed(H + rows)
        x, r = (3 * mx.random.normal((1, rows, H))).astype(dt), mx.random.normal((1, rows, H)).astype(dt)
        n = nn.RMSNorm(H)
        n.weight = (1 + 0.2 * mx.random.normal((H,))).astype(dt)
        h, y = DF.add_rms(x, r, n)
        assert mx.array_equal(h, x + r).item() and mx.array_equal(y, n(x + r)).item()


def test_add_rms_saturates_fp16_overflow():
    H = 256
    x = mx.full((1, 2, H), 60000.0).astype(mx.float16)
    r = mx.full((1, 2, H), 50000.0).astype(mx.float16)
    n = nn.RMSNorm(H)
    n.weight = mx.ones((H,), mx.float16)
    h, y = DF.add_rms(x, r, n)
    assert mx.all(h == 65504.0).item() and not mx.any(mx.isnan(y)).item() and not mx.any(mx.isinf(y)).item()


@pytest.mark.parametrize("rows", ROWS)
def test_foreign_forward_add_norm_bit_equal(rows):
    stock, fused = _model(), _model()
    assert DR.enable_foreign(fused) and not DR.enable_foreign(fused)
    cs, cf = _prefilled(stock), _prefilled(fused)
    x = mx.array((np.arange(rows) * 7 + 3) % 250)[None]
    a, b = stock(x, cache=cs), fused(x, cache=cf)
    assert mx.array_equal(a, b).item()
    assert fused.language_model.model.__class__.__name__.startswith("Kd")


def _captured(m, cache, x):
    QS.CAP = []
    try:
        out = m(x, cache=cache)
        mx.eval(out)
        return out, QS.CAP
    finally:
        cap, QS.CAP = QS.CAP, None


@pytest.fixture
def capture():
    QS.install_capture()
    yield
    QS.uninstall_capture()


@pytest.mark.parametrize("rows", ROWS)
def test_fused_gdn_rows_close_to_stock_capture(rows, capture, monkeypatch):
    m = _model()
    x = mx.array((np.arange(rows) * 5 + 2) % 250)[None]
    ca, cb = _prefilled(m), _prefilled(m)
    fused, rec = _captured(m, ca, x)
    assert any(isinstance(r[1], QS.FusedGdnCapture) for r in rec)
    monkeypatch.setattr(DF, "GDN", False)
    stock, rec2 = _captured(m, cb, x)
    assert not any(isinstance(r[1], QS.FusedGdnCapture) for r in rec2)
    f, s = np.array(fused.astype(mx.float32)), np.array(stock.astype(mx.float32))
    assert np.abs(f - s).max() < 0.05 * max(1.0, np.abs(s).max())
    for i, (u, v) in enumerate(zip(ca, cb)):
        if hasattr(u, "state") and u.state[0] is not None and len(u.state) == 2 and u.state[1].dtype == mx.float32:
            assert np.abs(np.array(u.state[1]) - np.array(v.state[1])).max() < 0.02 * max(1.0, np.abs(np.array(v.state[1])).max())


@pytest.mark.parametrize("rows,keep", [(6, 3), (16, 1), (16, 9), (48, 20)])
def test_fused_gdn_rollback_equals_prefix_pass(rows, keep, capture):
    m = _model()
    x = mx.array((np.arange(rows) * 5 + 2) % 250)[None]
    full, rec = _captured(m, c_full := _prefilled(m), x)
    _, _ = _captured(m, c_pref := _prefilled(m), x[:, :keep])
    lm = m.language_model.model
    idx = {id(l.linear_attn): i for i, l in enumerate(lm.layers) if l.is_linear}
    for r in rec:
        QS.rollback_gdn(r[0], c_full[idx[id(r[0])]], keep, r)
    for i in idx.values():
        mx.eval(c_full[i].state, c_pref[i].state)
        for j, tol in ((0, 0.02), (1, 0.02)):          # the dense projections round per row count (MLX GEMM tiles), so the two passes agree to rounding, not bits
            u, v = np.array(c_full[i][j].astype(mx.float32)), np.array(c_pref[i][j].astype(mx.float32))
            assert np.abs(u - v).max() < tol * max(1.0, np.abs(v).max())


@pytest.mark.parametrize("rows", ROWS)
def test_gdn_kernels_read_column_slices_in_place(rows):
    """The front end and the gated norm read qkv / a / b / z as column slices of a grouped projection output: same bits as on contiguous copies."""
    hk, hv = 2, 4
    C = 2 * hk * 128 + hv * 128
    mx.random.seed(rows)
    big = (mx.random.normal((1, rows, C + 1000)) * 0.5).astype(mx.float16)
    ab = mx.random.normal((1, rows, 2 * hv + 5)).astype(mx.float16)
    zbig = mx.random.normal((1, rows, hv * 128 + 300)).astype(mx.float16)
    qkv, a, b, z = big[..., 7:7 + C], ab[..., 2:2 + hv], ab[..., 2 + hv:2 + 2 * hv], zbig[..., 11:11 + hv * 128].reshape(1, rows, hv, 128)
    cs = mx.random.normal((1, 3, C)).astype(mx.float16)
    w = (mx.random.normal((C, 4, 1)) * 0.3).astype(mx.float16)
    A, dtb = mx.random.normal((hv,)).astype(mx.float32), mx.random.normal((hv,)).astype(mx.float16)
    ref = gdn_pf.pre(mx.contiguous(qkv), cs, w, mx.contiguous(a), mx.contiguous(b), A, dtb, hk, hv)
    got = gdn_pf.pre(qkv, cs, w, a, b, A, dtb, hk, hv)
    nw = (1 + 0.1 * mx.random.normal((128,))).astype(mx.float16)
    h = mx.random.normal((1, rows, hv, 128)).astype(mx.float16)
    n_ref, n_got = gdn_pf.gated_norm(h, mx.contiguous(z), nw, 1e-6), gdn_pf.gated_norm(h, z, nw, 1e-6)
    mx.eval(ref, got, n_ref, n_got)
    assert all(mx.array_equal(u, v).item() for u, v in zip(ref, got)) and mx.array_equal(n_ref, n_got).item()


def test_foreign_regions_equal_eager_decode(monkeypatch):
    """One-row decode steps in compiled regions (foreign_regions) give the eager body's logits and gated-delta states, step after step."""
    from kyojin_mlx import foreign_regions as FR
    eager, regions = _model(), _model()
    assert DR.enable_foreign(eager) and DR.enable_foreign(regions)
    ce, cr = _prefilled(eager), _prefilled(regions)
    fast0 = FR.STATS["fast"]
    for step in range(6):
        x = mx.array([[(step * 11 + 5) % 250]])
        monkeypatch.setattr(FR, "ENABLED", False)
        a = eager(x, cache=ce)
        monkeypatch.setattr(FR, "ENABLED", True)
        b = regions(x, cache=cr)
        mx.eval(a, b)
        assert mx.allclose(a, b, atol=2e-3, rtol=2e-3).item()
        assert mx.argmax(a, -1).item() == mx.argmax(b, -1).item()
    assert FR.STATS["fast"] - fast0 == 6
    for i, l in enumerate(regions.language_model.model.layers):
        if l.is_linear:
            assert mx.allclose(ce[i][1], cr[i][1], atol=2e-3, rtol=2e-3).item()


def test_foreign_regions_fall_back_for_rows_and_capture(monkeypatch):
    from kyojin_mlx import foreign_regions as FR
    m = _model()
    assert DR.enable_foreign(m)
    c = _prefilled(m)
    fast0 = FR.STATS["fast"]
    mx.eval(m(mx.array([[1, 2, 3]]), cache=c))              # 3 rows: eager body
    QS.install_capture()
    try:
        _captured(m, c, mx.array([[4]]))                    # a captured pass stays eager
    finally:
        QS.uninstall_capture()
    assert FR.STATS["fast"] == fast0
