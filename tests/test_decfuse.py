"""decfuse: fused same-input projections and residual add + RMSNorm in the compiled decode regions are bit-equal to the stock forward; the fused
gated-delta front end stays within rounding (tiny random hybrid model, bf16, MLX affine 4-bit linears, head dim 128 so every fused path is taken)."""
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_map
from mlx_lm.models import qwen3_5
from mlx_lm.models.cache import make_prompt_cache

from kyojin_mlx import decfuse as DF
from kyojin_mlx import step_regions as SR


def _model():
    tc = dict(model_type="qwen3_5", hidden_size=256, num_hidden_layers=4, intermediate_size=512, num_attention_heads=2, num_key_value_heads=1, head_dim=64,
              linear_num_value_heads=2, linear_num_key_heads=1, linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
              vocab_size=256, rms_norm_eps=1e-6, rope_theta=1000.0, partial_rotary_factor=0.5, max_position_embeddings=512,
              full_attention_interval=2, tie_word_embeddings=False, rope_parameters={"type": "default", "rope_theta": 1000.0, "partial_rotary_factor": 0.5})
    m = qwen3_5.Model(qwen3_5.ModelArgs.from_dict(dict(model_type="qwen3_5", text_config=tc)))
    mx.random.seed(5)
    m.update(tree_map(lambda a: (mx.random.normal(a.shape) * 0.05).astype(mx.bfloat16) if a.dtype == mx.float32 else a, m.parameters()))
    for l in m.language_model.model.layers:
        for n in (l.input_layernorm, l.post_attention_layernorm):
            n.weight = (1 + 0.1 * mx.random.normal(n.weight.shape)).astype(mx.bfloat16)
        if l.is_linear:
            l.linear_attn.A_log = mx.random.normal(l.linear_attn.A_log.shape).astype(mx.float32) * 0.5
    nn.quantize(m, group_size=64, bits=4, class_predicate=lambda p, mod: isinstance(mod, nn.Linear) and "lm_head" not in p)
    m.eval()
    mx.eval(m.parameters())
    return m


def _run(m, steps=12):
    cache = make_prompt_cache(m)
    ids = mx.array(np.arange(1, 30) % 250)[None]
    lg = m(ids, cache=cache)[:, -1]
    t = mx.argmax(lg, axis=-1)
    out = []
    for _ in range(steps):
        lg = m(t[:, None], cache=cache)[:, -1]
        t = mx.argmax(lg, axis=-1)
        mx.eval(lg, t)
        out.append(np.array(lg.astype(mx.float32)))
    return out


@pytest.mark.skipif(not (SR.ENABLED and DF.ENABLED), reason="step regions or decfuse off")
def test_proj_and_add_rms_bit_equal_gdn_close(monkeypatch):
    from kyojin_mlx import gdn_pf
    calls = []
    pre = gdn_pf.pre
    monkeypatch.setattr(gdn_pf, "pre", lambda *a: calls.append(1) or pre(*a))
    m = _model()
    ref = _run(m)
    gdn = DF.GDN
    try:
        DF.GDN = False
        assert SR.enable(m)
        tm = m.language_model.model
        assert sum(getattr(l.linear_attn if l.is_linear else l.self_attn, "_kr_fused", None) is not None for l in tm.layers) == 4
        got = _run(m)
        assert all(np.array_equal(a, b) for a, b in zip(ref, got))
        DF.GDN = True
        tm._kr_plan = SR._Plan(tm)
        got = _run(m)
        assert calls, "fused gated-delta front end not taken"
        assert all(np.argmax(a) == np.argmax(b) and np.abs(a - b).max() < 0.05 * max(1.0, np.abs(a).max()) for a, b in zip(ref, got))
        assert SR.STATS["fast"] >= 24
    finally:
        DF.GDN = gdn


def test_add_rms_matches_stock_pair():
    for H in (256, 2560, 4096, 5120):
        for dt in (mx.bfloat16, mx.float16):
            mx.random.seed(H)
            x, r = (3 * mx.random.normal((1, 1, H))).astype(dt), mx.random.normal((1, 1, H)).astype(dt)
            n = nn.RMSNorm(H)
            n.weight = (1 + 0.2 * mx.random.normal((H,))).astype(dt)
            h, y = DF.add_rms(x, r, n)
            assert mx.array_equal(h, x + r).item() and mx.array_equal(y, n(x + r)).item()
