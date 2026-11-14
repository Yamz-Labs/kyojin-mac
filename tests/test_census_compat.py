"""Compatibility fixes found by the 1.9.7 architecture census (tiny random packs): TurboQuant layer check without a head_dim attribute,
Qwen3.5 text-only model types, top-level config keys inherited by text_config, per-expert Qwen3.5-MoE keys, a pack without a chat template. CPU-only, no model download."""
import json
from types import SimpleNamespace as NS

import mlx.core as mx
import pytest


def test_tq_reads_head_dim_from_the_model_args_when_the_attention_has_none():
    """glm4_moe / phimoe attention modules carry no head_dim attribute: the model-wide value (args.head_dim, else hidden / heads) decides."""
    from kyojin_mlx import kvwin
    layers = [NS(self_attn=NS()) for _ in range(3)]
    assert kvwin._tq_unsupported_layers(NS(model=NS(layers=layers), args=NS(head_dim=8)), 3, 6, 6) == {0, 1, 2}
    assert kvwin._tq_unsupported_layers(NS(model=NS(layers=layers), args=NS(hidden_size=32, num_attention_heads=4)), 3, 6, 6) == {0, 1, 2}
    assert kvwin._tq_unsupported_layers(NS(model=NS(layers=layers), args=NS(head_dim=128)), 3, 6, 6) == set()
    assert kvwin._tq_unsupported_layers(NS(model=NS(layers=layers), args={"text_config": None, "head_dim": 8}), 3, 6, 6) == {0, 1, 2}


def test_qwen35_text_model_types_resolve_to_the_mlx_lm_classes():
    from mlx_lm import utils
    from kyojin_mlx import mlxpack
    mlxpack.remap_text_types()
    for mt, mod in (("qwen3_5_moe_text", "qwen3_5_moe"), ("qwen3_5_text", "qwen3_5")):
        cls, _ = utils._get_classes({"model_type": mt})
        assert cls.__module__.endswith(mod)


def test_top_level_tie_word_embeddings_reaches_text_config(tmp_path):
    from mlx_lm import utils
    from kyojin_mlx import mlxpack
    mlxpack.remap_text_types()
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_vl", "tie_word_embeddings": False, "text_config": {"hidden_size": 8}}))
    assert utils.load_config(tmp_path)["text_config"]["tie_word_embeddings"] is False
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_vl", "tie_word_embeddings": False, "text_config": {"tie_word_embeddings": True}}))
    assert utils.load_config(tmp_path)["text_config"]["tie_word_embeddings"] is True       # an explicit text_config value wins


def test_per_expert_keys_stack_into_the_fused_layout():
    from kyojin_mlx import mlxpack
    E, I, H = 3, 4, 5
    w = {}
    for e in range(E):
        for part, shape in (("gate", (I, H)), ("up", (I, H)), ("down", (H, I))):
            w[f"model.language_model.layers.0.mlp.experts.{e}.{part}_proj.weight"] = mx.full(shape, float(e * 10 + {"gate": 1, "up": 2, "down": 3}[part]))
    w["model.language_model.layers.0.mlp.gate.weight"] = mx.zeros((E, H))
    out = mlxpack._stack_experts(dict(w))
    gu = out["model.language_model.layers.0.mlp.experts.gate_up_proj"]
    dn = out["model.language_model.layers.0.mlp.experts.down_proj"]
    assert gu.shape == (E, 2 * I, H) and dn.shape == (E, H, I)
    assert float(gu[2, 0, 0]) == 21 and float(gu[2, I, 0]) == 22 and float(dn[1, 0, 0]) == 13       # gate rows first, then up (mlx-lm splits at mid)
    assert "model.language_model.layers.0.mlp.gate.weight" in out and not any(".experts.0." in k for k in out)


def test_fallback_family_without_a_template_serves_raw_and_rejects_chat(tmp_path):
    from kyojin_mlx.serve import chat as CH

    class Tok:
        chat_template = None
        special_tokens_map = {}
    fam = CH.JinjaFamily(Tok(), str(tmp_path), None, {})            # no chat_template.jinja, no tokenizer template: the server still starts
    with pytest.raises(CH.ChatError, match="no chat template"):
        fam.render({"messages": [{"role": "user", "content": "hi"}]})


def test_nemotron_h_transformers5_embedding_key_is_renamed():
    """transformers >= 5 saves NemotronH's embedding as backbone.embedding.weight; mlx-lm's class reads backbone.embeddings.weight."""
    from mlx_lm.models import nemotron_h
    from kyojin_mlx import mlxpack
    mlxpack.remap_text_types()
    fake = NS(args=NS(num_hidden_layers=0, n_routed_experts=0))
    out = nemotron_h.Model.sanitize(fake, {"backbone.embedding.weight": mx.zeros((4, 2)), "backbone.norm_f.weight": mx.ones((2,))})
    assert set(out) == {"backbone.embeddings.weight", "backbone.norm_f.weight"}


def test_fp8_block_sanitize_uses_the_pack_block():
    """FP8 packs with weight_block_size [32, 32] (tiny DeepseekV3 / GLM-5.3): the weights are dequantized with the pack's block before the class's
    own sanitize (mlx-lm hard-codes 128), the scale keys are dropped, and the class's sanitize is restored by the caller."""
    import mlx.core as mx
    import numpy as np
    from kyojin_mlx.serve import backends as B

    seen = {}

    class M:
        def sanitize(self, weights):
            seen.update(weights)
            return weights

    rng = np.random.default_rng(0)
    w = rng.uniform(-2, 2, (70, 40)).astype(np.float32)
    s = rng.uniform(0.5, 2.0, (3, 2)).astype(np.float32)                     # ceil(70 / 32) x ceil(40 / 32) blocks
    q = mx.to_fp8(mx.array(w))
    orig = M.sanitize
    prev = B._fp8_block_sanitize(M, {"quantization_config": {"quant_method": "fp8", "weight_block_size": [32, 32]}})
    assert prev is orig
    M().sanitize({"a.weight": q, "a.weight_scale_inv": mx.array(s), "b.bias": mx.zeros((2,))})
    M.sanitize = prev
    assert set(seen) == {"a.weight", "b.bias"}
    ref = np.array(mx.from_fp8(q, dtype=mx.float32)) * np.repeat(np.repeat(s, 32, 0), 32, 1)[:70, :40]
    assert np.allclose(np.array(seen["a.weight"].astype(mx.float32)), ref, rtol=1e-2, atol=1e-2)
    assert B._fp8_block_sanitize(M, {"quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}}) is None
