"""flint through the architecture kit with the mlx-vlm text class (CPU only; the module pins the MLX CPU device): config row (per-block sliding / rope patterns, leading dense blocks),
attention sinks (`attn_sinks` -> softmax_offset), the optional router bias, the tiny round trip class -> GGUF -> kit loader -> class logits, guards. Real header:
KYOJIN_TEST_FLINT_HDR, skipped when unset."""
from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

from kyojin_mlx import foreign_archkit as K, foreign_archkit_cpu as KC, foreign_cfgrows as CR, foreign_gguf as GF, foreign_synth as SY, foreign_vlm_classes as VC  # noqa: E402

pytest.importorskip("mlx_vlm.models.flint")
from mlx_vlm.models.flint import Model as VModel, ModelConfig  # noqa: E402

SW = [False, True, False, True]
RP = [True, False, True, True]
CFG = dict(model_type="flint", hidden_size=32, intermediate_size=48, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=50, rms_norm_eps=1e-6,
           max_position_embeddings=128, rope_theta=10000.0, layer_types=["sliding_attention" if x else "full_attention" for x in SW], no_rope_layer_indices=[i for i, x in enumerate(RP) if not x],
           use_sliding_window=True, sliding_window=3, mlp_only_layers=[0], num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16, shared_expert_intermediate_size=32,
           tie_word_embeddings=False)


@pytest.fixture(autouse=True)
def _switches(monkeypatch):
    monkeypatch.setenv("KYOJIN_KIT_UNPROVEN_ROWS", "1")
    monkeypatch.setenv("KYOJIN_KIT_VLM_CLASSES", "1")
    yield


def _model(seed=0):
    m = VModel(ModelConfig.from_dict(CFG))
    rng = np.random.default_rng(seed)
    vals = []
    for k, v in tree_flatten(m.parameters()):
        a = rng.standard_normal(v.shape) * (0.3 if v.ndim > 1 else 0.2) + (1.0 if v.ndim == 1 and "norm" in k else 0.0)
        vals.append((k, mx.array(np.zeros(v.shape, np.float32) if k.endswith("expert_bias") else a.astype(np.float32))))       # no router bias unless a test sets one
    m.update(tree_unflatten(vals))
    mx.eval(m.parameters())
    return m


_BLK = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output", "self_attn.q_norm": "attn_q_norm",
        "self_attn.k_norm": "attn_k_norm", "input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up",
        "mlp.down_proj": "ffn_down", "mlp.gate": "ffn_gate_inp", "mlp.switch_mlp.gate_proj": "ffn_gate_exps", "mlp.switch_mlp.up_proj": "ffn_up_exps", "mlp.switch_mlp.down_proj": "ffn_down_exps",
        "mlp.shared_expert.gate_proj": "ffn_gate_shexp", "mlp.shared_expert.up_proj": "ffn_up_shexp", "mlp.shared_expert.down_proj": "ffn_down_shexp"}


def to_gguf_tensors(model, bias=False):
    out = []
    for k, v in tree_flatten(model.parameters()):
        v, k = np.array(v), k[len("language_model."):]
        if k == "model.embed_tokens.weight":
            out.append(("token_embd.weight", v))
        elif k == "model.norm.weight":
            out.append(("output_norm.weight", v))
        elif k == "lm_head.weight":
            out.append(("output.weight", v))
        else:
            i, rest = k[len("model.layers."):].split(".", 1)
            if rest == "self_attn.softmax_offset":
                out.append((f"blk.{i}.attn_sinks.weight", v))              # conversion/flint.py: `.softmax_offset` + ".weight"
            elif rest == "mlp.expert_bias":
                if bias:
                    out.append((f"blk.{i}.exp_probs_b.bias", v))           # only when router_expert_bias
            else:
                out.append((f"blk.{i}." + _BLK[rest.rsplit(".", 1)[0]] + "." + rest.rsplit(".", 1)[1], v))
    return out


def kv_of(**over):
    kv = {"general.architecture": "flint", "flint.block_count": 4, "flint.context_length": 128, "flint.embedding_length": 32, "flint.feed_forward_length": 48,
          "flint.attention.head_count": 4, "flint.attention.head_count_kv": 2, "flint.rope.freq_base": 10000.0, "flint.attention.layer_norm_rms_epsilon": 1e-6,
          "flint.expert_count": 4, "flint.expert_used_count": 2, "flint.expert_gating_func": 2, "flint.attention.key_length": 8, "flint.attention.value_length": 8,
          "flint.expert_feed_forward_length": 16, "flint.expert_shared_feed_forward_length": 32, "flint.leading_dense_block_count": 1, "flint.expert_shared_count": 2,
          "flint.expert_weights_norm": True, "flint.expert_weights_scale": 1.0, "flint.attention.sliding_window": 3, "flint.attention.sliding_window_pattern": list(SW),
          "flint.attention.rope_pattern": list(RP), "flint.vocab_size": 50}
    kv.update(over)
    return kv


def write(path, model, kv=None, bias=False):
    SY.write_gguf(path, kv or kv_of(), [(n, tuple(reversed(a.shape)), "F32", np.ascontiguousarray(a, np.float32)) for n, a in to_gguf_tensors(model, bias)])


TOK = np.array([[3, 17, 5, 29, 8, 41, 2, 11, 30, 7]], np.int32)


def test_round_trip_logits_and_sinks(tmp_path):
    m = _model()
    p = str(tmp_path / "f.gguf")
    write(p, m)
    ref = np.array(m(mx.array(TOK)).logits)
    got, rep = KC.load_cpu(p)
    assert rep["model_type"] == "flint" and type(got).__module__.startswith("mlx_vlm.") and mx.default_device() == mx.cpu
    assert rep["config"]["no_rope_layer_indices"] == [1] and rep["config"]["mlp_only_layers"] == [0] and rep["config"]["layer_types"] == CFG["layer_types"]
    assert np.abs(np.array(got(mx.array(TOK)).logits) - ref).max() < 1e-4
    flat = dict(tree_flatten(got.parameters()))
    assert float(mx.abs(flat["language_model.model.layers.1.mlp.expert_bias"]).max()) == 0.0               # no exp_probs_b in the file: the class fills zeros
    m2 = _model()
    m2.update(tree_unflatten([(f"language_model.model.layers.{i}.self_attn.softmax_offset", mx.full((4,), 1.5)) for i in range(4)]))
    q = str(tmp_path / "f2.gguf")
    write(q, m2)
    ref2 = np.array(m2(mx.array(TOK)).logits)
    got2, _ = KC.load_cpu(q)
    assert np.abs(np.array(got2(mx.array(TOK)).logits) - ref2).max() < 1e-4 and np.abs(ref2 - ref).max() > 1e-3     # the sinks are read, and change the logits


def test_router_bias_is_kept_when_the_file_has_it(tmp_path):
    m = _model()
    m.update(tree_unflatten([(f"language_model.model.layers.{i}.mlp.expert_bias", mx.array([0.5, -0.5, 2.0, 0.0])) for i in range(1, 4)]))
    p = str(tmp_path / "f.gguf")
    write(p, m, bias=True)
    ref = np.array(m(mx.array(TOK)).logits)
    got, _ = KC.load_cpu(p)
    assert np.abs(np.array(got(mx.array(TOK)).logits) - ref).max() < 1e-4


@pytest.mark.parametrize("over,why", [({"flint.expert_weights_scale": 2.5}, "renormalises"), ({"flint.expert_weights_norm": False}, "renormalises"),
                                      ({"flint.attention.rope_pattern": [True]}, "one flag per block"), ({"flint.expert_gating_func": 1}, "sigmoid"),
                                      ({"flint.rope.scaling.type": "yarn"}, "rope.scaling")])
def test_guards_refuse(tmp_path, over, why):
    p = str(tmp_path / "f.gguf")
    write(p, _model(), kv_of(**over))
    with pytest.raises(GF.GgufError, match=why):
        KC.load_cpu(p)


def test_switches_off_are_the_old_error(tmp_path, monkeypatch):
    p = str(tmp_path / "f.gguf")
    write(p, _model())
    monkeypatch.setenv("KYOJIN_KIT_VLM_CLASSES", "0")
    with pytest.raises((K.KitError, GF.GgufError, ValueError)):
        KC.load_cpu(p)
    monkeypatch.setenv("KYOJIN_KIT_VLM_CLASSES", "1")
    monkeypatch.setenv("KYOJIN_KIT_UNPROVEN_ROWS", "0")
    with pytest.raises((K.KitError, GF.GgufError, ValueError)):
        KC.load_cpu(p)


def test_moe_variant_and_path():
    """256 routed experts top 6, sigmoid router, optional selection bias, renormalised, scale 1.0, 2 shared experts (one FFN of width 2 x 512), silu: the class block is not a softmax router
    (route_why), and the sigmoid select path of the description needs nvfp4 experts, so a K-quant file runs on the class's plain ops."""
    from kyojin_mlx import foreign_moe as FM, foreign_moe_generic as MG
    blk = _model().language_model.model.layers[1].mlp
    assert FM.route_why(blk) is not None
    spec = MG.MoeSpec(n_expert=4, top_k=2, gating="sigmoid", select_bias=True, norm_weights=True, act="silu", shared=True, n_dense_leading=1)
    assert MG.describe(spec) == {"kind": "glm_route", "why": []}


def test_real_header():
    f = os.environ.get("KYOJIN_TEST_FLINT_HDR")
    if not f:
        pytest.skip("KYOJIN_TEST_FLINT_HDR not set")
    g = GF.GgufFile(GF.find_gguf(f))
    cfg = CR.build("flint", g)
    Model, Args = VC.classes_for_config(cfg)
    names = [k[len("language_model."):] for k, _ in tree_flatten(Model(Args.from_dict(cfg)).parameters())]
    nm = K.resolve("flint", names, n_trunk=CR.n_trunk("flint", g.kv))
    assert {nm.hf(t) for t in g.tensors} <= set(names) and len(set(names) - {nm.hf(t) for t in g.tensors}) == 47
