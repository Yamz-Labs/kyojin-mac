"""gemma4 through the architecture kit (CPU only; the module pins the MLX CPU device): config row, converter rules (no norm shift, renamed router / expert-scale / layer-scale
tensors, fused gate_up experts, rope_freqs consumed by the config row), and the tiny round trip class -> GGUF -> kit loader -> class logits (dense per-layer-embedding variant and
the MoE variant). The real-header check reads KYOJIN_TEST_GEMMA4_HDR (a gemma4 MoE GGUF or a header-only copy of it) and is skipped when unset."""
from __future__ import annotations

import glob
import os

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402
from mlx_lm.models import gemma4_text as G4  # noqa: E402

from kyojin_mlx import foreign_archkit as K, foreign_archkit_cpu as KC, foreign_cfgrows as CR, foreign_gguf as GF, foreign_synth as SY, foreign_xform as XF  # noqa: E402

DENSE = dict(hidden_size=32, num_hidden_layers=6, intermediate_size=48, num_attention_heads=2, head_dim=16, global_head_dim=32, vocab_size=50, vocab_size_per_layer_input=50,
             num_key_value_heads=1, num_global_key_value_heads=1, num_kv_shared_layers=2, hidden_size_per_layer_input=8, sliding_window=8, sliding_window_pattern=3,
             max_position_embeddings=128, final_logit_softcapping=30.0, use_double_wide_mlp=True, rms_norm_eps=1e-6, global_partial_rotary_factor=0.5)
MOE = dict(DENSE, num_key_value_heads=2, num_kv_shared_layers=0, hidden_size_per_layer_input=0, use_double_wide_mlp=False, attention_k_eq_v=True, enable_moe_block=True,
           num_experts=4, top_k_experts=2, moe_intermediate_size=16, intermediate_size=24)


@pytest.fixture(autouse=True)
def _rows_on(monkeypatch):
    monkeypatch.setenv("KYOJIN_GEMMA4_ROWS", "1")           # the gemma4 rows are behind a switch (default off): these tests are the switch-on state; tests/test_arch_kit_switches.py has both states


def _model(cfg, seed=0):
    cfg = dict(cfg)
    cfg["rope_parameters"] = {"full_attention": {"partial_rotary_factor": cfg["global_partial_rotary_factor"], "rope_theta": 1e6, "rope_type": "proportional"},
                              "sliding_attention": {"partial_rotary_factor": 1.0, "rope_theta": 1e4, "rope_type": "default"}}
    m = G4.Model(G4.ModelArgs.from_dict(cfg))
    rng = np.random.default_rng(seed)
    m.update(tree_unflatten([(k, mx.array((rng.standard_normal(v.shape) * (0.3 if v.ndim > 1 else 0.2) + (1.0 if v.ndim == 1 else 0.0)).astype(np.float32))) for k, v in tree_flatten(m.parameters())]))
    mx.eval(m.parameters())
    return m, cfg


_BLK = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output", "self_attn.q_norm": "attn_q_norm",
        "self_attn.k_norm": "attn_k_norm", "input_layernorm": "attn_norm", "post_attention_layernorm": "post_attention_norm", "pre_feedforward_layernorm": "ffn_norm",
        "post_feedforward_layernorm": "post_ffw_norm", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down", "per_layer_input_gate": "inp_gate",
        "per_layer_projection": "proj", "post_per_layer_input_norm": "post_norm", "router.proj": "ffn_gate_inp", "pre_feedforward_layernorm_2": "pre_ffw_norm_2",
        "post_feedforward_layernorm_1": "post_ffw_norm_1", "post_feedforward_layernorm_2": "post_ffw_norm_2"}
_TOP = {"model.embed_tokens": "token_embd", "model.norm": "output_norm", "model.embed_tokens_per_layer": "per_layer_token_embd", "model.per_layer_model_projection": "per_layer_model_proj",
        "model.per_layer_projection_norm": "per_layer_proj_norm"}


def to_gguf_tensors(model):
    """What llama.cpp's converter (Gemma4Model) would write for these parameters: [(gguf name, numpy array in HF layout)]."""
    p = {k: np.array(v) for k, v in tree_flatten(model.parameters())}
    out = []
    for k, v in p.items():
        base, _, sfx = k.rpartition(".")
        if k in _TOP.values() or base in _TOP:
            out.append((_TOP[base] + "." + sfx, v))
            continue
        i, rest = k[len("model.layers."):].split(".", 1)
        if rest == "layer_scalar":
            out.append((f"blk.{i}.layer_output_scale.weight", v))
        elif rest == "router.scale":
            out.append((f"blk.{i}.ffn_gate_inp.scale", v))
        elif rest == "router.per_expert_scale":
            out.append((f"blk.{i}.ffn_down_exps.scale", v))
        elif ".switch_glu.gate_proj" in rest:
            out.append((f"blk.{i}.ffn_gate_up_exps.weight", np.concatenate([v, p[k.replace("gate_proj", "up_proj")]], axis=-2)))
        elif ".switch_glu.up_proj" in rest:
            continue
        elif ".switch_glu.down_proj" in rest:
            out.append((f"blk.{i}.ffn_down_exps.weight", v))
        else:
            out.append((f"blk.{i}." + _BLK[rest.rsplit(".", 1)[0]] + "." + sfx, v))
    return out


def kv_of(cfg):
    nt, ns = cfg["num_hidden_layers"], cfg["num_kv_shared_layers"]
    sp = cfg["sliding_window_pattern"]
    pat = [0 if (i + 1) % sp == 0 else 1 for i in range(nt)]
    hk = cfg["num_key_value_heads"] if cfg["num_key_value_heads"] == cfg["num_global_key_value_heads"] else [cfg["num_key_value_heads"] if s else cfg["num_global_key_value_heads"] for s in pat]
    ff = [cfg["intermediate_size"] * (2 if cfg["use_double_wide_mlp"] and i >= nt - ns > 0 else 1) for i in range(nt)] if cfg["use_double_wide_mlp"] else cfg["intermediate_size"]
    kv = {"general.architecture": "gemma4", "gemma4.block_count": nt, "gemma4.context_length": cfg["max_position_embeddings"], "gemma4.embedding_length": cfg["hidden_size"],
          "gemma4.feed_forward_length": ff, "gemma4.attention.head_count": cfg["num_attention_heads"], "gemma4.attention.head_count_kv": hk, "gemma4.rope.freq_base": 1e6,
          "gemma4.rope.freq_base_swa": 1e4, "gemma4.attention.layer_norm_rms_epsilon": 1e-6, "gemma4.attention.key_length": cfg["global_head_dim"],
          "gemma4.attention.value_length": cfg["global_head_dim"], "gemma4.attention.key_length_swa": cfg["head_dim"], "gemma4.attention.value_length_swa": cfg["head_dim"],
          "gemma4.final_logit_softcapping": 30.0, "gemma4.attention.sliding_window": cfg["sliding_window"], "gemma4.attention.shared_kv_layers": ns,
          "gemma4.embedding_length_per_layer_input": cfg["hidden_size_per_layer_input"], "gemma4.attention.sliding_window_pattern": pat,
          "gemma4.rope.dimension_count": cfg["global_head_dim"], "gemma4.rope.dimension_count_swa": cfg["head_dim"]}
    if cfg.get("enable_moe_block"):
        kv.update({"gemma4.expert_count": cfg["num_experts"], "gemma4.expert_used_count": cfg["top_k_experts"], "gemma4.expert_feed_forward_length": cfg["moe_intermediate_size"]})
    return kv


def write(path, model, cfg, kv=None, freqs=None):
    tl = [(n, tuple(reversed(a.shape)), "F32", np.ascontiguousarray(a, np.float32)) for n, a in to_gguf_tensors(model)]
    ghd = cfg["global_head_dim"]
    nrot = int(ghd * cfg["global_partial_rotary_factor"] / 2)
    f = np.array([1.0] * nrot + [1e30] * (ghd // 2 - nrot), np.float32) if freqs is None else freqs
    tl.append(("rope_freqs.weight", (len(f),), "F32", f))
    SY.write_gguf(path, kv or kv_of(cfg), tl)


TOK = np.array([[3, 17, 5, 29, 8, 41, 2, 11, 30, 7]], np.int32)


@pytest.mark.parametrize("cfg", [DENSE, MOE], ids=["dense_per_layer_embedding", "moe_26b_shape"])
def test_round_trip_logits(cfg, tmp_path):
    m, cfg = _model(cfg)
    p = str(tmp_path / "g.gguf")
    write(p, m, cfg)
    ref = np.array(m(mx.array(TOK)))
    got, rep = KC.load_cpu(p)
    assert rep["model_type"] == "gemma4_text" and mx.default_device() == mx.cpu
    assert np.abs(np.array(got(mx.array(TOK))) - ref).max() < 1e-4
    assert "rope_freqs.weight" in rep["skipped"]
    w = rep["weights"]
    assert np.array_equal(np.array(w["model.layers.0.input_layernorm.weight"]), np.array(m.model.layers[0].input_layernorm.weight))      # norm_shift = 0: no +1 on the stored norms
    assert got.args.global_partial_rotary_factor == cfg["global_partial_rotary_factor"]


def test_no_value_transform_in_the_xform_tables():
    assert K.vetted("gemma4", "xform", "") == ""
    assert XF.plan("blk.0.attn_norm.weight", "", {}) is None


def test_names_of_the_renamed_tensors(tmp_path):
    m, cfg = _model(MOE)
    nm = K.resolve("gemma4", [k for k, _ in tree_flatten(m.parameters())], n_trunk=6, check=False)
    assert nm.hf("blk.3.layer_output_scale.weight") == "model.layers.3.layer_scalar"
    assert nm.hf("blk.3.ffn_gate_inp.scale") == "model.layers.3.router.scale"
    assert nm.hf("blk.3.ffn_down_exps.scale") == "model.layers.3.router.per_expert_scale"
    assert nm.hf("blk.3.ffn_gate_up_exps.weight") == "model.layers.3.experts.gate_up_proj"
    assert nm.hf("blk.3.ffn_gate_inp.weight") == "model.layers.3.router.proj.weight"
    assert nm.hf("blk.3.post_ffw_norm_1.weight") == "model.layers.3.post_feedforward_layernorm_1.weight"
    assert nm.hf("blk.3.pre_ffw_norm_2.weight") == "model.layers.3.pre_feedforward_layernorm_2.weight"


def test_rope_freqs_rule(tmp_path):
    m, cfg = _model(DENSE)
    for nrot, want in ((4, 0.25), (16, 1.0)):
        p = str(tmp_path / f"f{nrot}.gguf")
        write(p, m, cfg, freqs=np.array([1.0] * nrot + [1e30] * (16 - nrot), np.float32))
        assert CR.build("gemma4", GF.GgufFile(p))["rope_parameters"]["full_attention"]["partial_rotary_factor"] == want
    bad = str(tmp_path / "bad.gguf")
    write(bad, m, cfg, freqs=np.array([1.0, 2.0] + [1e30] * 14, np.float32))
    with pytest.raises(GF.GgufError, match="not of the form"):
        CR.build("gemma4", GF.GgufFile(bad))
    with pytest.raises(K.KitError):                     # a file whose arch has no `consumed` rope_freqs row still refuses the tensor
        K.VETTED["gemma4"]["consumed"] = ()
        try:
            KC.load_cpu(str(tmp_path / "f4.gguf"))
        finally:
            K.VETTED["gemma4"]["consumed"] = ("rope_freqs.weight",)


@pytest.mark.parametrize("over, why", [({"gemma4.attention.sliding_window_pattern": [1, 0, 1, 1, 0, 1]}, "regular"), ({"gemma4.rope.scaling.type": "yarn"}, "rope.scaling"),
                                       ({"gemma4.rope.dimension_count_swa": 8}, "partial rope"), ({"gemma4.attention.value_length": 16}, "value_length"),
                                       ({"gemma4.attention.head_count_kv": [1, 1, 3, 1, 1, 2]}, "kv head count")])
def test_guards_refuse(over, why, tmp_path):
    m, cfg = _model(DENSE)
    p = str(tmp_path / "g.gguf")
    write(p, m, cfg, kv=dict(kv_of(cfg), **over))
    with pytest.raises(GF.GgufError, match=why.split()[0] if why != "kv head count" else "kv head count|head_count_kv"):
        CR.build("gemma4", GF.GgufFile(p))


def test_real_moe_header_maps_every_tensor():
    src = os.environ.get("KYOJIN_TEST_GEMMA4_HDR")
    if not src:
        pytest.skip("KYOJIN_TEST_GEMMA4_HDR not set")
    g = GF.GgufFile(src if src.endswith(".gguf") else glob.glob(os.path.join(src, "*.gguf"))[0])
    cfg = CR.build("gemma4", g)
    m = G4.Model(G4.ModelArgs.from_dict(cfg))
    names = [k for k, _ in tree_flatten(m.parameters())]
    nm = K.resolve("gemma4", names, n_trunk=CR.n_trunk("gemma4", g.kv), check=False)
    mapped = {n: nm.hf(n) for n in g.tensors if n != "rope_freqs.weight"}
    assert len(mapped) == len(g.tensors) - 1
    shapes = dict((k, tuple(v.shape)) for k, v in tree_flatten(m.parameters()))
    direct = [n for n, h in mapped.items() if h in shapes]
    assert all(tuple(reversed(g.tensors[n].ne)) == shapes[mapped[n]] for n in direct)
    experts = [n for n, h in mapped.items() if h.endswith(("experts.gate_up_proj", "experts.down_proj"))]
    assert len(direct) + len(experts) == len(mapped) and 3 * len(experts) // 2 == sum(1 for k in shapes if "switch_glu" in k)
