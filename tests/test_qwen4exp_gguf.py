# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next GGUF rows (foreign_qwen4exp_gguf): one test per converter rule on synthetic tensors, CPU only (numpy + mlx cpu)."""
import numpy as np
import pytest

from kyojin_mlx import foreign_linear as FL
from kyojin_mlx import foreign_names as FN
from kyojin_mlx import foreign_pack as FP
from kyojin_mlx import foreign_qwen4exp_gguf as Q
from kyojin_mlx import foreign_synth as SY
from kyojin_mlx import foreign_unproven as UP
from kyojin_mlx import foreign_xform as XF
from kyojin_mlx.foreign_gguf import GgufError

LM = "model.language_model."
KV = {"qwen4exp.ssm.group_count": 4, "qwen4exp.ssm.time_step_rank": 12, "qwen4exp.ssm.state_size": 8, "qwen4exp.ssm.inner_size": 96}


def _sync(monkeypatch, on):
    if on:
        monkeypatch.setenv("KYOJIN_KIT_UNPROVEN_ROWS", "1")
    else:
        monkeypatch.delenv("KYOJIN_KIT_UNPROVEN_ROWS", raising=False)
    Q.register()


@pytest.fixture(autouse=True)
def _rows_on(monkeypatch):
    _sync(monkeypatch, True)
    yield
    monkeypatch.undo()
    Q.register()


def test_switch_off_registers_nothing(monkeypatch):
    _sync(monkeypatch, False)
    assert "qwen4exp" not in FN.ARCHS and "qwen4exp" not in FN._CONFIG and "qwen4exp" not in XF._SETS
    assert FL.vlm_family("qwen4_exp") is None and FL.vlm_family("glm5_next") == "mlx_vlm.models.glm5_next" and FL.vlm_family("qwen3") is None
    assert UP.has_unproven("qwen4exp") and not UP.on("qwen4exp")                          # the hint line says the rows exist


def test_switch_on_registers():
    assert FN.ARCHS["qwen4exp"].model_type == "qwen4_exp" and "qwen4exp" in XF._SETS and FL.vlm_family("qwen4_exp") == "mlx_vlm.models.qwen4_exp"


@pytest.mark.parametrize("gguf,hf", [
    ("token_embd.weight", LM + "embed_tokens.weight"), ("output.weight", "lm_head.weight"),
    ("output_hc_down.weight", LM + "hyper_connection_mixer.input_mix_weight_down.weight"), ("output_hc_up.weight", LM + "hyper_connection_mixer.input_mix_weight_up.weight"),
    ("output_hc_norm.weight", LM + "hyper_connection_mixer.hc_norm.weight"),
    ("blk.3.hc_attn_down.weight", LM + "layers.3.attn_hyper_connection.input_mix_weight_down.weight"),
    ("blk.3.hc_attn_inject.weight", LM + "layers.3.attn_hyper_connection.block_inject_weight.weight"),
    ("blk.3.hc_ffn_up.weight", LM + "layers.3.mlp_hyper_connection.input_mix_weight_up.weight"),
    ("blk.3.hc_ffn_norm.weight", LM + "layers.3.mlp_hyper_connection.hc_norm.weight"),
    ("blk.0.attn_qkv.weight", LM + "layers.0.linear_attn.in_proj_qkv.weight"), ("blk.0.attn_gate.weight", LM + "layers.0.linear_attn.in_proj_z.weight"),
    ("blk.0.ssm_alpha.weight", LM + "layers.0.linear_attn.in_proj_a.weight"), ("blk.0.ssm_beta.weight", LM + "layers.0.linear_attn.in_proj_b.weight"),
    ("blk.0.ssm_a", LM + "layers.0.linear_attn.A_log"), ("blk.0.ssm_dt.bias", LM + "layers.0.linear_attn.dt_bias"),
    ("blk.3.indexer.q_norm.weight", LM + "layers.3.self_attn.indexer.q_layernorm.weight"), ("blk.3.indexer.k_proj.weight", LM + "layers.3.self_attn.indexer.k_proj.weight"),
    ("blk.1.ple_key.weight", LM + "layers.1.ple.key_proj.weight"), ("blk.1.ple_conv1d.weight", LM + "layers.1.ple.conv1d.weight"),
    ("blk.4.ffn_gate_exps.weight", LM + "layers.4.mlp.experts.gate_proj.weight"), ("blk.4.ffn_gate_inp_shexp.weight", LM + "layers.4.mlp.shared_expert_gate.weight"),
    ("blk.4.ffn_down_shexp.weight", LM + "layers.4.mlp.shared_expert.down_proj.weight"), ("blk.4.ffn_gate_inp.weight", LM + "layers.4.mlp.gate.weight"),
])
def test_names(gguf, hf):
    assert FN.gguf_name(gguf, "qwen4exp", 48) == hf


def test_unknown_name_refused():
    with pytest.raises(GgufError):
        FN.gguf_name("blk.0.surprise.weight", "qwen4exp", 48)


@pytest.mark.parametrize("name", ["output_hc_norm.weight", "blk.3.hc_attn_norm.weight", "blk.3.attn_q_norm.weight", "blk.3.indexer.k_norm.weight",
                                  "blk.1.ple_norm_key.weight", "blk.1.ple_norm_query.weight", "blk.1.ple_norm_conv.weight"])
def test_norm_plus_one_undone(name):
    """qwen.py:431-432 / qwen4exp.py:163-166: the converter stored w + 1; the class wants w (zero-centred (1 + w) norms)."""
    x = XF.plan(name, "qwen4exp", KV)
    assert x.post == "minus1"
    w = np.random.default_rng(0).normal(size=16).astype(np.float32)
    assert np.allclose(XF.POST["minus1"](w + 1.0), w, atol=1e-6)


def test_ssm_norm_not_shifted():
    assert XF.plan("blk.0.ssm_norm.weight", "qwen4exp", KV) is None


def test_a_log_and_dt_reorder_inverse():
    nk, nv = 4, 12
    x = XF.plan("blk.0.ssm_a", "qwen4exp", KV)
    hf = np.random.default_rng(1).normal(size=nv).astype(np.float32)
    # converter: A_log -> -exp(A_log), then the grouped -> tiled head reorder (conversion/qwen.py _reorder_v_heads)
    tiled = (-np.exp(hf)).reshape(nk, nv // nk).T.reshape(-1)
    back = XF.POST[x.post](tiled[x.rows])
    assert np.allclose(back, hf, atol=1e-5)
    assert XF.plan("blk.0.ssm_dt.bias", "qwen4exp", KV).rows is not None


def test_conv_and_qkv_rows():
    ds, nk, nv = 8, 4, 12
    x = XF.plan("blk.0.ssm_conv1d.weight", "qwen4exp", KV)
    C = 2 * ds * nk + 96
    assert len(x.rows) == C and sorted(x.rows) == list(range(C)) and (x.rows[:2 * ds * nk] == np.arange(2 * ds * nk)).all()
    out = XF.POST[x.post](np.zeros((C, 4), np.float32))
    assert out.shape == (C, 1, 4)
    assert XF.plan("blk.0.attn_qkv.weight", "qwen4exp", KV).rows.shape == (C,)
    assert XF.plan("blk.0.ssm_out.weight", "qwen4exp", KV).cols is not None            # out_proj columns: undone on the activation


def test_ple_conv_unsqueezed_and_shexp_gate_row():
    assert XF.POST[XF.plan("blk.1.ple_conv1d.weight", "qwen4exp", KV).post](np.zeros((12, 4), np.float32)).shape == (12, 1, 4)    # qwen4exp.py:168 squeeze undone
    assert XF.POST[XF.plan("blk.4.ffn_gate_inp_shexp.weight", "qwen4exp", KV).post](np.zeros(8, np.float32)).shape == (1, 8)


def test_indexer_q_k_concat_back():
    """qwen4exp.py:156-159: q rows [:n_q], k rows [n_q:]; the class holds `index_qk_proj` = q then k."""
    A = FN.ARCHS["qwen4exp"]
    q = FP.Pack("dense", 6, 4, np.arange(24, dtype=np.float16).reshape(6, 4))
    k = FP.Pack("dense", 2, 4, 100 + np.arange(8, dtype=np.float16).reshape(2, 4))
    src = FL._Src()
    p = LM + "layers.3.self_attn.indexer."
    src.packs = {p + "q_proj": q, p + "k_proj": k, LM + "layers.3.self_attn.o_proj": q}
    FL._apply_concat(src, A)
    got = FL._build(src.packs[p + "index_qk_proj"])
    assert (got.n, got.k) == (8, 4) and p + "q_proj" not in src.packs and p + "k_proj" not in src.packs
    assert np.array_equal(got.qs[:6], q.qs) and np.array_equal(got.qs[6:], k.qs)            # q rows first, then k


def test_ple_table_is_unserved(tmp_path):
    kv = {"general.architecture": "qwen4exp", "qwen4exp.block_count": 1}
    path = str(tmp_path / "m.gguf")
    SY.write_gguf(path, kv, [("per_layer_token_embd.weight", (32, 4), "F32", np.zeros(128, np.float32).view(np.uint8))])
    with pytest.raises(GgufError, match="no loader path"):
        FL._read_gguf(path, "qwen4exp")


def _kv():
    ratios = [0, 0, 0, 4] * 2
    k = {"block_count": 8, "context_length": 1024, "embedding_length": 64, "attention.head_count": 4, "attention.head_count_kv": 2, "attention.key_length": 16,
         "attention.value_length": 16, "attention.layer_norm_rms_epsilon": 1e-6, "rope.dimension_count": 4, "rope.dimension_sections": [1, 1, 0, 0], "rope.freq_base": 1e7,
         "expert_count": 8, "expert_used_count": 2, "expert_feed_forward_length": 32, "expert_shared_feed_forward_length": 32, "ssm.conv_kernel": 4, "ssm.state_size": 8,
         "ssm.group_count": 2, "ssm.time_step_rank": 6, "ssm.inner_size": 48, "full_attention_interval": 4, "hyper_connection.count": 4, "hyper_connection.low_rank": 8,
         "attention.indexer.head_count": 2, "attention.indexer.key_length": 8, "attention.indexer.top_k": 16, "attention.compress_ratios": ratios, "ple.layers": [1],
         "ple.ngram_size": 3, "ple.heads_per_ngram": 4, "ple.conv_kernel": 4, "ple.eos_token_id": 5, "embedding_length_per_layer_input": 8}
    return {"general.architecture": "qwen4exp", **{f"qwen4exp.{a}": b for a, b in k.items()}}


class _G:
    def __init__(self, kv, out=True):
        class T:
            ne = [64, 100]
        self.kv, self.tensors = kv, {"token_embd.weight": T(), **({"output.weight": T()} if out else {})}


def test_config_from_metadata():
    c = Q.config(_G(_kv()), "qwen4exp")
    tc = c["text_config"]
    assert c["model_type"] == "qwen4_exp" and tc["num_hidden_layers"] == 8 and tc["ple_layer_ids"] == [2] and tc["indexer_compress_ratio"] == 4
    assert tc["linear_value_head_dim"] == 8 and tc["vocab_size"] == 100 and tc["tie_word_embeddings"] is False and tc["rope_parameters"]["partial_rotary_factor"] == 0.25
    assert Q.config(_G(_kv(), out=False), "qwen4exp")["tie_word_embeddings"] is True


@pytest.mark.parametrize("over", [{"rope.scaling.type": "yarn"}, {"expert_gating_func": 2}, {"expert_weights_scale": 2.5}, {"attention.value_length": 8},
                                  {"attention.compress_ratios": [0, 4, 0, 0] * 2}, {"ssm.inner_size": 40}, {"embedding_length_per_layer_input": 7}])
def test_config_refusals(over):
    kv = _kv(); kv.update({f"qwen4exp.{a}": b for a, b in over.items()})
    with pytest.raises(GgufError):
        Q.config(_G(kv), "qwen4exp")


def test_class_builds_from_the_config_row():
    """the mlx-vlm class accepts the row's config (CPU, parameters stay lazy) and has the names the rows map to"""
    mod = pytest.importorskip("mlx_vlm.models.qwen4_exp")
    import mlx.core as mx
    from mlx.utils import tree_flatten
    mx.set_default_device(mx.cpu)
    c = Q.config(_G(_kv()), "qwen4exp")
    m = mod.Model(mod.ModelConfig.from_dict(c))
    names = {k for k, _ in tree_flatten(m.language_model.parameters())}
    assert "model.layers.3.self_attn.indexer.index_qk_proj.weight" in names and "model.layers.0.linear_attn.in_proj_qkv.weight" in names
    assert "model.layers.1.ple.key_proj.weight" in names and "model.hyper_connection_mixer.hc_norm.weight" in names and "model.layers.5.mlp.switch_mlp.down_proj.weight" in names
    assert sum(1 for n in names if ".ngram_embedding.shards." in n) == 128
