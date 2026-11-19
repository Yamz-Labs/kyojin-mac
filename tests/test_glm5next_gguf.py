# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash GGUF rows (foreign_glm_gguf): one test per converter rule, synthetic tensors, numpy only (CPU)."""
import numpy as np
import pytest

from kyojin_mlx import foreign_glm_gguf as GG
from kyojin_mlx import foreign_names as FN
from kyojin_mlx import foreign_xform as XF
from kyojin_mlx.foreign_gguf import GgufError

LM = "model.language_model."


@pytest.fixture(autouse=True)
def _unproven_rows_on(monkeypatch):
    monkeypatch.setenv("KYOJIN_KIT_UNPROVEN_ROWS", "1")       # the rows have no Metal proof yet: registered only behind the one switch (tests/test_arch_kit_unproven.py covers off)
    GG.register()
    yield
    monkeypatch.undo()
    GG.register()


def test_registered_without_the_unproven_switch(monkeypatch):
    from kyojin_mlx import foreign_unproven as UP
    monkeypatch.delenv("KYOJIN_KIT_UNPROVEN_ROWS", raising=False)
    GG.register()
    assert "glm5-next" not in UP.UNPROVEN_ARCHS and not UP.has_unproven("glm5-next")
    assert FN.ARCHS["glm5-next"].model_type == "glm5_next" and "glm5next" in FN._CONFIG and "glm5next" in XF._SETS


def test_registered():
    assert FN.ARCHS["glm5-next"].model_type == "glm5_next" and "glm5next" in FN._CONFIG and "glm5next" in XF._SETS


@pytest.mark.parametrize("gguf,hf", [
    ("token_embd.weight", LM + "embed_tokens.weight"), ("output_norm.weight", LM + "norm.weight"), ("output.weight", "lm_head.weight"),
    ("blk.3.attn_q_a.weight", LM + "layers.3.self_attn.q_a_proj.weight"),
    ("blk.3.attn_k_b.weight", LM + "layers.3.self_attn.embed_q.weight"),          # glm.py:543-552: k_b is the transposed nope rows = embed_q
    ("blk.3.attn_v_b.weight", LM + "layers.3.self_attn.unembed_out.weight"),
    ("blk.0.ssm_a", LM + "layers.0.self_attn.A_log"), ("blk.0.ssm_dt.bias", LM + "layers.0.self_attn.dt_bias"),   # glm.py:563-568
    ("blk.0.ssm_conv1d_q.weight", LM + "layers.0.self_attn.q_conv1d.weight"),
    ("blk.0.hc_attn_fn.weight", LM + "layers.0.hc_attn_fn"), ("blk.7.hc_ffn_scale.weight", LM + "layers.7.hc_ffn_scale"),   # glm.py:570-572
    ("blk.3.indexer_compressor_ape.weight", LM + "layers.3.self_attn.indexer.index_kpool_compress_ape"),
    ("blk.3.indexer.k_norm.bias", LM + "layers.3.self_attn.indexer.k_norm.bias"),
    ("blk.4.ffn_gate_exps.weight", LM + "layers.4.mlp.experts.gate_proj.weight"),   # stacked experts stay one tensor (glm.py:525-540)
    ("blk.4.exp_probs_b.bias", LM + "layers.4.mlp.gate.e_score_correction_bias"),
    ("blk.4.ffn_down_shexp.weight", LM + "layers.4.mlp.shared_experts.down_proj.weight"),
])
def test_names(gguf, hf):
    assert FN.gguf_name(gguf, "glm5-next", 45) == hf


def test_nextn_block_skipped():
    for t in GG.NEXTN + ("attn_q_a.weight", "ffn_gate_exps.weight"):
        assert FN.gguf_name("blk.45." + t, "glm5-next", 45) is None
    with pytest.raises(GgufError):
        FN.gguf_name("blk.3.not_a_tensor.weight", "glm5-next", 45)


def test_a_log_rule_inverse():
    a_log = np.random.RandomState(0).randn(64).astype(np.float32)
    x = XF.plan("blk.0.ssm_a", "glm5next", {})                                      # converter: -exp(A_log) (glm.py:563-565)
    assert x.post == "a_log" and x.rows is None
    np.testing.assert_allclose(XF.POST[x.post](-np.exp(a_log)), a_log, atol=1e-6)
    for n in ("blk.0.ssm_dt.bias", "blk.0.ssm_conv1d_k.weight", "blk.3.attn_k_b.weight", "blk.5.hc_attn_fn.weight", "blk.4.ffn_up_exps.weight"):
        assert XF.plan(n, "glm5next", {}) is None


def test_conv_reshape_is_same_bytes():
    hf = np.random.RandomState(1).randn(8, 1, 4).astype(np.float32)             # HF [C, 1, K]
    assert np.array_equal(hf.reshape(1, 8, 1, 4).reshape(8, 1, 4), hf)           # glm.py:554-561


def test_kv_b_split_equals_sanitize():
    H, nope, v, r = 3, 4, 5, 6
    w = np.random.RandomState(2).randn(H * (nope + v), r).astype(np.float32).reshape(H, nope + v, r)
    k_b, v_b = w[:, :nope].transpose(0, 2, 1), w[:, nope:]                       # glm.py:548-551 (converter)
    embed_q, unembed_out = np.ascontiguousarray(w[:, :nope].swapaxes(-1, -2)), w[:, nope:]   # mlx-vlm glm5_next language.py kv_b_proj branch (class)
    assert np.array_equal(k_b, embed_q) and np.array_equal(v_b, unembed_out) and k_b.shape == (H, r, nope)


def test_expert_stack_layout_and_dense_kinds():
    per = [np.full((3, 2), e, np.float32) for e in range(4)]
    st = np.stack(per)                                                           # glm.py:539: [E, I, H]; GGUF ne = [H, I, E]
    assert st.shape == (4, 3, 2) and all(np.array_equal(st[e], per[e]) for e in range(4))
    assert FN.is_linear(LM + "layers.4.mlp.experts.gate_proj.weight", 3, "glm5-next")
    for dense in ("self_attn.q_conv1d.weight", "self_attn.embed_q.weight", "self_attn.unembed_out.weight"):
        assert not FN.is_linear(LM + "layers.3." + dense, 3, "glm5-next")


class _G:
    def __init__(self, kv):
        self.kv = {f"glm5-next.{k}": v for k, v in kv.items()}
        self.tensors = {"token_embd.weight": type("T", (), {"ne": (64, 100)})()}


def _kv(**o):
    kv = {"block_count": 8, "embedding_length": 64, "feed_forward_length": 128, "expert_feed_forward_length": 32, "expert_count": 8, "expert_used_count": 2,
          "expert_shared_count": 1, "expert_weights_scale": 2.5, "expert_weights_norm": True, "expert_gating_func": 2, "expert_group_count": 1,
          "nextn_predict_layers": 1, "context_length": 1024, "attention.head_count": 4, "attention.head_count_kv": [0, 0, 0, 1, 0, 0, 0, 1],
          "attention.q_lora_rank": 16, "attention.kv_lora_rank": 12, "rope.dimension_count": 0, "attention.key_length_mla": 8, "attention.value_length_mla": 8,
          "attention.layer_norm_rms_epsilon": 1e-5, "ssm.conv_kernel": 4, "kda.head_dim": 8, "kda.gate_lower_bound": -5.0, "attention.indexer.head_count": 2,
          "attention.indexer.key_length": 8, "attention.indexer.top_k": 16, "attention.indexer.kpool": 4, "hyper_connection.count": 4, "hyper_connection.epsilon": 1e-6,
          "hyper_connection.sinkhorn_iterations": 20, "leading_dense_block_count": 1, "swiglu_clamp_exp": [10.0] * 8, "vocab_size": 100}
    kv.update(o)
    return _G(kv)


def test_config_from_metadata():
    c = GG.config(_kv(), "glm5-next")["text_config"]
    assert c["num_hidden_layers"] == 7 and c["layer_types"][3] == "deepseek_sparse_attention" and c["layer_types"][0] == "linear_attention"
    assert c["linear_attn_config"]["kda_layers"] == [0, 1, 2, 4, 5, 6] and c["linear_attn_config"]["full_attn_layers"] == [3]
    assert (c["qk_nope_head_dim"], c["v_head_dim"], c["qk_rope_head_dim"], c["swiglu_limit"], c["index_kpool"], c["first_k_dense_replace"]) == (8, 8, 0, 10.0, 4, 1)
    assert c["indexer_types"] == ["full"] * 7 and c["scoring_func"] == "sigmoid"


@pytest.mark.parametrize("over", [{"expert_gating_func": 1}, {"expert_group_count": 2}, {"rope.dimension_count": 4},
                                  {"attention.head_count_kv": [0, 2, 0, 1, 0, 0, 0, 1]}, {"swiglu_clamp_exp": [10.0, 9.0] + [10.0] * 6}])
def test_config_refusals(over):
    with pytest.raises(GgufError):
        GG.config(_kv(**over), "glm5-next")
