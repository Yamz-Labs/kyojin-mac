"""lfm2moe through the architecture kit (CPU only; the module pins the MLX CPU device): config row, the converter rules of conversion/lfm2.py LFM2MoeModel
(conv weight squeezed, experts stacked, `.expert_bias` renamed), and the tiny round trip class -> GGUF -> kit loader -> class logits. The real-header check reads
KYOJIN_TEST_LFM2MOE_HDR (an lfm2moe GGUF or a header-only copy of it) and is skipped when unset."""
from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

@pytest.fixture(autouse=True)
def _unproven_rows_on(monkeypatch):
    monkeypatch.setenv("KYOJIN_KIT_UNPROVEN_ROWS", "1")       # these rows have no Metal proof yet: in force only behind the one switch (tests/test_arch_kit_unproven.py covers off)
    pass
    yield

from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402
from mlx_lm.models import lfm2_moe as L  # noqa: E402

from kyojin_mlx import foreign_archkit as K, foreign_archkit_cpu as KC, foreign_cfgrows as CR, foreign_gguf as GF, foreign_synth as SY, foreign_xform as XF  # noqa: E402

HK = [0, 0, 2, 0, 2]
CFG = dict(model_type="lfm2_moe", vocab_size=50, hidden_size=32, intermediate_size=48, moe_intermediate_size=16, num_hidden_layers=5, num_experts=4, num_experts_per_tok=2,
           norm_topk_prob=True, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128, use_expert_bias=True, num_dense_layers=1, norm_eps=1e-5,
           conv_bias=False, conv_L_cache=3, rope_theta=1e6, full_attn_idxs=[i for i, h in enumerate(HK) if h])


def _model(seed=0):
    m = L.Model(L.ModelArgs.from_dict(dict(CFG, layer_types=["full_attention" if h else "conv" for h in HK])))
    rng = np.random.default_rng(seed)
    m.update(tree_unflatten([(k, mx.array((rng.standard_normal(v.shape) * (0.3 if v.ndim > 1 else 0.2) + (1.0 if v.ndim == 1 else 0.0)).astype(np.float32))) for k, v in tree_flatten(m.parameters())]))
    mx.eval(m.parameters())
    return m


_BLK = {"operator_norm": "attn_norm", "ffn_norm": "ffn_norm", "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v", "self_attn.out_proj": "attn_output",
        "self_attn.q_layernorm": "attn_q_norm", "self_attn.k_layernorm": "attn_k_norm", "conv.in_proj": "shortconv.in_proj", "conv.out_proj": "shortconv.out_proj",
        "feed_forward.gate_proj": "ffn_gate", "feed_forward.up_proj": "ffn_up", "feed_forward.down_proj": "ffn_down", "feed_forward.gate": "ffn_gate_inp",
        "feed_forward.switch_mlp.gate_proj": "ffn_gate_exps", "feed_forward.switch_mlp.up_proj": "ffn_up_exps", "feed_forward.switch_mlp.down_proj": "ffn_down_exps"}


def to_gguf_tensors(model):
    """What conversion/lfm2.py LFM2MoeModel writes for these parameters (the HF checkpoint is the class's pre-sanitize layout): [(gguf name, numpy array)]."""
    out = []
    for k, v in tree_flatten(model.parameters()):
        v = np.array(v)
        if k == "model.embed_tokens.weight":
            out.append(("token_embd.weight", v))
        elif k == "model.embedding_norm.weight":
            out.append(("token_embd_norm.weight", v))
        else:
            i, rest = k[len("model.layers."):].split(".", 1)
            if rest == "feed_forward.expert_bias":
                out.append((f"blk.{i}.exp_probs_b.bias", v))                       # lfm2.py:314-315 rename
            elif rest == "conv.conv.weight":
                out.append((f"blk.{i}.shortconv.conv.weight", np.ascontiguousarray(v.transpose(0, 2, 1)).squeeze(1)))     # class [C, K, 1] <- HF [C, 1, K]; lfm2.py:321-322 squeeze(1)
            else:
                out.append((f"blk.{i}." + _BLK[rest.rsplit(".", 1)[0]] + "." + rest.rsplit(".", 1)[1], v))       # experts: lfm2.py:345 torch.stack(dim=0) = the class's stacked switch_mlp
    return out


def kv_of(hk=HK, **over):
    kv = {"general.architecture": "lfm2moe", "lfm2moe.block_count": 5, "lfm2moe.context_length": 128, "lfm2moe.embedding_length": 32, "lfm2moe.feed_forward_length": 48,
          "lfm2moe.attention.head_count": 4, "lfm2moe.attention.head_count_kv": list(hk), "lfm2moe.rope.freq_base": 1e6, "lfm2moe.attention.layer_norm_rms_epsilon": 1e-5,
          "lfm2moe.expert_count": 4, "lfm2moe.expert_used_count": 2, "lfm2moe.expert_feed_forward_length": 16, "lfm2moe.leading_dense_block_count": 1, "lfm2moe.expert_gating_func": 2,
          "lfm2moe.vocab_size": 50, "lfm2moe.shortconv.l_cache": 3}
    kv.update(over)
    return kv


def write(path, model, kv=None):
    SY.write_gguf(path, kv or kv_of(), [(n, tuple(reversed(a.shape)), "F32", np.ascontiguousarray(a, np.float32)) for n, a in to_gguf_tensors(model)])


TOK = np.array([[3, 17, 5, 29, 8, 41, 2, 11, 30, 7]], np.int32)


def test_round_trip_logits(tmp_path):
    m = _model()
    p = str(tmp_path / "l.gguf")
    write(p, m)
    ref = np.array(m(mx.array(TOK)))
    got, rep = KC.load_cpu(p)
    assert rep["model_type"] == "lfm2_moe" and mx.default_device() == mx.cpu
    assert np.abs(np.array(got(mx.array(TOK))) - ref).max() < 1e-4
    assert rep["config"]["full_attn_idxs"] == [2, 4] and rep["config"]["num_key_value_heads"] == 2


def test_conv_rule_inverts_the_squeeze():
    hf = np.random.default_rng(1).standard_normal((8, 1, 3)).astype(np.float32)        # HF depthwise Conv1d weight [C, 1, K]
    stored = hf.squeeze(1)                                                              # conversion/lfm2.py:322
    x = XF.plan("blk.2.shortconv.conv.weight", "lfm2moe", {})
    assert np.array_equal(XF.POST[x.post](stored), hf)
    assert XF.plan("blk.2.shortconv.in_proj.weight", "lfm2moe", {}) is None and XF.plan("blk.2.attn_norm.weight", "lfm2moe", {}) is None       # nothing else is touched (no norm shift)


def test_expert_stack_and_bias_name():
    m = _model()
    w = {n: a for n, a in to_gguf_tensors(m)}
    per_expert = [np.random.default_rng(e).standard_normal((16, 32)).astype(np.float32) for e in range(4)]
    assert np.stack(per_expert, 0).shape == w["blk.1.ffn_gate_exps.weight"].shape                    # [expert, ff, dim], torch.stack(dim=0) at lfm2.py:345
    nm = K.resolve("lfm2moe", [k for k, _ in tree_flatten(m.parameters())])
    assert nm.hf("blk.1.exp_probs_b.bias") == "model.layers.1.feed_forward.expert_bias"
    assert nm.hf("token_embd_norm.weight") == "model.embedding_norm.weight" and nm.hf("blk.0.shortconv.conv.weight") == "model.layers.0.conv.conv.weight"


@pytest.mark.parametrize("over,hk,why", [({"lfm2moe.expert_gating_func": 1}, HK, "sigmoid"), ({"lfm2moe.expert_weights_scale": 2.5}, HK, "expert_weights_scale"),
                                         ({}, [0, 0, 2, 0, 4], "kv head count"), ({"lfm2moe.rope.scaling.type": "yarn"}, HK, "rope.scaling"),
                                         ({"lfm2moe.attention.sliding_window": 8}, HK, "sliding")])
def test_guards_refuse(tmp_path, over, hk, why):
    m = _model()
    p = str(tmp_path / "l.gguf")
    write(p, m, kv_of(hk, **over))
    with pytest.raises(GF.GgufError, match=why):
        KC.load_cpu(p)


def test_real_header():
    f = os.environ.get("KYOJIN_TEST_LFM2MOE_HDR")
    if not f:
        pytest.skip("KYOJIN_TEST_LFM2MOE_HDR not set")
    g = GF.GgufFile(GF.find_gguf(f))
    cfg = CR.build("lfm2moe", g)
    Model, Args, _ = KC._classes("lfm2moe")
    names = [k for k, _ in tree_flatten(Model(Args.from_dict(cfg)).parameters())]       # lazy arrays: nothing is evaluated
    nm = K.resolve("lfm2moe", names, n_trunk=CR.n_trunk("lfm2moe", g.kv))
    assert {nm.hf(t) for t in g.tensors} == set(names)


def test_moe_variant_description():
    from kyojin_mlx import foreign_moe_generic as MG
    spec = MG.MoeSpec(n_expert=64, top_k=4, gating="sigmoid", select_bias=True, norm_weights=True, act="silu", n_dense_leading=2)       # the variant of the file: no shared expert, no scale
    assert MG.describe(spec) == {"kind": "glm_route", "why": []}


@pytest.fixture(autouse=True)
def _unproven_rows_on(monkeypatch):
    monkeypatch.setenv("KYOJIN_KIT_UNPROVEN_ROWS", "1")       # these rows have no Metal proof yet: in force only behind the one switch (tests/test_arch_kit_unproven.py covers off)
