"""nemotron_h_moe (Nemotron-3.5-Lightning) through the architecture kit with the mlx-lm nemotron_h class (CPU only; the module pins the MLX CPU device): config row, the converter
rules of conversion/nemotron.py NemotronHModel (A_log, D, conv, gated norm), the tiny round trip class -> GGUF -> kit loader -> class logits (Mamba-2 blocks run as plain MLX ops on
the CPU device), guards. Real-header check: KYOJIN_TEST_NEMOTRONHMOE_HDR, skipped when unset."""
from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402
from mlx_lm.models import nemotron_h as NH  # noqa: E402

from kyojin_mlx import foreign_archkit as K, foreign_archkit_cpu as KC, foreign_cfgrows as CR, foreign_gguf as GF, foreign_synth as SY, foreign_xform as XF  # noqa: E402

PAT = ["M", "E", "*", "M", "E"]
CFG = dict(model_type="nemotron_h", vocab_size=50, hidden_size=32, intermediate_size=24, max_position_embeddings=128, num_attention_heads=4, num_key_value_heads=2, attention_bias=False,
           mamba_num_heads=4, mamba_head_dim=8, mamba_proj_bias=False, ssm_state_size=8, conv_kernel=4, n_groups=2, mlp_bias=False, layer_norm_epsilon=1e-5, use_bias=False, use_conv_bias=True,
           hybrid_override_pattern=PAT, head_dim=8, moe_intermediate_size=24, moe_shared_expert_intermediate_size=40, n_routed_experts=4, n_shared_experts=1, n_group=1, topk_group=1,
           num_experts_per_tok=2, norm_topk_prob=True, routed_scaling_factor=2.5)


@pytest.fixture(autouse=True)
def _unproven_rows_on(monkeypatch):
    monkeypatch.setenv("KYOJIN_KIT_UNPROVEN_ROWS", "1")          # no Metal proof yet: the rows exist only behind the one switch
    yield


def _model(seed=0):
    m = NH.Model(NH.ModelArgs.from_dict(CFG))
    rng = np.random.default_rng(seed)
    vals = []
    for k, v in tree_flatten(m.parameters()):
        a = rng.standard_normal(v.shape) * (0.2 if v.ndim > 1 else 0.1)
        if k.endswith("A_log"):
            a = np.log(rng.uniform(1.0, 4.0, v.shape))
        elif k.endswith(".norm.weight") or k.endswith("norm_f.weight"):
            a = a + 1.0
        vals.append((k, mx.array(a.astype(np.float32))))
    m.update(tree_unflatten(vals))
    mx.eval(m.parameters())
    return m


_TOP = {"backbone.embeddings.weight": "token_embd.weight", "backbone.norm_f.weight": "output_norm.weight", "lm_head.weight": "output.weight"}
_BLK = {"norm": "attn_norm", "mixer.in_proj": "ssm_in", "mixer.out_proj": "ssm_out", "mixer.conv1d": "ssm_conv1d", "mixer.q_proj": "attn_q", "mixer.k_proj": "attn_k", "mixer.v_proj": "attn_v",
        "mixer.o_proj": "attn_output", "mixer.gate": "ffn_gate_inp", "mixer.switch_mlp.fc1": "ffn_up_exps", "mixer.switch_mlp.fc2": "ffn_down_exps",
        "mixer.shared_experts.up_proj": "ffn_up_shexp", "mixer.shared_experts.down_proj": "ffn_down_shexp"}


def to_gguf_tensors(model):
    """What conversion/nemotron.py NemotronHModel writes for these parameters: [(gguf name, numpy array)]."""
    out = []
    for k, v in tree_flatten(model.parameters()):
        v = np.array(v)
        if k in _TOP:
            out.append((_TOP[k], v))
            continue
        i, rest = k[len("backbone.layers."):].split(".", 1)
        if rest == "mixer.A_log":
            out.append((f"blk.{i}.ssm_a", (-np.exp(v)).reshape(-1, 1)))                 # -exp(A_log), reshape(-1, 1)
        elif rest == "mixer.D":
            out.append((f"blk.{i}.ssm_d", v.reshape(-1, 1)))
        elif rest == "mixer.dt_bias":
            out.append((f"blk.{i}.ssm_dt.bias", v))
        elif rest == "mixer.gate.e_score_correction_bias":
            out.append((f"blk.{i}.exp_probs_b.bias", v))
        elif rest == "mixer.norm.weight":
            out.append((f"blk.{i}.ssm_norm.weight", v.reshape(CFG["n_groups"], -1)))     # reshape(n_group, -1)
        elif rest == "mixer.conv1d.weight":
            out.append((f"blk.{i}.ssm_conv1d.weight", np.ascontiguousarray(v[..., 0])))   # class [C, K, 1] <- HF [C, 1, K] squeezed to [C, K]
        else:
            out.append((f"blk.{i}." + _BLK[rest.rsplit(".", 1)[0]] + "." + rest.rsplit(".", 1)[1], v))
    return out


def kv_of(**over):
    hk = [0, 0, 2, 0, 0]
    ff = [0, 24, 0, 0, 24]
    kv = {"general.architecture": "nemotron_h_moe", "nemotron_h_moe.block_count": 5, "nemotron_h_moe.context_length": 128, "nemotron_h_moe.embedding_length": 32,
          "nemotron_h_moe.feed_forward_length": ff, "nemotron_h_moe.attention.head_count": 4, "nemotron_h_moe.attention.head_count_kv": hk, "nemotron_h_moe.rope.freq_base": 10000.0,
          "nemotron_h_moe.attention.layer_norm_rms_epsilon": 1e-5, "nemotron_h_moe.attention.layer_norm_epsilon": 1e-5, "nemotron_h_moe.expert_count": 4, "nemotron_h_moe.expert_used_count": 2,
          "nemotron_h_moe.expert_group_count": 1, "nemotron_h_moe.expert_group_used_count": 1, "nemotron_h_moe.vocab_size": 50, "nemotron_h_moe.ssm.conv_kernel": 4,
          "nemotron_h_moe.ssm.state_size": 8, "nemotron_h_moe.ssm.group_count": 2, "nemotron_h_moe.ssm.inner_size": 32, "nemotron_h_moe.ssm.time_step_rank": 4,
          "nemotron_h_moe.attention.key_length": 8, "nemotron_h_moe.attention.value_length": 8, "nemotron_h_moe.expert_feed_forward_length": 24,
          "nemotron_h_moe.expert_shared_feed_forward_length": 40, "nemotron_h_moe.expert_shared_count": 1, "nemotron_h_moe.expert_weights_norm": True, "nemotron_h_moe.expert_weights_scale": 2.5}
    kv.update(over)
    return kv


def write(path, model, kv=None):
    SY.write_gguf(path, kv or kv_of(), [(n, tuple(reversed(a.shape)), "F32", np.ascontiguousarray(a, np.float32)) for n, a in to_gguf_tensors(model)])


TOK = np.array([[3, 17, 5, 29, 8, 41, 2, 11, 30, 7]], np.int32)


def test_round_trip_logits(tmp_path):
    m = _model()
    p = str(tmp_path / "n.gguf")
    write(p, m)
    ref = np.array(m(mx.array(TOK)))
    got, rep = KC.load_cpu(p)
    assert rep["model_type"] == "nemotron_h" and mx.default_device() == mx.cpu and rep["config"]["hybrid_override_pattern"] == PAT
    assert np.abs(np.array(got(mx.array(TOK))) - ref).max() < 1e-4


def test_converter_rules_invert():
    rng = np.random.default_rng(3)
    a_log = np.log(rng.uniform(1, 4, 6)).astype(np.float32)
    stored = (-np.exp(a_log)).reshape(-1, 1)
    x = XF.plan("blk.2.ssm_a", "nemotron_h_moe", {})
    assert np.allclose(XF.POST[x.post](stored), a_log, atol=1e-6) and XF.POST[x.post](stored).shape == (6,)
    with pytest.raises(GF.GgufError):
        XF.POST[x.post](np.ones((6, 1), np.float32))
    assert XF.POST[XF.plan("blk.0.ssm_d", "nemotron_h_moe", {}).post](np.ones((6, 1))).shape == (6,)
    assert XF.POST[XF.plan("blk.0.ssm_norm.weight", "nemotron_h_moe", {}).post](np.ones((2, 8))).shape == (16,)
    assert XF.POST[XF.plan("blk.0.ssm_conv1d.weight", "nemotron_h_moe", {}).post](np.ones((5, 4))).shape == (5, 1, 4)
    assert XF.plan("blk.0.ssm_in.weight", "nemotron_h_moe", {}) is None and XF.plan("blk.0.attn_norm.weight", "nemotron_h_moe", {}) is None


@pytest.mark.parametrize("over,why", [({"nemotron_h_moe.moe_latent_size": 8}, "latent"), ({"nemotron_h_moe.expert_gating_func": 1}, "sigmoid"),
                                      ({"nemotron_h_moe.feed_forward_length": [0, 24]}, "one value per block")])
def test_guards_refuse(tmp_path, over, why):
    p = str(tmp_path / "n.gguf")
    write(p, _model(), kv_of(**over))
    with pytest.raises(GF.GgufError, match=why):
        KC.load_cpu(p)


def test_switch_off_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("KYOJIN_KIT_UNPROVEN_ROWS", "0")
    p = str(tmp_path / "n.gguf")
    write(p, _model())
    with pytest.raises((K.KitError, GF.GgufError)):
        KC.load_cpu(p)


def test_moe_variant_and_path():
    """128 routed experts top 6 + 1 shared, sigmoid router with a selection bias, renormalised, scaled 2.5, non-gated relu^2 experts: not the fused softmax path (route_why), and
    the generic description puts it on the plain path (non-gated experts have no pair kernel)."""
    from kyojin_mlx import foreign_moe as FM, foreign_moe_generic as MG
    blk = next(l.mixer for l in _model().layers if l.block_type == "E")
    assert FM.route_why(blk) is not None
    spec = MG.MoeSpec(n_expert=4, top_k=2, gating="sigmoid", select_bias=True, norm_weights=True, weight_scale=2.5, act="relu2", gated=False, shared=True, n_dense_leading=0)
    d = MG.describe(spec)
    assert d["kind"] == "stock" and any("non-gated" in w for w in d["why"])


def test_real_header():
    f = os.environ.get("KYOJIN_TEST_NEMOTRONHMOE_HDR")
    if not f:
        pytest.skip("KYOJIN_TEST_NEMOTRONHMOE_HDR not set")
    g = GF.GgufFile(GF.find_gguf(f))
    cfg = CR.build("nemotron_h_moe", g)
    Model, Args, _ = KC._classes("nemotron_h_moe")
    names = [k for k, _ in tree_flatten(Model(Args.from_dict(cfg)).parameters())]
    nm = K.resolve("nemotron_h_moe", names, n_trunk=CR.n_trunk("nemotron_h_moe", g.kv))
    assert len({nm.hf(t) for t in g.tensors if nm.hf(t)}) == len(names)
