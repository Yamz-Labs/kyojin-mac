"""cohere2moe (North-Mini-Code) through the architecture kit, with the text class of mlx-vlm (CPU only; the module pins the MLX CPU device): the class lookup switch
(KYOJIN_KIT_VLM_CLASSES), the config row, the tiny round trip class -> GGUF -> kit loader -> class logits, the guards, and the census proof that no architecture that resolves in
mlx-lm today changes class. The real-header check reads KYOJIN_TEST_COHERE2MOE_HDR (a cohere2moe GGUF or a header-only copy) and is skipped when unset."""
from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

from kyojin_mlx import foreign_arch_gen as G, foreign_archkit as K, foreign_archkit_cpu as KC, foreign_cfgrows as CR, foreign_gguf as GF, foreign_synth as SY, foreign_vlm_classes as VC  # noqa: E402

pytest.importorskip("mlx_vlm.models.cohere2_moe")
from mlx_vlm.models.cohere2_moe import Model as VModel, ModelConfig  # noqa: E402

PATTERN = [False, True, False, True]            # block 0 = dense lead (global + rope); block 2 is global WITHOUT rope: the class default pattern (4) would make blocks 0-2 sliding
CFG = dict(model_type="cohere2_moe", hidden_size=32, head_dim=8, num_hidden_layers=4, intermediate_size=16, prefix_dense_intermediate_size=48, num_attention_heads=4, num_key_value_heads=2,
           vocab_size=50, num_experts=4, num_experts_per_tok=2, moe_num_shared_experts=0, first_k_dense_replace=1, rms_norm_eps=1e-6, layer_norm_eps=1e-5, logit_scale=0.5, sliding_window=4,
           norm_topk_prob=False, moe_gate_act="sigmoid", rope_theta=50000.0,
           layer_types=["sliding_attention" if p else "full_attention" for p in PATTERN])


@pytest.fixture(autouse=True)
def _switches(monkeypatch):
    monkeypatch.setenv("KYOJIN_KIT_UNPROVEN_ROWS", "1")        # the row and the vetted line carry proven=False
    monkeypatch.setenv("KYOJIN_KIT_VLM_CLASSES", "1")
    yield


def _model(seed=0):
    m = VModel(ModelConfig.from_dict(CFG))
    rng = np.random.default_rng(seed)
    m.update(tree_unflatten([(k, mx.array((rng.standard_normal(v.shape) * (0.3 if v.ndim > 1 else 0.2) + (1.0 if v.ndim == 1 else 0.0)).astype(np.float32))) for k, v in tree_flatten(m.parameters())]))
    mx.eval(m.parameters())
    return m


_BLK = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output", "input_layernorm": "attn_norm",
        "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down", "mlp.gate": "ffn_gate_inp",
        "mlp.switch_mlp.gate_proj": "ffn_gate_exps", "mlp.switch_mlp.up_proj": "ffn_up_exps", "mlp.switch_mlp.down_proj": "ffn_down_exps"}


def to_gguf_tensors(model):
    out = []
    for k, v in tree_flatten(model.parameters()):
        v, k = np.array(v), k[len("language_model."):]
        if k == "model.embed_tokens.weight":
            out.append(("token_embd.weight", v))
        elif k == "model.norm.weight":
            out.append(("output_norm.weight", v))
        else:
            i, rest = k[len("model.layers."):].split(".", 1)
            out.append((f"blk.{i}." + _BLK[rest.rsplit(".", 1)[0]] + "." + rest.rsplit(".", 1)[1], v))          # experts: command_r.py torch.stack(dim=0) = the class's stacked switch_mlp
    return out


def kv_of(**over):
    kv = {"general.architecture": "cohere2moe", "cohere2moe.block_count": 4, "cohere2moe.context_length": 128, "cohere2moe.embedding_length": 32, "cohere2moe.feed_forward_length": 48,
          "cohere2moe.attention.head_count": 4, "cohere2moe.attention.head_count_kv": 2, "cohere2moe.rope.freq_base": 50000.0, "cohere2moe.attention.layer_norm_rms_epsilon": 1e-6,
          "cohere2moe.attention.layer_norm_epsilon": 1e-5, "cohere2moe.expert_count": 4, "cohere2moe.expert_used_count": 2, "cohere2moe.attention.key_length": 8,
          "cohere2moe.attention.value_length": 8, "cohere2moe.logit_scale": 0.5, "cohere2moe.attention.sliding_window": 4, "cohere2moe.attention.sliding_window_pattern": list(PATTERN),
          "cohere2moe.vocab_size": 50, "cohere2moe.expert_feed_forward_length": 16, "cohere2moe.leading_dense_block_count": 1, "cohere2moe.expert_weights_norm": False,
          "cohere2moe.expert_gating_func": 2, "cohere2moe.rope.dimension_count": 8, "cohere2moe.rope.scaling.type": "none"}
    kv.update(over)
    return kv


def write(path, model, kv=None):
    SY.write_gguf(path, kv or kv_of(), [(n, tuple(reversed(a.shape)), "F32", np.ascontiguousarray(a, np.float32)) for n, a in to_gguf_tensors(model)])


TOK = np.array([[3, 17, 5, 29, 8, 41, 2, 11, 30, 7]], np.int32)


def test_round_trip_logits(tmp_path):
    m = _model()
    p = str(tmp_path / "c.gguf")
    write(p, m)
    ref = np.array(m(mx.array(TOK)).logits)
    got, rep = KC.load_cpu(p)
    assert rep["model_type"] == "cohere2_moe" and type(got).__module__.startswith("mlx_vlm.") and mx.default_device() == mx.cpu
    assert rep["config"]["layer_types"] == CFG["layer_types"] and rep["config"]["moe_num_shared_experts"] == 0
    assert np.abs(np.array(got(mx.array(TOK)).logits) - ref).max() < 1e-4
    default = dict(CFG, layer_types=None, sliding_window_pattern=4)                                   # what the class does without the file's pattern: another model
    other = VModel(ModelConfig.from_dict(default))
    other.update(m.parameters())
    assert np.abs(np.array(other(mx.array(TOK)).logits) - ref).max() > 1e-3


def test_switch_off_is_the_old_error(tmp_path, monkeypatch):
    monkeypatch.setenv("KYOJIN_KIT_VLM_CLASSES", "0")
    p = str(tmp_path / "c.gguf")
    write(p, _model())
    with pytest.raises((K.KitError, GF.GgufError, ValueError)):
        KC.load_cpu(p)
    monkeypatch.setenv("KYOJIN_KIT_UNPROVEN_ROWS", "0")
    monkeypatch.setenv("KYOJIN_KIT_VLM_CLASSES", "1")
    with pytest.raises((K.KitError, GF.GgufError, ValueError)):
        KC.load_cpu(p)


@pytest.mark.parametrize("over,why", [({"cohere2moe.expert_gating_func": 1}, "sigmoid"), ({"cohere2moe.rope.scaling.type": "yarn"}, "rope.scaling"),
                                      ({"cohere2moe.expert_weights_scale": 2.5}, "expert_weights_scale"), ({"cohere2moe.attention.sliding_window_pattern": [False, True]}, "sliding_window_pattern"),
                                      ({"cohere2moe.rope.dimension_count": 4}, "partial rotary")])
def test_guards_refuse(tmp_path, over, why):
    p = str(tmp_path / "c.gguf")
    write(p, _model(), kv_of(**over))
    with pytest.raises(GF.GgufError, match=why):
        KC.load_cpu(p)


def test_vision_tower_is_dropped_only_without_vision_tensors():
    import mlx.nn as nn

    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = nn.Linear(2, 2)
            self.vision_tower = nn.Linear(2, 2)
    w = Wrap()
    assert VC.drop_vision(w, ["token_embd.weight"]) and not any(k.startswith("vision_tower") for k, _ in tree_flatten(w.parameters()))
    w = Wrap()
    assert not VC.drop_vision(w, ["token_embd.weight", "v.blk.0.attn_q.weight"]) and any(k.startswith("vision_tower") for k, _ in tree_flatten(w.parameters()))


def test_census_no_architecture_changes_class(monkeypatch):
    """Every GGUF architecture of the generated tables: the class mlx-lm gives with the switch off is the same with it on; only architectures that raise today may resolve in mlx-vlm."""
    off, on = {}, {}
    for arch in sorted(G.CONVERTER):
        for env, out in (("0", off), ("1", on)):
            monkeypatch.setenv("KYOJIN_KIT_VLM_CLASSES", env)
            try:
                c = KC._classes(arch)
                out[arch] = (c[0].__module__, c[2])
            except K.KitError:
                out[arch] = None
    assert len(off) > 100
    changed = [a for a in off if off[a] is not None and off[a] != on[a]]
    assert changed == []
    gained = sorted(a for a in off if off[a] is None and on[a] is not None)
    assert not {"bert", "clef", "modern-bert"} & set(gained)                                       # encoders have no `language` module
    assert "cohere2moe" in gained and "flint" in gained and all(on[a][0].startswith("mlx_vlm.") for a in gained)


def test_real_header():
    f = os.environ.get("KYOJIN_TEST_COHERE2MOE_HDR")
    if not f:
        pytest.skip("KYOJIN_TEST_COHERE2MOE_HDR not set")
    g = GF.GgufFile(GF.find_gguf(f))
    cfg = CR.build("cohere2moe", g)
    Model, Args = VC.classes_for_config(cfg)
    names = [k[len("language_model."):] for k, _ in tree_flatten(Model(Args.from_dict(cfg)).parameters())]
    nm = K.resolve("cohere2moe", names, n_trunk=CR.n_trunk("cohere2moe", g.kv))
    assert all(nm.hf(n) in names for n in g.tensors)


def test_moe_variant_and_path():
    """128 routed experts top 8, sigmoid router, no renormalisation, no selection bias, no shared expert: the class block is not a softmax router (route_why), so the fused softmax
    kernels do not take it; the sigmoid select path (glm_route) exists for nvfp4 experts only, so a K-quant file is served by the class's plain ops."""
    from kyojin_mlx import foreign_moe as FM, foreign_moe_generic as MG
    blk = _model().language_model.model.layers[1].mlp
    assert FM.route_why(blk) == "router_not_softmax"
    spec = MG.MoeSpec(n_expert=4, top_k=2, gating="sigmoid", norm_weights=False, act="silu", n_dense_leading=1)
    assert MG.describe(spec) == {"kind": "glm_route", "why": []}
