"""cggufarch2: qwen35moe, glm4moe, deepseek2, mistral3, mistral4, gemma4 (names, converter transforms undone, config from metadata) and the generic serving route.
Same method as test_gguf_arch.py: a tiny mlx-lm model, the converter replayed forward (separate transcription of conversion/*.py at foreign_names.LLAMA_CPP_COMMIT),
a synthetic GGUF, load_foreign, parameters equal + logits. CPU mlx only (no Metal)."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from kyojin_mlx import foreign_gguf as GF, foreign_names as N, foreign_synth as SY, foreign_xform as XF

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
import test_gguf_arch as T          # noqa: E402  (helpers: _convert_qwen35, _hf_qwen35, _kv35, _ref_model, C, ...)

C = T.C


# ------------------------------------------------------------------------------------------------------------------ qwen35moe
E_, K_, FE, FS = 4, 2, 32, 48      # experts, used, expert ff, shared ff


def _hf_moe(seed=0):
    hf = T._hf_qwen35(seed, mtp=False)
    r = np.random.default_rng(seed + 1)
    q = lambda *s: (r.integers(-24, 25, s) / 64).astype(np.float32)
    H = C["H"]
    for k in [k for k in hf if ".mlp." in k]:
        del hf[k]
    for i in range(C["L"]):
        p = f"model.layers.{i}.mlp."
        hf.update({p + "gate.weight": q(E_, H), p + "experts.gate_proj.weight": q(E_, FE, H), p + "experts.up_proj.weight": q(E_, FE, H),
                   p + "experts.down_proj.weight": q(E_, H, FE), p + "shared_expert.gate_proj.weight": q(FS, H), p + "shared_expert.up_proj.weight": q(FS, H),
                   p + "shared_expert.down_proj.weight": q(H, FS), p + "shared_expert_gate.weight": q(1, H)})
    return hf


_MOE_GG = {"mlp.gate.weight": "ffn_gate_inp.weight", "mlp.experts.gate_proj.weight": "ffn_gate_exps.weight", "mlp.experts.up_proj.weight": "ffn_up_exps.weight",
           "mlp.experts.down_proj.weight": "ffn_down_exps.weight", "mlp.shared_expert.gate_proj.weight": "ffn_gate_shexp.weight",
           "mlp.shared_expert.up_proj.weight": "ffn_up_shexp.weight", "mlp.shared_expert.down_proj.weight": "ffn_down_shexp.weight",
           "mlp.shared_expert_gate.weight": "ffn_gate_inp_shexp.weight"}


def _kv_moe(**over):
    kv = T._kv35(False, **{"qwen35.block_count": C["L"]})
    kv = {k.replace("qwen35.", "qwen35moe."): v for k, v in kv.items()}
    kv["general.architecture"] = "qwen35moe"
    kv.pop("qwen35moe.feed_forward_length", None)
    kv.update({"qwen35moe.expert_count": E_, "qwen35moe.expert_used_count": K_, "qwen35moe.expert_feed_forward_length": FE, "qwen35moe.expert_shared_feed_forward_length": FS})
    kv.update(over)
    return kv


def _fused(hf):
    """The HF checkpoint layout of the experts (what mlx-lm sanitize reads): gate_up_proj [E, 2 * ff, H] and down_proj, no separate gate / up."""
    out = {k: v for k, v in hf.items() if ".experts." not in k}
    for i in range(C["L"]):
        p = f"model.layers.{i}.mlp.experts."
        out[p + "gate_up_proj"] = np.concatenate([hf[p + "gate_proj.weight"], hf[p + "up_proj.weight"]], axis=1)
        out[p + "down_proj"] = hf[p + "down_proj.weight"]
    return out


def _write_moe(path, hf, kv=None):
    old = dict(T._BLK35)
    T._BLK35.update(_MOE_GG)
    try:
        flat = {k: v for k, v in hf.items() if ".experts." not in k}
        g = T._convert_qwen35(flat, dict(C), C["L"])
        for k, v in hf.items():
            if ".experts." in k:
                _, _, i, rest = k.split(".", 3)
                g[f"blk.{i}.{_MOE_GG[rest]}"] = v
    finally:
        T._BLK35.clear()
        T._BLK35.update(old)
    # shared_expert_gate [1, H] is stored as a vector (llama.cpp {n_embd})
    g = {n: (a.reshape(-1) if n.endswith("ffn_gate_inp_shexp.weight") else a) for n, a in g.items()}
    SY.write_gguf(path, kv or _kv_moe(), [(n, tuple(reversed(a.shape)), "F32", np.ascontiguousarray(a, np.float32)) for n, a in g.items()])


def test_qwen35moe_gguf_loads_to_the_same_parameters_and_logits(tmp_path):
    from mlx.utils import tree_flatten
    from kyojin_mlx import foreign_linear as FL
    hf = _hf_moe()
    p = tmp_path / "m.gguf"
    _write_moe(p, hf)
    d = N.gguf_serve_dir(str(p), str(tmp_path / "cache"))
    cfg = json.load(open(os.path.join(d, "config.json")))
    assert cfg["model_type"] == "qwen3_5_moe" and cfg["num_experts"] == E_ and cfg["moe_intermediate_size"] == FE and cfg["shared_expert_intermediate_size"] == FS
    ref, rp = T._ref_model(_fused(hf), cfg, tmp_path)
    model, _ = FL.load_foreign(d, log=lambda *a: None)
    got = dict(tree_flatten(model.parameters()))
    lin = {n: m for n, m in model.named_modules() if isinstance(m, FL.ForeignLinear)}
    sw = dict(model.named_modules())
    for k, v in rp.items():
        if k in got:
            if k.endswith("A_log"):
                np.testing.assert_allclose(np.array(got[k]), np.array(v), rtol=0, atol=2e-6, err_msg=k)
            else:
                np.testing.assert_array_equal(np.array(got[k].astype(mx.float32)), np.array(v.astype(mx.float32)), err_msg=k)
        elif ".switch_mlp." in k:
            blk, proj = k.split(".switch_mlp.")
            s = sw[blk + ".switch_mlp"]
            w = {"gate_proj.weight": s.g, "up_proj.weight": s.u, "down_proj.weight": s.d}[proj].w
            np.testing.assert_array_equal(np.array(w.astype(mx.float32)), np.array(v.astype(mx.float32)), err_msg=k)
        else:
            assert k.endswith(".weight") and k[:-7] in lin, k
            np.testing.assert_array_equal(np.array(lin[k[:-7]].dev.w.astype(mx.float32)), np.array(v.astype(mx.float32)), err_msg=k)
    ids = mx.array([[3, 7, 1, 30, 22, 5, 9, 2]])
    try:
        a = np.array(ref(ids).astype(mx.float32))
    except RuntimeError:
        ref.set_dtype(mx.float32)
        a = np.array(ref(ids))
    b = np.array(model(ids).astype(mx.float32))
    assert np.abs(a - b).max() <= 3e-2 * max(1.0, np.abs(a).max())


def test_qwen35moe_draft_block_is_skipped_and_fused_experts_refused(tmp_path):
    from kyojin_mlx import foreign_linear as FL
    hf = _hf_moe()
    p = tmp_path / "m.gguf"
    _write_moe(p, hf)
    g = GF.GgufFile(p)
    tl = [(n, ti.ne, "F32", np.ascontiguousarray(g.tensor_f32(n))) for n, ti in g.tensors.items()]
    extra = [(n.replace("blk.3.", "blk.4."), ne, t, a) for n, ne, t, a in tl if n.startswith("blk.3.")] + [("blk.4.nextn.eh_proj.weight", (128, 64), "F32", np.zeros((64, 128), np.float32))]
    SY.write_gguf(tmp_path / "n.gguf", dict(g.kv, **{"qwen35moe.block_count": C["L"] + 1, "qwen35moe.nextn_predict_layers": 1}), tl + extra)
    src = FL._read_gguf(str(tmp_path / "n.gguf"), "qwen35moe")
    assert not any(k.startswith("model.layers.4") for k in list(src.dense) + list(src.packs) + list(src.experts)) and any("draft-head block" in n for n in src.notes)
    c = N.gguf_config(GF.GgufFile(tmp_path / "n.gguf"))
    assert c["num_hidden_layers"] == C["L"] and c["mtp_num_hidden_layers"] == 0
    with pytest.raises(GF.GgufError):
        N.gguf_name("blk.0.ffn_gate_up_exps.weight", "qwen35moe", C["L"])
    with pytest.raises(GF.GgufError):
        SY.write_gguf(tmp_path / "s.gguf", _kv_moe(**{"qwen35moe.expert_weights_scale": 2.5}), [("token_embd.weight", (64, 48), "F32", np.zeros((48, 64), np.float32))])
        N.gguf_config(GF.GgufFile(tmp_path / "s.gguf"))


# ------------------------------------------------------------------------------------------------------------------ glm4moe
def _moe_gname(k):
    if k.endswith("mlp.gate.e_score_correction_bias"):         # conversion/base.py:763
        return f"blk.{k.split('.')[2]}.exp_probs_b.bias"
    return None


def test_glm4moe_gguf_round_trip_with_expert_bias_shared_expert_and_dense_lead(tmp_path):
    cfg = dict(hidden_size=64, num_hidden_layers=3, intermediate_size=96, num_attention_heads=4, num_key_value_heads=2, head_dim=16, attention_bias=True, use_qk_norm=True,
               partial_rotary_factor=0.5, max_position_embeddings=128, rope_theta=10000.0, rope_scaling=None, tie_word_embeddings=False, n_routed_experts=4,
               num_experts_per_tok=2, moe_intermediate_size=32, n_shared_experts=1, first_k_dense_replace=1, routed_scaling_factor=1.5, norm_topk_prob=True, n_group=1,
               topk_group=1, scoring_func="sigmoid", topk_method="noaux_tc")
    kv = {"glm4moe.rope.dimension_count": 8, "glm4moe.attention.key_length": 16, "glm4moe.expert_count": 4, "glm4moe.expert_used_count": 2, "glm4moe.expert_feed_forward_length": 32,
          "glm4moe.expert_shared_count": 1, "glm4moe.leading_dense_block_count": 1, "glm4moe.expert_weights_scale": 1.5, "glm4moe.expert_weights_norm": True,
          "glm4moe.expert_gating_func": 2, "glm4moe.attention.layer_norm_rms_epsilon": 1e-6}
    ref, got, d = T._tiny_dense("glm4moe", tmp_path, "glm4_moe", cfg, kv, alias={"mlp.switch_mlp": "mlp.experts"}, gname_of=_moe_gname)
    T._same_model(ref, got)
    c = json.load(open(os.path.join(d, "config.json")))
    assert c["model_type"] == "glm4_moe" and c["first_k_dense_replace"] == 1 and c["routed_scaling_factor"] == 1.5 and c["norm_topk_prob"] is True
    assert c["use_qk_norm"] is True and c["attention_bias"] is True and c["n_shared_experts"] == 1 and c["partial_rotary_factor"] == 0.5
    # the correction bias must reach the router: a zero bias gives other logits than the stored one
    bias = np.array(got.model.layers[1].mlp.gate.e_score_correction_bias)
    assert np.abs(bias).max() > 0
    np.testing.assert_array_equal(bias, np.array(ref.model.layers[1].mlp.gate.e_score_correction_bias.astype(mx.float32)))


def test_glm4moe_refuses_what_the_class_cannot_run(tmp_path):
    base = {"general.architecture": "glm4moe", "glm4moe.block_count": 1, "glm4moe.embedding_length": 8, "glm4moe.attention.head_count": 2, "glm4moe.context_length": 8,
            "glm4moe.feed_forward_length": 8, "glm4moe.expert_count": 4, "glm4moe.expert_used_count": 2}
    for extra in ({"glm4moe.expert_gating_func": 1}, {"glm4moe.rope.scaling.type": "yarn"}, {"glm4moe.expert_used_count": 1, "glm4moe.expert_weights_norm": True},
                  {"glm4moe.rope.dimension_sections": [4, 4, 0, 0]}):
        p = tmp_path / "x.gguf"
        SY.write_gguf(p, {**base, **extra}, [("token_embd.weight", (8, 4), "F32", np.zeros((4, 8), np.float32))])
        with pytest.raises(GF.GgufError):
            N.gguf_config(GF.GgufFile(p))


# ------------------------------------------------------------------------------------------------------------------ deepseek2 (MLA kv_b split)
def _split_kv_b_like_the_converter(kv_b, n_head, v_dim, nope):
    """conversion/deepseek.py DeepseekV2Model.modify_tensors: view, split on dim 1, k_b transposed (torch, transcribed to numpy)."""
    v = kv_b.reshape(n_head, v_dim + nope, kv_b.shape[-1])
    k_b, v_b = v[:, :nope, :], v[:, nope:, :]
    return np.ascontiguousarray(k_b.transpose(0, 2, 1)), np.ascontiguousarray(v_b)


def _ds_tr(k, w, allw):
    """Make the GGUF side from a kv_b_proj rebuilt out of the model's embed_q / unembed_out the way HF stores it, then run the converter's split."""
    if k.endswith("self_attn.embed_q.weight") or k.endswith("self_attn.unembed_out.weight"):
        pre = k.rsplit("self_attn.", 1)[0] + "self_attn."
        wk, wv = allw[pre + "embed_q.weight"], allw[pre + "unembed_out.weight"]
        H, nope = wk.shape[0], wk.shape[2]
        kv_b = np.concatenate([wk.transpose(0, 2, 1), wv], axis=1).reshape(H * (nope + wv.shape[1]), wk.shape[1])      # HF layout [H * (nope + v), kv_lora]
        k_b, v_b = _split_kv_b_like_the_converter(kv_b, H, wv.shape[1], nope)
        return k_b if k.endswith("embed_q.weight") else v_b
    return w


_ds_tr.wants_all = True


def _ds_gname(k):
    g = _moe_gname(k)
    if g:
        return g
    for a, b in (("embed_q", "attn_k_b"), ("unembed_out", "attn_v_b")):
        if k.endswith(f"self_attn.{a}.weight"):
            return f"blk.{k.split('.')[2]}.{b}.weight"
    return None


def test_deepseek2_gguf_round_trip_splits_kv_b_without_a_transpose(tmp_path):
    cfg = dict(hidden_size=64, num_hidden_layers=3, intermediate_size=96, moe_intermediate_size=32, num_attention_heads=4, num_key_value_heads=4,
               n_shared_experts=1, n_routed_experts=4, routed_scaling_factor=2.5, kv_lora_rank=24, q_lora_rank=16, qk_rope_head_dim=8, v_head_dim=12, qk_nope_head_dim=16,
               topk_method="noaux_tc", scoring_func="sigmoid", norm_topk_prob=True, n_group=1, topk_group=1, num_experts_per_tok=2, moe_layer_freq=1, first_k_dense_replace=1,
               max_position_embeddings=128, rope_scaling=None, attention_bias=False, tie_word_embeddings=False)
    kv = {"deepseek2.rope.dimension_count": 8, "deepseek2.attention.key_length_mla": 24, "deepseek2.attention.value_length_mla": 12, "deepseek2.attention.kv_lora_rank": 24,
          "deepseek2.attention.q_lora_rank": 16, "deepseek2.expert_count": 4, "deepseek2.expert_used_count": 2, "deepseek2.expert_feed_forward_length": 32,
          "deepseek2.expert_shared_count": 1, "deepseek2.leading_dense_block_count": 1, "deepseek2.expert_weights_scale": 2.5, "deepseek2.expert_weights_norm": True,
          "deepseek2.expert_gating_func": 2, "deepseek2.attention.layer_norm_rms_epsilon": 1e-6}
    ref, got, d = T._tiny_dense("deepseek2", tmp_path, "deepseek_v3", cfg, kv, alias={"mlp.switch_mlp": "mlp.experts"}, tr=_ds_tr, gname_of=_ds_gname)
    T._same_model(ref, got)
    c = json.load(open(os.path.join(d, "config.json")))
    assert c["model_type"] == "deepseek_v3" and c["qk_nope_head_dim"] == 16 and c["qk_rope_head_dim"] == 8 and c["v_head_dim"] == 12 and c["kv_lora_rank"] == 24
    assert c["q_lora_rank"] == 16 and c["first_k_dense_replace"] == 1 and c["routed_scaling_factor"] == 2.5 and c["n_group"] == 1
    # a stored transposed k_b would be caught: swapping the roles must change the weights
    wk = np.array(got.model.layers[0].self_attn.embed_q.weight.astype(mx.float32))
    assert wk.shape == (4, 24, 16) and np.array(got.model.layers[0].self_attn.unembed_out.weight).shape == (4, 12, 24)


def test_deepseek2_split_is_the_inverse_of_mlx_lm_sanitize():
    rng = np.random.default_rng(3)
    H, nope, vd, kvl = 4, 16, 12, 24
    kv_b = rng.standard_normal((H * (nope + vd), kvl)).astype(np.float32)
    k_b, v_b = _split_kv_b_like_the_converter(kv_b, H, vd, nope)
    from mlx_lm.models.deepseek_v3 import Model, ModelArgs
    args = ModelArgs(hidden_size=8, num_hidden_layers=1, num_attention_heads=H, kv_lora_rank=kvl, qk_nope_head_dim=nope, v_head_dim=vd, n_routed_experts=None, vocab_size=8)
    w = Model.sanitize(type("M", (), {"args": args})(), {"model.layers.0.self_attn.kv_b_proj.weight": mx.array(kv_b)})
    np.testing.assert_array_equal(np.array(w["model.layers.0.self_attn.embed_q.weight"]), k_b)
    np.testing.assert_array_equal(np.array(w["model.layers.0.self_attn.unembed_out.weight"]), v_b)


def test_deepseek2_refuses_yarn_softmax_and_legacy_kv_b(tmp_path):
    base = {"general.architecture": "deepseek2", "deepseek2.block_count": 1, "deepseek2.embedding_length": 8, "deepseek2.attention.head_count": 2, "deepseek2.context_length": 8,
            "deepseek2.feed_forward_length": 8, "deepseek2.rope.dimension_count": 4, "deepseek2.attention.key_length_mla": 8, "deepseek2.attention.value_length_mla": 4,
            "deepseek2.attention.kv_lora_rank": 8, "deepseek2.expert_count": 4, "deepseek2.expert_used_count": 2, "deepseek2.expert_gating_func": 2}
    z = np.zeros((4, 8), np.float32)
    ok = [("token_embd.weight", (8, 4), "F32", z), ("blk.0.attn_k_b.weight", (4, 8, 2), "F32", np.zeros((2, 8, 4), np.float32)), ("blk.0.exp_probs_b.bias", (4,), "F32", np.zeros(4, np.float32))]
    p = tmp_path / "x.gguf"
    SY.write_gguf(p, base, ok)
    assert N.gguf_config(GF.GgufFile(p))["model_type"] == "deepseek_v3"
    for extra, tens in (({"deepseek2.rope.scaling.type": "yarn"}, ok), ({"deepseek2.rope.scaling.yarn_log_multiplier": 0.1}, ok), ({"deepseek2.expert_gating_func": 1}, ok), ({}, ok[:1]),
                        ({}, [ok[0], ok[2]])):
        SY.write_gguf(p, {**base, **extra}, tens)
        with pytest.raises(GF.GgufError):
            N.gguf_config(GF.GgufFile(p))
    with pytest.raises(GF.GgufError):
        N.gguf_name("blk.0.attn_kv_b.weight", "deepseek2", 1)


# ------------------------------------------------------------------------------------------------------------------ mistral3
def test_mistral3_gguf_round_trip_undoes_the_q_k_permute(tmp_path):
    cfg = dict(hidden_size=64, num_hidden_layers=2, intermediate_size=96, num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=128,
               tie_word_embeddings=False, attention_bias=False, mlp_bias=False)

    def tr(k, w):
        return T._permute(w, 4, 4) if k.endswith("q_proj.weight") else T._permute(w, 4, 2) if k.endswith("k_proj.weight") else w
    kv = {"mistral3.attention.key_length": 16}
    ref, got, d = T._tiny_dense("mistral3", tmp_path, "llama", cfg, kv, tr=tr)
    T._same_model(ref, got)
    c = json.load(open(os.path.join(d, "config.json")))
    assert c["model_type"] == "llama" and c["head_dim"] == 16 and c["num_key_value_heads"] == 2


def test_mistral3_refuses_ministral3_yarn_and_attention_temperature(tmp_path):
    base = {"general.architecture": "mistral3", "mistral3.block_count": 1, "mistral3.embedding_length": 8, "mistral3.attention.head_count": 2, "mistral3.context_length": 8,
            "mistral3.feed_forward_length": 8, "mistral3.attention.head_count_kv": 2}
    for extra in ({"mistral3.rope.scaling.type": "yarn"}, {"mistral3.attention.temperature_scale": 0.1}, {"mistral3.rope.scaling.yarn_log_multiplier": 1.0}):
        p = tmp_path / "x.gguf"
        SY.write_gguf(p, {**base, **extra}, [("token_embd.weight", (8, 4), "F32", np.zeros((4, 8), np.float32))])
        with pytest.raises(GF.GgufError):
            N.gguf_config(GF.GgufFile(p))


# ------------------------------------------------------------------------------------------------------------------ SentencePiece vocabulary (tokenizer.ggml.model = llama)
def _spm_kv(tmp_path, prefix=True, vocab=400):
    sp = pytest.importorskip("sentencepiece")
    corpus = tmp_path / "c.txt"
    words = ["the quick brown fox jumps over the lazy dog", "hello world this is a tiny corpus", "kyojin serves gguf files from llama cpp", "naive café déjà vu 12345",
             "a b c d e f g h i j k l m n o p q r s t u v w x y z"]
    corpus.write_text("\n".join(words * 20))
    sp.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(tmp_path / "m"), vocab_size=vocab, model_type="bpe", byte_fallback=True, add_dummy_prefix=prefix,
                                  character_coverage=1.0, remove_extra_whitespaces=False, normalization_rule_name="identity", unk_id=0, bos_id=1, eos_id=2, pad_id=-1, hard_vocab_limit=False)
    s = sp.SentencePieceProcessor(model_file=str(tmp_path / "m.model"))
    n = s.get_piece_size()
    ty = []
    for i in range(n):                                              # llama.cpp conversion/base.py _create_vocab_sentencepiece token types
        ty.append(2 if s.is_unknown(i) else 3 if s.is_control(i) else 6 if s.is_byte(i) else 5 if s.is_unused(i) else 1)
    kv = {"tokenizer.ggml.model": "llama", "tokenizer.ggml.tokens": [s.id_to_piece(i) for i in range(n)], "tokenizer.ggml.scores": [float(s.get_score(i)) for i in range(n)],
          "tokenizer.ggml.token_type": ty, "tokenizer.ggml.bos_token_id": 1, "tokenizer.ggml.eos_token_id": 2, "tokenizer.ggml.unknown_token_id": 0,
          "tokenizer.ggml.add_bos_token": True, "tokenizer.ggml.add_space_prefix": prefix, "tokenizer.chat_template": "{{ bos_token }}{% for m in messages %}[{{ m['role'] }}] {{ m['content'] }}{% endfor %}"}
    return s, kv


@pytest.mark.parametrize("prefix", [True, False])
def test_spm_vocab_builds_a_tokenizer_json_that_encodes_like_sentencepiece(tmp_path, prefix):
    from transformers import AutoTokenizer
    s, kv = _spm_kv(tmp_path, prefix)
    out = tmp_path / "tok"
    out.mkdir()
    N.write_gguf_tokenizer(kv, str(out))
    tok = AutoTokenizer.from_pretrained(str(out))
    texts = ["the quick brown fox", "hello world", "kyojin serves gguf", "naive café déjà vu", "12345 and 678", "a b c", "zzz qqq xxx", "日本語 text", "the  double  space", "ends with space "]
    bad = []
    for t in texts:
        ref = s.encode(t)
        got = tok.encode(t, add_special_tokens=False)
        if ref != got:
            bad.append((t, ref, got))
    assert not bad, bad
    assert tok.encode("hello", add_special_tokens=True)[0] == 1 and tok.bos_token == "<s>"
    assert tok.decode(tok.encode("hello world", add_special_tokens=False)) == "hello world"
    assert json.load(open(out / "generation_config.json"))["eos_token_id"] == [2]


# ------------------------------------------------------------------------------------------------------------------ bare GGUF -> backend row -> chat family -> stop ids
def _bare_llama(tmp_path, arch="llama", model_type="llama", chat="{{ bos_token }}{% for m in messages %}[{{ m['role'] }}] {{ m['content'] }}{% endfor %}{% if add_generation_prompt %}[assistant]{% endif %}"):
    s, tk = _spm_kv(tmp_path, True)
    cfg = dict(hidden_size=64, num_hidden_layers=2, intermediate_size=96, num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=128,
               tie_word_embeddings=False, attention_bias=False, mlp_bias=False)
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_lm.utils import _get_classes
    full = dict(model_type=model_type, vocab_size=len(tk["tokenizer.ggml.tokens"]), rms_norm_eps=1e-6, hidden_act="silu", rope_theta=10000.0, **cfg)
    mc, ac = _get_classes(config=dict(full))
    m = mc(ac.from_dict(dict(full)))
    inv, top = T._inv_names(arch)
    tl = []
    for k, v in tree_flatten(m.parameters()):
        w = np.array(v.astype(mx.float32))
        if k.rsplit(".", 1)[0] in top:
            g = top[k.rsplit(".", 1)[0]] + "." + k.rsplit(".", 1)[1]
        else:
            _, _, i, rest = k.split(".", 3)
            mod, sfx = rest.rsplit(".", 1)
            g = f"blk.{i}.{inv[mod]}.{sfx}"
        tl.append((g, tuple(reversed(w.shape)), "F32", np.ascontiguousarray(w, np.float32)))
    kv = {"general.architecture": arch, f"{arch}.block_count": 2, f"{arch}.context_length": 128, f"{arch}.embedding_length": 64, f"{arch}.feed_forward_length": 96,
          f"{arch}.attention.head_count": 4, f"{arch}.attention.head_count_kv": 2, f"{arch}.attention.key_length": 16, f"{arch}.attention.layer_norm_rms_epsilon": 1e-6,
          f"{arch}.rope.freq_base": 10000.0, **dict(tk, **{"tokenizer.chat_template": chat})}
    p = tmp_path / f"{arch}.gguf"
    SY.write_gguf(p, kv, tl)
    return p


def test_deepseek2_gguf_is_routed_to_the_generic_loader(tmp_path, monkeypatch):
    """MLA (deepseek_v3 class) is served: Q35Backend runs its attention per user under a RowsKV (multiuser.install_mla, tests/test_mla_rows.py)."""
    from kyojin_mlx.serve import backends as B
    from kyojin_mlx import foreign_linear as FL
    p = tmp_path / "d.gguf"
    SY.write_gguf(p, {"general.architecture": "deepseek2"}, [])
    seen = {}
    monkeypatch.setattr(B, "wire_memory", lambda log=print: 0)
    monkeypatch.setattr(N, "gguf_config", lambda g: {"model_type": "deepseek_v3"})
    monkeypatch.setattr(FL, "is_foreign", lambda path: "gguf")
    monkeypatch.setattr(GF, "GgufFile", lambda path: None)
    monkeypatch.setattr(GF, "find_gguf", lambda path: str(p))
    monkeypatch.setattr(N, "gguf_has_mtp", lambda path: False)
    monkeypatch.setattr(B, "load_q35", lambda path, **kw: seen.update(path=path) or ("backend", {"model_type": "deepseek_v3"}))
    assert B.load_backend(str(p))[0] == "backend" and seen["path"] == str(p)


@pytest.mark.parametrize("arch,model_type", [("llama", "llama"), ("mistral3", "llama"), ("qwen2", "qwen2")])
def test_a_bare_gguf_of_a_mapped_family_is_routed_to_the_generic_loader_without_a_flag(tmp_path, monkeypatch, arch, model_type):
    from kyojin_mlx.serve import backends as B
    cfg = dict(hidden_size=64)
    p = _bare_llama(tmp_path, arch, model_type)
    seen = {}
    monkeypatch.setattr(B, "wire_memory", lambda log=print: 0)
    monkeypatch.setattr(B, "load_q35", lambda path, **kw: seen.update(path=path, kw=kw) or ("backend", {"model_type": model_type}))
    monkeypatch.delenv("KYOJIN_GGUF_GENERIC", raising=False)
    be, _ = B.load_backend(str(p))
    assert be == "backend" and seen["path"] == str(p)
    # an unmapped architecture is still refused by name, before any weight is read
    SY.write_gguf(tmp_path / "x.gguf", {"general.architecture": "mamba2"}, [])
    with pytest.raises(GF.GgufError):
        B.load_backend(str(tmp_path / "x.gguf"))


def test_generic_chat_family_uses_the_file_template_and_the_file_stop_ids(tmp_path):
    from transformers import AutoTokenizer
    from kyojin_mlx.serve import chat as CH, server as SV
    p = _bare_llama(tmp_path)
    d = N.gguf_serve_dir(str(p), str(tmp_path / "cache"))
    cfg = json.load(open(os.path.join(d, "config.json")))
    assert SV._family_name(d, cfg) == "generic" and CH.family_of_gguf("glm4") == "generic" and CH.family_of_gguf("glm4_moe") == "generic"
    assert CH.family_of_gguf("qwen2") == "qwen" and CH.family_of_gguf("qwen2_moe") == "qwen" and CH.family_of_gguf("gemma3_text") == "generic"
    tok = AutoTokenizer.from_pretrained(d)
    fam = CH.FAMILIES["generic"](tok, d, None, {})
    r = fam.render({"messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}], "tools": [{"type": "function"}]})
    assert r.prompt == "[system] be brief[user] hi[assistant]" and r.thinking_first is False and r.parse_tools is False      # the BOS the template prints is left to the tokenizer
    ids = tok.encode(r.prompt)
    assert ids[0] == 1 and ids[1] != 1 and ids.count(1) == 1                  # exactly one BOS reaches the model
    assert SV.stop_token_ids(d, cfg, tok, "generic") == {2}                   # eos from tokenizer.ggml.eos_token_id
    from kyojin_mlx.serve import autoconfig as AC
    assert AC.family_of_pack(cfg, d) == "generic"
    assert AC.family_of_pack({"model_type": "llama"}) is None               # a safetensors llama pack: unchanged (no family)


def test_rotating_cache_forks_and_continues_like_the_original():
    """gemma3 sliding layers hold a RotatingKVCache: multiuser.fork_cache (prefix snapshots, batching) used to raise on it."""
    from mlx_lm.models.cache import RotatingKVCache
    from kyojin_mlx import multiuser as MU
    c = RotatingKVCache(max_size=4)
    rng = np.random.default_rng(0)
    kv = lambda n: (mx.array(rng.standard_normal((1, 2, n, 8)).astype(np.float32)), mx.array(rng.standard_normal((1, 2, n, 8)).astype(np.float32)))
    c.update_and_fetch(*kv(6))                                               # prompt longer than the window: the ring has wrapped
    c.update_and_fetch(*kv(1))
    f = MU.fork_cache([c])[0]
    assert (f.offset, f._idx, f.max_size, f.keep) == (c.offset, c._idx, c.max_size, c.keep)
    k2, v2 = kv(1)
    a = c.update_and_fetch(k2, v2)
    b = f.update_and_fetch(k2, v2)
    np.testing.assert_array_equal(np.array(a[0]), np.array(b[0]))
    np.testing.assert_array_equal(np.array(a[1]), np.array(b[1]))
    assert f.offset == c.offset and f._idx == c._idx
    f.update_and_fetch(*kv(1))                                               # writing the fork must not touch the original
    assert c.offset == 8


def test_generic_stop_ids_include_end_of_turn_by_text_when_the_metadata_has_no_eot(tmp_path):
    """llama-vocab.cpp:3266-3296 puts <end_of_turn> / <|eot_id|> / <|im_end|> ... into special_eog_ids by their text; gemma and GLM files carry no eot key."""
    from transformers import AutoTokenizer
    from kyojin_mlx.serve import server as SV
    s, kv = _spm_kv(tmp_path, False)
    n = len(kv["tokenizer.ggml.tokens"])
    kv["tokenizer.ggml.tokens"] = kv["tokenizer.ggml.tokens"] + ["<start_of_turn>", "<end_of_turn>"]
    kv["tokenizer.ggml.scores"] = kv["tokenizer.ggml.scores"] + [0.0, 0.0]
    kv["tokenizer.ggml.token_type"] = kv["tokenizer.ggml.token_type"] + [3, 3]
    out = tmp_path / "t"
    out.mkdir()
    N.write_gguf_tokenizer(kv, str(out))
    tok = AutoTokenizer.from_pretrained(str(out))
    assert SV.stop_token_ids(str(out), {}, tok, "generic") == {2, n + 1}


# ------------------------------------------------------------------------------------------------------------------ Q35Backend on CPU: prefill -> fork -> batch -> decode vs the model's own forward
def _tiny_backend_models(tmp_path):
    cfg = dict(hidden_size=64, num_hidden_layers=2, intermediate_size=96, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128, tie_word_embeddings=False,
               attention_bias=False, mlp_bias=False)
    yield "llama", T._tiny_dense("llama", tmp_path / "a", "llama", cfg, {})[1]
    g = dict(hidden_size=64, num_hidden_layers=6, intermediate_size=96, num_attention_heads=4, num_key_value_heads=2, head_dim=16, sliding_window=4, sliding_window_pattern=6,
             query_pre_attn_scalar=16.0, rope_theta=1000000.0, rope_local_base_freq=10000.0, max_position_embeddings=128)
    kv = {"gemma3.attention.key_length": 16, "gemma3.attention.value_length": 16, "gemma3.attention.sliding_window": 4, "gemma3.rope.freq_base": 1000000.0,
          "gemma3.rope.freq_base_swa": 10000.0, "gemma3.block_count": 6}
    yield "gemma3", T._tiny_dense("gemma3", tmp_path / "b", "gemma3_text", g, kv, tr=_norm_plus1_)[1]
    yield "glm4moe", _glm4moe_model(tmp_path / "c")


def _glm4moe_model(tmp_path):
    cfg = dict(hidden_size=64, num_hidden_layers=3, intermediate_size=96, num_attention_heads=4, num_key_value_heads=2, head_dim=16, attention_bias=True, use_qk_norm=True,
               partial_rotary_factor=0.5, max_position_embeddings=128, rope_theta=10000.0, rope_scaling=None, tie_word_embeddings=False, n_routed_experts=4,
               num_experts_per_tok=2, moe_intermediate_size=32, n_shared_experts=1, first_k_dense_replace=1, routed_scaling_factor=1.5, norm_topk_prob=True, n_group=1,
               topk_group=1, scoring_func="sigmoid", topk_method="noaux_tc")
    kv = {"glm4moe.rope.dimension_count": 8, "glm4moe.attention.key_length": 16, "glm4moe.expert_count": 4, "glm4moe.expert_used_count": 2, "glm4moe.expert_feed_forward_length": 32,
          "glm4moe.expert_shared_count": 1, "glm4moe.leading_dense_block_count": 1, "glm4moe.expert_weights_scale": 1.5, "glm4moe.expert_weights_norm": True, "glm4moe.expert_gating_func": 2}
    return T._tiny_dense("glm4moe", tmp_path, "glm4_moe", cfg, kv, alias={"mlp.switch_mlp": "mlp.experts"}, gname_of=_moe_gname)[1]


def _norm_plus1_(k, w):
    return w + 1 if k.endswith("norm.weight") else w


def test_q35backend_serves_the_generic_models_on_cpu_and_matches_the_models_own_forward(tmp_path):
    """llama, glm4moe and gemma3 (sliding window 4 < the 8-token prompt: the ring has wrapped, and the fork/batch run on RotatingKVCache). Solo prefill, fork, 2-user batched decode."""
    from kyojin_mlx.serve import backends as B
    for tmp in ("a", "b", "c"):
        (tmp_path / tmp).mkdir()
    prompt, nxt = [3, 7, 1, 30, 22, 5, 9, 2], 4
    dev = mx.default_device()
    mx.set_default_device(mx.cpu)               # Metal: stock mlx-lm gemma3 prefill+step vs full forward already differ by 0.0039 in float32 (no Kyojin code involved)
    try:
        _serve_generic_models_cpu(tmp_path, prompt, nxt)
    finally:
        mx.set_default_device(dev)


def _serve_generic_models_cpu(tmp_path, prompt, nxt):
    from kyojin_mlx.serve import backends as B
    for name, model in _tiny_backend_models(tmp_path):
        model.set_dtype(mx.float32)
        want = np.array(model(mx.array([prompt + [nxt]]))[:, -1])
        be = B.Q35Backend(model)
        cache = be.make_cache()
        be.prefill_chunk(cache, prompt)
        bc = be.batch([cache, be.fork(cache)])
        got = np.array(be.step(bc, mx.array([nxt, nxt], dtype=mx.int32)))
        np.testing.assert_allclose(got[0], want[0], atol=2e-4, rtol=1e-3, err_msg=name)
        np.testing.assert_allclose(got[1], want[0], atol=2e-4, rtol=1e-3, err_msg=name)


def test_freqrope_matches_mlx_llama3_rope_and_plain_rope():
    """llama.cpp rope_freqs = Llama3RoPE freqs / base^(2i/d): FreqRoPE with them is mlx-lm's Llama3RoPE; all ones is the plain rope."""
    import mlx.nn as nn
    from mlx_lm.models.rope_utils import Llama3RoPE
    from kyojin_mlx.foreign_linear import FreqRoPE
    d, base = 64, 500000.0
    ref = Llama3RoPE(d, 131072, False, base, {"factor": 32.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8192})
    ff = ref._freqs / (base ** (mx.arange(0, d, 2, dtype=mx.float32) / d))
    x = mx.random.normal((1, 4, 7, d))
    np.testing.assert_allclose(np.array(FreqRoPE(d, base, False, ff)(x, offset=5)), np.array(ref(x, offset=5)), atol=1e-5)
    np.testing.assert_allclose(np.array(FreqRoPE(d, base, False, mx.ones((d // 2,)))(x, offset=3)), np.array(nn.RoPE(d, base=base)(x, offset=3)), atol=1e-5)
