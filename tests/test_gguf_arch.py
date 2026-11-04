"""GGUF architectures beyond qwen3 / qwen3moe: the name tables against llama.cpp's own tensor map, the converter transforms replayed in the forward direction
(a separate transcription of conversion/qwen.py at foreign_names.LLAMA_CPP_COMMIT) and undone by the loader, the tokenizer table. Synthetic files only; CPU
mlx / mlx-lm (no Metal): the quantised-kernel launches themselves are not run here."""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest

from kyojin_mlx import foreign_gguf as GF, foreign_names as N, foreign_synth as SY, foreign_xform as XF

SNAP = json.load(open(os.path.join(os.path.dirname(__file__), "data", "gguf_names_llamacpp.json")))
mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")


# ------------------------------------------------------------------------------------------------------------------ names vs llama.cpp
_RENAMED = {"linear_attn.dt_bias": "linear_attn.dt_proj", "self_attn.embed_q": "self_attn.k_b_proj", "self_attn.unembed_out": "self_attn.v_b_proj"}      # conversion/qwen.py Qwen3NextModel.modify_tensors: `.dt_bias` -> `.dt_proj.bias` before the map


def _hf_forms(h: str) -> set:
    return {h, "model.language_model." + h[len("model."):] if h.startswith("model.") else h}


@pytest.mark.parametrize("arch", sorted(N.ARCHS))
def test_every_table_row_is_a_llamacpp_mapping(arch):
    A, snap = N.ARCHS[arch], SNAP[arch]
    for kind, mod in A.blk.items():
        row = snap.get(f"blk.0.{kind}")
        assert row is not None, f"{arch}: llama.cpp has no tensor blk.N.{kind}"
        assert f"model.layers.0.{_RENAMED.get(mod, mod)}" in row, (arch, kind, mod, row)
    for key, mod in A.exact.items():
        kind = key.rpartition(".")[0] if key.endswith((".weight", ".bias")) else key
        assert f"model.layers.0.{_RENAMED.get(mod.rsplit('.', 1)[0] + '.' + mod.rsplit('.', 1)[1], mod)}" in snap[f"blk.0.{kind}"] or \
            f"model.layers.0.{_RENAMED.get(mod, mod)}" in snap[f"blk.0.{kind}"], (arch, key, mod)
    for g, h in A.top.items():
        assert h.rsplit(".", 1)[0] in snap[g.rsplit(".", 1)[0]], (arch, g, h)
    for g, h in (A.mtp or {}).items():                  # filter_tensors renames mtp.fc ... to eh_proj ... of the first draft block, and tensor_mapping maps those
        kind = g.rpartition(".")[0]
        assert any(c.endswith(h_) for c in snap[f"blk.0.{kind}"] for h_ in ({"mtp.fc.weight": "eh_proj", "mtp.pre_fc_norm_embedding.weight": "enorm",
                                                                             "mtp.pre_fc_norm_hidden.weight": "hnorm", "mtp.norm.weight": "shared_head.norm"}[h],)), (arch, g)


def test_qwen35_names_of_the_real_file():
    real = """output.weight output_norm.weight token_embd.weight blk.0.attn_gate.weight blk.0.attn_norm.weight blk.0.attn_qkv.weight blk.0.ffn_down.weight
        blk.0.ffn_gate.weight blk.0.ffn_up.weight blk.0.post_attention_norm.weight blk.0.ssm_a blk.0.ssm_alpha.weight blk.0.ssm_beta.weight blk.0.ssm_conv1d.weight
        blk.0.ssm_dt.bias blk.0.ssm_norm.weight blk.0.ssm_out.weight blk.3.attn_k.weight blk.3.attn_k_norm.weight blk.3.attn_output.weight blk.3.attn_q.weight
        blk.3.attn_q_norm.weight blk.3.attn_v.weight blk.64.attn_k.weight blk.64.attn_k_norm.weight blk.64.attn_norm.weight blk.64.attn_output.weight
        blk.64.attn_q.weight blk.64.attn_q_norm.weight blk.64.attn_v.weight blk.64.ffn_down.weight blk.64.ffn_gate.weight blk.64.ffn_up.weight
        blk.64.nextn.eh_proj.weight blk.64.nextn.enorm.weight blk.64.nextn.hnorm.weight blk.64.nextn.shared_head_norm.weight blk.64.post_attention_norm.weight""".split()
    got = {n: N.gguf_name(n, "qwen35", 64) for n in real}
    assert got["blk.0.attn_qkv.weight"] == "model.layers.0.linear_attn.in_proj_qkv.weight"
    assert got["blk.0.attn_gate.weight"] == "model.layers.0.linear_attn.in_proj_z.weight"
    assert got["blk.0.ssm_alpha.weight"] == "model.layers.0.linear_attn.in_proj_a.weight" and got["blk.0.ssm_beta.weight"] == "model.layers.0.linear_attn.in_proj_b.weight"
    assert got["blk.0.ssm_a"] == "model.layers.0.linear_attn.A_log" and got["blk.0.ssm_dt.bias"] == "model.layers.0.linear_attn.dt_bias"
    assert got["blk.64.nextn.eh_proj.weight"] == "mtp.fc.weight" and got["blk.64.attn_q.weight"] == "mtp.layers.0.self_attn.q_proj.weight"
    assert got["blk.64.post_attention_norm.weight"] == "mtp.layers.0.post_attention_layernorm.weight" and got["blk.3.attn_q.weight"] == "model.layers.3.self_attn.q_proj.weight"
    assert len(set(got.values())) == len(real)


def test_qwen35_refuses_what_it_does_not_map():
    with pytest.raises(GF.GgufError):
        N.gguf_name("blk.0.ffn_gate_exps.weight", "qwen35", 64)         # the MoE variant is a different architecture (qwen35moe)
    with pytest.raises(GF.GgufError):
        N.gguf_name("blk.64.nextn.embed_tokens.weight", "qwen35", 64)   # dedicated draft embeddings: the q35 draft head has none
    with pytest.raises(GF.GgufError):
        N.gguf_name("blk.64.nextn.eh_proj.weight", "qwen3", 64)         # no draft head mapping for this arch


def test_v_head_index_is_the_inverse_of_the_converter_reorder():
    for nk, r, hd in ((2, 3, 1), (16, 3, 128), (4, 2, 5)):
        t = np.random.default_rng(0).standard_normal(nk * r * hd)
        fwd = _reorder(t, 0, nk, r, hd)                                  # qwen.py _reorder_v_heads, transcribed below
        np.testing.assert_array_equal(fwd[XF.v_grouped_index(nk, nk * r, hd)], t)
        np.testing.assert_array_equal(t[XF.v_tiled_index(nk, nk * r, hd)], fwd)
        if r > 1:                                                        # applying the converter's reorder again is NOT the inverse
            assert not np.array_equal(_reorder(fwd, 0, nk, r, hd), t)


# ------------------------------------------------------------------------------------------------------------------ the converter, forward
def _reorder(t, dim, nk, r, hd):
    """conversion/qwen.py `_LinearAttentionVReorderBase._reorder_v_heads` (non-LoRA branch), numpy."""
    shape = list(t.shape)
    if dim < 0:
        dim += len(shape)
    new_shape = shape[:dim] + [nk, r, hd] + shape[dim + 1:]
    t = t.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return np.ascontiguousarray(t.transpose(*perm)).reshape(*shape)


def _convert_qwen35(hf: dict, c: dict, n_layer: int) -> dict:
    """HF tensors (checkpoint names, mtp.* included) -> {gguf name: float32 array} the way convert_hf_to_gguf.py does for Qwen3_5TextModel with an MTP head."""
    nk, nv, kd, vd = c["nk"], c["nv"], c["kd"], c["vd"]
    r = nv // nk
    out = {}
    for name, t in hf.items():
        t = np.asarray(t, np.float32)
        if name.startswith("mtp."):                                     # _QwenMtpMixin.filter_tensors
            rem = {"mtp.fc.weight": "eh_proj.weight", "mtp.pre_fc_norm_embedding.weight": "enorm.weight", "mtp.pre_fc_norm_hidden.weight": "hnorm.weight",
                   "mtp.norm.weight": "shared_head.norm.weight"}
            name = f"model.layers.{n_layer}.{rem[name]}" if name in rem else f"model.layers.{n_layer + int(name.split('.')[2])}.{name.split('.', 3)[3]}"
        if "linear_attn." in name and nk != nv:                         # _LinearAttentionVReorderBase.modify_tensors
            if ".in_proj_qkv." in name:
                q = t[:kd * nk]
                k = t[kd * nk:2 * kd * nk]
                t = np.concatenate([q, k, _reorder(t[2 * kd * nk:], 0, nk, r, vd)], 0)
            elif ".in_proj_z." in name:
                t = _reorder(t, 0, nk, r, vd)
            elif ".in_proj_b." in name or ".in_proj_a." in name:
                t = _reorder(t, 0, nk, r, 1)
            elif ".A_log" in name or ".dt_bias" in name:
                t = _reorder(t[:, None], 0, nk, r, 1)[:, 0]
            elif ".conv1d" in name:
                d = t.squeeze()
                t = np.concatenate([d[:2 * kd * nk], _reorder(d[2 * kd * nk:], 0, nk, r, vd)], 0)
            elif ".out_proj." in name:
                t = _reorder(t, 1, nk, r, vd)
        if name.endswith(".A_log"):                                     # Qwen3NextModel.modify_tensors
            t = -np.exp(t)
        elif name.endswith(".dt_bias"):
            name = name.rpartition(".dt_bias")[0] + ".dt_proj.bias"
        elif "conv1d" in name:
            t = t.squeeze()
        elif name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
            t = t + 1
        out[_gguf_of(name)] = t
    return out


_TOP35 = {"model.embed_tokens.weight": "token_embd.weight", "model.norm.weight": "output_norm.weight", "lm_head.weight": "output.weight"}
_BLK35 = {  # gguf-py tensor_mapping.py (the pinned commit), written out per HF suffix
    "input_layernorm.weight": "attn_norm.weight", "post_attention_layernorm.weight": "post_attention_norm.weight", "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight", "self_attn.v_proj.weight": "attn_v.weight", "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight", "self_attn.k_norm.weight": "attn_k_norm.weight", "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight", "mlp.down_proj.weight": "ffn_down.weight", "linear_attn.in_proj_qkv.weight": "attn_qkv.weight",
    "linear_attn.in_proj_z.weight": "attn_gate.weight", "linear_attn.in_proj_a.weight": "ssm_alpha.weight", "linear_attn.in_proj_b.weight": "ssm_beta.weight",
    "linear_attn.out_proj.weight": "ssm_out.weight", "linear_attn.norm.weight": "ssm_norm.weight", "linear_attn.conv1d.weight": "ssm_conv1d.weight",
    "linear_attn.A_log": "ssm_a", "linear_attn.dt_proj.bias": "ssm_dt.bias", "eh_proj.weight": "nextn.eh_proj.weight", "enorm.weight": "nextn.enorm.weight",
    "hnorm.weight": "nextn.hnorm.weight", "shared_head.norm.weight": "nextn.shared_head_norm.weight"}


def _gguf_of(name):
    if name in _TOP35:
        return _TOP35[name]
    _, _, i, rest = name.split(".", 3)
    return f"blk.{i}.{_BLK35[rest]}"


# ------------------------------------------------------------------------------------------------------------------ a tiny qwen35
C = dict(H=64, V=48, I=96, nh=4, nkv=2, hd=16, nk=2, nv=6, kd=16, vd=16, K=4, L=4, rd=8, sec=[2, 1, 1, 0])


def _hf_qwen35(seed=0, mtp=True):
    """Random HF Qwen3.5 text tensors; values multiples of 1/64 so fp16 / the +1 / -1 round trips are exact."""
    r = np.random.default_rng(seed)
    q = lambda *s: (r.integers(-24, 25, s) / 64).astype(np.float32)
    H, V, I = C["H"], C["V"], C["I"]
    conv = 2 * C["kd"] * C["nk"] + C["nv"] * C["vd"]
    t = {"model.embed_tokens.weight": q(V, H), "model.norm.weight": q(H), "lm_head.weight": q(V, H)}

    def mlp_attn(p, full):
        d = {p + "input_layernorm.weight": q(H), p + "post_attention_layernorm.weight": q(H), p + "mlp.gate_proj.weight": q(I, H), p + "mlp.up_proj.weight": q(I, H),
             p + "mlp.down_proj.weight": q(H, I)}
        if full:
            d.update({p + "self_attn.q_proj.weight": q(C["nh"] * C["hd"] * 2, H), p + "self_attn.k_proj.weight": q(C["nkv"] * C["hd"], H),
                      p + "self_attn.v_proj.weight": q(C["nkv"] * C["hd"], H), p + "self_attn.o_proj.weight": q(H, C["nh"] * C["hd"]),
                      p + "self_attn.q_norm.weight": q(C["hd"]), p + "self_attn.k_norm.weight": q(C["hd"])})
        return d

    for i in range(C["L"]):
        p = f"model.layers.{i}."
        full = (i + 1) % 4 == 0
        t.update(mlp_attn(p, full))
        if not full:
            a = p + "linear_attn."
            t.update({a + "in_proj_qkv.weight": q(conv, H), a + "in_proj_z.weight": q(C["nv"] * C["vd"], H), a + "in_proj_a.weight": q(C["nv"], H),
                      a + "in_proj_b.weight": q(C["nv"], H), a + "conv1d.weight": q(conv, 1, C["K"]), a + "A_log": r.integers(-8, 8, C["nv"]).astype(np.float32) / 8,
                      a + "dt_bias": q(C["nv"]), a + "norm.weight": q(C["vd"]), a + "out_proj.weight": q(H, C["nv"] * C["vd"])})
    if mtp:
        m = mlp_attn("mtp.layers.0.", True)
        m.update({"mtp.fc.weight": q(H, 2 * H), "mtp.norm.weight": q(H), "mtp.pre_fc_norm_embedding.weight": q(H), "mtp.pre_fc_norm_hidden.weight": q(H)})
        t.update(m)
    return t


def _kv35(mtp=True, **over):
    kv = {"general.architecture": "qwen35", "qwen35.block_count": C["L"] + (1 if mtp else 0), "qwen35.context_length": 128, "qwen35.embedding_length": C["H"],
          "qwen35.feed_forward_length": C["I"], "qwen35.attention.head_count": C["nh"], "qwen35.attention.head_count_kv": C["nkv"],
          "qwen35.attention.key_length": C["hd"], "qwen35.attention.value_length": C["hd"], "qwen35.attention.layer_norm_rms_epsilon": 1e-6,
          "qwen35.rope.freq_base": 1e7, "qwen35.rope.dimension_count": C["rd"], "qwen35.rope.dimension_sections": C["sec"], "qwen35.ssm.conv_kernel": C["K"],
          "qwen35.ssm.state_size": C["kd"], "qwen35.ssm.group_count": C["nk"], "qwen35.ssm.time_step_rank": C["nv"], "qwen35.ssm.inner_size": C["nv"] * C["vd"],
          "qwen35.full_attention_interval": 4, **_tiny_tokenizer_kv()}
    if mtp:
        kv["qwen35.nextn_predict_layers"] = 1
    kv.update(over)
    return kv


def _write(path, hf, kv=None, qtype=None, mtp=True):
    """GGUF file of the converted tensors; F32 everywhere, or `qtype` blocks (random) for the listed names {gguf name: type}."""
    g = _convert_qwen35(hf, dict(C), C["L"])
    tl = []
    for n, a in g.items():
        if qtype and n in qtype:
            n_, k_ = a.shape
            tl.append((n, (k_, n_), qtype[n], SY.random_blocks(qtype[n], n_, k_, np.random.default_rng(len(n)))))
        else:
            tl.append((n, tuple(reversed(a.shape)), "F32", np.ascontiguousarray(a, np.float32)))
    SY.write_gguf(path, kv or _kv35(mtp), tl)
    return g


def test_qwen35_config_from_metadata(tmp_path):
    p = tmp_path / "m.gguf"
    _write(p, _hf_qwen35())
    cfg = N.gguf_config(GF.GgufFile(p))
    assert cfg["model_type"] == "qwen3_5" and cfg["num_hidden_layers"] == C["L"] and cfg["mtp_num_hidden_layers"] == 1       # block_count 5 = 4 trunk + 1 nextn
    assert cfg["linear_num_key_heads"] == 2 and cfg["linear_num_value_heads"] == 6 and cfg["linear_key_head_dim"] == 16 and cfg["linear_value_head_dim"] == 16
    assert cfg["partial_rotary_factor"] == 0.5 and cfg["rope_parameters"]["mrope_section"] == [2, 1, 1] and cfg["tie_word_embeddings"] is False
    assert cfg["attn_output_gate"] is True and cfg["head_dim"] == 16 and cfg["rope_theta"] == 1e7
    # the real 27B numbers: 65 blocks, nextn 1, inner 6144 / rank 48, rope dims 64 of head size 256, sections [11, 11, 10, 0]
    real = N._cfg_qwen35(type("G", (), {"kv": _kv35(**{"qwen35.block_count": 65, "qwen35.embedding_length": 5120, "qwen35.attention.key_length": 256,
                                                       "qwen35.attention.value_length": 256, "qwen35.rope.dimension_count": 64, "qwen35.ssm.state_size": 128,
                                                       "qwen35.ssm.group_count": 16, "qwen35.ssm.time_step_rank": 48, "qwen35.ssm.inner_size": 6144,
                                                       "qwen35.rope.dimension_sections": [11, 11, 10, 0]}),
                                        "tensors": {"token_embd.weight": type("T", (), {"ne": (5120, 248320)})()}})(), "qwen35")
    assert real["num_hidden_layers"] == 64 and real["partial_rotary_factor"] == 0.25 and real["linear_value_head_dim"] == 128


@pytest.mark.parametrize("over, why", [({"qwen35.rope.scaling.type": "yarn"}, "rope"), ({"prism.hadamard.block_size": 16}, "hadamard"),
                                       ({"qwen35.ssm.inner_size": 7 * 16}, "ssm"), ({"qwen35.rope.dimension_sections": [11, 11, 10, 0]}, "sections"),
                                       ({"qwen35.attention.recurrent_layers": [True, True, True, True, False]}, "recurrent")])
def test_qwen35_config_refuses(tmp_path, over, why):
    p = tmp_path / "m.gguf"
    _write(p, _hf_qwen35(), kv=_kv35(**over))
    with pytest.raises(GF.GgufError):
        N.gguf_config(GF.GgufFile(p))


def _tiny_tokenizer_kv():
    ab = [chr(c) for c in range(97, 123)]
    pairs = [ab[i] + ab[i + 1] for i in range(C["V"] - 4 - 26)]
    toks = ab + pairs + ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>"]
    return {"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "qwen35", "tokenizer.ggml.tokens": toks, "tokenizer.ggml.merges": [f"{p[0]} {p[1]}" for p in pairs],
            "tokenizer.ggml.token_type": [1] * (len(toks) - 4) + [3, 3, 3, 4], "tokenizer.ggml.eos_token_id": len(toks) - 2,
            "tokenizer.ggml.padding_token_id": len(toks) - 4, "tokenizer.chat_template": "{{ messages }}"}


def _ref_model(hf, cfg, tmp_path):
    from mlx.utils import tree_flatten
    from mlx_lm.utils import _get_classes
    mc, ac = _get_classes(config=dict(cfg))
    m = mc(ac.from_dict(dict(cfg)))
    ck = {k: mx.array(v) for k, v in hf.items() if not k.startswith("mtp.")}      # what the safetensors path feeds: HF names, HF layouts
    w = m.sanitize(ck)
    m.load_weights([(k, (v.astype(mx.float16) if v.ndim > 1 else v.astype(mx.float32))) for k, v in w.items()], strict=False)
    return m, dict(tree_flatten(m.parameters()))


def test_qwen35_gguf_loads_to_the_same_parameters_as_the_hf_checkpoint(tmp_path):
    from mlx.utils import tree_flatten
    from kyojin_mlx import foreign_linear as FL
    hf = _hf_qwen35()
    p = tmp_path / "m.gguf"
    _write(p, hf)
    d = N.gguf_serve_dir(str(p), str(tmp_path / "cache"))
    kv = _tiny_tokenizer_kv()
    cfg = json.load(open(os.path.join(d, "config.json")))
    ref, rp = _ref_model(hf, cfg, tmp_path)
    model, _ = FL.load_foreign(d, log=lambda *a: None)
    head = FL.take_mtp(model)
    got = dict(tree_flatten(model.parameters()))
    # linears became ForeignLinear (dense fp16 packs here): compare their weights through the module tree
    lin = {n: m for n, m in model.named_modules() if isinstance(m, FL.ForeignLinear)}
    assert len(lin) > 0
    for k, v in rp.items():
        if k in got:
            if k.endswith("A_log"):                                      # log(-ssm_a): a float32 round trip, not bit-exact (the file holds -exp(A_log))
                np.testing.assert_allclose(np.array(got[k]), np.array(v), rtol=0, atol=2e-6, err_msg=k)
            else:
                np.testing.assert_array_equal(np.array(got[k].astype(mx.float32)), np.array(v.astype(mx.float32)), err_msg=k)
        else:
            mod = k[:-len(".weight")]
            assert k.endswith(".weight") and mod in lin, k
            w = lin[mod].dev.w
            np.testing.assert_array_equal(np.array(w.astype(mx.float32)), np.array(v.astype(mx.float32)), err_msg=k)
    # draft head: the checkpoint's own mtp.* values, norms without the converter's +1, linears fp16
    assert head is not None
    for k, v in hf.items():
        if k.startswith("mtp."):
            np.testing.assert_array_equal(np.array(head[k].astype(mx.float32)), np.asarray(v, np.float32), err_msg=k)
    assert set(head) == {k for k in hf if k.startswith("mtp.")}


def test_qwen35_gguf_forward_matches_the_hf_checkpoint(tmp_path):
    from kyojin_mlx import foreign_linear as FL
    hf = _hf_qwen35()
    p = tmp_path / "m.gguf"
    _write(p, hf)
    d = N.gguf_serve_dir(str(p), str(tmp_path / "cache"))
    cfg = json.load(open(os.path.join(d, "config.json")))
    ref, _ = _ref_model(hf, cfg, tmp_path)
    model, _ = FL.load_foreign(d, log=lambda *a: None)
    ids = mx.array([[3, 7, 1, 30, 22, 5, 9, 2]])
    a = np.array(ref(ids).astype(mx.float32))
    b = np.array(model(ids).astype(mx.float32))
    assert np.abs(a - b).max() <= 3e-2 * max(1.0, np.abs(a).max()), np.abs(a - b).max()
    # the wrong inverse (running the converter's reorder again) must be caught by this very test
    from kyojin_mlx import foreign_xform as X
    good = X.v_grouped_index
    X.v_grouped_index = lambda nk, nv, hd: X.v_tiled_index(nk, nv, hd)
    try:
        bad, _ = FL.load_foreign(d, log=lambda *a: None)
        c = np.array(bad(ids).astype(mx.float32))
    finally:
        X.v_grouped_index = good
    assert np.abs(a - c).max() > 10 * np.abs(a - b).max()


def test_qwen35_equal_head_counts_apply_no_reorder(tmp_path):
    kv = {"qwen35.ssm.group_count": 2, "qwen35.ssm.time_step_rank": 2, "qwen35.ssm.inner_size": 32, "qwen35.ssm.state_size": 16}
    x = XF.plan("blk.0.attn_qkv.weight", "qwen35", _kv35(**kv))
    assert x is None
    x = XF.plan("blk.0.ssm_a", "qwen35", _kv35(**kv))
    assert x.rows is None and x.post == "a_log"


def test_qwen35_row_permutation_acts_on_the_stored_blocks_and_ssm_out_on_the_activation(tmp_path):
    from kyojin_mlx import foreign_linear as FL
    hf = _hf_qwen35()
    p = tmp_path / "q.gguf"
    names = {"blk.0.attn_qkv.weight": "Q4_0", "blk.0.attn_gate.weight": "Q4_0", "blk.0.ssm_out.weight": "Q4_0", "blk.0.ssm_alpha.weight": "Q4_0"}
    g = _write(p, hf, qtype=names)
    src = FL._read_gguf(str(p), "qwen35")
    gf = GF.GgufFile(p)
    # qkv: Q rows and K rows untouched, V rows permuted to the grouped order
    raw = gf.raw("blk.0.attn_qkv.weight")
    inv = XF.v_grouped_index(C["nk"], C["nv"], C["vd"])
    n_qk = 2 * C["kd"] * C["nk"]
    want = np.concatenate([raw[:n_qk], raw[n_qk:][inv]])
    ref = GF.to_pack("Q4_0", want, raw.shape[1] // 18 * 32)
    pk = FL._build(src.packs["model.layers.0.linear_attn.in_proj_qkv"])  # packs are built lazily, one at a time
    np.testing.assert_array_equal(pk.qs, ref.qs)
    np.testing.assert_array_equal(pk.meta, ref.meta)
    # ssm_out: a kernel type is kept as stored and carries the activation gather
    assert "model.layers.0.linear_attn.out_proj" in src.in_perm
    idx = src.in_perm["model.layers.0.linear_attn.out_proj"]
    np.testing.assert_array_equal(idx, XF.v_tiled_index(C["nk"], C["nv"], C["vd"]))
    # and that gather is the right one: dense W_hf @ x == W_stored @ x[idx]
    rng = np.random.default_rng(1)
    w_hf = rng.standard_normal((8, C["nv"] * C["vd"])).astype(np.float32)
    w_st = _reorder(w_hf, 1, C["nk"], C["nv"] // C["nk"], C["vd"])
    x = rng.standard_normal(C["nv"] * C["vd"]).astype(np.float32)
    np.testing.assert_allclose(w_st @ x[idx], w_hf @ x, rtol=1e-5, atol=1e-5)
    lin = FL.ForeignLinear(FL.FP.Pack("dense", 8, w_st.shape[1], w_st.astype(np.float16)), in_perm=idx)
    got = np.array(lin(mx.array(x.astype(np.float16))[None]).astype(mx.float32))[0]
    np.testing.assert_allclose(got, w_hf.astype(np.float16).astype(np.float32) @ x.astype(np.float16).astype(np.float32), rtol=3e-2, atol=3e-2)


def test_qwen35_a_log_round_trip_is_within_float32(tmp_path):
    p = tmp_path / "m.gguf"
    hf = _hf_qwen35()
    g = _write(p, hf)
    from kyojin_mlx import foreign_linear as FL
    src = FL._read_gguf(str(p), "qwen35")
    for i in range(3):
        a = np.array(src.dense[f"model.layers.{i}.linear_attn.A_log"])
        np.testing.assert_allclose(a, hf[f"model.layers.{i}.linear_attn.A_log"], rtol=0, atol=2e-6)
        np.testing.assert_allclose(-np.exp(a), -np.exp(hf[f"model.layers.{i}.linear_attn.A_log"]), rtol=1e-5)          # the value llama.cpp runs with
        np.testing.assert_allclose(-np.exp(a)[XF.v_tiled_index(2, 6, 1)], g[f"blk.{i}.ssm_a"], rtol=1e-5)               # and the file's own, in its (tiled) order


def test_qwen35_loader_does_not_turn_conv1d_into_a_linear(tmp_path):
    from kyojin_mlx import foreign_linear as FL
    p = tmp_path / "m.gguf"
    _write(p, _hf_qwen35())
    src = FL._read_gguf(str(p), "qwen35")
    assert "model.layers.0.linear_attn.conv1d" not in src.packs and src.dense["model.layers.0.linear_attn.conv1d.weight"].shape == (2 * 16 * 2 + 6 * 16, 1, 4)
    assert not any(k.startswith("model.layers.4") for k in list(src.dense) + list(src.packs)) and set(src.mtp) >= {"mtp.fc.weight", "mtp.norm.weight"}


def test_gguf_has_mtp(tmp_path):
    p, q = tmp_path / "with.gguf", tmp_path / "without.gguf"
    _write(p, _hf_qwen35())
    _write(q, {k: v for k, v in _hf_qwen35(mtp=False).items()}, kv=_kv35(False))
    assert N.gguf_has_mtp(str(p)) is True and N.gguf_has_mtp(str(q)) is False and N.gguf_has_mtp(str(tmp_path / "none")) is False


# ------------------------------------------------------------------------------------------------------------------ tokenizer
def test_qwen35_pre_has_its_own_regex_and_builds(tmp_path):
    from tokenizers import Tokenizer
    from kyojin_mlx import foreign_pre as PRE
    assert PRE.PRE_NAMES["qwen35"][0] == "QWEN35" and PRE.PRE_REGEX["QWEN35"] != PRE.PRE_REGEX["QWEN2"]
    N.write_gguf_tokenizer(_tiny_tokenizer_kv(), str(tmp_path))
    t = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    ids = t.encode("ab<|im_start|><think>").ids
    assert ids[0] == 26
    assert ids[-2:] == [C["V"] - 3, C["V"] - 1]
    tc = json.load(open(tmp_path / "tokenizer_config.json"))
    gc = json.load(open(tmp_path / "generation_config.json"))
    assert tc["chat_template"] == "{{ messages }}" and tc["eos_token"] == "<|im_end|>" and "bos_token" not in tc
    assert gc == {"eos_token_id": [C["V"] - 2]}                      # stop ids from metadata; no sampling values invented


def test_pre_regexes_all_compile_or_are_refused_by_name():
    from tokenizers import Regex
    from kyojin_mlx import foreign_pre as PRE
    bad = []
    for k, (be, ex) in PRE.PRE_REGEX.items():
        for e in ex:
            try:
                Regex(e)
            except Exception:
                bad.append(k)
    assert bad == ["AFMOE"]
    assert all(t in PRE.PRE_REGEX for t, *_ in PRE.PRE_NAMES.values())


def test_llama3_pre_adds_bos_and_ignores_merges(tmp_path):
    from tokenizers import Tokenizer
    toks = [chr(c) for c in range(33, 60)] + ["ab", "<|begin_of_text|>", "<|eot_id|>"]
    kv = {"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "llama-bpe", "tokenizer.ggml.tokens": toks, "tokenizer.ggml.merges": ["a b"] if "a" in toks else [],
          "tokenizer.ggml.token_type": [1] * 28 + [3, 3], "tokenizer.ggml.bos_token_id": 28, "tokenizer.ggml.eos_token_id": 29, "tokenizer.ggml.eot_token_id": 29}
    N.write_gguf_tokenizer(kv, str(tmp_path))
    t = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    assert t.encode("!\"").ids[0] == 28 and t.encode("!\"", add_special_tokens=False).ids[0] != 28
    assert json.load(open(tmp_path / "tokenizer_config.json"))["bos_token"] == "<|begin_of_text|>"
    kv["tokenizer.ggml.add_bos_token"] = False                           # the file's own flag wins over the pre default
    N.write_gguf_tokenizer(kv, str(tmp_path))
    assert Tokenizer.from_file(str(tmp_path / "tokenizer.json")).encode("!\"").ids[0] != 28


@pytest.mark.parametrize("pre", ["gemma4", "afmoe", "no-such-pre"])
def test_tokenizer_refuses_what_it_cannot_build(tmp_path, pre):
    with pytest.raises(GF.GgufError):
        N.write_gguf_tokenizer(dict(_tiny_tokenizer_kv(), **{"tokenizer.ggml.pre": pre}), str(tmp_path))
    with pytest.raises(GF.GgufError):
        N.write_gguf_tokenizer(dict(_tiny_tokenizer_kv(), **{"tokenizer.ggml.model": "wpm"}), str(tmp_path))


QWEN2_SHA = {"generation_config.json": "52664c15568574061ce54c02b7b99f2b3aac5aa69eb633165a34c258d575d146",
             "tokenizer.json": "d2786f1afb35b814d4e0cc51b76a4bcf34801fb7c5861fa96ef5b3d23dd4b709",
             "tokenizer_config.json": "737bcf5c536aae42078b904445f1a0416d35d1f88fc966d65c04aa819ef04fba"}


def test_qwen2_tokenizer_files_are_unchanged(tmp_path):
    """sha256 of the three files the code before this change wrote for the same metadata (foreign_names at origin/cloud/cgguftypes)."""
    toks = [chr(c) for c in range(33, 127)] + ["Ġ", "Ġa", "ab", "<|endoftext|>", "<|im_end|>", "<think>"]
    kv = {"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "qwen2", "tokenizer.ggml.tokens": toks, "tokenizer.ggml.merges": ["a b", "Ġ a"],
          "tokenizer.ggml.token_type": [1] * (len(toks) - 3) + [3, 3, 4], "tokenizer.ggml.eos_token_id": len(toks) - 2,
          "tokenizer.ggml.padding_token_id": len(toks) - 3, "tokenizer.ggml.bos_token_id": len(toks) - 3, "tokenizer.chat_template": "{{x}}"}
    N.write_gguf_tokenizer(kv, str(tmp_path))
    got = {f: hashlib.sha256(open(tmp_path / f, "rb").read()).hexdigest() for f in sorted(os.listdir(tmp_path))}
    assert got == QWEN2_SHA


# ------------------------------------------------------------------------------------------------------------------ dense archs: qwen2, llama
def _inv_names(arch):
    """HF module suffix -> gguf kind, from llama.cpp's own map (the snapshot), not from foreign_names."""
    inv = {}
    for g, hfs in SNAP[arch].items():
        if g.startswith("blk.0."):
            for h in hfs:
                if h.startswith("model.layers.0."):
                    inv[h[len("model.layers.0."):]] = g[len("blk.0."):]
    top = {}
    for g, hfs in SNAP[arch].items():
        if not g.startswith("blk."):
            for h in hfs:
                top[h] = g
    return inv, top


def _permute(w, n_head, n_head_kv):
    """conversion/llama.py LlamaModel.permute, numpy."""
    if n_head_kv is not None and n_head != n_head_kv:
        n_head = n_head_kv
    return w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def _tiny_dense(arch, tmp_path, model_type, cfg_kw, kv_extra, tr=None, vocab=48, alias=None, gname_of=None):
    """Random tiny mlx-lm model -> GGUF (llama.cpp names from the snapshot, `tr` the converter's transform) -> load_foreign. Returns (ref model, loaded model, path)."""
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_lm.utils import _get_classes
    from kyojin_mlx import foreign_linear as FL
    cfg = dict(model_type=model_type, vocab_size=vocab, rms_norm_eps=1e-6, hidden_act="silu", **{"rope_theta": 10000.0, **cfg_kw})
    mc, ac = _get_classes(config=dict(cfg))
    m = mc(ac.from_dict(dict(cfg)))
    rng = np.random.default_rng(0)
    flat = [(k, mx.array((rng.integers(-24, 25, v.shape) / 64).astype(np.float16))) for k, v in tree_flatten(m.parameters())]
    m.update(tree_unflatten(flat))
    mx.eval(m.parameters())
    inv, top = _inv_names(arch)
    tl = []
    allw = {k: np.array(v.astype(mx.float32)) for k, v in flat}
    for k, v in flat:
        w = np.array(v.astype(mx.float32))
        if gname_of is not None and gname_of(k) is not None:
            gname = gname_of(k)
        elif k.rsplit(".", 1)[0] in top:
            gname = top[k.rsplit(".", 1)[0]] + "." + k.rsplit(".", 1)[1]
        else:
            _, _, i, rest = k.split(".", 3)
            mod, sfx = rest.rsplit(".", 1)
            for a_, b_ in (alias or {}).items():
                mod = mod.replace(a_, b_)
            gname = f"blk.{i}.{inv[mod]}.{sfx}"
        if tr is not None:
            w = tr(k, w, allw) if getattr(tr, "wants_all", False) else tr(k, w)
        tl.append((gname, tuple(reversed(w.shape)), "F32", np.ascontiguousarray(w, np.float32)))
    kv = {"general.architecture": arch, f"{arch}.block_count": cfg["num_hidden_layers"], f"{arch}.context_length": 128, f"{arch}.embedding_length": cfg["hidden_size"],
          f"{arch}.feed_forward_length": cfg["intermediate_size"], f"{arch}.attention.head_count": cfg["num_attention_heads"],
          f"{arch}.attention.head_count_kv": cfg["num_key_value_heads"], f"{arch}.attention.layer_norm_rms_epsilon": 1e-6, f"{arch}.rope.freq_base": 10000.0,
          **kv_extra, **{k: v for k, v in _tiny_tokenizer_kv().items() if k != "tokenizer.ggml.pre"}, "tokenizer.ggml.pre": "qwen2"}
    kv["tokenizer.ggml.tokens"] = kv["tokenizer.ggml.tokens"][:vocab] + [f"x{i}" for i in range(max(0, vocab - len(kv["tokenizer.ggml.tokens"])))]
    kv["tokenizer.ggml.token_type"] = (kv["tokenizer.ggml.token_type"] + [1] * vocab)[:vocab]
    p = tmp_path / f"{arch}.gguf"
    SY.write_gguf(p, kv, tl)
    d = N.gguf_serve_dir(str(p), str(tmp_path / "cache"))
    loaded, _ = FL.load_foreign(d, log=lambda *a: None)
    return m, loaded, d


def _same_model(ref, got, tol=3e-2):
    from mlx.utils import tree_flatten
    from kyojin_mlx import foreign_linear as FL
    lin = {n: m for n, m in got.named_modules() if isinstance(m, FL.ForeignLinear)}
    gp = dict(tree_flatten(got.parameters()))
    for k, v in tree_flatten(ref.parameters()):
        if k in gp:
            np.testing.assert_array_equal(np.array(gp[k].astype(mx.float32)), np.array(v.astype(mx.float32)), err_msg=k)
        elif ".switch_mlp." in k:                                         # routed experts live in a ForeignSwitchGLU (dense fp16 stacks here)
            blk, proj = k.split(".switch_mlp.")
            sw = dict(got.named_modules())[blk + ".switch_mlp"]
            w = {"gate_proj.weight": sw.g, "up_proj.weight": sw.u, "down_proj.weight": sw.d}[proj].w
            np.testing.assert_array_equal(np.array(w.astype(mx.float32)), np.array(v.astype(mx.float32)), err_msg=k)
        elif k.endswith(".bias"):                                      # a linear's bias lives inside its ForeignLinear
            np.testing.assert_array_equal(np.array(lin[k[:-5]]._bias.astype(mx.float32)), np.array(v.astype(mx.float32)), err_msg=k)
        else:
            assert k.endswith(".weight") and k[:-7] in lin, k
            np.testing.assert_array_equal(np.array(lin[k[:-7]].dev.w.astype(mx.float32)), np.array(v.astype(mx.float32)), err_msg=k)
    ids = mx.array([[3, 7, 1, 30, 22, 5, 9, 2]])
    try:
        a = np.array(ref(ids).astype(mx.float32))
    except RuntimeError:                                                # CPU gather_mm has no fp16: run the reference in float32 (the weights were compared above)
        ref.set_dtype(mx.float32)
        a = np.array(ref(ids))
    b = np.array(got(ids).astype(mx.float32))
    assert np.abs(a - b).max() <= tol * max(1.0, np.abs(a).max())
    return a


def test_qwen2_gguf_round_trip(tmp_path):
    cfg = dict(hidden_size=64, num_hidden_layers=2, intermediate_size=96, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
               tie_word_embeddings=False)
    ref, got, d = _tiny_dense("qwen2", tmp_path, "qwen2", cfg, {})
    _same_model(ref, got)
    c = json.load(open(os.path.join(d, "config.json")))
    assert c["model_type"] == "qwen2" and c["num_hidden_layers"] == 2 and c["tie_word_embeddings"] is False and c["head_dim"] == 16
    from kyojin_mlx import modeldesc as MD
    assert MD.backend_desc("qwen2") is None and MD.chat_desc("qwen2").server == "qwen"        # no table row: a bare GGUF is routed by serve.backends.load_backend (cggufarch2)
    assert MD.chat_desc("llama") is None and MD.chat_desc("gemma3_text") is None and MD.chat_desc("glm4").server == "glm"     # no chat family / the GLM-5 one: stay refused


def test_llama_gguf_round_trip_undoes_the_q_k_permute(tmp_path):
    cfg = dict(hidden_size=64, num_hidden_layers=2, intermediate_size=96, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
               tie_word_embeddings=False, attention_bias=False, mlp_bias=False)

    def tr(k, w):                                       # LlamaModel.modify_tensors (llama.py:261-265): n_head = 4, n_kv_head = 2 (heads != kv heads on purpose)
        if k.endswith("q_proj.weight"):
            return _permute(w, 4, 4)
        if k.endswith("k_proj.weight"):
            return _permute(w, 4, 2)
        return w
    ref, got, d = _tiny_dense("llama", tmp_path, "llama", cfg, {}, tr=tr)
    a = _same_model(ref, got)
    # without the inverse the same file must not match: serve it with the transform turned off
    old = XF._SETS["llama"]
    XF._SETS["llama"] = lambda name, kv: None
    try:
        from kyojin_mlx import foreign_linear as FL
        bad, _ = FL.load_foreign(d, log=lambda *a: None)
    finally:
        XF._SETS["llama"] = old
    assert np.abs(np.array(bad(mx.array([[3, 7, 1, 30]])).astype(mx.float32)) - a[:, :4]).max() > 1e-2


def test_llama_permute_index_is_the_inverse():
    for rows, h in ((64, 4), (32, 2), (48, 3)):
        w = np.random.default_rng(0).standard_normal((rows, 5))
        stored = _permute(w, h, h)
        x = XF.plan("blk.0.attn_q.weight", "llama", {"llama.attention.head_count": h}, rows)
        np.testing.assert_array_equal(stored[x.rows], w)
    x = XF.plan("blk.0.attn_k.weight", "llama", {"llama.attention.head_count": 4, "llama.attention.head_count_kv": 2}, 32)
    w = np.arange(32.0)[:, None]
    np.testing.assert_array_equal(_permute(w, 4, 2)[x.rows], w)
    assert XF.plan("blk.0.attn_v.weight", "llama", {"llama.attention.head_count": 4}, 32) is None


def test_llama_refuses_rope_freqs_scaling_and_moe(tmp_path):
    p = tmp_path / "x.gguf"
    kv = {"general.architecture": "llama", "llama.block_count": 1, "llama.embedding_length": 8, "llama.attention.head_count": 2, "llama.context_length": 8,
          "llama.feed_forward_length": 8, "llama.attention.head_count_kv": 2}
    for extra in ({"llama.rope.scaling.type": "yarn"}, {"llama.expert_count": 8}, {"llama.rope.dimension_count": 2}):
        SY.write_gguf(p, {**kv, **extra}, [("token_embd.weight", (8, 4), "F32", np.zeros((4, 8), np.float32))])
        with pytest.raises(GF.GgufError):
            N.gguf_config(GF.GgufFile(p))
    with pytest.raises(GF.GgufError):
        N.gguf_name("rope_freqs.weight", "llama")        # Llama 3.1 style rope_freqs tensor: scaling the config cannot express


def _norm_plus1(k, w):
    return w + 1 if k.endswith("norm.weight") else w


def test_glm4_gguf_round_trip_skips_the_nextn_block(tmp_path):
    cfg = dict(hidden_size=64, num_hidden_layers=2, intermediate_size=96, num_attention_heads=4, num_key_value_heads=2, head_dim=16, attention_bias=True,
               partial_rotary_factor=0.5, max_position_embeddings=128, rope_theta=10000.0)
    ref, got, d = _tiny_dense("glm4", tmp_path, "glm4", cfg, {"glm4.rope.dimension_count": 8, "glm4.attention.key_length": 16, "glm4.attention.layer_norm_rms_epsilon": 1e-6,
                                                              "glm4.block_count": 2})
    _same_model(ref, got)
    c = json.load(open(os.path.join(d, "config.json")))
    assert c["partial_rotary_factor"] == 0.5 and c["attention_bias"] is True and c["num_hidden_layers"] == 2 and c["model_type"] == "glm4"
    # a file with a nextn block (block_count 3, nextn 1): the trunk is 2 layers and blk.2.* is skipped, not loaded as layer 2
    from kyojin_mlx import foreign_linear as FL
    g = GF.GgufFile(tmp_path / "glm4.gguf")
    tl = [(n, ti.ne, "F32", np.ascontiguousarray(g.tensor_f32(n))) for n, ti in g.tensors.items()]
    extra = [(n.replace("blk.1.", "blk.2."), ne, t, a) for n, ne, t, a in tl if n.startswith("blk.1.")] + [("blk.2.nextn.eh_proj.weight", (128, 64), "F32", np.zeros((64, 128), np.float32))]
    kv = dict(g.kv, **{"glm4.block_count": 3, "glm4.nextn_predict_layers": 1})
    SY.write_gguf(tmp_path / "n.gguf", kv, tl + extra)
    src = FL._read_gguf(str(tmp_path / "n.gguf"), "glm4")
    assert not any(k.startswith("model.layers.2") for k in list(src.dense) + list(src.packs)) and any("draft-head block" in n for n in src.notes)
    assert N.gguf_config(GF.GgufFile(tmp_path / "n.gguf"))["num_hidden_layers"] == 2


def test_glm4_refuses_mrope_permuted_files(tmp_path):
    p = tmp_path / "g.gguf"
    SY.write_gguf(p, {"general.architecture": "glm4", "glm4.block_count": 1, "glm4.embedding_length": 8, "glm4.attention.head_count": 2, "glm4.context_length": 8,
                      "glm4.feed_forward_length": 8, "glm4.rope.dimension_sections": [4, 4, 0, 0]}, [("token_embd.weight", (8, 4), "F32", np.zeros((4, 8), np.float32))])
    with pytest.raises(GF.GgufError):
        N.gguf_config(GF.GgufFile(p))


def test_gemma3_gguf_round_trip_undoes_the_norm_shift(tmp_path):
    cfg = dict(hidden_size=64, num_hidden_layers=6, intermediate_size=96, num_attention_heads=4, num_key_value_heads=2, head_dim=16, sliding_window=4,
               sliding_window_pattern=6, query_pre_attn_scalar=16.0, rope_theta=1000000.0, rope_local_base_freq=10000.0, max_position_embeddings=128,
               rope_scaling={"type": "linear", "rope_type": "linear", "factor": 2.0})
    kv = {"gemma3.attention.key_length": 16, "gemma3.attention.value_length": 16, "gemma3.attention.sliding_window": 4, "gemma3.rope.freq_base": 1000000.0,
          "gemma3.rope.freq_base_swa": 10000.0, "gemma3.rope.scaling.type": "linear", "gemma3.rope.scaling.factor": 2.0, "gemma3.block_count": 6}
    ref, got, d = _tiny_dense("gemma3", tmp_path, "gemma3_text", cfg, kv, tr=_norm_plus1)
    _same_model(ref, got)
    c = json.load(open(os.path.join(d, "config.json")))
    for k in ("sliding_window", "sliding_window_pattern", "rope_theta", "rope_local_base_freq", "query_pre_attn_scalar", "head_dim"):
        assert c[k] == cfg[k], k
    assert c["rope_scaling"]["factor"] == 2.0 and c["model_type"] == "gemma3_text"
    # the 62-layer model scales queries by 1/sqrt(n_embd / n_head)
    g = GF.GgufFile(tmp_path / "gemma3.gguf")
    big = N._cfg_gemma3(type("G", (), {"kv": dict(g.kv, **{"gemma3.block_count": 62}), "tensors": g.tensors})(), "gemma3")
    assert big["query_pre_attn_scalar"] == 16.0 and N._cfg_gemma3(g, "gemma3")["query_pre_attn_scalar"] == 16.0   # 64 / 4 == head size here: use a distinguishing size below
    big = N._cfg_gemma3(type("G", (), {"kv": dict(g.kv, **{"gemma3.block_count": 62, "gemma3.embedding_length": 128}), "tensors": g.tensors})(), "gemma3")
    assert big["query_pre_attn_scalar"] == 32.0


def test_gemma3_refuses_what_the_class_cannot_run(tmp_path):
    base = {"general.architecture": "gemma3", "gemma3.block_count": 1, "gemma3.embedding_length": 8, "gemma3.attention.head_count": 2, "gemma3.context_length": 8,
            "gemma3.feed_forward_length": 8, "gemma3.attention.key_length": 4}
    for extra, tensors in (({"gemma3.final_logit_softcapping": 30.0}, []), ({"gemma3.rope.scaling.type": "yarn"}, []),
                           ({"gemma3.attention.sliding_window_pattern": [1, 0]}, [])):
        p = tmp_path / "x.gguf"
        SY.write_gguf(p, {**base, **extra}, [("token_embd.weight", (8, 4), "F32", np.zeros((4, 8), np.float32))] + tensors)
        with pytest.raises(GF.GgufError):
            N.gguf_config(GF.GgufFile(p))


def test_qwen2moe_gguf_round_trip(tmp_path):
    cfg = dict(hidden_size=64, num_hidden_layers=2, intermediate_size=96, num_attention_heads=4, num_key_value_heads=2, num_experts=4, num_experts_per_tok=2,
               moe_intermediate_size=32, shared_expert_intermediate_size=48, max_position_embeddings=128, tie_word_embeddings=False)
    kv = {"qwen2moe.expert_count": 4, "qwen2moe.expert_used_count": 2, "qwen2moe.expert_feed_forward_length": 32, "qwen2moe.expert_shared_feed_forward_length": 48}
    ref, got, d = _tiny_dense("qwen2moe", tmp_path, "qwen2_moe", cfg, kv, alias={"mlp.switch_mlp": "mlp.experts"})
    _same_model(ref, got)
    c = json.load(open(os.path.join(d, "config.json")))
    assert c["num_experts"] == 4 and c["shared_expert_intermediate_size"] == 48 and c["model_type"] == "qwen2_moe"
