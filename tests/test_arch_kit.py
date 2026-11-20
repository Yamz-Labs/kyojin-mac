"""Architecture import kit: the tables generated from llama.cpp (foreign_arch_gen.py) against the hand-written tables of the architectures the engine
already maps, the resolver against the real mlx-lm classes, and one architecture (olmo2) that has no hand-written row, through a synthetic GGUF.
CPU only; own tests only. The generator test needs a llama.cpp checkout (KYOJIN_LLAMACPP_SRC) and is skipped without one."""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
mx.set_default_device(mx.cpu)

from mlx.utils import tree_flatten  # noqa: E402

from kyojin_mlx import foreign_arch_gen as G, foreign_archkit as K, foreign_names as N, foreign_synth as SY  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
LLAMACPP = os.environ.get("KYOJIN_LLAMACPP_SRC", "")

# Hand rows whose HF name the converter does not map onto that GGUF tensor: a rename done by a converter transform or by the mlx-lm class. Complete list.
KNOWN_NOT_CANDIDATE = {
    "qwen35": {"ssm_dt.bias", "blk.<nextn>.nextn.eh_proj.weight", "blk.<nextn>.nextn.enorm.weight", "blk.<nextn>.nextn.hnorm.weight", "blk.<nextn>.nextn.shared_head_norm.weight"},
    "qwen35moe": {"ssm_dt.bias"},
    "deepseek2": {"attn_k_b", "attn_v_b"},
}


def test_generated_tables_cover_llamacpp():
    assert len(G.ARCH_NAMES) >= 169 and set(G.ARCH_TENSORS) == set(G.ARCH_NAMES)
    assert len(G.CONVERTER) >= 161 and len(G.LOADER) >= 161
    assert "{arch}.attention.head_count" in G.WRITER_KEYS["add_head_count"]
    assert G.LLAMA_CPP_COMMIT != "UNKNOWN"
    for a in N.ARCHS:                                   # every architecture the engine maps is in the generated tables
        assert K.have(a), a


@pytest.mark.parametrize("arch", sorted(N.ARCHS))
def test_hand_rows_are_generated_rows(arch):
    d = K.compare_hand(arch, N.ARCHS[arch])
    assert d["unknown_kind"] == []
    assert {x[0] for x in d["not_candidate"]} == KNOWN_NOT_CANDIDATE.get(arch, set())


_BASE = dict(hidden_size=64, num_hidden_layers=2, intermediate_size=96, num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=48, rms_norm_eps=1e-6,
             rope_theta=10000.0, hidden_act="silu", max_position_embeddings=128, tie_word_embeddings=False, attention_bias=False)
_EXTRA = {"qwen3_moe": dict(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32, decoder_sparse_step=1, mlp_only_layers=[], norm_topk_prob=True),
          "qwen2_moe": dict(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32, shared_expert_intermediate_size=32),
          "gemma3_text": dict(query_pre_attn_scalar=16, sliding_window=8, sliding_window_pattern=2), "glm4": dict(partial_rotary_factor=0.5)}
# glm4_moe / deepseek_v3 / qwen3_5(_moe): their tiny configs are nested or need MLA fields; they are covered by the table-level test above only.
_RESOLVABLE = [a for a, A in N.ARCHS.items() if A.model_type not in ("qwen3_5", "qwen3_5_moe", "glm4_moe", "deepseek_v3")]


@pytest.mark.parametrize("arch", _RESOLVABLE)
def test_resolver_reproduces_the_hand_table_on_the_real_mlx_lm_class(arch):
    from mlx_lm.utils import _get_classes
    A = N.ARCHS[arch]
    cfg = dict(model_type=A.model_type, **_BASE, **_EXTRA.get(A.model_type, {}))
    mc, ac = _get_classes(config=dict(cfg))
    names = [k for k, _ in tree_flatten(mc(ac.from_dict(dict(cfg))).parameters())]
    nm = K.resolve(arch, names)                                           # strict: raises on an ambiguous or a missing required tensor
    for kind, hf in A.blk.items():
        assert nm.blk[kind].replace(".switch_mlp.", ".experts.") == "model.layers.{bid}." + hf, kind
    assert set(nm.blk) == set(A.blk)                                      # the kit finds no tensor kind the hand table lacks, for these classes
    for g, hf in A.top.items():
        assert nm.hf(g) == hf


def test_a_converter_that_changes_values_is_refused_until_vetted():
    kv = {"llama.block_count": 2, "llama.embedding_length": 64}
    with pytest.raises(K.KitError, match="permute"):
        K.kit_config("llama", kv)
    with pytest.raises(K.KitError, match="norm_shift"):
        K.kit_config("gemma3", {"gemma3.block_count": 2})
    cfg, _ = K.kit_config("llama", kv, fields={"num_hidden_layers", "hidden_size"}, allow_flags=K.value_transforms("llama"))      # a vetted transform lets it through
    assert cfg["num_hidden_layers"] == 2


def test_a_file_key_nobody_explains_is_refused():
    kv = {"olmo2.block_count": 2, "olmo2.embedding_length": 64, "olmo2.attention.sliding_window": 128, "olmo2.attention.sliding_window_pattern": [True, False]}
    with pytest.raises(K.KitError, match="sliding_window_pattern"):
        K.kit_config("olmo2", kv)


def test_a_tensor_the_class_lacks_is_refused():
    with pytest.raises(K.KitError, match="requires"):
        K.resolve("olmo2", ["model.embed_tokens.weight", "lm_head.weight"])


# olmo2 tensor names as llama.cpp's converter writes them (listing of a real OLMo-2-0425-1B file): written here by hand, not taken from the tables
_O2 = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj", "attn_q_norm": "self_attn.q_norm",
       "attn_k_norm": "self_attn.k_norm", "post_attention_norm": "post_attention_layernorm", "post_ffw_norm": "post_feedforward_layernorm",
       "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}


def test_olmo2_loads_through_the_generated_tables_and_matches_the_source_model(tmp_path):
    from mlx_lm.models import olmo2
    from kyojin_mlx import foreign_archkit_cpu as KC
    H, L, V = 64, 2, 48
    args = olmo2.ModelArgs(model_type="olmo2", hidden_size=H, num_hidden_layers=L, intermediate_size=96, num_attention_heads=4, num_key_value_heads=4, rms_norm_eps=1e-6,
                           vocab_size=V, rope_theta=500000.0, max_position_embeddings=128, tie_word_embeddings=False)
    ref = olmo2.Model(args)
    rng = np.random.default_rng(1)
    flat = dict(tree_flatten(ref.parameters()))
    from mlx.utils import tree_unflatten
    ref.update(tree_unflatten([(k, mx.array((rng.standard_normal(v.shape) * (0.3 if v.ndim > 1 else 0.1) + (1.0 if v.ndim == 1 else 0.0)).astype(np.float32))) for k, v in flat.items()]))
    mx.eval(ref.parameters())
    ws = dict(tree_flatten(ref.parameters()))
    inv = {v: k for k, v in _O2.items()}
    tl = []
    for k, v in ws.items():
        w = np.array(v)
        if k.startswith("model.layers."):
            _, _, i, rest = k.split(".", 3)
            mod, sfx = rest.rsplit(".", 1)
            g = f"blk.{i}.{inv[mod]}.{sfx}"
        else:
            g = {"model.embed_tokens.weight": "token_embd.weight", "model.norm.weight": "output_norm.weight", "lm_head.weight": "output.weight"}[k]
        tl.append((g, tuple(reversed(w.shape)), "F32", np.ascontiguousarray(w, np.float32)))
    kv = {"general.architecture": "olmo2", "olmo2.block_count": L, "olmo2.context_length": 128, "olmo2.embedding_length": H, "olmo2.feed_forward_length": 96,
          "olmo2.attention.head_count": 4, "olmo2.attention.head_count_kv": 4, "olmo2.rope.freq_base": 500000.0, "olmo2.attention.layer_norm_rms_epsilon": 1e-6}
    p = tmp_path / "olmo2.gguf"
    SY.write_gguf(p, kv, tl)
    got, rep = KC.load_cpu(str(p))
    assert rep["model_type"] == "olmo2" and rep["n_tensors"] == len(ws) and rep["names"].ambiguous == {}
    x = mx.array(rng.integers(0, V, (1, 12)).astype(np.int32))
    a, b = np.array(ref(x)), np.array(got(x))
    assert np.abs(a - b).max() < 1e-4 * max(1.0, np.abs(a).max())
    # the serve path: the same tables become a foreign_names.Arch row; it must equal the independent hand listing above and name every tensor like the model's own parameter
    from kyojin_mlx import foreign_archkit_serve as KS
    try:
        assert KS.register(str(p)) == "olmo2"
        A = N.ARCHS["olmo2"]
        assert A.blk == _O2 and A.model_type == "olmo2" and A.xform == "" and not A.exact and not A.dense
        assert {N.gguf_name(g, "olmo2") for g, *_ in tl} == set(ws)
        assert N.gguf_config(__import__("kyojin_mlx.foreign_gguf", fromlist=["x"]).GgufFile(str(p)))["hidden_size"] == H
    finally:
        N.ARCHS.pop("olmo2", None)
        N._CONFIG.pop("kit:olmo2", None)


def test_generator_reproduces_the_committed_module(tmp_path):
    if not LLAMACPP or not os.path.isdir(os.path.join(LLAMACPP, "gguf-py")):
        pytest.skip("no llama.cpp checkout (KYOJIN_LLAMACPP_SRC)")
    out = tmp_path / "gen.py"
    r = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "gen_arch_tables.py"), LLAMACPP, "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]
    want = open(os.path.join(ROOT, "kyojin_mlx", "foreign_arch_gen.py")).read()
    got = out.read_text()
    if f"LLAMA_CPP_COMMIT = {G.LLAMA_CPP_COMMIT!r}" not in got:
        pytest.skip("the checkout is not the commit the module was generated from")
    assert got == want


def test_q8k_activation_rounding_matches_a_port_of_ggml():
    """tools/archkit_logit_check.q8k_round = quantize_row_q8_K_ref (ggml-quants.c): the oracle-side rounding the K-quant CPU dot products apply."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import archkit_logit_check as L
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 512)).astype(np.float32) * np.float32(3.0)
    x[0, 7] = 40.0
    x[2, :256] = 0.0
    want = np.zeros_like(x)
    for r in range(x.shape[0]):
        for b in range(x.shape[1] // 256):
            v = x[r, b * 256:(b + 1) * 256]
            i = int(np.argmax(np.abs(v)))
            if abs(v[i]) == 0:
                continue
            iscale = np.float32(-127.0) / v[i]
            q = np.minimum(127, np.rint(iscale * v)).astype(np.float32)
            want[r, b * 256:(b + 1) * 256] = q * (np.float32(1.0) / iscale)
    got = np.array(L.q8k_round(mx.array(x)))
    assert np.allclose(got, want, rtol=1e-6, atol=1e-7)
    assert len(np.unique(got[1, :256])) <= 255


def test_expression_rows_and_vetted_derivations():
    assert K._plain_hparams("hparams['hidden_size']") == ("hidden_size",)
    assert K._plain_hparams("self.hparams['num_key_value_heads'] if 'num_key_value_heads' in hparams else hparams['num_attention_heads']") == ("num_key_value_heads", "num_attention_heads")
    assert K._plain_hparams("hparams['a'] + os.system('x')") == ()
    assert K._hparam_arith("hparams['hidden_size'] // hparams['num_attention_heads']", {"hidden_size": 1024, "num_attention_heads": 16}) == 64
    assert K._hparam_arith("__import__('os')", {}) is None
    fields = {"hidden_size", "num_hidden_layers", "intermediate_size", "num_attention_heads", "num_key_value_heads", "head_dim", "rms_norm_eps", "query_pre_attn_scalar",
              "attn_logit_softcapping", "final_logit_softcapping", "vocab_size"}
    kv = {"gemma2.context_length": 8192, "gemma2.embedding_length": 2304, "gemma2.block_count": 26, "gemma2.feed_forward_length": 9216, "gemma2.attention.head_count": 8,
          "gemma2.attention.head_count_kv": 4, "gemma2.attention.layer_norm_rms_epsilon": 1e-6, "gemma2.attention.key_length": 256, "gemma2.attention.value_length": 256,
          "gemma2.attn_logit_softcapping": 50.0, "gemma2.final_logit_softcapping": 30.0, "gemma2.attention.sliding_window": 4096}
    cfg, _ = K.kit_config("gemma2", kv, fields, vocab=100)
    assert cfg["query_pre_attn_scalar"] == 256 and cfg["num_key_value_heads"] == 4 and cfg["head_dim"] == 256
    cfg, _ = K.kit_config("gemma2", {**kv, "gemma2.block_count": 46}, fields, vocab=100)
    assert cfg["query_pre_attn_scalar"] == 2304 // 8                                  # the 27B: 1 / sqrt(n_embd / n_head) (src/models/gemma2.cpp)
    with pytest.raises(K.KitError, match="not explained"):
        K.kit_config("gemma2", {**kv, "gemma2.attention.surprise": 1}, fields, vocab=100)
    from kyojin_mlx import foreign_xform as XF
    assert XF.plan("blk.0.post_ffw_norm.weight", K.vetted("gemma2", "xform", ""), kv).post == "minus1" and XF.plan("blk.0.attn_q.weight", "gemma2", kv) is None
