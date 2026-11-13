"""Serving an architecture that has no router row, out of the box: fallback backend / chat family on by default (opt-out 0), the per-user attention hook on every
module that holds an attention call (not only type(model).__module__), mlx-lm's own model_type resolution in load_generic, and plain bf16 / fp16 HF packs.
Each case drives the Backend API the engine uses (make_cache, chunked prefill_chunk, fork, a 2-row batched step against the solo step). Tiny random packs only:
synthetic ones written here, and the public tiny-random repos (skipped when they cannot be fetched)."""
import json
import os

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

pytest.importorskip("mlx_lm")
pytest.importorskip("mlx_vlm")
from tools.dayone import engine  # noqa: E402

V = 96
LLAMA = dict(model_type="llama", architectures=["LlamaForCausalLM"], hidden_size=64, num_hidden_layers=4, intermediate_size=128, num_attention_heads=4,
             num_key_value_heads=2, rms_norm_eps=1e-6, vocab_size=V, max_position_embeddings=512, tie_word_embeddings=False)
SYNTH = {
    "mistral": dict(LLAMA, model_type="mistral", architectures=["MistralForCausalLM"]),                       # mlx-lm remaps mistral -> llama
    "smollm3": dict(LLAMA, model_type="smollm3", architectures=["SmolLM3ForCausalLM"], no_rope_layers=[1, 1, 1, 0], no_rope_layer_interval=4,
                    layer_types=["full_attention"] * 4, rope_theta=10000.0),                                # attention module is llama.py's
    "falcon_mamba": dict(model_type="falcon_mamba", architectures=["FalconMambaForCausalLM"], hidden_size=32, num_hidden_layers=2, intermediate_size=64,
                         state_size=8, conv_kernel=4, time_step_rank=2, vocab_size=V, use_bias=False, use_conv_bias=True, layer_norm_epsilon=1e-5,
                         tie_word_embeddings=False, rescale_prenorm_residual=False, residual_in_fp32=True, mixer_rms_eps=1e-6),   # SSM state cache, mlx-lm remap -> mamba
}
HF = {"ministral3": "optimum-intel-internal-testing/tiny-random-ministral3",
      "gemma4": "optimum-intel-internal-testing/tiny-random-gemma4",
      "gemma4_unified": "optimum-intel-internal-testing/tiny-random-gemma4-unified-it",
      "olmo3": "trl-internal-testing/tiny-Olmo3ForCausalLM"}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("KYOJIN_BACKEND_FALLBACK", "KYOJIN_CHAT_FALLBACK"):
        monkeypatch.delenv(k, raising=False)                      # the defaults decide
    monkeypatch.setenv("KYOJIN_WIRED_MB", "0")


def _two_row_step(path, tokens=((3, 5, 7, 9, 11, 13, 15, 17), (4, 6, 8, 10, 12, 14, 16, 18, 20, 22))):
    from kyojin_mlx.serve import backends
    be, cfg = backends.load_backend(path, log=lambda *a: None)
    caches, firsts = [], []
    for p in tokens:
        c = be.make_cache()
        lg = be.prefill_chunk(c, list(p))
        caches.append(c)
        firsts.append(int(mx.argmax(lg[0]).item()))
    solo = [int(mx.argmax(be.step(be.batch([be.fork(c)]), mx.array([t])), axis=-1)[0]) for c, t in zip(caches, firsts)]
    got = [int(t) for t in mx.argmax(be.step(be.batch([be.fork(c) for c in caches]), mx.array(firsts)), axis=-1).tolist()]
    assert got == solo, (got, solo)
    return be


@pytest.mark.parametrize("name", sorted(SYNTH))
def test_no_row_arch_serves_by_default(name, tmp_path):
    """Default env (no KYOJIN_*_FALLBACK): GenericBackend, the Jinja chat family, 2-row step == solo."""
    cfg = SYNTH[name]
    engine.write_pack(cfg, str(tmp_path))
    from kyojin_mlx.serve import autoconfig
    assert autoconfig.read_pack(str(tmp_path)).family is not None
    be = _two_row_step(str(tmp_path))
    assert type(be).__name__ == "GenericBackend"


def test_remap_goes_through_mlx_lm(tmp_path):
    """mistral loads on mlx-lm's llama class (its MODEL_REMAPPING), not on the mlx-vlm class the literal import fell to."""
    engine.write_pack(SYNTH["mistral"], str(tmp_path))
    from kyojin_mlx.serve import backends
    be, _ = backends.load_backend(str(tmp_path), log=lambda *a: None)
    assert type(be.model).__module__ == "mlx_lm.models.llama"


@pytest.mark.parametrize("var", ["KYOJIN_BACKEND_FALLBACK", "KYOJIN_CHAT_FALLBACK"])
def test_fallback_opt_out(var, tmp_path, monkeypatch):
    """0 keeps the old refusals."""
    from kyojin_mlx.serve import backends, chat
    monkeypatch.setenv(var, "0")
    engine.write_pack(SYNTH["mistral"], str(tmp_path))
    if var == "KYOJIN_BACKEND_FALLBACK":
        with pytest.raises(ValueError, match="no serving backend"):
            backends.load_backend(str(tmp_path), log=lambda *a: None)
    else:
        with pytest.raises(ValueError, match="no chat family"):
            chat.family_of("mistral")


def test_fallback_flags_default_on():
    from kyojin_mlx import flags as FL
    assert FL.FLAGS["backend_fallback"].default == "1" and FL.FLAGS["chat_fallback"].default == "1"


def test_hook_patches_the_module_that_holds_the_attention(tmp_path):
    """The attention of smollm3 lives in llama.py: after the backend loads, that module's sdpa takes RowsKV (it used to keep the single-user wrapper)."""
    engine.write_pack(SYNTH["smollm3"], str(tmp_path))
    be = _two_row_step(str(tmp_path))
    import sys
    fn = sys.modules["mlx_lm.models.llama"].scaled_dot_product_attention
    assert getattr(fn, "_kyojin_mu", False) or fn is be.MU._mu_sdpa


def test_plain_bf16_pack_loads_unquantized(tmp_path):
    """A bf16 HF checkpoint without quantization_config (Qwen3 dense) is read by mlx-lm as stored instead of KeyError: quantization_config (EXL3 loader)."""
    cfg = dict(model_type="qwen3", architectures=["Qwen3ForCausalLM"], hidden_size=64, num_hidden_layers=2, intermediate_size=128, num_attention_heads=4,
               num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6, vocab_size=V, max_position_embeddings=512, rope_theta=10000.0, tie_word_embeddings=False)
    model, _, _ = engine.write_pack(cfg, str(tmp_path))
    flat = {k: v.astype(mx.bfloat16) for k, v in tree_flatten(model.parameters())}
    mx.save_safetensors(str(tmp_path / "model.safetensors"), flat, {"format": "pt"})
    assert "quantization_config" not in json.load(open(tmp_path / "config.json"))
    from kyojin_mlx import mlxpack
    assert mlxpack.is_plain_pack(str(tmp_path)) and not mlxpack.is_mlx_pack(str(tmp_path))
    be = _two_row_step(str(tmp_path))
    assert type(be.model).__module__ == "mlx_lm.models.qwen3"
    assert be.model.model.layers[0].self_attn.q_proj.weight.dtype == mx.bfloat16


def test_quantized_config_is_not_plain(tmp_path):
    from kyojin_mlx import mlxpack
    for key in ("quantization", "quantization_config"):
        d = tmp_path / key
        d.mkdir()
        json.dump({"model_type": "qwen3", key: {"bits": 4}}, open(d / "config.json", "w"))
        open(d / "model.safetensors", "wb").close()
        assert not mlxpack.is_plain_pack(str(d))


@pytest.fixture(scope="module")
def hf_pack():
    hub = pytest.importorskip("huggingface_hub")

    def get(name):
        local = os.path.join(os.environ.get("KYOJIN_TINY_PACKS", ""), name)          # a directory of already fetched packs (name -> pack dir)
        if os.environ.get("KYOJIN_TINY_PACKS") and os.path.isdir(local):
            return local
        try:
            return hub.snapshot_download(HF[name])
        except Exception as e:                                    # offline / repo gone
            pytest.skip(f"{HF[name]} not available: {type(e).__name__}")
    return get


@pytest.mark.parametrize("name", sorted(HF))
def test_public_tiny_random_packs(name, hf_pack):
    """ministral3 (llama.py attention under a mistral3 wrapper), gemma4 (KV-sharing layers, rotating windows), gemma4_unified (remap), olmo3."""
    be = _two_row_step(hf_pack(name))
    assert type(be).__name__ == "GenericBackend"


def test_fallback_chat_family_hands_the_special_tokens_to_the_template(tmp_path):
    """The server encodes a rendered prompt without special tokens: a template that prints {{ bos_token }} (Gemma, Llama, Mistral) must see it, else the model reads no BOS."""
    from kyojin_mlx.serve import chat as CH
    (tmp_path / "chat_template.jinja").write_text("{{ bos_token }}{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}{% endfor %}{% if add_generation_prompt %}<model>{% endif %}")

    class Tok:
        chat_template = None
        special_tokens_map = {"bos_token": "<s>", "eos_token": "</s>", "additional_special_tokens": ["<x>"]}
    assert CH.family_of("some_new_arch") == "jinja"
    fam = CH.FAMILIES["jinja"](Tok(), str(tmp_path), None, {})
    r = fam.render({"messages": [{"role": "user", "content": "hi"}]})
    assert r.prompt == "<s><user>hi<model>"
    qwen = CH.FAMILIES["qwen"](Tok(), str(tmp_path), None, {})                # the rows' family is unchanged: no special tokens handed over
    assert qwen.render({"messages": [{"role": "user", "content": "hi"}]}).prompt == "<user>hi<model>"


def test_fallback_chat_family_plain_chat_renders_no_tools(tmp_path):
    """Llama 3.x tests `tools is not none`: a request without tools must render tools=None (transformers' apply_chat_template), not [], else the tool-calling system prompt lands in every chat."""
    from kyojin_mlx.serve import chat as CH
    (tmp_path / "chat_template.jinja").write_text("{% if tools is not none %}<tooluse>{% endif %}{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}{% endfor %}<model>")

    class Tok:
        chat_template = None
        special_tokens_map = {}
    fam = CH.FAMILIES["jinja"](Tok(), str(tmp_path), None, {})
    msgs = [{"role": "user", "content": "hi"}]
    assert fam.render({"messages": msgs}).prompt == "<user>hi<model>"
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]
    assert fam.render({"messages": msgs, "tools": tools}).prompt == "<tooluse><user>hi<model>"


def test_fallback_parser_splits_a_think_block_the_model_opens():
    """SmolLM3 /think: the generation prompt has no <think>, the model writes it. The fallback family moves that block to reasoning; the Qwen family keeps the text."""
    from kyojin_mlx.serve import chat as CH
    for think_open, want in ((True, [("reasoning", "plan it"), ("content", "Answer.")]), (False, [("content", "\n<think>\nplan it\n</think>\nAnswer.")])):
        p = CH.QwenParser(False, None, True, think_open=think_open)
        ev = []
        for piece in ("\n<th", "ink>\npla", "n it\n</think", ">\nAnswer."):
            ev += p.feed(piece)
        ev += p.finish()
        merged = []
        for k, t in ev:
            if merged and merged[-1][0] == k:
                merged[-1] = (k, merged[-1][1] + t)
            else:
                merged.append((k, t))
        assert merged == want, (think_open, merged)


TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]


def _run(parser, pieces):
    ev = []
    for p in pieces:
        ev += parser.feed(p)
    return ev + parser.finish()


@pytest.mark.parametrize("pieces, want_args", [
    (["[TOOL_CALLS]get_weather[ARGS]", '{"city": "Paris"}'], {"city": "Paris"}),                                     # Mistral / Ministral (real Ministral-3 reply)
    (["[TOOL_CALLS][", '{"name": "get_weather", "arguments": {"city": "Paris"}}]'], {"city": "Paris"}),
    (["<|python_tag|>", '{"type": "function", "function": "get_weather", "parameters": {"city": "Paris"}}'], {"city": "Paris"}),   # Llama 3.2 (real reply)
    (['{"name": "get_weather", "parameters": {"city": "Par', 'is"}}'], {"city": "Paris"}),
    (["<|tool_call>call:get_weather{city:<|", '"|>Paris<|"|>}<tool_call|>'], {"city": "Paris"}),                       # Gemma 4 (real reply)
])
def test_fallback_parser_reads_the_call_formats(pieces, want_args):
    from kyojin_mlx.serve import chat as CH
    ev = _run(CH.FormatQwenParser(False, TOOLS, True, think_open=True), pieces)
    starts = [e for e in ev if e[0] == "tool_start"]
    assert len(starts) == 1 and starts[0][3] == "get_weather"
    assert json.loads("".join(e[2] for e in ev if e[0] == "tool_args")) == want_args
    assert not [e for e in ev if e[0] == "content"]


def test_fallback_parser_leaves_other_text_alone():
    from kyojin_mlx.serve import chat as CH
    for text in ('{"name": "other_tool", "parameters": {}}', "[TOOL_CALLS]nope[ARGS]{}", "Use [TOOL_CALLS] to call.", "Hello <|tool"):
        ev = _run(CH.FormatQwenParser(False, TOOLS, True, think_open=True), [text[:5], text[5:]])
        assert "".join(e[1] for e in ev if e[0] == "content") == text and not [e for e in ev if e[0] == "tool_start"]
    ev = _run(CH.FormatQwenParser(False, None, True, think_open=True), ["[TOOL_CALLS]get_weather[ARGS]{}"])    # no tools in the request: plain text
    assert ev == [("content", "[TOOL_CALLS]get_weather[ARGS]{}")]
    ev = _run(CH.FormatQwenParser(False, TOOLS, True, think_open=True), ["Sure.[TOOL_CALLS]get_weather[ARGS]", '{"city": "Rome"}'])
    assert ev[0] == ("content", "Sure.") and [e[3] for e in ev if e[0] == "tool_start"] == ["get_weather"]


def test_tq_skips_layers_without_a_codebook_for_their_head_dim():
    """Gemma 4: sliding layers 256, global layers 512 (no TurboQuant rotation); KV-shared layers have no cache entry of their own."""
    from types import SimpleNamespace as NS
    from kyojin_mlx import kvwin
    layer = lambda d: NS(self_attn=NS(head_dim=d))                             # noqa: E731
    model = NS(model=NS(layers=[layer(256), layer(512), layer(128), layer(96), layer(256)]))
    assert kvwin._tq_unsupported_layers(model, 4, 6, 6) == {1, 3}
    assert kvwin._tq_unsupported_layers(NS(model=NS(layers=[NS(mixer=None)])), 1, 6, 6) == set()     # no head_dim to read: the layer stays on TurboQuant


def test_fallback_parser_splits_the_gemma4_thought_channel():
    from kyojin_mlx.serve import chat as CH
    p = CH.FormatQwenParser(False, TOOLS, True, think_open=True)
    ev = _run(p, ["<|chan", "nel>thought\nPlan ", "it.<channel|>", "<|tool_call>call:get_weather{city:<|\"|>Rome<|\"|>}<tool_call|>"])
    assert ("".join(e[1] for e in ev if e[0] == "reasoning"), [e[3] for e in ev if e[0] == "tool_start"]) == ("Plan it.", ["get_weather"])
    ev = _run(CH.FormatQwenParser(False, None, True, think_open=True), ["<|channel>thought\n<channel|>Hi."])      # thinking off: empty block, then the answer
    assert [e for e in ev if e[0] == "content"] == [("content", "Hi.")]
