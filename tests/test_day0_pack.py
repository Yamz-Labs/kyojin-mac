"""tools/day0_pack.py + the safetensors / tokenizer / vision / ceiling / serve-command paths of tools/day0_inspect.py, on synthetic headers (CPU, no weight byte read)."""
from __future__ import annotations

import json
import os
import struct
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
mx.set_default_device(mx.cpu)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import day0_inspect as D  # noqa: E402
import day0_pack as PK  # noqa: E402

_SZ = {"F32": 4, "BF16": 2, "F16": 2, "U8": 1, "I32": 4, "U32": 4, "F8_E4M3": 1, "I64": 8}


def st_write(path, tensors):
    """Safetensors file with a real header and zero data: tensors = {name: (dtype, shape)}."""
    hdr, off = {}, 0
    for n, (dt, sh) in tensors.items():
        nb = int(np.prod(sh, dtype=np.int64)) * _SZ[dt]
        hdr[n] = {"dtype": dt, "shape": list(sh), "data_offsets": [off, off + nb]}
        off += nb
    raw = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw + b"\0" * off)


def cfg_qwen(**kw):
    c = dict(model_type="qwen3_5", architectures=["Qwen3_5ForCausalLM"], hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
             head_dim=8, vocab_size=100, full_attention_interval=2)
    c.update(kw)
    return c


def nvfp4(p, out, inn):
    return {p + ".weight": ("U8", (out, inn // 2)), p + ".weight_scale": ("F8_E4M3", (out, inn // 16)), p + ".weight_scale_2": ("F32", ())}


def make_pack(tmp, cfg, tensors, name="pack"):
    d = tmp / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(cfg))
    st_write(str(d / "model.safetensors"), tensors)
    return str(d)


def nv_tensors():
    t = {"model.embed_tokens.weight": ("BF16", (100, 64)), "lm_head.weight": ("BF16", (100, 64)), "model.layers.0.input_layernorm.weight": ("BF16", (64,))}
    t.update(nvfp4("model.layers.0.self_attn.q_proj", 64, 64))
    for e in range(4):
        for pj in ("gate_proj", "up_proj", "down_proj"):
            t.update(nvfp4(f"model.layers.0.mlp.experts.{e}.{pj}", 64, 64))
    t.update(nvfp4("mtp.layers.0.mlp.up_proj", 64, 64))
    return t


# ------------------------------------------------------------------------------------------------------------------------------ item 2
def test_nvfp4_pack_roles_formats_and_verdict(tmp_path):
    p = make_pack(tmp_path, cfg_qwen(num_experts=4, num_experts_per_tok=1), nv_tensors())
    r = PK.inspect_pack(p)
    assert r["pack_family"] == "foreign" and r["backend_row"]["backend"] == "q35"
    fm = {(e["role"], e["format"]): e for e in r["formats"]}
    assert ("routed experts", "nvfp4") in fm and fm[("routed experts", "nvfp4")]["tensors"] == 4 * 3 * 3
    assert ("trunk", "nvfp4") in fm and ("embeddings", "dense_bf16") in fm and ("head", "dense_bf16") in fm and ("mtp", "nvfp4") in fm
    assert r["verdict"] == "READY" and r["mtp_bytes"] > 0
    assert r["bytes_by_role"]["routed experts"] == fm[("routed experts", "nvfp4")]["bytes"]


def test_header_only(tmp_path, monkeypatch):
    p = make_pack(tmp_path, cfg_qwen(), nv_tensors())
    import mmap
    monkeypatch.setattr(mmap, "mmap", lambda *a, **k: (_ for _ in ()).throw(AssertionError("data mapped")))
    info, files = PK.read_headers(p)
    assert len(files) == 1 and info["lm_head.weight"] == ("BF16", (100, 64), 100 * 64 * 2)


def test_foreign_without_row_needs_a_row_and_wrong_backend_needs_model_code(tmp_path):
    p = make_pack(tmp_path, cfg_qwen(model_type="zzznew", architectures=["ZzzForCausalLM"]), nv_tensors(), "a")
    assert PK.inspect_pack(p)["verdict"] == "ROW NEEDED"
    p2 = make_pack(tmp_path, cfg_qwen(model_type="ernie4_5_moe", architectures=["Ernie4_5_MoeForCausalLM"]), nv_tensors(), "b")
    r = PK.inspect_pack(p2)
    assert r["verdict"] == "MODEL CODE NEEDED" and "generic" in r["why"]


def test_exl3_and_mlx_packs(tmp_path):
    t = {"model.embed_tokens.weight": ("BF16", (100, 64))}
    for k in ("trellis",):
        t["model.layers.0.mlp.up_proj." + k] = ("I32", (4, 4, 32))
    t["model.layers.0.mlp.up_proj.suh"] = ("F16", (64,))
    t["model.layers.0.mlp.up_proj.svh"] = ("F16", (64,))
    c = cfg_qwen(quantization_config={"quant_method": "exl3", "version": "1.4"})
    r = PK.inspect_pack(make_pack(tmp_path, c, t, "e"))
    assert r["pack_family"] == "exl3" and ("trunk", "exl3") in {(e["role"], e["format"]) for e in r["formats"]} and r["verdict"] == "READY"
    r = PK.inspect_pack(make_pack(tmp_path, cfg_qwen(model_type="zzznew", quantization_config={"quant_method": "exl3"}), t, "e2"))
    assert r["verdict"] == "MODEL CODE NEEDED"
    m = {"model.layers.0.mlp.up_proj.weight": ("U32", (64, 8)), "model.layers.0.mlp.up_proj.scales": ("BF16", (64, 1)), "model.layers.0.mlp.up_proj.biases": ("BF16", (64, 1))}
    r = PK.inspect_pack(make_pack(tmp_path, cfg_qwen(quantization={"group_size": 64, "bits": 4}), m, "m"))
    assert r["pack_family"] == "mlx" and {e["format"] for e in r["formats"]} == {"mlx_quant"} and r["verdict"] == "READY"
    r = PK.inspect_pack(make_pack(tmp_path, cfg_qwen(model_type="zzznew", quantization={"group_size": 64, "bits": 4}), m, "m2"))
    assert r["verdict"] == "MODEL CODE NEEDED" and "no modeldesc row and no mlx_lm" in r["why"]


def test_type_needed_and_does_not_fit(tmp_path):
    t = {"model.layers.0.mlp.up_proj.weight": ("U8", (64, 32))}                       # integer payload with no scale: GPTQ / AWQ style
    r = PK.inspect_pack(make_pack(tmp_path, cfg_qwen(), t, "g"))
    assert r["verdict"] == "TYPE NEEDED"
    p = make_pack(tmp_path, cfg_qwen(), nv_tensors(), "f")
    assert PK.inspect_pack(p, limit_gib=1e-9)["verdict"] == "DOES NOT FIT"


def test_pack_roofline_by_hand(tmp_path):
    t = nv_tensors()
    info = {k: (v[0], v[1], int(np.prod(v[1], dtype=np.int64)) * _SZ[v[0]]) for k, v in t.items()}
    c = cfg_qwen(num_experts=4, num_experts_per_tok=1)
    rl = PK.roofline_pack(info, c, 100.0, None)
    exp = sum(nb for n, (_, _, nb) in info.items() if ".experts." in n)
    assert rl["parts"]["routed_experts"] == exp * 1 / 4
    assert rl["parts"]["head"] == 100 * 64 * 2 and rl["parts"]["embedding_row"] == 64 * 2
    assert rl["draft_head_bytes"] == sum(nb for n, (_, _, nb) in info.items() if n.startswith("mtp."))
    assert any("no --ctx" in x for x in rl["terms_not_counted"]) and "ceiling_with_kv_tok_s" not in rl


# ------------------------------------------------------------------------------------------------------------------------------ item 3
def test_kv_term_hand_numbers():
    c = cfg_qwen()                               # 4 layers, interval 2 -> 2 full layers; per token per layer 2 * 2 heads * 8 dims * 2 bytes = 64
    k = PK.kv_term_pack(c, 100)
    assert k["bytes"] == 2 * 100 * 64 and k["full_layers"] == 2
    c = cfg_qwen(full_attention_interval=None, layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"], sliding_window=10)
    assert PK.kv_term_pack(c, 100)["bytes"] == (2 * 100 + 2 * 10) * 64
    c = cfg_qwen(full_attention_interval=None, kv_lora_rank=32, qk_rope_head_dim=8)
    assert PK.kv_term_pack(c, 10)["bytes"] == 4 * 10 * (32 + 8) * 2
    assert "error" in PK.kv_term_pack(cfg_qwen(index_n_heads=4), 10)
    assert "error" in PK.kv_term_pack({"hidden_size": 8}, 10)


def test_ceiling_adds_kv_and_lists_terms(tmp_path):
    p = make_pack(tmp_path, cfg_qwen(num_experts=4, num_experts_per_tok=1), nv_tensors())
    info, _ = PK.read_headers(p)
    a = PK.roofline_pack(info, cfg_qwen(num_experts=4, num_experts_per_tok=1), 100.0, 1000)
    assert a["bytes_per_step_with_kv"] == a["bytes_per_step"] + 2 * 1000 * 64
    assert a["ceiling_with_kv_tok_s"] < a["ceiling_tok_s"]
    assert any("KV read at ctx 1000" in x for x in a["terms_counted"]) and any("draft head declared" in x for x in a["terms_counted"])
    text = D.render_pack(dict(D.inspect_pack(p, bw_gbs=100.0, ctx=1000, tokenizer=False)))
    assert "+ KV / attention read at ctx 1000" in text and "NOT counted:" in text


# ------------------------------------------------------------------------------------------------------------------------------ item 1
def tok_dir(d, template, extra_cfg=None):
    from tokenizers import Tokenizer, models, pre_tokenizers
    vocab = {w: i for i, w in enumerate(["<unk>", "hello", "world", "what", "is", "the", "weather", "in", "paris", "<|im_end|>"])}
    t = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    t.save(str(d / "tokenizer.json"))
    tc = {"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "<|im_end|>", "unk_token": "<unk>"}
    if template is not None:
        tc["chat_template"] = template
    (d / "tokenizer_config.json").write_text(json.dumps(tc))


TPL = "{% if tools %}TOOLS:{% for t in tools %}{{ t.function.name }} {% endfor %}\n{% endif %}{% for m in messages %}<{{ m.role }}>{{ m.content }}\n{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}"


def test_tokenizer_ok_tools_and_failures(tmp_path):
    d = tmp_path / "t1"
    d.mkdir()
    tok_dir(d, TPL)
    cfg = dict(model_type="zzznew", vocab_size=10)
    r = PK.check_tokenizer(str(d), cfg)
    assert r["verdict"] == "TOKENIZER OK" and r["family"] == "jinja" and r["tools_in_prompt"] is True
    d2 = tmp_path / "t2"
    d2.mkdir()
    tok_dir(d2, "{% for m in messages %}{{ m.role }}:{{ m.content }}\n{% endfor %}")
    r = PK.check_tokenizer(str(d2), cfg)
    assert r["verdict"] == "TOKENIZER OK" and r["tools_in_prompt"] is False and "ignores a tools list" in r["why"]
    d3 = tmp_path / "t3"
    d3.mkdir()
    tok_dir(d3, "{% for m in messages %}{{ m.nonexistent.deep.x() }}{% endfor %}{% if %}")
    assert PK.check_tokenizer(str(d3), cfg)["verdict"] == "TEMPLATE FAILS"
    d4 = tmp_path / "t4"
    d4.mkdir()
    tok_dir(d4, None)
    r = PK.check_tokenizer(str(d4), cfg)
    assert r["verdict"] == "TEMPLATE FAILS" and "template" in r["why"].lower()
    d5 = tmp_path / "t5"
    d5.mkdir()
    r = PK.check_tokenizer(str(d5), cfg)
    assert r["verdict"] == "TOKENIZER NEEDED" and "--tokenizer" in r["why"]


def test_gguf_tokenizer_from_metadata(tmp_path):
    from kyojin_mlx import foreign_synth as SY
    kv = {"general.architecture": "zzznotanarch", "tokenizer.ggml.model": "wpm", "tokenizer.ggml.tokens": ["a", "b"], "tokenizer.ggml.token_type": [1, 1]}
    r = PK.check_tokenizer(None, {"model_type": "zzznotanarch"}, gguf_kv=kv, family="generic")
    assert r["verdict"] == "TOKENIZER NEEDED" and "not mapped" in r["why"]
    r = PK.check_tokenizer(None, {}, gguf_kv={"general.architecture": "x"}, family="generic")
    assert r["verdict"] == "TOKENIZER NEEDED" and "no tokenizer.ggml.tokens" in r["why"]
    kv = {"general.architecture": "zzznotanarch", "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "no-such-pre", "tokenizer.ggml.tokens": ["a", "b"], "tokenizer.ggml.token_type": [1, 1]}
    r = PK.check_tokenizer(None, {}, gguf_kv=kv, family="generic")
    assert r["verdict"] == "TOKENIZER NEEDED" and "no-such-pre" in r["why"]
    assert SY is not None


def test_gguf_byte_level_tokenizer_and_template(tmp_path):
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))       # the GPT-2 byte-to-unicode map
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    toks = [chr(c) for _, c in sorted(zip(bs, cs))] + ["<|im_end|>"]
    kv = {"general.architecture": "zzznotanarch", "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "qwen2", "tokenizer.ggml.tokens": toks,
          "tokenizer.ggml.token_type": [1] * 256 + [3], "tokenizer.ggml.merges": [], "tokenizer.ggml.eos_token_id": 256, "tokenizer.chat_template": TPL}
    r = PK.check_tokenizer(None, {"model_type": "zzznotanarch"}, gguf_kv=kv, family="generic")
    assert r["verdict"] == "TOKENIZER OK" and r["family"] == "generic" and r["tools_in_prompt"] is True, r
    assert r["gguf_tokenizer"]["pre"] == "qwen2"


# ------------------------------------------------------------------------------------------------------------------------------ item 4
def test_vision_verdicts(tmp_path):
    t = nv_tensors()
    t["model.visual.blocks.0.attn.qkv.weight"] = ("BF16", (8, 8))
    c = cfg_qwen(architectures=["Qwen3_5ForConditionalGeneration"], vision_config={"depth": 1})
    r = D.inspect_pack(make_pack(tmp_path, c, t), tokenizer=False)
    assert r["vision"]["verdict"].startswith("vision supported") and r["vision"]["vision_parts"]["tower_tensors"] == ["visual"] and "UNVERIFIED" in r["vision"]["why"]
    assert "visual" in r["side_tensors"] or r["side_tensors"].get("vision")
    v = PK.check_vision("safetensors", dict(model_type="gpt_oss", architectures=["GptOssForCausalLM"], vision_config={"depth": 1}), ["x.weight"], [], ["preprocessor_config.json"])
    assert v["verdict"] == "text only loads" and "KYOJIN_VISIONGEN" in v["why"]
    v = PK.check_vision("gguf", None, ["blk.0.attn_k.weight", "v.blk.0.attn_q.weight", "mm.0.weight"], ["mmproj-x.gguf"], [])
    assert v["verdict"] == "text only loads" and v["vision_parts"]["mmproj_files"] == ["mmproj-x.gguf"] and v["vision_parts"]["tower_tensors"] == ["mm", "v"]
    assert PK.check_vision("safetensors", cfg_qwen(), ["a.weight"], [], [])["verdict"] == "text only"


# ------------------------------------------------------------------------------------------------------------------------------ item 5
def test_serve_command(tmp_path):
    p = make_pack(tmp_path, cfg_qwen(), nv_tensors())
    s = PK.serve_command(p, 8391, 4096, "q35", "READY", dict(verdict="TOKENIZER OK"))
    assert s["command"].endswith(f"-m kyojin_mlx.serve --model {os.path.realpath(p)} --port 8391 --ctx 4096") and s["runnable"] and not s["unknown"]
    assert s["derived"]["ctx"] == 4096
    s = PK.serve_command(p, 8391, None, "q35", "READY", dict(verdict="TOKENIZER NEEDED"))
    assert not s["runnable"] and any("--tokenizer" in u and "UNKNOWN" in u for u in s["unknown"])
    s = PK.serve_command(p, 8391, None, "q35", "ROW NEEDED", None)
    assert not s["runnable"] and "not claimed" in s["note"]
    s = PK.serve_command(p, 8391, None, "vlm", "READY", None)
    assert s["unknown"] == [] or "vlm_pkg" in s["unknown"][0]


def test_cli_pack_json_and_text(tmp_path, capsys):
    p = make_pack(tmp_path, cfg_qwen(num_experts=4, num_experts_per_tok=1), nv_tensors())
    assert D.main([p, "--roofline", "--bw-gbs", "100", "--ctx", "512", "--serve-cmd", "--no-tokenizer"]) == 0
    out = capsys.readouterr().out
    assert "VERDICT   READY" in out and "serve     " in out and "KV / attention read at ctx 512" in out and "vision    text only" in out
    assert D.main([p, "--json", "--no-tokenizer"]) == 0
    assert json.loads(capsys.readouterr().out)["pack_family"] == "foreign"
    assert D.is_pack(p) and not D.is_pack(str(tmp_path))


# ------------------------------------------------------------------------------------------------------------------------------ GGUF side of items 1, 3, 4, 5
def test_gguf_ctx_term_mmproj_and_serve(tmp_path):
    import test_day0_inspect as TD
    p, _ = TD._qwen35(tmp_path)
    (tmp_path / "mmproj-m.gguf").write_bytes(b"x")                      # side file: never opened as a model
    a = D.inspect(str(tmp_path), bw_gbs=100.0, ctx=1000, serve_cmd=True)
    b = D.inspect(str(tmp_path), bw_gbs=100.0, ctx=2000)
    ka, kb = a["roofline"]["kv"], b["roofline"]["kv"]
    assert ka["bytes"] > 0 and kb["bytes"] == 2 * ka["bytes"]
    assert a["roofline"]["bytes_per_step_with_kv"] == a["roofline"]["bytes_per_step"] + ka["bytes"]
    assert a["vision"]["verdict"] == "text only loads" and a["vision"]["vision_parts"]["mmproj_files"] == ["mmproj-m.gguf"]
    assert "--model" in a["serve"]["command"] and "--ctx 1000" in a["serve"]["command"]
    assert a["roofline"]["draft_head_bytes"] > 0 and a["mtp_layers"] == 1
    assert "KV / attention read at ctx 1000" in D.render(a)
    assert D.inspect(str(p), bw_gbs=100.0)["roofline"]["kv"] is None
