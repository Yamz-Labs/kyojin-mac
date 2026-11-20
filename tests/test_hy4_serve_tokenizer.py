"""Hy4-flash (`hy_v4_flash`, tokenizer `gpt2` + pre `hy_v4`) on the serve path's tokenizer: foreign_names.gguf_serve_dir builds the directory the server reads
(config from the skeleton spec, tokenizer.json / tokenizer_config.json / generation_config.json from the GGUF metadata), behind KYOJIN_SKELETON and
KYOJIN_KIT_UNPROVEN_ROWS. The real file is compared token for token with llama.cpp's llama-tokenize (vocab-only load) on the corpus of
tools/hy4_tok_oracle.py (tests/data/hy4_tok_oracle.json): with special-token parsing (what the chat server feeds) and without it."""
import importlib.util
import json
import os

import numpy as np
import pytest

from kyojin_mlx import foreign_archkit as K
from kyojin_mlx import foreign_names as FN

HERE = os.path.dirname(__file__)
ORACLE = os.path.join(HERE, "data", "hy4_tok_oracle.json")
START, END, PAD = "<｜hy_start:6124c78e｜>", "<｜hy_end:6124c78e｜>", "<｜hy_pad:6124c78e｜>"


def _e2e():
    spec = importlib.util.spec_from_file_location("hy4_e2e_helpers", os.path.join(HERE, "test_skeleton_hy4_e2e.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _tok_kv(n):
    toks = ["1", "2", "3", "4", "34", "a", "b", "ab", "Ġ", "Ġab", "!", START, END, PAD]
    types = [1] * 11 + [3, 3, 3]
    toks += [f"f{i}" for i in range(n - len(toks))]
    types += [1] * (n - len(types))
    return {"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "hy_v4", "tokenizer.ggml.tokens": toks, "tokenizer.ggml.token_type": types,
            "tokenizer.ggml.merges": ["3 4", "a b", "Ġ ab"], "tokenizer.ggml.bos_token_id": 11, "tokenizer.ggml.eos_token_id": 12,
            "tokenizer.ggml.padding_token_id": 13, "tokenizer.ggml.add_bos_token": False, "tokenizer.ggml.add_eos_token": False,
            "tokenizer.chat_template": "{{ messages }}"}


@pytest.fixture
def flags_on(monkeypatch):
    monkeypatch.setenv("KYOJIN_SKELETON", "1")
    monkeypatch.setenv("KYOJIN_KIT_UNPROVEN_ROWS", "1")


@pytest.fixture
def synth(tmp_path):
    m = _e2e()
    rng = np.random.default_rng(7)
    t = m._model_tensors(rng)
    table = rng.integers(0, 256, (sum(m.PRIMES), m.HD)).astype(np.uint8)
    table[(table & 127) == 127] = 0x38
    kv = {**m._kv(), **_tok_kv(m.V)}
    out = [(m.EG.TABLE, (m.HD, table.shape[0]), "F8_E4M3", table)] + [(n, a.shape[::-1], "F32", a.astype(np.float32)) for n, a in t.items()]
    p = tmp_path / "model" / "hy4_2l.gguf"
    p.parent.mkdir()
    m.SY.write_gguf(str(p), kv, out)
    return m, str(p), str(tmp_path / "cache")


def test_hy4_has_no_kit_table():
    assert not K.have("hy_v4_flash")                                    # the skeleton is its only serve route


def test_serve_dir_refused_with_flags_off(synth, monkeypatch):
    monkeypatch.delenv("KYOJIN_SKELETON", raising=False)
    monkeypatch.delenv("KYOJIN_KIT_UNPROVEN_ROWS", raising=False)
    _, p, cache = synth
    with pytest.raises(K.KitError):
        FN.gguf_serve_dir(os.path.dirname(p), cache=cache)


def test_serve_dir_synthetic(synth, flags_on):
    tk = pytest.importorskip("tokenizers")
    m, p, cache = synth
    out = FN.gguf_serve_dir(os.path.dirname(p), cache=cache)
    assert sorted(os.listdir(out)) == sorted(["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "hy4_2l.gguf"])
    assert "hy_v4_flash" not in FN.ARCHS                                # no loader row registered on the way
    cfg = json.load(open(os.path.join(out, "config.json")))
    assert cfg["gguf_architecture"] == "hy_v4_flash" and cfg["model_type"] == "kyojin_skeleton"
    assert cfg["num_hidden_layers"] == 2 and cfg["hidden_size"] == m.D and cfg["max_position_embeddings"] == 64
    assert (cfg["mla_kv_lora"], cfg["kpool"], cfg["hc_count"], cfg["engram_layers"]) == (m.KVL, m.KP, m.HC, [1])
    assert cfg["moe"]["n_expert"] == m.E and cfg["moe"]["latent"] and cfg["moe"]["clamp_mode"] == "pre_act" and cfg["moe"]["dense_layers"] == [0]
    t = tk.Tokenizer.from_file(os.path.join(out, "tokenizer.json"))
    assert t.encode("1234").ids == [0, 1, 2, 3]                         # hy_v4 splits digits by 3 first: "123" | "4", so the "3 4" merge cannot fire
    assert t.encode(START + "ab ab").ids == [11, 7, 9]
    assert json.load(open(os.path.join(out, "generation_config.json"))) == {"eos_token_id": [12]}
    tc = json.load(open(os.path.join(out, "tokenizer_config.json")))
    assert tc["eos_token"] == END and tc["pad_token"] == PAD and tc["chat_template"] == "{{ messages }}" and "bos_token" not in tc


def test_oracle_corpus_covers_the_cases():
    o = json.load(open(ORACLE))
    texts = [c["text"] for c in o["cases"]]
    assert len(texts) >= 200 and len(set(texts)) == len(texts) and len(o["llama_cpp"]) == 40
    has = lambda f: sum(1 for s in texts if f(s))                       # noqa: E731
    assert has(lambda s: any("\u4e00" <= ch <= "\u9fa5" for ch in s)) >= 20           # Chinese (the hy_v4 CJK split)
    assert has(lambda s: any(ch in "éèàçêôœ" for ch in s)) >= 15                      # French
    assert has(lambda s: any(ord(ch) >= 0x1F300 for ch in s)) >= 20                    # emoji
    assert has(lambda s: "def " in s or "{" in s or "();" in s or "#include" in s) >= 15    # code
    assert has(lambda s: "<｜hy_" in s) >= 15                                          # special tokens
    sp = [c for c in o["cases"] if START in c["text"]]
    assert sp and all(120000 in c["ids"] and 120000 not in c["ids_plain"] for c in sp)  # parsed as one id only with special parsing


def _real():
    p = os.environ.get("KYOJIN_TEST_HY4_GGUF")
    if not p or not os.path.exists(p):
        pytest.skip("KYOJIN_TEST_HY4_GGUF not set (a Hy4-flash GGUF file or directory; only its headers and vocab are read)")
    return p


def test_real_gguf_vs_llama_cpp(flags_on, tmp_path):
    p = _real()
    tk = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    out = FN.gguf_serve_dir(p, cache=str(tmp_path))
    o = json.load(open(ORACLE))
    t = tk.Tokenizer.from_file(os.path.join(out, "tokenizer.json"))
    bad = [c["text"] for c in o["cases"] if t.encode(c["text"], add_special_tokens=False).ids != c["ids"]]
    assert not bad, f"{len(bad)} / {len(o['cases'])} differ from llama-tokenize, first {bad[:3]!r}"
    t.encode_special_tokens = True                                      # control tokens as text (llama-tokenize --no-parse-special)
    bad = [c["text"] for c in o["cases"] if t.encode(c["text"], add_special_tokens=False).ids != c["ids_plain"]]
    assert not bad, f"{len(bad)} differ with --no-parse-special, first {bad[:3]!r}"
    t.encode_special_tokens = False
    assert all(t.decode(c["ids"], skip_special_tokens=False) == c["text"] for c in o["cases"])      # streaming detokenization gives the text back
    ht = transformers.AutoTokenizer.from_pretrained(out)                # what serve.server loads (tokenizer_config.json present)
    assert all(ht.encode(c["text"], add_special_tokens=False) == c["ids"] for c in o["cases"])
    assert ht.eos_token_id == 120025 and ht.pad_token == PAD and ht.bos_token is None
    assert json.load(open(os.path.join(out, "generation_config.json"))) == {"eos_token_id": [120025]}    # llama.cpp special_eog_ids = {eos} for this vocab
    assert "reasoning_effort" in json.load(open(os.path.join(out, "tokenizer_config.json")))["chat_template"]
    cfg = json.load(open(os.path.join(out, "config.json")))
    assert (cfg["gguf_architecture"], cfg["num_hidden_layers"], cfg["vocab_size"], cfg["nextn_predict_layers"]) == ("hy_v4_flash", 42, 120832, 1)
    assert cfg["max_position_embeddings"] == 1048576 and cfg["moe"]["n_expert"] == 512
