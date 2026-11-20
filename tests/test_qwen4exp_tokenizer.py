"""qwen4exp's tokenizer: `gpt2` byte-level BPE with pre-tokenizer `qwen35`, built from the GGUF metadata by the existing path (foreign_names.write_gguf_tokenizer)."""
import json
import os

import numpy as np
import pytest

from kyojin_mlx import foreign_names as FN
from kyojin_mlx import foreign_pre as PRE


def _kv():
    toks = ["a", "b", "Ġ", "ab", "Ġab", "<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    return {"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "qwen35", "tokenizer.ggml.tokens": toks, "tokenizer.ggml.token_type": np.array([1, 1, 1, 1, 1, 3, 3, 3]),
            "tokenizer.ggml.merges": ["a b", "Ġ ab"], "tokenizer.ggml.eos_token_id": 7, "tokenizer.ggml.padding_token_id": 5, "tokenizer.ggml.bos_token_id": 5,
            "tokenizer.ggml.add_bos_token": False, "tokenizer.chat_template": "{{ messages }}"}


def test_pre_type_is_known():
    assert "qwen35" in PRE.PRE_NAMES and PRE.PRE_NAMES["qwen35"][0] == "QWEN35" and "qwen35" in FN._NFC_PRES


def test_builds_from_synthetic_metadata(tmp_path):
    tk = pytest.importorskip("tokenizers")
    FN.write_gguf_tokenizer(_kv(), str(tmp_path))
    t = tk.Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    assert t.encode("<|im_start|>ab ab").ids == [6, 3, 4]                       # the special token is one id; "ab" and " ab" merge
    assert json.load(open(tmp_path / "generation_config.json")) == {"eos_token_id": [7]}
    assert json.load(open(tmp_path / "tokenizer_config.json"))["eos_token"] == "<|im_end|>"


def test_real_shard1_if_present():
    p = os.environ.get("KYOJIN_TEST_Q4EXP_SHARD1")
    if not p or not os.path.exists(p):
        pytest.skip("KYOJIN_TEST_Q4EXP_SHARD1 not set (shard 1 of a Qwen3.8-Flash-Next GGUF, ~11 MB: it holds the tokenizer)")
    import tempfile
    tk = pytest.importorskip("tokenizers")
    from kyojin_mlx import foreign_gguf as GF
    g = GF.GgufFile(p, _shard=True)
    with tempfile.TemporaryDirectory() as d:
        FN.write_gguf_tokenizer(g.kv, d)
        t = tk.Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
        assert g.kv["tokenizer.ggml.pre"] == "qwen35" and len(g.kv["tokenizer.ggml.tokens"]) == 248320
        assert t.encode("<|im_start|>user\nhi<|im_end|>").ids[0] == 248045 and t.encode("<|im_end|>").ids == [248046]
        assert json.load(open(os.path.join(d, "generation_config.json"))) == {"eos_token_id": [248046]}
