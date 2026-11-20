# SPDX-License-Identifier: Apache-2.0
"""day0_pack on a qwen4_exp safetensors header: ROW NEEDED with the unproven switch off, UNVERIFIED (naming what is unproven) with it on, never READY. Header-only, CPU."""
import json
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import day0_pack as D  # noqa: E402

LM = "model.language_model."


def _pack(tmp_path):
    t, off = {}, 0

    def add(name, dt, shape, nbytes):
        nonlocal off
        t[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + nbytes]}
        off += nbytes
    for e in range(2):
        b = f"{LM}layers.0.mlp.experts.{e}.gate_proj"
        add(b + ".weight", "U8", [8, 32], 256)
        add(b + ".weight_scale", "F8_E4M3", [8, 4], 32)
        add(b + ".weight_scale_2", "F32", [], 4)
    add(LM + "embed_tokens.weight", "BF16", [16, 64], 2048)
    h = json.dumps(t).encode()
    with open(tmp_path / "model-00001-of-00001.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"\0" * off)
    json.dump({"model_type": "qwen4_exp", "architectures": ["Qwen4ExpForConditionalGeneration"], "text_config": {"model_type": "qwen4_exp_text", "hidden_size": 64, "num_hidden_layers": 1}},
              open(tmp_path / "config.json", "w"))
    return str(tmp_path)


def test_never_ready(tmp_path, monkeypatch):
    p = _pack(tmp_path)
    monkeypatch.delenv("KYOJIN_KIT_UNPROVEN_ROWS", raising=False)
    off = D.inspect_pack(p)
    monkeypatch.setenv("KYOJIN_KIT_UNPROVEN_ROWS", "1")
    on = D.inspect_pack(p)
    assert off["verdict"] == "ROW NEEDED" and "KYOJIN_KIT_UNPROVEN_ROWS" in off["why"]
    assert on["verdict"] == "UNVERIFIED" and "Metal identity" in on["why"] and "k % 1024" in on["why"]
