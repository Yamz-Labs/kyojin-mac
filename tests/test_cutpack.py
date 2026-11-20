import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import cutpack as CP  # noqa: E402


def _write(path, tensors):
    hdr, off, blobs = {}, 0, []
    for k, (dt, shape, b) in tensors.items():
        hdr[k] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    hdr["__metadata__"] = {"format": "pt"}
    hb = json.dumps(hdr).encode()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb + b"".join(blobs))


def _src(tmp):
    s = tmp / "src"
    names = {"model.embed.weight": 0}
    for L in (0, 1, 2, 3, 45):
        names[f"model.language_model.layers.{L}.mlp.w"] = L
        names[f"model.language_model.layers.{L}.mlp.w_scale"] = L
    names["model.visual.blocks.0.w"] = 9
    names["model.language_model.layers.1.odd"] = 1
    rng = np.random.default_rng(0)
    a, b = {}, {}
    for i, (k, L) in enumerate(names.items()):
        n = 5 if k.endswith("odd") else 64
        t = ("U8", [n], rng.integers(0, 255, n, dtype=np.uint8).tobytes()) if i % 2 else ("F32", [n // 4 or 1], rng.random(n // 4 or 1, dtype=np.float32).tobytes())
        (a if i % 3 else b)[k] = t
    _write(str(s / "model-00001-of-00002.safetensors"), a)
    _write(str(s / "tensors" / "x.safetensors"), b)
    wm = {k: "model-00001-of-00002.safetensors" for k in a} | {k: "tensors/x.safetensors" for k in b}
    json.dump({"metadata": {}, "weight_map": wm}, open(s / "model.safetensors.index.json", "w"))
    json.dump({"text_config": {"num_hidden_layers": 46, "layer_types": list("abcdef") * 8, "mlp_layer_types": list("abcdef") * 8, "x": 1}}, open(s / "config.json", "w"))
    (s / "tokenizer.json").write_text("{}")
    return s, wm


def test_cutpack(tmp_path):
    s, wm = _src(tmp_path)
    out = tmp_path / "cut"
    CP.build(str(s), str(out), 2, verify=True, log=lambda *a: None)
    idx = json.load(open(out / "model.safetensors.index.json"))["weight_map"]
    keep = CP.keep_fn(2)
    assert set(idx) == {k for k in wm if keep(k)}
    assert any(".layers.45." in k for k in idx) and not any("visual" in k or ".layers.3." in k for k in idx)
    cfg = json.load(open(out / "config.json"))["text_config"]
    assert cfg["num_hidden_layers"] == 2 and len(cfg["layer_types"]) == 2 and cfg["x"] == 1
    assert (out / "tokenizer.json").exists() and not (out / "tensors").exists()
    m1 = os.stat(out / "cutpack_manifest.json").st_mtime_ns
    CP.build(str(s), str(out), 2, log=lambda *a: None)                       # idempotent: untouched
    assert os.stat(out / "cutpack_manifest.json").st_mtime_ns == m1
    CP.build(str(s), str(tmp_path / "cutnm"), 2, mtp=False, verify=True, log=lambda *a: None)
    assert not any(".layers.45." in k for k in json.load(open(tmp_path / "cutnm" / "model.safetensors.index.json"))["weight_map"])
    os.utime(s / "model-00001-of-00002.safetensors", (1, 1))                # source changed -> stale
    msgs = []
    CP.build(str(s), str(out), 2, log=msgs.append)
    assert any("stale" in m for m in msgs)
