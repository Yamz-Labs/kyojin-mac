import json, struct, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import hf_slice as S


def test_keep_selects_embedding_head_norm_and_first_blocks_only():
    ks = ["model.language_model.embed_tokens.weight", "lm_head.weight", "model.language_model.norm.weight", "model.language_model.layers.3.mlp.experts.down_proj",
          "model.language_model.layers.4.mlp.experts.down_proj", "model.visual.blocks.0.attn.qkv.weight", "mtp.fc.weight", "mtp.layers.0.self_attn.q_proj.weight"]
    assert [k for k in ks if S.keep(k, 4)] == ks[:4]


def test_slice_writes_a_valid_safetensors_from_range_reads(tmp_path, monkeypatch):
    import types, huggingface_hub as H
    names = {"model.language_model.embed_tokens.weight": [4, 8], "model.language_model.layers.0.mlp.gate.weight": [2, 8], "model.language_model.layers.1.mlp.gate.weight": [2, 8]}
    blob, hdr, off = b"", {}, 0
    for k, sh in names.items():
        raw = bytes((off + i) % 251 for i in range(sh[0] * sh[1] * 2)); hdr[k] = {"dtype": "BF16", "shape": sh, "data_offsets": [off, off + len(raw)]}; blob += raw; off += len(raw)
    h = json.dumps(hdr).encode(); shard = struct.pack("<Q", len(h)) + h + blob
    monkeypatch.setattr(S, "_get", lambda url, lo, hi: shard[lo:hi + 1])
    files = {"config.json": json.dumps({"text_config": {"num_hidden_layers": 2, "layer_types": ["a", "b"], "mtp_num_hidden_layers": 1}, "vision_config": {}}),
             "model.safetensors.index.json": json.dumps({"weight_map": {k: "s0.safetensors" for k in names}})}
    def dl(repo, f, **kw):
        p = tmp_path / "dl" / f; p.parent.mkdir(exist_ok=True); p.write_text(files[f])
        if kw.get("local_dir"): (Path(kw["local_dir"]) / f).write_text(files[f])
        return str(p)
    info = types.SimpleNamespace(sha="abc", siblings=[types.SimpleNamespace(rfilename=f) for f in files])
    monkeypatch.setattr(S, "hf_hub_download", dl); monkeypatch.setattr(S, "hf_hub_url", lambda r, f, revision=None: "u")
    monkeypatch.setattr(H.HfApi, "model_info", lambda self, repo, revision=None: info)
    out = tmp_path / "out"
    r = S.slice_repo("x/y", out, 1)
    d = (out / "model.safetensors").read_bytes(); n = struct.unpack("<Q", d[:8])[0]; hh = json.loads(d[8:8 + n])
    assert sorted(k for k in hh if k != "__metadata__") == ["model.language_model.embed_tokens.weight", "model.language_model.layers.0.mlp.gate.weight"]
    a, b = hh["model.language_model.layers.0.mlp.gate.weight"]["data_offsets"]
    assert d[8 + n + a:8 + n + b] == blob[64:64 + 32] and r["layers"] == 1
    c = json.loads((out / "config.json").read_text())
    assert c["text_config"]["num_hidden_layers"] == 1 and c["text_config"]["layer_types"] == ["a"] and "vision_config" not in c
