"""Pack metadata: everything the later stages need, read from the pack directory (no GPU, no weights loaded).

Port of the CUDA/ROCm version (tools/qualify/pack.py, same keys): the flags, the chat template and the draft depth are derived from the pack.
Per-arch knowledge is the FAMILIES table: which server family, how the KV cache is chosen (kvwin affine pool on the full-attention layers of the
Qwen3.5 hybrid; the native cache of DS4.1 / GLM-5.3 has no KV-bit choice) and which streamed BF16 reference adapter exists (refs/<name>.py).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

KVWIN_GROUP = 64
KVWIN_SINK, KVWIN_WINDOW = 4, 128             # the fp16 overlay of KYOJIN_KVWIN=kbits:vbits:group:sink:window (feat/kvwin defaults)
KV_SCALE_OVERHEAD_BITS = 32 / KVWIN_GROUP     # affine pool: fp16 scale + fp16 bias per group of 64 = 0.5 bit per element

# architectures -> family, derived from the model table (kyojin_mlx/modeldesc.py: one row per architecture, nothing to edit here).
# kv: "kvwin" = quantized pool chosen by the Q6 rule; "native" = the model's own cache (no bits to choose).
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kyojin_mlx import modeldesc as _md          # mlx-free

FAMILIES = {arch: dict(family=d.family, server=d.server, ref=d.ref, kv=d.kv, vision=vision, row=d)
            for d in _md.DESCS for arch, vision in d.architectures.items()}
SERVER_MODULE = "kyojin_mlx.serve.server"


def _text_cfg(cfg: dict) -> dict:
    return cfg.get("text_config") or cfg


def kv_layers(t: dict) -> int:
    """Number of layers that hold a growing KV cache (full attention only)."""
    n = t["num_hidden_layers"]
    lt = t.get("layer_types")
    if lt:
        return sum(1 for x in lt if x in ("full_attention", "attention"))
    iv = t.get("full_attention_interval")
    return n // iv if iv else n


def kv_bytes_per_token(meta: dict, k_bits: int, v_bits: int) -> float | None:
    """Bytes of KV per token of context. None for the native caches (DS4.1 ring + compressed pools, GLM MLA): the max-ctx stage measures instead."""
    if meta.get("kv_mode") != "kvwin":
        return None
    elems = meta["kv_layers"] * meta["kv_heads"] * meta["head_dim"]
    return elems * ((k_bits + KV_SCALE_OVERHEAD_BITS) + (v_bits + KV_SCALE_OVERHEAD_BITS)) / 8


def weights_dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.glob("*.safetensors"))


def template_source(p: Path, arch: str) -> str | None:
    """Where the chat template lives: the pack's own jinja file, tokenizer_config.json, or the family's built-in renderer (DS4.1 encoder)."""
    if (p / "chat_template.jinja").exists():
        return "chat_template.jinja"
    tc = p / "tokenizer_config.json"
    if tc.exists() and json.loads(tc.read_text()).get("chat_template"):
        return "tokenizer_config.json"
    row = _md.by_architecture(arch)
    if row is not None and row.builtin_template:
        return "builtin:" + row.builtin_template
    return None


def read_pack(path: str | os.PathLike) -> dict:
    p = Path(path).expanduser()
    cfg = json.loads((p / "config.json").read_text())
    t = _text_cfg(cfg)
    arch = (cfg.get("architectures") or ["?"])[0]
    fam = FAMILIES.get(arch, dict(family="unknown", server=None, ref=None, kv="native", vision=False))
    heads = t["num_attention_heads"]
    qc = {}
    qf = p / "quantization_config.json"
    if qf.exists():
        qc = json.loads(qf.read_text())
    qc = qc or cfg.get("quantization_config") or {}
    idx = p / "model.safetensors.index.json"
    if idx.exists():
        ih = hashlib.sha1(idx.read_bytes()).hexdigest()[:12]
    else:                                # single-file packs (no index): name + size of the shards stand for it
        ih = hashlib.sha1("|".join(f"{f.name}:{f.stat().st_size}" for f in sorted(p.glob("*.safetensors"))).encode()).hexdigest()[:12]
    meta = dict(
        name=p.name, path=str(p), arch=arch, family=fam["family"], server_family=fam["server"], ref_adapter=fam["ref"], kv_mode=fam["kv"],
        model_type=t.get("model_type"), layers=t["num_hidden_layers"], kv_layers=kv_layers(t),
        heads=heads, kv_heads=t.get("num_key_value_heads", heads), head_dim=t.get("head_dim") or t["hidden_size"] // heads,
        hidden=t["hidden_size"], max_pos=t.get("max_position_embeddings"), vocab=t.get("vocab_size"),
        quant=qc.get("quant_method"), bpw=qc.get("bits"), head_bits=qc.get("head_bits"),
        vision=bool(fam["vision"] and cfg.get("vision_config")),
        mtp_layers=int(t.get("mtp_num_hidden_layers") or t.get("num_nextn_predict_layers") or 0),
        weights_bytes=weights_dir_bytes(p), index_sha=ih, license=_license(p), template=template_source(p, arch),
        has_index=idx.exists(),
    )
    meta["problems"] = problems(meta)
    return meta


def _license(p: Path) -> str | None:
    for f in ("LICENSE", "LICENSE.md", "LICENSE.txt"):
        if (p / f).exists():
            return f
    rd = p / "README.md"
    if rd.exists():
        for line in rd.read_text(errors="replace").splitlines()[:20]:
            if line.lower().startswith("license:"):
                return line.split(":", 1)[1].strip()
    return None


def problems(meta: dict) -> list[str]:
    out = []
    if meta["quant"] != "exl3":
        out.append(f"quant_method={meta['quant']!r}: kyojin_mlx.serve loads EXL3 packs")
    if not meta["server_family"]:
        out.append(f"no server family for architecture {meta['arch']}: add it to pack.FAMILIES")
    if not meta["template"]:
        out.append("pack has no chat template (chat_template.jinja or tokenizer_config.json chat_template)")
    return out


# --mtp-depth per architecture: ModelDesc.mtp_depth (DS4.1: 3, DAY1_RUNBOOK step 10; the depth adapts to each session's acceptance)


def mtp_depth(meta: dict) -> int:
    """The server's --mtp-depth needs draft-head tensors in the pack (DS4.1: mtp.*); the Qwen3.5 server path has no speculative decode, so a vendor
    mtp_num_hidden_layers there stays unused (per-arch)."""
    row = _md.by_architecture(meta["arch"])
    return (row.mtp_depth or 0) if row is not None and meta["mtp_layers"] > 0 else 0


def draft_policy(meta: dict) -> str:
    return f"mtp{mtp_depth(meta)}" if mtp_depth(meta) else "off"


def kvwin_env(k_bits: int, v_bits: int) -> dict:
    """The KV choice as the server reads it (KYOJIN_KVWIN=kbits:vbits:group:sink:window). 16 = fp16 cache (switch unset)."""
    if k_bits >= 16:
        return {}
    return {"KYOJIN_KVWIN": f"{k_bits}:{v_bits}:{KVWIN_GROUP}:{KVWIN_SINK}:{KVWIN_WINDOW}"}


def serve_env(meta: dict, k_bits: int, v_bits: int) -> dict:
    return kvwin_env(k_bits, v_bits) if meta["kv_mode"] == "kvwin" else {}


def serve_flags(meta: dict, *, port: int, ctx: int, sessions: int, k_bits: int, v_bits: int, model_id: str = "qual",
                chunk: int | None = None) -> list[str]:
    """argv of the server (after the python executable) for this pack. No hand flags: every value is derived. The KV choice is the environment
    (serve_env); `k_bits` / `v_bits` are accepted for symmetry with the other backends."""
    f = [meta["serve_entry"]] if meta.get("serve_entry") else ["-m", SERVER_MODULE]      # serve_entry: tests point it at a fake server script
    f += [meta["path"], "--port", str(port), "--model-id", model_id, "--ctx", str(ctx), "--sessions", str(sessions)]
    if chunk:
        f += ["--prefill-chunk", str(chunk), "--prefix-block", str(chunk)]
    if mtp_depth(meta):
        f += ["--mtp-depth", str(mtp_depth(meta))]
    row = _md.by_architecture(meta["arch"])
    if row is not None and row.own_tokenizer and not (Path(meta["path"]) / "tokenizer_config.json").exists():
        f += ["--tokenizer", str(Path(os.environ.get("KYOJIN_TOKENIZER", Path.home() / "kyojin-mlx/models" / row.own_tokenizer)))]
    return f


def verify_shards(path: str | os.PathLike) -> list[str]:
    """Problems found when comparing the shard files with model.safetensors.index.json (names + existence + non-empty); packs without an index are
    checked for at least one non-empty *.safetensors."""
    p = Path(path).expanduser()
    idx = p / "model.safetensors.index.json"
    if not idx.exists():
        files = sorted(f.name for f in p.glob("*.safetensors"))
        return [] if files and all((p / f).stat().st_size for f in files) else [f"{p}: no safetensors shard (and no model.safetensors.index.json)"]
    files = sorted(set(json.loads(idx.read_text())["weight_map"].values()))
    bad = []
    for f in files:
        fp = p / f
        if not fp.exists():
            bad.append(f"missing shard {f}")
        elif fp.stat().st_size == 0:
            bad.append(f"empty shard {f}")
    return bad


def disk_free_gb(path: str | os.PathLike) -> float:
    q = Path(path).expanduser()
    while not q.exists():
        q = q.parent
    return shutil.disk_usage(q).free / 1e9
