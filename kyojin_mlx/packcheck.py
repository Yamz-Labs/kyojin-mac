# SPDX-License-Identifier: Apache-2.0
"""Loader checks: does a pack carry what its descriptor promises? Reads `config.json`, the safetensors index (or the safetensors headers) and the
tokenizer JSON only: no weight is read, no format is touched, nothing imports mlx (the vendor-class check imports the model package when it is there).

Every finding names the descriptor field that made the promise, so the fix is a pack fix or a descriptor edit, never a guess:

    row            ModelDesc found by model_type / architectures[0]                      (DESCS)
    arch           the row's `arch` has an ArchSpec                                       (ModelDesc.arch, archs.ARCHS)
    config         the config keys the geometry needs                                     (modeldesc.geometry)
    layers         every decoder layer 0..n-1 has tensors in the pack                     (ArchSpec.layer_re)
    moe            "yes": experts in the config AND expert tensors in the pack; "no": none (ModelDesc.moe)
    hc             "yes": hc_mult in the config AND hc tensors                            (ModelDesc.hc)
    index          index_n_heads in the config -> indexer tensors in the pack
    engram         engram_layer_ids in the config -> engram tensors in the pack
    mtp            draft head requested: ArchSpec.mtp, its config keys, `mtp.*` tensors  (ArchSpec.mtp, MtpSpec.keys / prefix)
    ced            the ArchSpec promises CED: `ced.explain(config)` must be None          (ArchSpec.ced, CedSpec.mid)
    vendor         every class role of ArchSpec.vendor exists in the model package        (ArchSpec.vendor)
    tokenizer      a tokenizer ships with the pack, or the row names one                  (ModelDesc.own_tokenizer)
    stop_tokens    every stop string of the row is in the tokenizer vocabulary            (ModelDesc.stop_tokens)
    template       a chat template ships with the pack, or the row names a builtin        (ModelDesc.builtin_template)

`enforce()`: flag KYOJIN_DESC_STRICT (default "0", docs/cloud/LEVERS.md): "0" logs every finding and loads on, "1" raises PackError listing all of them."""
from __future__ import annotations

import glob
import json
import os
import re
import struct
from dataclasses import dataclass

from . import archs, modeldesc

STRICT_DEFAULT = "0"                                   # the flag default (flags.py desc_strict); the module literal equals it (tests/test_packcheck.py)


class PackError(ValueError):
    pass


@dataclass(frozen=True)
class Finding:
    field: str                                         # the descriptor field / check name above
    msg: str

    def __str__(self):
        return f"[{self.field}] {self.msg}"


def strict() -> bool:
    return os.environ.get("KYOJIN_DESC_STRICT", STRICT_DEFAULT) not in ("0", "")


# ---------------------------------------------------------------------------------------------------------------- pack reading
def _header(path: str) -> dict:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


def tensor_names(pack_dir: str) -> set:
    """Tensor names of the pack from its index, else from the safetensors headers (the `mtp/` sidecar of a Qwen pack included)."""
    names: set = set()
    ip = os.path.join(pack_dir, "model.safetensors.index.json")
    if os.path.isfile(ip):
        names.update(json.load(open(ip)).get("weight_map", {}))
    else:
        for f in sorted(glob.glob(os.path.join(pack_dir, "*.safetensors"))):
            names.update(k for k in _header(f) if k != "__metadata__")
    for f in sorted(glob.glob(os.path.join(pack_dir, "mtp", "*.safetensors"))):
        names.update("mtp." + k for k in _header(f) if k != "__metadata__")
    return names


def _vocab_has(pack_dir: str, token: str):
    """True / False from tokenizer.json (added tokens + vocab); None when there is no tokenizer.json to look at."""
    p = os.path.join(pack_dir, "tokenizer.json")
    if not os.path.isfile(p):
        return None
    tj = json.load(open(p))
    if any(t.get("content") == token for t in tj.get("added_tokens", [])):
        return True
    voc = (tj.get("model") or {}).get("vocab")
    if isinstance(voc, dict):
        return token in voc
    if isinstance(voc, list):
        return any(isinstance(v, list) and v and v[0] == token for v in voc)
    return False


# ---------------------------------------------------------------------------------------------------------------- the checks
def check(pack_dir: str, config: dict | None = None, mtp: bool = False, ced: bool = False, names: set | None = None) -> list:
    """All findings for `pack_dir` (empty = the pack has what the descriptor promises). mtp / ced: what the caller is about to switch on."""
    out: list = []
    add = lambda field, msg: out.append(Finding(field, msg))           # noqa: E731
    cfg = config if config is not None else json.load(open(os.path.join(pack_dir, "config.json")))
    t = modeldesc.text_config(cfg)
    row, _ = None, None
    row = modeldesc.by_model_type(t.get("model_type", "") or cfg.get("model_type", "")) or modeldesc.by_architecture((cfg.get("architectures") or ["?"])[0])
    if row is None:
        add("row", f"model_type {cfg.get('model_type')!r} / architectures {cfg.get('architectures')} match no ModelDesc in modeldesc.DESCS: add a row")
        return out
    spec = archs.ARCHS.get(row.arch) if row.arch else None
    if row.arch and spec is None:
        add("arch", f"row {row.name!r} says arch={row.arch!r} but archs.ARCHS has {sorted(archs.ARCHS)}")
    try:
        g = modeldesc.geometry(cfg)
    except (KeyError, TypeError, ValueError) as e:
        add("config", f"config.json lacks a key the geometry needs: {e!r} (num_hidden_layers, hidden_size, num_attention_heads)")
        return out
    if names is None:
        names = tensor_names(pack_dir)
    if not names:
        add("layers", f"{pack_dir}: no safetensors tensors found (no index, no *.safetensors)")
        return out
    has = lambda sub: any(sub in n for n in names)                      # noqa: E731

    # ---- layers
    layer_re = re.compile(spec.layer_re if spec else r"layers\.(\d+)\.")
    seen = {int(m[1]) for n in names if (m := layer_re.search(n))}
    missing = [i for i in range(g.layers) if i not in seen]
    if missing:
        add("layers", f"config says {g.layers} layers; the pack has no tensor for layer(s) {missing[:8]}{' ...' if len(missing) > 8 else ''} (layer_re {layer_re.pattern!r})")

    # ---- MoE
    if row.moe == "yes":
        if g.experts <= 0:
            add("moe", f"row {row.name!r} promises MoE but the config has no routed-expert count (n_routed_experts / num_experts / num_local_experts)")
        elif not (has("experts") or has("switch_mlp")):
            add("moe", f"config has {g.experts} routed experts but the pack has no expert tensors (names containing 'experts' / 'switch_mlp')")
    elif row.moe == "no" and g.experts > 0:
        add("moe", f"row {row.name!r} promises a dense model but the config has {g.experts} routed experts")

    # ---- hyper-connections / indexer / engram
    if row.hc == "yes":
        if g.hc_mult <= 0:
            add("hc", f"row {row.name!r} promises hyper-connections but the config has no hc_mult")
        elif not has("hc_"):
            add("hc", f"config has hc_mult={g.hc_mult} but the pack has no hc_* tensors")
    elif row.hc == "no" and g.hc_mult > 0:
        add("hc", f"row {row.name!r} promises no hyper-connections but the config has hc_mult={g.hc_mult}")
    if g.index_heads > 0 and not has("indexer"):
        add("index", f"config has index_n_heads={g.index_heads} but the pack has no indexer tensors")
    eng = list(t.get("engram_layer_ids") or ())
    if eng and not has("engram"):
        add("engram", f"config lists Engram layers {eng[:6]} but the pack has no engram tensors")

    # ---- draft head
    if mtp:
        ms = spec.mtp if spec else None
        if row.arch and ms is None:
            add("mtp", f"draft head requested but ArchSpec {row.arch!r} has no MtpSpec")
        prefix = ms.prefix if ms else "mtp."
        if not any(n.startswith(prefix) for n in names):
            add("mtp", f"draft head requested but the pack has no tensors under {prefix!r} (MtpSpec.prefix)")
        if ms:
            lack = [f"{role}={key}" for role, key in ms.keys.items() if t.get(key) is None]
            if lack:
                add("mtp", f"MtpSpec.keys name config keys this config lacks: {', '.join(lack)}")
            elif int(t.get(ms.keys.get("stages", ""), 0) or 0) <= 0:
                add("mtp", f"config {ms.keys['stages']} = 0: no draft stages")

    # ---- CED
    if spec is not None and spec.ced is not None and ced:
        from . import ced as C
        import types
        c = types.SimpleNamespace(**{**t, "num_hidden_layers": g.layers})
        try:
            mid, _ = C.spec_defaults(spec, None, None)
            why = C.explain(c, mid)
        except (AttributeError, TypeError, ValueError) as e:
            why = f"config lacks a key CED reads: {e!r}"
        if why:
            add("ced", f"ArchSpec {spec.name!r} promises CED but this config's layer layout does not support it: {why}")

    # ---- vendor classes
    if spec is not None and spec.vendor is not None:
        try:
            lang = spec.vendor.lang()
        except Exception as e:                                          # noqa: BLE001  (a missing model package is the finding)
            add("vendor", f"cannot import {spec.vendor.module}: {e!r}")
        else:
            for role in ("block", "attn", "moe", "cache", "batch_cache"):
                name = getattr(spec.vendor, role)
                if name and not hasattr(lang, name):
                    add("vendor", f"{spec.vendor.module} has no class {name!r} (ArchSpec.vendor.{role}); the model code changed or the ArchSpec is wrong")

    # ---- tokenizer, stop tokens, template
    has_tok = any(os.path.isfile(os.path.join(pack_dir, f)) for f in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model"))
    if not has_tok and not row.own_tokenizer:
        add("tokenizer", f"no tokenizer file in {pack_dir} and row {row.name!r} names none (ModelDesc.own_tokenizer)")
    for tok in row.stop_tokens:
        if _vocab_has(pack_dir, tok) is False:
            add("stop_tokens", f"stop token {tok!r} (ModelDesc.stop_tokens of {row.name!r}) is not in the pack's tokenizer.json")
    tmpl = any(os.path.isfile(os.path.join(pack_dir, f)) for f in ("chat_template.jinja", "chat_template.json"))
    if not tmpl and os.path.isfile(os.path.join(pack_dir, "tokenizer_config.json")):
        tmpl = bool(json.load(open(os.path.join(pack_dir, "tokenizer_config.json"))).get("chat_template"))
    if not tmpl and not row.builtin_template:
        add("template", f"no chat template in {pack_dir} and row {row.name!r} names no builtin (ModelDesc.builtin_template)")
    return out


def enforce(findings: list, log=print, strict_: bool | None = None) -> list:
    """strict (flag desc_strict = 1): raise PackError listing every finding. Otherwise log them one per line and return them."""
    if not findings:
        return findings
    if strict() if strict_ is None else strict_:
        raise PackError("pack does not match its descriptor:\n  " + "\n  ".join(str(f) for f in findings))
    for f in findings:
        log(f"[kyojin] pack check (KYOJIN_DESC_STRICT=0, loading on): {f}")
    return findings


def check_pack(pack_dir: str, mtp: bool = False, ced: bool = False, log=print, strict_: bool | None = None) -> list:
    return enforce(check(pack_dir, mtp=mtp, ced=ced), log, strict_)


if __name__ == "__main__":
    import argparse
    import sys
    ap = argparse.ArgumentParser(description="header-only check of a pack against its modeldesc row / ArchSpec (no weight is read)")
    ap.add_argument("pack")
    ap.add_argument("--mtp", action="store_true", help="the draft head will be switched on")
    ap.add_argument("--ced", action="store_true", help="CED prefill will be switched on")
    a = ap.parse_args()
    fs = check(a.pack, mtp=a.mtp, ced=a.ced)
    for f in fs:
        print(f)
    print(f"{len(fs)} finding(s)")
    sys.exit(1 if fs else 0)
