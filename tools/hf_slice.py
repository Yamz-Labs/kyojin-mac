#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""First-N-layers slice of a sharded HF checkpoint, fetched tensor by tensor with HTTP range requests (no full shard ever lands on disk):
the text embedding, the final norm, the head, the first N decoder blocks; no vision tower, no MTP. Writes DEST/model.safetensors (BF16 bytes as in the source),
a config.json with num_hidden_layers / layer_types cut to N, and the small tokenizer / template files.
usage: hf_slice.py REPO DEST N [--revision SHA]"""
from __future__ import annotations

import argparse, json, re, struct, urllib.request
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, hf_hub_url

SMALL = (".json", ".txt", ".jinja", ".model")


def _get(url: str, lo: int, hi: int) -> bytes:
    h = {"Range": f"bytes={lo}-{hi}"}
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=600) as r:
        return r.read()


def keep(name: str, n: int) -> bool:
    m = re.match(r"^(?:model\.language_model|model)\.layers\.(\d+)\.", name)
    if m:
        return int(m[1]) < n
    return name.startswith(("model.language_model.embed_tokens.", "model.language_model.norm.", "model.embed_tokens.", "model.norm.", "lm_head."))


def slice_repo(repo: str, dest: Path, n: int, revision: str | None = None) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    info = api.model_info(repo, revision=revision)
    sib = [s.rfilename for s in info.siblings]
    for f in sib:
        if f.endswith(SMALL) and "/" not in f and f != "model.safetensors.index.json":
            hf_hub_download(repo, f, revision=info.sha, local_dir=str(dest))
    index = json.loads(Path(hf_hub_download(repo, "model.safetensors.index.json", revision=info.sha)).read_text())["weight_map"]
    want = {k: f for k, f in index.items() if keep(k, n)}
    plan, off = [], 0                                                        # (name, dtype, shape, url, abs lo, abs hi, out offset); headers first, then one tensor in memory at a time
    for shard in sorted(set(want.values())):
        url = hf_hub_url(repo, shard, revision=info.sha)
        hl = struct.unpack("<Q", _get(url, 0, 7))[0]
        hdr = json.loads(_get(url, 8, 8 + hl - 1))
        for k in sorted(k for k, f in want.items() if f == shard):
            t = hdr[k]; lo, hi = t["data_offsets"]
            plan.append((k, t["dtype"], t["shape"], url, 8 + hl + lo, 8 + hl + hi, off)); off += hi - lo
    meta = {k: {"dtype": dt, "shape": sh, "data_offsets": [o, o + (hi - lo)]} for k, dt, sh, _, lo, hi, o in plan}
    hdr = json.dumps({"__metadata__": {"format": "pt"}, **meta}, separators=(",", ":")).encode()
    hdr += b" " * (-len(hdr) % 8)
    nbytes = off
    with open(dest / "model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(hdr))); f.write(hdr)
        for i, (k, dt, sh, url, lo, hi, o) in enumerate(plan):
            if hi > lo:
                f.write(_get(url, lo, hi - 1))
            if i % 10 == 9 or i == len(plan) - 1:
                print(f"[slice] {i + 1}/{len(plan)} tensors, {(o + hi - lo) / 1e9:.2f} of {nbytes / 1e9:.2f} GB", flush=True)
    blobs = plan
    cfg = json.loads((dest / "config.json").read_text())
    tc = cfg.get("text_config", cfg)
    tc["num_hidden_layers"] = n
    if "layer_types" in tc:
        tc["layer_types"] = tc["layer_types"][:n]
    tc["mtp_num_hidden_layers"] = 0
    cfg.pop("vision_config", None)
    (dest / "config.json").write_text(json.dumps(cfg, indent=2))
    (dest / "slice.json").write_text(json.dumps({"repo": repo, "revision": info.sha, "layers": n, "tensors": len(blobs), "bytes": nbytes}))
    return dict(repo=repo, revision=info.sha, layers=n, bytes=nbytes)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("repo"); ap.add_argument("dest"); ap.add_argument("n", type=int); ap.add_argument("--revision", default=None)
    a = ap.parse_args()
    print(json.dumps(slice_repo(a.repo, Path(a.dest), a.n, a.revision)))
