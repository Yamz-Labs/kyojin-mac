#!/usr/bin/env python3
"""BOX side of the Engram cross-check: ground truth from the box engine for a few token sequences, as one small npz for the Mac.

usage (inside the engine env, CPU only, reads ~20k random rows of the 190 GiB tables):
  python tools/engram_cut.py PACK_DIR ENGRAM_DIR OUT.npz --ids-json ref.json [--extra-random 4096] [--seed 1]

--ids-json: ref.json of tools/ref_ds41_box.py (prompt_ids + out_ids per prompt) or a json list of token id lists.
Output (all numpy, no torch needed to read it):
  seq<k>_ids        int64 [L]                 token ids of sequence k
  seq<k>_hash       int64 [L, n_layers, 24]   table rows from the BOX NgramHasher (exllamav3.dsv41.engram)
  rows<p>           int64 [n]                 sorted unique rows touched by all sequences + the random extra rows, per Engram layer position
  raw<p>            uint8 [n, 264]            the engine store's rows: 256 e4m3 values then 8 e8m0 scales
  deq<p>            float32 [n, 256]          engine dequantisation of those rows
  hdr_<shard>       uint8 [h]                 the first 8 + header-length bytes of each Engram shard (the Mac rebuilds sparse shards from them)
  meta              json string               config values, token map size, shard names
The Mac side (tools/engram_check.py) rebuilds sparse shards with the real headers and compares three implementations row for row."""
import argparse
import glob
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
ROOT = os.environ.get("DS41_ROOT", os.path.expanduser("~/kyojin-work/ds41golive"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "ds41"))

import torch
from engram_subset import token_map_from_file
from exllamav3.dsv41.config import DSV41Config
from exllamav3.dsv41.engram import EngramLayout, NgramHasher, RawBinEngramStore, compute_multipliers, dequant_rows

ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("engram_dir"); ap.add_argument("out")
ap.add_argument("--ids-json", required=True)
ap.add_argument("--extra-random", type=int, default=4096)
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()

cfg = DSV41Config.from_file(os.path.join(a.pack, "config.json"))
tm, nv = token_map_from_file(os.path.join(a.pack, "tokenizer.json"))
assert nv == cfg.engram_compressed_vocab_size, (nv, cfg.engram_compressed_vocab_size)
layout = EngramLayout.build(cfg.engram_layer_ids, cfg.engram_num_embeddings, cfg.engram_max_ngram_size, cfg.engram_vocab_size, cfg.engram_n_heads,
                            cfg.engram_head_dim, None)
mult = compute_multipliers(layout.layer_ids, layout.max_ngram_size, nv)

data = json.load(open(a.ids_json))
if isinstance(data, dict) and "prompts" in data:
    seqs = [p["prompt_ids"] + p["out_ids"] for p in data["prompts"]]
else:
    seqs = data
out, per = {}, [[] for _ in layout.layer_ids]
for k, ids in enumerate(seqs):
    t = torch.tensor(ids, dtype=torch.int64).view(1, -1)
    h = NgramHasher(layout, tm, cfg.engram_pad_token_id, nv, mult, 1, t.shape[1], "cpu")(t, 0)[0]      # [L, nl, 24]
    out[f"seq{k}_ids"], out[f"seq{k}_hash"] = np.array(ids, dtype=np.int64), h.numpy()
    for p in range(len(layout.layer_ids)):
        per[p].append(h[:, p, :].reshape(-1).numpy())
rng = np.random.default_rng(a.seed)
store = RawBinEngramStore.open(a.engram_dir, cfg.engram_layer_ids, cfg.engram_head_dim, "auto")
for p, lid in enumerate(layout.layer_ids):
    rows = np.unique(np.concatenate(per[p] + [rng.integers(0, cfg.engram_num_embeddings[p], a.extra_random)]))
    raw = store.fetch(p, torch.from_numpy(rows))
    deq = dequant_rows(raw, torch.float32, "fp8")
    out[f"rows{p}"], out[f"raw{p}"], out[f"deq{p}"] = rows, raw.numpy(), deq.numpy()
    print(f"layer {lid}: {rows.size} rows, raw {tuple(raw.shape)}", flush=True)
shards = []
for f in sorted(glob.glob(os.path.join(a.engram_dir, "*.safetensors"))):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        fh.seek(0)
        out["hdr_" + os.path.basename(f)] = np.frombuffer(fh.read(8 + n), dtype=np.uint8)
    shards.append(os.path.basename(f))
out["meta"] = np.array(json.dumps(dict(layer_ids=list(layout.layer_ids), num_embeddings=list(cfg.engram_num_embeddings), n_seq=len(seqs), shards=shards,
                                       compressed_vocab=nv, token_map_len=len(tm), max_ngram=layout.max_ngram_size, n_heads=layout.n_heads,
                                       head_dim=layout.head_dim, pad_token_id=cfg.engram_pad_token_id, vocab_size=cfg.engram_vocab_size)))
np.savez_compressed(a.out, **out)
print("saved", a.out, round(os.path.getsize(a.out) / 2 ** 20, 1), "MiB")
