#!/usr/bin/env python3
"""Make a converted DS4.1 pack directory ready for Engram: link the two official Engram shards, write the merged index, the token map, validate.
usage: engram_index.py MODEL_DIR [--engram-dir DIR] [--check-only]
  --engram-dir  directory with the official shards (model-00047 / 00048, F8 embed tables); symlinked into MODEL_DIR. Skipped when the shards
                are already in MODEL_DIR.
  writes        MODEL_DIR/model.safetensors.index.json (header-only merge; the pack wins on q/k/wkv, the Engram shards give the embed tables)
                MODEL_DIR/engram_token_map.json (from MODEL_DIR/tokenizer.json; mlx-vlm reads it instead of rebuilding the map)
  validates     dtype F8_E4M3 / F8_E8M0, shape [rows, 256] / [rows, 8] and byte sizes against config.json, file sizes
Safe to re-run. Exit 1 on any problem."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kyojin_mlx import engram_rows as ER

ap = argparse.ArgumentParser()
ap.add_argument("model_dir")
ap.add_argument("--engram-dir")
ap.add_argument("--check-only", action="store_true")
a = ap.parse_args()

cfg = json.load(open(os.path.join(a.model_dir, "config.json")))
spec = ER.EngramSpec.from_config(cfg)
if spec is None:
    if not a.check_only:
        print("config has no Engram layer; index:", ER.write_merged_index(a.model_dir, ER.merged_index(a.model_dir)))
    sys.exit(0)
if not a.check_only:
    if a.engram_dir and os.path.realpath(a.engram_dir) != os.path.realpath(a.model_dir):
        print("linked", ER.link_engram_shards(a.model_dir, a.engram_dir))
    wmap = ER.merged_index(a.model_dir, spec)
    print("index:", ER.write_merged_index(a.model_dir, wmap), f"({len(wmap)} tensors)")
    tok = os.path.join(a.model_dir, "tokenizer.json")
    out = os.path.join(a.model_dir, "engram_token_map.json")
    if os.path.exists(tok) and not os.path.exists(out):
        from transformers import PreTrainedTokenizerFast
        from mlx_vlm.models.deepseek_v41.engram import build_compressed_token_map
        tmap, nv = build_compressed_token_map(PreTrainedTokenizerFast(tokenizer_file=tok))
        assert nv == spec.compressed_vocab_size, (nv, spec.compressed_vocab_size)
        json.dump([int(x) for x in tmap], open(out + ".tmp", "w"))
        os.replace(out + ".tmp", out)
        print("token map:", out, f"({len(tmap)} tokens -> {nv} ids)")
bad = ER.EngramTables(a.model_dir, spec).validate()
for b in bad:
    print("PROBLEM:", b)
print("Engram tables OK" if not bad else f"{len(bad)} problem(s)")
sys.exit(1 if bad else 0)
