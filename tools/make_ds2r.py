#!/usr/bin/env python3
"""Build a ratio-4 variant of a shrunk DS4.1 pack (tools/shrink_pack.py) for the compressor-state checks: a ratio > 1 needs a compressor gate
weight and the shrunk pack (compress_ratios 1) has none. Shards are symlinked; only wgate (a copy of wkv, same shape), the config and the index are
written. usage: make_ds2r.py SRC_PACK OUT_DIR"""
import json
import os
import shutil
import sys

import mlx.core as mx

src, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
ix = json.load(open(os.path.join(src, "model.safetensors.index.json")))
for f in os.listdir(src):
    if f.endswith(".safetensors") and not os.path.lexists(os.path.join(out, f)):
        os.symlink(os.path.realpath(os.path.join(src, f)), os.path.join(out, f))
cfg = json.load(open(os.path.join(src, "config.json")))
cfg["text_config"]["compress_ratios"] = [4, 4]
cfg["text_config"]["index_topk"] = 16
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
t = mx.load(os.path.join(src, "shrunk.safetensors"))
extra = {}
for l in (0, 1):
    kk = f"layers.{l}.attn.compressor.wkv.weight"
    if kk in t:                                        # only a kv-source layer owns a compressor
        extra[f"layers.{l}.attn.compressor.wgate.weight"] = t[kk]
mx.save_safetensors(os.path.join(out, "extra.safetensors"), extra)
wm = dict(ix["weight_map"])
wm.update({k: "extra.safetensors" for k in extra})
json.dump({**ix, "weight_map": wm}, open(os.path.join(out, "model.safetensors.index.json"), "w"))
print("wrote", out, len(extra), "tensors")
