#!/usr/bin/env python3
"""Variants of a shrunk DS4.1 pack (tools/shrink_pack.py, 2 layers) for the batched-core exactness checks.
usage: make_core_packs.py SRC_PACK OUT_DIR
  OUT_DIR/reindex   layer 1 gets indexer wq_b / weights_proj (copies of layer 0): run with index_source_layer_ids=0:1 (layer 1 reuses the keys of
                    layer 0, `reindex` and candidate-consumer paths), optionally candidate_topk_blocks=2..3
  OUT_DIR/ratio1    compressor.wgate of layer 0 dropped: run with compress_ratios=1:1 (ratio 1 has no gate)
Shards are symlinked, only config / index / the extra tensors are written."""
import json
import os
import shutil
import sys

import mlx.core as mx


def make(src, out, add_layer1_indexer=False, drop=()):
    ix = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    wm = {k: v for k, v in ix["weight_map"].items() if k not in drop}
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(src):
        if f.endswith(".safetensors") and not os.path.lexists(os.path.join(out, f)):
            os.symlink(os.path.realpath(os.path.join(src, f)), os.path.join(out, f))
    shutil.copy(os.path.join(src, "config.json"), os.path.join(out, "config.json"))
    if add_layer1_indexer:
        t = mx.load(os.path.join(src, "shrunk.safetensors"))
        extra = {k.replace("layers.0.", "layers.1."): v for k, v in t.items() if k.startswith("layers.0.attn.indexer.") and ("wq_b" in k or "weights_proj" in k)}
        mx.save_safetensors(os.path.join(out, "extra.safetensors"), extra)
        wm.update({k: "extra.safetensors" for k in extra})
    json.dump({**ix, "weight_map": wm}, open(os.path.join(out, "model.safetensors.index.json"), "w"))


if __name__ == "__main__":
    src, out = sys.argv[1:3]
    make(src, os.path.join(out, "reindex"), add_layer1_indexer=True)
    make(src, os.path.join(out, "ratio1"), drop=("layers.0.attn.compressor.wgate.weight",))
    print(f"{out}/reindex {out}/ratio1")
