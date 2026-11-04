"""Derive a 2-layer DS4.1 slice pack with another compressor / indexer layout from a shrunk pack (dense and quantized compressor / indexer
tensors are copied between layers, so every mode of DeepseekV41Attention can be exercised by a slice: full, reindex, reuse, local).
usage: slice_variant.py SRC OUT --layer0 full|local|keep --layer1 full|reindex|reuse [--ratio1 0,1]
  full     compressor + indexer with its own keys (kv source and index source)
  reindex  indexer.wq_b + weights_proj only (index source that reads the keys of another layer)
  reuse    nothing (shares the pool and the top-k of the source)
  local    nothing, compress ratio 0 (window attention only)
Stacked files are symlinked, shrunk.safetensors is rewritten. The config is NOT changed: pass compress_ratios / kv_source_layer_ids /
index_source_layer_ids / candidate_source_layer_id with `--set` when you use the pack."""
import argparse, json, os, shutil
import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--layer0", default="keep"); ap.add_argument("--layer1", default="reuse")
ap.add_argument("--ratio1", default="", help="comma list of layers whose compress ratio is 1 (no wgate)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))
wm = idx["weight_map"]
t = mx.load(os.path.join(a.src, "shrunk.safetensors"))
full = {k: v for k, v in t.items() if k.startswith("layers.0.attn.compressor.") or k.startswith("layers.0.attn.indexer.")}
assert full, "source layer 0 has no compressor / indexer tensors"
KEEP_RE = {"full": lambda s: True, "reindex": lambda s: s.startswith("indexer.wq_b.") or s.startswith("indexer.weights_proj."), "reuse": lambda s: False, "local": lambda s: False}
for k in [k for k in t if (".attn.compressor." in k or ".attn.indexer." in k)]:
    del t[k]
for layer, mode in ((0, a.layer0), (1, a.layer1)):
    if mode == "keep":
        mode = "full" if layer == 0 else "reuse"
    for k, v in full.items():
        suffix = k.split("layers.0.attn.", 1)[1]
        if suffix.endswith("compressor.wgate.weight") and str(layer) in a.ratio1.split(","):
            continue
        if KEEP_RE[mode](suffix):
            t[f"layers.{layer}.attn.{suffix}"] = v
mx.save_safetensors(os.path.join(a.out, "shrunk.safetensors"), t)
new = {k: v for k, v in wm.items() if v != "shrunk.safetensors"}
new.update({k: "shrunk.safetensors" for k in t})
idx["weight_map"] = new
json.dump(idx, open(os.path.join(a.out, "model.safetensors.index.json"), "w"))
shutil.copy(os.path.join(a.src, "config.json"), os.path.join(a.out, "config.json"))
for f in os.listdir(a.src):
    if f.startswith("stacked.") and not os.path.exists(os.path.join(a.out, f)):
        os.symlink(os.path.abspath(os.path.join(a.src, f)), os.path.join(a.out, f))
print("wrote", a.out, len(t), "dense/side tensors")
