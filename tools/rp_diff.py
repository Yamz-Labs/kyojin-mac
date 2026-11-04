"""Record N consecutive decode steps of a slice and diff the command streams (what changes per step).
usage: PYTHONPATH=.:../../vlm_pkg python tools/rp_diff.py models/ds2 [--len 136] [--steps 4] [--ratios 0,0] [--dump DIR]"""
import argparse, ctypes, json, os, sys, tempfile, re, collections
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "100000"); os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "1000000")
lib = ctypes.CDLL(os.path.join(here, "..", "replay", "build", "librp.dylib"))
lib.rp_install(1)
import numpy as np
import mlx.core as mx
from kyojin_mlx import patch as P, rows as R
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--len", type=int, default=136); ap.add_argument("--steps", type=int, default=4); ap.add_argument("--ratios"); ap.add_argument("--dump", default="/tmp/rp_dump")
a = ap.parse_args()
if a.ratios:
    src, a.pack = a.pack, tempfile.mkdtemp(prefix="rp_")
    for f in os.listdir(src):
        if f != "config.json": os.symlink(os.path.abspath(os.path.join(src, f)), os.path.join(a.pack, f))
    c = json.load(open(os.path.join(src, "config.json"))); (c.get("text_config") or c)["compress_ratios"] = [int(v) for v in a.ratios.split(",")]
    json.dump(c, open(os.path.join(a.pack, "config.json"), "w"))
model, spec, cfg = P.build_model(a.pack); P.patch_model(model, a.pack, spec, cfg); R.enable(model, spec)
lm = model.language_model
ids = np.random.default_rng(0).integers(16, cfg["text_config"]["vocab_size"] - 16, a.len).tolist()
cache = lm.make_cache()
lg = lm(mx.array(ids)[None], cache=cache).logits[:, -1].astype(mx.float32); mx.eval(lg)
tok = mx.argmax(lg, axis=-1)
for i in range(4):                                              # warm: kernels built, compiled regions traced
    out = R.step(model, cache, tok); tok = mx.argmax(out, axis=-1); mx.eval(tok)
os.makedirs(a.dump, exist_ok=True)
mx.set_cache_limit(0); mx.clear_cache()
for s in range(a.steps):
    mx.synchronize()
    lib.rp_rec_begin(s)
    out = R.step(model, cache, tok); tok = mx.argmax(out, axis=-1); mx.eval(tok); mx.synchronize()
    n = lib.rp_rec_end()
    print(f"step {s}: cmds {n} encoders {lib.rp_rec_info(s,1)} fences {lib.rp_rec_info(s,2)} other {lib.rp_rec_info(s,3)} arena {lib.rp_rec_info(s,4)} born {lib.rp_rec_info(s,5)}")
    lib.rp_dump(s, os.path.join(a.dump, f"step{s}.txt").encode())
