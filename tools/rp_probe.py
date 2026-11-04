"""Count the Metal calls MLX makes for one decode step of a slice (ds2 / ds2e): what must be recorded for a replay.
usage: PYTHONPATH=.:../../vlm_pkg python tools/rp_probe.py models/ds2 [--len 136] [--steps 3]"""
import argparse, ctypes, os, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "100000"); os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "1000000")
lib = ctypes.CDLL(os.path.join(here, "..", "replay", "build", "librp_probe.dylib"))
lib.rp_probe_install()
import numpy as np
import mlx.core as mx
from kyojin_mlx import patch as P, rows as R
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--len", type=int, default=136); ap.add_argument("--steps", type=int, default=3)
a = ap.parse_args()
model, spec, cfg = P.build_model(a.pack); P.patch_model(model, a.pack, spec, cfg); R.enable(model, spec)
lm = model.language_model
ids = np.random.default_rng(0).integers(16, cfg["text_config"]["vocab_size"] - 16, a.len).tolist()
cache = lm.make_cache()
lg = lm(mx.array(ids)[None], cache=cache).logits[:, -1].astype(mx.float32); mx.eval(lg)
tok = mx.argmax(lg, axis=-1)
for i in range(a.steps + 4):
    out = R.step(model, cache, tok); tok = mx.argmax(out, axis=-1); mx.eval(tok)
    if i >= 3 and i < 3 + a.steps:
        pass
    if i == 3:
        mx.synchronize()
        print(f"--- recording step {i}", file=sys.stderr)
        lib.rp_probe_begin()
        out = R.step(model, cache, tok); tok = mx.argmax(out, axis=-1); mx.eval(tok); mx.synchronize()
        lib.rp_probe_end()
