"""Experiment: marginal cost of one extra launch in a real decode step, GPU-light (CPU-bound) and full. Adds k trivial elementwise ops per layer."""
import argparse, os, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("M10_LIGHT"):
    import light
import numpy as np
import mlx.core as mx
from kyojin_mlx import patch as P, rows as R
import mlx_vlm.models.deepseek_v41.language as L

ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--len", type=int, default=135); ap.add_argument("--reps", type=int, default=25); ap.add_argument("--ks", default="0,50,100")
a = ap.parse_args()
model, spec, cfg = P.build_model(a.pack); P.patch_model(model, a.pack, spec, cfg); R.enable(model, spec)
lm = model.language_model
rng = np.random.default_rng(0)
ids = rng.integers(16, cfg["text_config"]["vocab_size"] - 16, a.len).tolist()
cache = lm.make_cache()
lg = lm(mx.array(ids)[None], cache=cache).logits[:, -1].astype(mx.float32); mx.eval(lg)
tok = mx.argmax(lg, axis=-1)
orig = L.DeepseekV41Block.__call__
st = {"k": 0}
def call(self, h, pre_mix, image_mask, cache):
    h2, p2 = orig(self, h, pre_mix, image_mask, cache)
    if h2.shape[1] == 1:
        for _ in range(st["k"]):
            h2 = h2 * 1.0001
    return h2, p2
L.DeepseekV41Block.__call__ = call
ks = [int(v) for v in a.ks.split(",")]
res = {k: [] for k in ks}
nl = len(lm.layers)
for rep in range(a.reps + 2):
    for k in ks:
        st["k"] = k
        c = R.fork(cache)
        t = time.perf_counter(); lg = R.step(model, c, tok); mx.eval(lg); dt = time.perf_counter() - t
        if rep >= 2: res[k].append(dt)
    tok = mx.argmax(lg, axis=-1)
base = statistics.median(res[ks[0]])
for k in ks:
    m = statistics.median(res[k])
    print(f"k={k:3d} extra ops/layer: median {1e3 * m:.2f} ms" + (f"  marginal {1e6 * (m - base) / (k * nl):.2f} us per extra op" if k else ""))
