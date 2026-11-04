"""Experiment: does an async_eval after every layer overlap the python build with the GPU? usage: exp_async.py PACK [--every 1]"""
import argparse, os, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("M10_LIGHT"):
    import light
import numpy as np
import mlx.core as mx
from kyojin_mlx import patch as P, rows as R
import mlx_vlm.models.deepseek_v41.language as L

ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--every", type=int, default=1); ap.add_argument("--len", type=int, default=135); ap.add_argument("--reps", type=int, default=30)
a = ap.parse_args()
model, spec, cfg = P.build_model(a.pack); P.patch_model(model, a.pack, spec, cfg); R.enable(model, spec)
lm = model.language_model
rng = np.random.default_rng(0)
ids = rng.integers(16, cfg["text_config"]["vocab_size"] - 16, a.len).tolist()
cache = lm.make_cache()
lg = lm(mx.array(ids)[None], cache=cache).logits[:, -1].astype(mx.float32); mx.eval(lg)
tok = mx.argmax(lg, axis=-1)
orig = L.DeepseekV41Block.__call__
state = {"on": False, "i": 0}
def call(self, h, pre_mix, image_mask, cache):
    h2, p2 = orig(self, h, pre_mix, image_mask, cache)
    if state["on"] and h2.shape[1] == 1 and (self.layer_idx + 1) % a.every == 0:
        mx.async_eval(h2, p2)
    return h2, p2
L.DeepseekV41Block.__call__ = call
res = {False: [], True: []}
for rep in range(a.reps + 2):
    for on in (False, True):
        state["on"] = on
        c = R.fork(cache)
        t = time.perf_counter()
        lg = R.step(model, c, tok); mx.eval(lg)
        dt = time.perf_counter() - t
        if rep >= 2: res[on].append(dt)
    tok = mx.argmax(lg, axis=-1)
for on in (False, True):
    print(f"per-layer async_eval {on}: median {1e3 * statistics.median(res[on]):.2f} ms  min {1e3 * min(res[on]):.2f} ms")
