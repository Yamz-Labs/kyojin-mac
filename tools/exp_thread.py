"""Experiment: commit each layer from a worker thread so the encode of layer i overlaps the python build of layer i+1.
usage: exp_thread.py PACK [--every 1] [--reps 30]. Prints median / min step time per variant and whether the logits are bit-identical."""
import faulthandler; faulthandler.dump_traceback_later(60, exit=True)
import argparse, os, queue, statistics, sys, threading, time
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
gpu = mx.new_stream(mx.gpu)
mx.set_default_stream(gpu)
q = queue.Queue()
def worker():
    with mx.stream(gpu):
        while True:
            it = q.get()
            if it is None:
                break
            if it == "sync":
                done.set(); continue
            mx.async_eval(*it)
th = threading.Thread(target=worker, daemon=True); th.start()
done = threading.Event()
state = {"mode": "base"}
def call(self, h, pre_mix, image_mask, cache):
    h2, p2 = orig(self, h, pre_mix, image_mask, cache)
    if h2.shape[1] == 1 and (self.layer_idx + 1) % a.every == 0:
        if state["mode"] == "thread":
            q.put((h2, p2))
        elif state["mode"] == "inline":
            mx.async_eval(h2, p2)
    return h2, p2
L.DeepseekV41Block.__call__ = call
modes = ["base", "thread", "inline"]
res = {m: [] for m in modes}; outs = {}
for rep in range(a.reps + 2):
    for m in modes:
        state["mode"] = m
        c = R.fork(cache)
        t = time.perf_counter()
        lg = R.step(model, c, tok)
        if m == "thread":
            done.clear(); q.put("sync"); done.wait()
        mx.eval(lg)
        dt = time.perf_counter() - t
        if rep >= 2: res[m].append(dt)
        outs[m] = np.array(lg)
    tok = mx.argmax(lg, axis=-1)
for m in modes:
    print(f"{m:7s} median {1e3 * statistics.median(res[m]):.2f} ms  min {1e3 * min(res[m]):.2f} ms  logits==base {bool((outs[m] == outs['base']).all())}")
q.put(None)
