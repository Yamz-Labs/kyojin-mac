"""A/B of decode-step variants in ONE process, interleaved (the Air is loaded by other agents; separate runs are not comparable).
usage: bench_step.py PACK [--variants base,s5] [--reps 30] [--len 135] [--bit]   (M10_LIGHT=F for the CPU-bound proxy)
Each variant is a name in VARIANTS: a function that sets flags (module attributes) before the step. Per variant: median / min of the whole step
(build + commit + wait), of the python build, of the async_eval return (encode), launches per step, and whether the logits are bit-identical to `base`."""
import argparse, os, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("M10_LIGHT"):
    import light
import numpy as np
import mlx.core as mx
from kyojin_mlx import patch as P, rows as R
from kyojin_mlx import fused_group as FG

ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--variants", default="base,s5,all"); ap.add_argument("--reps", type=int, default=30)
ap.add_argument("--len", type=int, default=135); ap.add_argument("--no-install", action="store_true")
a = ap.parse_args()
model, spec, cfg = P.build_model(a.pack); P.patch_model(model, a.pack, spec, cfg); R.enable(model, spec)
info = getattr(lm_ := model.language_model, "_kyojin_s5", None) or (FG.install(model) if not a.no_install else {})
print("S5 install:", {k: v for k, v in info.items() if k != "names"}, flush=True)
lm = model.language_model
ids = np.random.default_rng(0).integers(16, cfg["text_config"]["vocab_size"] - 16, a.len).tolist()
cache = lm.make_cache()
FG.ENABLED = False
lg = lm(mx.array(ids)[None], cache=cache).logits[:, -1].astype(mx.float32); mx.eval(lg)
tok = mx.argmax(lg, axis=-1)

from kyojin_mlx import fuse as FU
FEATURES = {"s5": lambda on: setattr(FG, "ENABLED", on), "router": lambda on: setattr(FU, "ROUTER", on)}
for _f in ("MHC", "ATTN", "FQ", "ENGRAM"):                      # added as the fusions land
    if hasattr(FU, _f):
        FEATURES[_f.lower()] = (lambda f: lambda on: setattr(FU, f, on))(_f)
def variant(name):
    on = set(name.split("+")) - {"base"}
    def setup():
        for f, fn in FEATURES.items():
            fn(f in on or "all" in on)
    return setup
VARIANTS = {}
names = a.variants.split(",")
for n in names:
    VARIANTS[n] = variant(n)
res = {n: dict(step=[], build=[], enc=[]) for n in names}; outs = {}
for rep in range(a.reps + 3):
    for n in names:
        VARIANTS[n]()
        c = R.fork(cache)
        t0 = time.perf_counter(); lg = R.step(model, c, tok); t1 = time.perf_counter(); mx.async_eval(lg); t2 = time.perf_counter(); mx.eval(lg); t3 = time.perf_counter()
        if rep >= 3:
            res[n]["step"].append(t3 - t0); res[n]["build"].append(t1 - t0); res[n]["enc"].append(t2 - t1)
        outs[n] = np.array(lg)
    tok = mx.argmax(lg, axis=-1)
for n in names:
    r = res[n]
    print(f"{n:10s} step med {1e3 * statistics.median(r['step']):6.2f} min {1e3 * min(r['step']):6.2f} | build med {1e3 * statistics.median(r['build']):5.2f} | encode med {1e3 * statistics.median(r['enc']):5.2f} | cpu {1e3 * (statistics.median(r['build']) + statistics.median(r['enc'])):5.2f} ms | logits==base {bool((outs[n] == outs[names[0]]).all())}")
