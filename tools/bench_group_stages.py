#!/usr/bin/env python3
"""Stage-by-stage GPU cost of the S5 groups of a real EXL3 model: rot / GEMV / finalize alone (inputs of the stage already evaluated), separate members vs
the fused group. usage: bench_group_stages.py MODEL_DIR [--reps 40]"""
import argparse
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from kyojin_mlx import fused_group as FG
from kyojin_mlx import kernels as KN
from kyojin_mlx.exl3_linear import load_exl3

ap = argparse.ArgumentParser()
ap.add_argument("model_dir"); ap.add_argument("--reps", type=int, default=40)
a = ap.parse_args()
model, _ = load_exl3(a.model_dir)
groups = FG.discover(model, lambda: model(mx.array([[0]]), cache=make_prompt_cache(model)))
rng = np.random.default_rng(0)
xs = [mx.array((rng.standard_normal((1, g[0][1].in_features)) * 2).astype(np.float16)) for g in groups]
fused = [FG.Exl3Group([m for _, m in g]) for g in groups]
gs = [[m._gemv(1) for _, m in g] for g in groups]
xh_sep = [[KN.rot(x, m.suh) for m in ms] for ms, x in zip(gs, xs)]
xh_fus = [f.rot(x) for f, x in zip(fused, xs)]
part_sep = [[m.partial(h) for m, h in zip(ms, hs)] for ms, hs in zip(gs, xh_sep)]
part_fus = [f.gemv(h, 1) for f, h in zip(fused, xh_fus)]
mx.eval(*[t for l in xh_sep + part_sep for t in l], *xh_fus, *part_fus)
stages = {
    "rot   sep": lambda: [KN.rot(x, m.suh) for ms, x in zip(gs, xs) for m in ms],
    "rot   fus": lambda: [f.rot(x) for f, x in zip(fused, xs)],
    "gemv  sep": lambda: [m.partial(h) for ms, hs in zip(gs, xh_sep) for m, h in zip(ms, hs)],
    "gemv  fus": lambda: [f.gemv(h, 1) for f, h in zip(fused, xh_fus)],
    "final sep": lambda: [m.finalize(p) for ms, ps in zip(gs, part_sep) for m, p in zip(ms, ps)],
    "final fus": lambda: [y for f, p in zip(fused, part_fus) for y in f.fin(p, 1)],
}
for fn in stages.values():
    mx.eval(*fn())
ts = {k: [] for k in stages}
for _ in range(a.reps):
    for k, fn in stages.items():
        outs = fn(); t0 = time.perf_counter(); mx.eval(*outs); ts[k].append(time.perf_counter() - t0)
n = len(groups)
for k in stages:
    print(f"{k}: min {min(ts[k]) * 1e3:7.3f} ms  median {statistics.median(ts[k]) * 1e3:7.3f} ms   ({n} groups, {min(ts[k]) / n * 1e6:6.1f} us per group)")
