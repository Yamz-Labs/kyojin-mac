#!/usr/bin/env python3
"""Host (python graph build) cost per 1-row EXL3 linear call, no GPU wait: lin = Exl3Linear.__call__, gf = Mul1Gemv.fused, up to the cache.
usage: host_build.py [--calls 2000]"""
import argparse, statistics, time
import numpy as np
import mlx.core as mx
from exl3_roof import synth

ap = argparse.ArgumentParser(); ap.add_argument("--calls", type=int, default=1500); ap.add_argument("--reps", type=int, default=9); ap.add_argument("--rows", type=int, default=1)
a = ap.parse_args()
mx.set_cache_limit(256 << 20)
ls = [synth(2560, 2560, 4, 10 + i) for i in range(4)]
mx.eval(*[l._words for l in ls])
x0 = mx.array(np.random.default_rng(5).standard_normal((a.rows, 2560)).astype(np.float16)); mx.eval(x0)
fs = {"lin": lambda l, x: l(x), "gf": lambda l, x: l._g[1].fused(x)}
T = {n: [] for n in fs}
for rep in range(a.reps):
    for n, f in fs.items():
        outs = []
        t0 = time.perf_counter()
        for i in range(a.calls):
            outs.append(f(ls[i & 3], x0))
        T[n].append((time.perf_counter() - t0) / a.calls * 1e6)
        mx.eval(outs)
for n in fs: print(f"  {n:4s} host build min {min(T[n]):6.2f} med {statistics.median(T[n]):6.2f} us per call", flush=True)
