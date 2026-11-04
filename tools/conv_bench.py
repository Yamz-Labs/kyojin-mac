"""Viterbi throughput on this machine (tiles/s, mul1; the work per tile is 512 steps x 65536 edge evaluations for every K).
impl 2 = register-resident codebook + threadgroup cost row (K2-5, default), impl 1 = device-memory kernel (table lookup; K1, K6-8 fall back to it).
usage: python tools/conv_bench.py [--tiles 768] [--K 2 3 4] [--impl 1 2] [--reps 5]   (run under lockf on the Mac). Rounds are interleaved over (K, impl); min and p25 are reported."""
import argparse, sys, time
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kyojin_mlx import conv_viterbi as V

ap = argparse.ArgumentParser()
ap.add_argument("--tiles", type=int, default=768); ap.add_argument("--K", type=int, nargs="+", default=[2, 3])
ap.add_argument("--impl", type=int, nargs="+", default=[1, 2]); ap.add_argument("--reps", type=int, default=5)
a = ap.parse_args()
x = np.random.default_rng(0).standard_normal((a.tiles, 256)).astype(np.float32)
combos = [(K, im) for K in a.K for im in a.impl]
ts = {c: [] for c in combos}
for K, im in combos:
    V.CFG["impl"] = im; V.quantize_tiles(x[:8], K)
for _ in range(a.reps):
    for K, im in combos:
        V.CFG["impl"] = im
        t0 = time.perf_counter(); w, i = V.quantize_tiles(x, K); mx.eval(w, i); ts[(K, im)].append(a.tiles / (time.perf_counter() - t0))
for (K, im), v in ts.items():
    v.sort(reverse=True)
    print(f"K={K} impl={im}: min-time {v[0]:.0f} p25 {v[len(v)//4]:.0f} tiles/s", flush=True)
