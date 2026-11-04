#!/usr/bin/env python3
"""MMA-GEMV verify linear (rot + partial + finalize): chain up(2560->9216) -> down, per-piece cost, ABAB in one process. usage: mma_parts.py [--rows 5] [--variants full,part,rotfin]"""
import argparse, statistics, time
import numpy as np
import mlx.core as mx
from kyojin_mlx import kernels as KN, mma_gemv as MM
from exl3_roof import synth

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=5); ap.add_argument("--pool", type=int, default=12); ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--chain", type=int, default=12); ap.add_argument("--variants", default="full,part,rotfin")
    a = ap.parse_args()
    mx.set_cache_limit(256 << 20)
    R = a.rows
    ups = [synth(2560, 9216, 4, 10 + i) for i in range(a.pool)]; dns = [synth(9216, 2560, 4, 100 + i) for i in range(a.pool)]
    mx.eval(*[l._words for l in ups + dns])
    x0 = mx.array(np.random.default_rng(5).standard_normal((R, 2560)).astype(np.float16)); mx.eval(x0)
    def full(l, x, ws=4): return MM.finalize_rows(l._g[1], MM.partial_mma(l._g[1], KN.rot(x, l._g[1].suh), R, ws), R)
    def part(l, x, ws=4):
        g = l._g[1]; xh = x.astype(mx.float32)
        if xh.shape[1] < g.kt * 16: xh = mx.pad(xh, [(0, 0), (0, g.kt * 16 - xh.shape[1])])
        p = MM.partial_mma(g, xh, R, ws)[0]
        return p[:, : (2560 if l.out_features == 9216 else 9216)] if False else p
    def rotfin(l, x):
        g = l._g[1]; xh = KN.rot(x, g.suh)
        return MM.finalize_rows(g, mx.zeros((g.tgk, R, g.n), dtype=mx.float32), R)
    fs = {"full": full, "part": part, "rotfin": rotfin}
    for w in (1, 2, 4):
        fs[f"full{w}"] = (lambda w: lambda l, x: full(l, x, w))(w); fs[f"part{w}"] = (lambda w: lambda l, x: part(l, x, w))(w)
    names = a.variants.split(",")
    def once(n, k):
        f = fs[n]; x = x0
        for i in range(k):
            h = f(ups[i % a.pool], x)
            h = h[:, :9216] if h.shape[1] >= 9216 else mx.pad(h, [(0, 0), (0, 9216 - h.shape[1])])
            y = f(dns[i % a.pool], h.astype(mx.float16) if not n.startswith("part") else h)
            x = y[:, :2560] if y.shape[1] >= 2560 else y
        return x
    for n in names: mx.eval(once(n, 2))
    T = {n: [] for n in names}
    for rep in range(a.reps):
        for n in names:
            t0 = time.perf_counter(); mx.eval(once(n, a.chain)); T[n].append((time.perf_counter() - t0) / a.chain * 1e6 / 2)
    print(f"R={R}")
    for n in names: print(f"  {n:7s} min {min(T[n]):7.1f} med {statistics.median(T[n]):7.1f} us per linear", flush=True)
main()
