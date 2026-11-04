#!/usr/bin/env python3
"""A/B in one process of the chained verify of n rows (n = 1..16), GEMV chunks of 4 rows vs the fused NAX kernel for the dense linears above 4 rows
(rows.enable(nax=...)), interleaved so that the load of the Air hits both alike. Routed experts keep their GEMV path in both arms.
usage: verify_cost_nax.py PACK [--rows 1,2,3,4,5,6,8,12,16] [--reps 9] [--len 135] [--union]
Prints median ms per arm, the ratio to the 1-row verify of the same arm, and the NAX gain."""
import argparse, statistics, time
import mlx.core as mx
import numpy as np
from kyojin_mlx import exl3_linear as EL, patch as P, rows as R, spec as S
import count_launches as CL

ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("--rows", default="1,2,3,4,5,6,8,12,16"); ap.add_argument("--reps", type=int, default=9)
ap.add_argument("--len", type=int, default=135); ap.add_argument("--union", action="store_true")
a = ap.parse_args()
mx.set_cache_limit(512 << 20)
model, spec, cfg = P.build_model(a.pack, {})
P.patch_model(model, a.pack, spec, cfg)
R.enable(model, spec, union=a.union, nax=False)
S.install(model, spec, tap_layers=())
tc = cfg["text_config"]
rng = np.random.default_rng(1)
prompt = rng.integers(16, tc["vocab_size"] - 16, a.len).tolist()
cache, lg, taps, first = S.prefill(model, prompt, ())
rows_l = [int(v) for v in a.rows.split(",")]
tok = [int(mx.argmax(lg[0]).item())] + rng.integers(16, tc["vocab_size"] - 16, max(rows_l)).tolist()
launches = {}
for n in rows_l:
    for nx in (False, True):
        EL.ROWS_NAX = nx
        lgs, _, _ = S.verify(model, cache, tok[:n])
        launches[(n, nx)] = CL.graph_census(lgs, n)["dispatches"]
        mx.eval(lgs)
times = {(n, nx): [] for n in rows_l for nx in (False, True)}
for rep in range(a.reps + 2):
    for n in rows_l:
        for nx in (False, True):
            EL.ROWS_NAX = nx
            t = time.perf_counter(); lgs, _, _ = S.verify(model, cache, tok[:n]); mx.eval(lgs); dt = time.perf_counter() - t
            if rep >= 2:
                times[(n, nx)].append(dt)
med = {k: 1e3 * statistics.median(v) for k, v in times.items()}
b0, n0 = med[(rows_l[0], False)], med[(rows_l[0], True)]
print(f"{'rows':>4s} {'GEMV ms':>9s} {'xrow1':>6s} {'NAX ms':>9s} {'xrow1':>6s} {'gain':>6s} {'launches GEMV/NAX':>18s}")
for n in rows_l:
    g, x = med[(n, False)], med[(n, True)]
    print(f"{n:4d} {g:9.2f} {g / b0:6.2f} {x:9.2f} {x / n0:6.2f} {g / x:6.2f} {launches[(n, False)]:9d}/{launches[(n, True)]:<8d}")
print(f"peak memory {mx.get_peak_memory() / 2**30:.2f} GiB")
