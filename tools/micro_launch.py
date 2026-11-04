#!/usr/bin/env python3
"""Cost of one launch on this machine: N dependent tiny ops in one graph, built-in op vs custom Metal kernel (the kind Exl3Linear uses).
usage: micro_launch.py [--n 600] [--reps 15]
Per op, microseconds: build = Python + C++ node creation (wall), submit_cpu = main-thread CPU of mx.async_eval (graph walk + Metal command encoding),
gpu = wall clock from submit to completion of the whole chain divided by N (the dispatch gap of dependent kernels on the GPU; tiny work per kernel)."""
import argparse
import statistics
import time

import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=600); ap.add_argument("--reps", type=int, default=15)
a = ap.parse_args()

src = "uint i = thread_position_in_grid.x; y[i] = x[i] + 1.0f;"
kern = mx.fast.metal_kernel(name="micro_inc", input_names=["x"], output_names=["y"], source=src)
kern4 = mx.fast.metal_kernel(name="micro_inc4", input_names=["x", "a", "b", "c"], output_names=["y"], source=src)
x0 = mx.zeros((2048,), dtype=mx.float32)
aux = [mx.zeros((2048,), dtype=mx.float32) for _ in range(3)]
mx.eval(x0, *aux)


def chain(kind, n):
    x = x0
    for _ in range(n):
        if kind == "builtin add":
            x = x + 1.0
        elif kind == "custom kernel (1 input)":
            x = kern(inputs=[x], grid=(2048, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(2048,)], output_dtypes=[mx.float32])[0]
        else:
            x = kern4(inputs=[x, *aux], grid=(2048, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(2048,)], output_dtypes=[mx.float32])[0]
    return x


res = {}
for kind in ("builtin add", "custom kernel (1 input)", "custom kernel (4 inputs)"):
    b, s, g = [], [], []
    for _ in range(a.reps + 3):
        t0 = time.perf_counter(); x = chain(kind, a.n); t1 = time.perf_counter()
        c0 = time.thread_time(); mx.async_eval(x); c1 = time.thread_time(); t2 = time.perf_counter()
        mx.eval(x); t3 = time.perf_counter()
        b.append((t1 - t0) / a.n * 1e6); s.append((c1 - c0) / a.n * 1e6); g.append((t3 - t2) / a.n * 1e6)
    res[kind] = (statistics.median(b[3:]), statistics.median(s[3:]), statistics.median(g[3:]))
    print(f"{kind:26s} build {res[kind][0]:6.2f} us/op  submit cpu {res[kind][1]:6.2f} us/op  gpu (dependent chain, incl. residual submit wait) {res[kind][2]:6.2f} us/op", flush=True)
