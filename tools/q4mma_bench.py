"""q4mma vs mx.quantized_matmul: error and time per call for the 4B shapes. usage: q4mma_bench.py [--rows 1,2,4,8] [--nt 4] [--ws 4]"""
import argparse, time, statistics
import mlx.core as mx
from kyojin_mlx import q4mma as Q
ap = argparse.ArgumentParser(); ap.add_argument("--rows", default="1,2,4,8"); ap.add_argument("--nt", type=int, default=4); ap.add_argument("--ws", type=int, default=4); ap.add_argument("--shapes", default="all"); a = ap.parse_args()
SH = {"qkv": (8192, 2560), "z": (4096, 2560), "out": (2560, 4096), "gate": (9216, 2560), "down": (2560, 9216), "kv": (1024, 2560), "head": (248320, 2560)}
if a.shapes != "all": SH = {k: SH[k] for k in a.shapes.split(",")}
def bench(fn, n, reps=7):
    ts = []
    for r in range(reps + 2):
        t0 = time.perf_counter(); mx.eval([fn(i) for i in range(n)]); dt = time.perf_counter() - t0
        if r >= 2: ts.append(dt / n * 1e3)
    return statistics.median(ts)
for name, (N, K) in SH.items():
    L = 4 if name == "head" else 12
    ws = [(mx.random.randint(0, 2**31 - 1, (N, K // 8)).astype(mx.uint32) * mx.array(2654435761, dtype=mx.uint32)) for _ in range(L)]
    scs = [(mx.random.uniform(0.001, 0.02, (N, K // 64))).astype(mx.bfloat16) for _ in range(L)]
    bis = [(mx.random.uniform(-0.1, 0.0, (N, K // 64))).astype(mx.bfloat16) for _ in range(L)]
    mx.eval(ws, scs, bis)
    for R in [int(r) for r in a.rows.split(",")]:
        x = mx.random.normal((R, K)).astype(mx.bfloat16); mx.eval(x)
        ref = mx.quantized_matmul(x, ws[0], scs[0], bis[0], transpose=True, group_size=64, bits=4)
        xt = Q.prep_x(x); y = Q.q4mma(xt, ws[0], scs[0], bis[0], R, a.nt, a.ws)
        err = float(mx.max(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32))) / (mx.max(mx.abs(ref.astype(mx.float32))) + 1e-9))
        t_ref = bench(lambda i: mx.quantized_matmul(x, ws[i % L], scs[i % L], bis[i % L], transpose=True, group_size=64, bits=4), L)
        t_prep = bench(lambda i: Q.prep_x(x + i), 8)
        t_new = bench(lambda i: Q.q4mma(xt, ws[i % L], scs[i % L], bis[i % L], R, a.nt, a.ws), L)
        gb = N * K / 2 / 1e6
        print(f"{name:5s} N={N:6d} K={K} R={R}: mlx {t_ref*1e3:7.0f} us  q4mma {t_new*1e3:7.0f} us ({gb/t_new:5.0f} GB/s)  prep {t_prep*1e3:4.0f} us  relerr {err:.1e}", flush=True)
