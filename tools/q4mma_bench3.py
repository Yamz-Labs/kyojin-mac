"""Interleaved microbench: every pass cycles all (rows, impl) for one shape. Prints median over passes and min. usage: q4mma_bench3.py --shape gate --rows 1,2,3,4,6,8 --impls mlx,f,v1:4:4,v2:4:1 --passes 30"""
import argparse, time, statistics
import mlx.core as mx
from kyojin_mlx import q4mma as Q
ap = argparse.ArgumentParser(); ap.add_argument("--shape", default="gate"); ap.add_argument("--warm", type=float, default=0); ap.add_argument("--rows", default="1,2,3,4,6,8"); ap.add_argument("--impls", default="mlx,f,v1:4:4"); ap.add_argument("--passes", type=int, default=30); a = ap.parse_args()
SH = {"qkv": (8192, 2560), "z": (4096, 2560), "out": (2560, 4096), "gate": (9216, 2560), "down": (2560, 9216), "kv": (1024, 2560), "head": (248320, 2560)}
if a.warm:
    z = mx.random.normal((2048, 2048)).astype(mx.float16); t0 = time.perf_counter()
    while time.perf_counter() - t0 < a.warm: mx.eval([z @ z for _ in range(8)])
N, K = SH[a.shape]; L = 3 if a.shape == "head" else 24
ws = [mx.random.randint(0, 2**31 - 1, (N, K // 8)).astype(mx.uint32) * mx.array(2654435761, dtype=mx.uint32) for _ in range(L)]
scs = [mx.random.uniform(0.001, 0.02, (N, K // 64)).astype(mx.bfloat16) for _ in range(L)]
bis = [mx.random.uniform(-0.1, 0.0, (N, K // 64)).astype(mx.bfloat16) for _ in range(L)]
mx.eval(ws, scs, bis)
rows = [int(r) for r in a.rows.split(",")]; impls = a.impls.split(",")
xs = {R: mx.random.normal((R, K)).astype(mx.bfloat16) for R in rows}
px1 = {R: Q.prep_x(xs[R]) for R in rows}; px2 = {R: Q.prep_x2(xs[R]) for R in rows}
def mk(impl, R):
    x = xs[R]
    if impl == "mlx": return lambda i: mx.quantized_matmul(x, ws[i % L], scs[i % L], bis[i % L], transpose=True, group_size=64, bits=4)
    p = impl.split(":")
    if p[0] == "f": return lambda i: Q.q4fma(x, ws[i % L], scs[i % L], bis[i % L])
    if p[0] == "v1": return lambda i: Q.q4mma(px1[R], ws[i % L], scs[i % L], bis[i % L], R, int(p[1]), int(p[2]))
    if p[0] == "v4": return lambda i: Q.q4mma4(x, ws[i % L], scs[i % L], bis[i % L], R, int(p[1]), int(p[2]))
    if p[0] == "v3": return lambda i: Q.q4mma3(px1[R], ws[i % L], scs[i % L], bis[i % L], R, int(p[1]), int(p[2]))
    if p[0] == "v2": return lambda i: Q.q4mma2(px2[R], ws[i % L], scs[i % L], bis[i % L], R, int(p[1]), int(p[2]))
fns = {(R, im): mk(im, R) for R in rows for im in impls if not (im == "f" and R > 4)}
res = {k: [] for k in fns}
ref = {R: mx.quantized_matmul(xs[R], ws[0], scs[0], bis[0], transpose=True, group_size=64, bits=4).astype(mx.float32) for R in rows}
for (R, im), f in fns.items():
    e = float(mx.max(mx.abs(f(0).astype(mx.float32) - ref[R])) / mx.max(mx.abs(ref[R]))); print(f"err R={R} {im}: {e:.1e}", flush=True) if e > 1.5e-2 else None
for p in range(a.passes + 2):
    for k, f in fns.items():
        t0 = time.perf_counter(); mx.eval([f(i) for i in range(L)]); t = (time.perf_counter() - t0) / L * 1e6
        if p >= 2: res[k].append(t)
print(a.shape, "median us (min):")
for R in rows:
    print(f"  R={R}: " + "  ".join(f"{im} {statistics.median(res[(R, im)]):6.0f}({min(res[(R, im)]):4.0f})" for im in impls if (R, im) in res))
