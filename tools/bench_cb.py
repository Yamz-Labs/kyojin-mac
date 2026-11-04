"""Partial-GEMV microbench per codebook / K / rows. usage: bench_cb.py [cb ...]  (mul1 mcg 3inst), shapes of the Qwen3.5-2B test model.
Prints Gweights/s and GB/s of trellis traffic. Random trellis (any bit pattern is a valid stream)."""
import sys, time, numpy as np, mlx.core as mx
from kyojin_mlx import kernels as KN, ref_numpy as R
def timeit(fn, iters=200):
    for _ in range(10): mx.eval(fn())
    mx.synchronize(); t0 = time.perf_counter(); o = [fn() for _ in range(iters)]; mx.eval(o); mx.synchronize()
    return (time.perf_counter() - t0) / iters
cbs = sys.argv[1:] or ["mul1", "mcg"]
shapes = [(2048, 6144, (2, 3)), (6144, 2048, (2, 3)), (2048, 248320, (6,))]
for cb in cbs:
    for k, n, Ks in shapes:
        for K in Ks:
            t, su, sv = R.random_linear(k, n, K, seed=K)
            for rows in (1, 4):
                lin = KN.Mul1Gemv(t, su, sv, rows=rows, cb=cb)
                x = mx.array(np.random.default_rng(0).standard_normal((rows, k)).astype(np.float16))
                xh = KN.rot(x, lin.suh); mx.eval(xh)
                dt = min(timeit(lambda: lin.partial(xh)) for _ in range(3))
                print(f"{cb:5s} K={K} {k}x{n} rows={rows}: {dt*1e6:8.1f} us {k*n/dt/1e9:6.1f} Gw/s {k*n*K/8/dt/1e9:6.1f} GB/s", flush=True)
