"""Cost of one dependent small GPU launch in an MLX graph (chain of n tiny ops, one eval): the per-launch floor the step cannot go below."""
import time, statistics
import mlx.core as mx
x0 = mx.ones((1, 5120), mx.float16); mx.eval(x0)
for n in (500, 2000, 500, 2000, 4000):
    ts = []
    for _ in range(15):
        x = x0
        t = time.perf_counter()
        for _ in range(n):
            x = x * 1.0001
        mx.eval(x)
        ts.append((time.perf_counter() - t) / n)
    print(f"chain of {n} dependent elementwise launches: {1e6 * statistics.median(ts):.2f} us per launch (graph build + encode + run)")
