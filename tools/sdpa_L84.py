import sys, time, mlx.core as mx
S = mx.fast.scaled_dot_product_attention
def bench(fn, it=3, blocks=4):
    mx.eval(fn()); best = 1e9
    for _ in range(blocks):
        mx.synchronize(); t0 = time.perf_counter(); ys = [fn() for _ in range(it)]; mx.eval(ys); best = min(best, (time.perf_counter() - t0) / it)
    return best
for o in (8192, 24576):
    for L in (512, 1024, 2048, 4096, 8192):
        N = o + L; q = mx.random.normal((1, 16, L, 256)).astype(mx.bfloat16); k = mx.random.normal((1, 4, N, 256)).astype(mx.bfloat16); v = mx.random.normal((1, 4, N, 256)).astype(mx.bfloat16); mx.eval(q, k, v)
        t = bench(lambda: S(q, k, v, scale=0.0625, mask="causal")); fl = 4 * L * (o + (L + 1) / 2) * 16 * 256
        print(f"o={o} L={L:5d}: {t*1e3:8.2f} ms  {fl/t/1e12:5.2f} TF", flush=True)
