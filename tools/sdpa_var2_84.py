"""sdpa launch-shape sweep: base | per kv group (4 q heads) | 2 q heads per call | 1 q head per call; offsets o. usage: sdpa_var2_84.py [o,o,..]"""
import sys, time, mlx.core as mx
offs = [int(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 2048, 4096, 8192, 16384, 30720]
def bench(fn, it=4, blocks=5):
    mx.eval(fn()); best = 1e9
    for _ in range(blocks):
        mx.synchronize(); t0 = time.perf_counter(); ys = [fn() for _ in range(it)]; mx.eval(ys); best = min(best, (time.perf_counter() - t0) / it)
    return best
L = 2048; sc = 256 ** -0.5; S = mx.fast.scaled_dot_product_attention
for o in offs:
    N = o + L; dt = mx.bfloat16
    q = mx.random.normal((1, 16, L, 256)).astype(dt); k = mx.random.normal((1, 4, N, 256)).astype(dt); v = mx.random.normal((1, 4, N, 256)).astype(dt); mx.eval(q, k, v)
    fl = 4 * L * (o + (L + 1) / 2) * 16 * 256
    def split(hq):
        def f():
            return mx.concatenate([S(q[:, h:h + hq], k[:, h // 4:h // 4 + 1], v[:, h // 4:h // 4 + 1], scale=sc, mask="causal") for h in range(0, 16, hq)], axis=1)
        return f
    line = f"o={o:6d}"
    for name, fn in (("base", lambda: S(q, k, v, scale=sc, mask="causal")), ("g4", split(4)), ("g2", split(2)), ("g1", split(1))):
        t = bench(fn); line += f" | {name} {t*1e3:7.2f} ms {fl/t/1e12:5.2f} TF"
    print(line, flush=True)
