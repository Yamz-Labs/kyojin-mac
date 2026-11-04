"""GDN core / conv / norm microbench at the Qwen3.5 GDN shape (Hk 16, Hv 32, D 128), T rows. usage: gdn84.py [T,T,..]
stock = mlx_lm gated_delta_update (step kernel), fast = mx.fast.gated_delta_update. min over blocks, back-to-back evals (sustained-ish)."""
import sys, time, mlx.core as mx
from mlx_lm.models import gated_delta as G
Ts = [int(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 else [512, 2048, 4096]
def bench(fn, it=6, blocks=4):
    mx.eval(fn()); best = 1e9
    for _ in range(blocks):
        mx.synchronize(); t0 = time.perf_counter(); ys = [fn() for _ in range(it)]; mx.eval(ys); best = min(best, (time.perf_counter() - t0) / it)
    return best
for T in Ts:
    q = mx.random.normal((1, T, 16, 128)).astype(mx.bfloat16); k = mx.random.normal((1, T, 16, 128)).astype(mx.bfloat16); v = mx.random.normal((1, T, 32, 128)).astype(mx.bfloat16)
    a = mx.random.normal((1, T, 32)).astype(mx.bfloat16); b = mx.random.normal((1, T, 32)).astype(mx.bfloat16)
    A_log = mx.random.normal((32,)).astype(mx.float32); dtb = mx.random.normal((32,)).astype(mx.bfloat16); st = mx.zeros((1, 32, 128, 128), dtype=mx.float32)
    q, k = G.normalize_qk(q, k, inv_scale=128 ** -0.5, eps=1e-6)
    mx.eval(q, k, v, a, b, A_log, dtb, st)
    g = G.compute_g(A_log, a, dtb); beta = mx.sigmoid(b); mx.eval(g, beta)
    ts = bench(lambda: G.gated_delta_update(q, k, v, a, b, A_log, dtb, st)[0])
    tf = bench(lambda: mx.fast.gated_delta_update(q, k, v, g, beta, st)[0])
    tg = bench(lambda: G.compute_g(A_log, a, dtb) + mx.sigmoid(b)[..., :1] * 0)
    y1 = G.gated_delta_update(q, k, v, a, b, A_log, dtb, st)[0].astype(mx.float32); y2 = mx.fast.gated_delta_update(q, k, v, g, beta, st)[0].astype(mx.float32)
    err = float(mx.max(mx.abs(y1 - y2))); ref = float(mx.max(mx.abs(y1)))
    fl = T * 5.8e6
    print(f"T={T:5d}: stock {ts*1e3:7.3f} ms | fast {tf*1e3:7.3f} ms ({fl/tf/1e12:5.2f} TF, {ts/tf:.2f}x) | g/beta prep {tg*1e3:6.3f} ms | max|dy| {err:.3g} of {ref:.3g}", flush=True)
