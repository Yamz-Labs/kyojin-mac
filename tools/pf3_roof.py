"""Linear roof: sum over the model's Exl3Linear of an fp16 [R,K]@[K,N] MLX matmul (resident weights) vs FLOP roof. usage: pf3_roof.py MODEL_DIR R"""
import sys, time, collections, mlx.core as mx
from kyojin_mlx import exl3_linear as EL
from kyojin_mlx.serve.backends import load_backend
be, _ = load_backend(sys.argv[1]); R = int(sys.argv[2]); mx.set_cache_limit(256 * 2 ** 20)
shapes = collections.Counter()
for _, m in be.model.named_modules():
    if isinstance(m, EL.Exl3Linear) and m.out_features < 100000: shapes[(m.in_features, m.out_features)] += 1
fl = sum(2 * R * k * n * c for (k, n), c in shapes.items())
tot = 0
for (k, n), c in shapes.items():
    x = mx.random.normal((R, k)).astype(mx.float16); w = mx.random.normal((n, k)).astype(mx.float16); mx.eval(x, w)
    f = lambda: x @ w.T
    mx.eval(f()); ts = []
    for _ in range(6):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(*[f() for _ in range(4)]); ts.append((time.perf_counter() - t0) / 4)
    t = min(ts); tot += t * c
    print(f"  K={k} N={n} x{c}: {t*1e3:.3f} ms  {2*R*k*n/t/1e12:.1f} TF")
print(f"R={R}: {sum(shapes.values())} linears, {fl/1e12:.2f} TFLOP, matmul-roof total {tot*1e3:.0f} ms, {fl/tot/1e12:.1f} TF avg")
