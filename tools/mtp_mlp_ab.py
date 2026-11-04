import argparse, time, statistics
import mlx.core as mx, mlx.nn as nn
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--rows", default="1,2,4,8"); ap.add_argument("--passes", type=int, default=6); a = ap.parse_args()
from kyojin_mlx import mlxpack, q4mma as QM
model, tok = mlxpack.load(a.pack); lm = model.language_model
QM.install(model); L = lm.model.layers
def bench(fn, n=3, reps=4):
    ts = []
    for r in range(reps + 1):
        t0 = time.perf_counter(); mx.eval([fn() for _ in range(n)]); dt = time.perf_counter() - t0
        if r: ts.append(dt / n * 1e3)
    return statistics.median(ts)
res = {}
for ps in range(a.passes):
    for R in [int(r) for r in a.rows.split(",")]:
        x0 = mx.random.normal((1, R, 2560)).astype(mx.bfloat16)
        def chain():
            x = x0
            for l in L: x = x + l.mlp(l.post_attention_layernorm(x))
            return x
        for v in ("stock", "mma"):
            QM.MIN_ROWS = 2 if v == "mma" else 99
            res.setdefault((R, v), []).append(bench(chain))
for R in [int(r) for r in a.rows.split(",")]:
    print(f"R={R}: 32 MLP blocks stock {statistics.median(res[(R,'stock')]):.2f} ms  mma {statistics.median(res[(R,'mma')]):.2f} ms")
