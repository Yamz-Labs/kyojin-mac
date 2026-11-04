"""Sustained trunk probe: forward of S rows (q4mma for S>=5), 25 s phases, ms per forward in 5 s slices. Shows whether S=5 slows down over time and whether S=1 slows down after S=5 heat.
usage: clock_probe2.py PACK [--plan 1,5,1,5] [--phase 25]"""
import argparse, time, mlx.core as mx
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--plan", default="1,5,1,5"); ap.add_argument("--phase", type=float, default=25); a = ap.parse_args()
from kyojin_mlx import mlxpack, q4mma as QM
from mlx_lm.models.cache import make_prompt_cache
model, tok = mlxpack.load(a.pack); lm = model.language_model
QM.install(model); cache = make_prompt_cache(model)
mx.eval(lm.model(mx.arange(100)[None] % 1000 + 100, cache))
L = lm.model.layers
for S in [int(v) for v in a.plan.split(",")]:
    QM.MIN_ROWS = 99 if S == 1 else 1
    x = mx.array([[5] * S]); sl = []; t0 = time.perf_counter(); cur = []; ts = t0
    while time.perf_counter() - t0 < a.phase:
        s = time.perf_counter(); mx.eval(lm.model(x, cache)); cur.append((time.perf_counter() - s) * 1e3)
        for l, c in zip(L, cache):
            if not l.is_linear: c.trim(S)
        if time.perf_counter() - ts > 5: sl.append(sum(cur) / len(cur)); cur = []; ts = time.perf_counter()
    print(f"S={S}: " + " ".join(f"{v:.1f}" for v in sl) + " ms/fwd per 5 s slice", flush=True)
