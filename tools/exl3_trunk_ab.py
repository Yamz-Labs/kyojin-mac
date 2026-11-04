"""Trunk forward (S rows, + head) ms of an EXL3 Qwen3.5 pack for several KYOJIN_MMA_ROWS thresholds (0 = GEMV), alternating in one process.
usage: exl3_trunk_ab.py PACK [--rows 1,2,3,4,5,6,8] [--mma 0,1,2,4] [--passes 3] [--warm SECONDS]"""
import argparse, time, statistics
import mlx.core as mx, mlx.nn as nn
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--rows", default="1,2,3,4,5,6,8"); ap.add_argument("--mma", default="0,1,2,4")
ap.add_argument("--passes", type=int, default=3); ap.add_argument("--warm", type=float, default=0); a = ap.parse_args()
from kyojin_mlx import exl3_linear as EL
from mlx_lm.models.cache import make_prompt_cache
model, cfg = EL.load_exl3(a.pack); lm = model.language_model
head = lm.__dict__.get("lm_head") or (lambda h: lm.model.embed_tokens.as_linear(h))
cache = make_prompt_cache(model)
mx.eval(lm.model(mx.arange(200)[None] % 1000 + 100, cache))
def bench(fn, n=6, reps=5):
    ts = []
    for r in range(reps + 1):
        t0 = time.perf_counter(); outs = [fn() for _ in range(n)]; mx.eval(outs); dt = time.perf_counter() - t0
        if r: ts.append(dt / n * 1e3)
    return statistics.median(ts)
rows = [int(x) for x in a.rows.split(",")]; modes = [int(x) for x in a.mma.split(",")]
t0 = time.time()
while time.time() - t0 < a.warm: bench(lambda: lm.model(mx.array([[5] * 4]), cache), n=4, reps=1)
res = {}
for ps in range(a.passes):
    for S in rows:
        x = mx.array([[5] * S])
        for m in modes:
            EL.MMA_ROWS = m
            res.setdefault((S, m), []).append((bench(lambda: lm.model(x, cache)), bench(lambda: head(lm.model(x, cache)))))
for S in rows:
    print(f"S={S} " + "  ".join(f"mma{m}: {statistics.median(x[1] for x in res[(S, m)]):6.2f}" for m in modes) + "  (trunk+head ms)", flush=True)
