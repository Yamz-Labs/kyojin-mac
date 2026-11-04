"""Interleaved A/B of one trunk pass at S rows on an EXL3 Qwen3.5 pack: variants set exl3_linear knobs. usage: exl3_rows_ab.py PACK [--rows 5] [--reps 15]"""
import argparse, statistics, time
import mlx.core as mx
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--rows", type=int, default=5); ap.add_argument("--reps", type=int, default=15)
ap.add_argument("--variants", default="nax,def"); ap.add_argument("--chain", type=int, default=3); a = ap.parse_args()
from kyojin_mlx import exl3_linear as EL
from mlx_lm.models.cache import make_prompt_cache
model, cfg = EL.load_exl3(a.pack); lm = model.language_model
cache = make_prompt_cache(model)
mx.eval(lm.model(mx.arange(200)[None] % 1000 + 100, cache))
head = lm.__dict__.get("lm_head") or lm.lm_head
def setv(v):
    EL.MMA_ROWS, EL.NAX_KS = 0, 0
    if v == "def": EL.MMA_ROWS, EL.NAX_KS = -1, -1
    if v == "mma": EL.MMA_ROWS = 5
    elif v.startswith("ks"): EL.NAX_KS = int(v[2:])
S = a.rows; tok = mx.array([[5] * S])
def one(): return head(lm.model(tok, cache))
vs = a.variants.split(",")
for v in vs: setv(v); mx.eval(one())
T = {v: [] for v in vs}
for r in range(a.reps):
    for v in vs:
        setv(v); t0 = time.perf_counter()
        for _ in range(a.chain): mx.eval(one())
        T[v].append((time.perf_counter() - t0) / a.chain * 1e3)
for v in vs: print(f"S={S} {v:5s} min {min(T[v]):6.2f} med {statistics.median(T[v]):6.2f} ms")
