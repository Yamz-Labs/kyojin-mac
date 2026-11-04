"""Trunk forward (S rows, + head) ms: stock MLX qmv vs q4mma rows, alternating (same thermal state). usage: mtp_trunk_ab.py PACK [--rows 1,2,3,4,5,6,8] [--passes 3]"""
import argparse, time, statistics
import mlx.core as mx, mlx.nn as nn
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--rows", default="1,2,3,4,5,6,8"); ap.add_argument("--passes", type=int, default=3); a = ap.parse_args()
from kyojin_mlx import mlxpack, q4mma as QM
from mlx_lm.models.cache import make_prompt_cache
model, tok = mlxpack.load(a.pack); lm = model.language_model
class _T(nn.Module):
    def __init__(s, e): super().__init__(); s.__dict__["_e"] = e
    def __call__(s, h): return s._e.as_linear(h)
lm.__dict__["lm_head"] = _T(lm.model.embed_tokens)
print("installed", QM.install(model), flush=True)
cache = make_prompt_cache(model)
p = tok.apply_chat_template([{"role": "user", "content": "Explain how a hash map handles collisions."}], tokenize=True, add_generation_prompt=True, enable_thinking=False)
p = p["input_ids"] if not isinstance(p, list) else p
QM.MIN_ROWS = 99
mx.eval(lm.model(mx.array(p)[None], cache))
def bench(fn, n=6, reps=5):
    ts = []
    for r in range(reps + 1):
        t0 = time.perf_counter(); outs = [fn() for _ in range(n)]; mx.eval(outs); dt = time.perf_counter() - t0
        if r: ts.append(dt / n * 1e3)
    return statistics.median(ts)
rows = [int(x) for x in a.rows.split(",")]
res = {}
for ps in range(a.passes):
    for S in rows:
        x = mx.array([[p[-1]] * S])
        for v in ("stock", "mma"):
            QM.MIN_ROWS = 1 if v == "mma" else 99
            t_tr = bench(lambda: lm.model(x, cache)); t_h = bench(lambda: lm.lm_head(lm.model(x, cache)))
            res.setdefault((S, v), []).append((t_tr, t_h))
for S in rows:
    for v in ("stock", "mma"):
        if (S, v) in res:
            r = res[(S, v)]; print(f"S={S} {v:5s}: trunk {statistics.median(x[0] for x in r):6.2f} ms  trunk+head {statistics.median(x[1] for x in r):6.2f} ms")
