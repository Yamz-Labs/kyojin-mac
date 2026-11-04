"""Launches and CPU ms of the DSpark draft call and of the drafter's `append` (rows 1..block), eager vs compiled, at ADVANCING positions (a retrace would show as a spike
and in dspark.STATS). usage: draft_census.py PACK [--len 135] [--reps 40] [--json F]; run it twice, KYOJIN_DSPARK_COMPILE=0 and 1.
Per call: build = python graph build, enc = async_eval encode (CPU), launches = dispatches of the graph (views excluded). Medians and minima over reps."""
import argparse, json, os, statistics, sys, time
os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "100000"); os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "1000000")     # one command buffer: encode never waits for the GPU
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("M10_LIGHT"):
    import light

import mlx.core as mx
import numpy as np

from count_launches import graph_census
from kyojin_mlx import dspark as D, patch as P, rows as R, spec as S

ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("--len", type=int, default=135); ap.add_argument("--reps", type=int, default=40); ap.add_argument("--json"); ap.add_argument("--types", type=int, default=0)
a = ap.parse_args()
model, spec, cfg = P.build_model(a.pack)
P.patch_model(model, a.pack, spec, cfg)
R.enable(model, spec)
tc = cfg["text_config"]
drafter = D.build_drafter(model, a.pack, spec, cfg, conf_threshold=0.0)
S.install(model, spec, tap_layers=set(drafter.tap_ids))
rng = np.random.default_rng(1)
prompt = rng.integers(16, tc["vocab_size"] - 16, a.len).tolist()
cache, lg, taps, first = S.prefill(model, prompt, drafter.tap_ids)
drafter.append([taps[l] for l in drafter.tap_ids], first)
mx.eval(drafter.ctx.rows)
H = tc["hidden_size"]
real_eval = mx.eval
res = {}


def timed_draft(p0):
    mark = {}

    def hook(*xs):
        mark["b"] = time.perf_counter()
        cen = graph_census(list(xs)); mark["c"] = cen["dispatches"]
        if a.types and "printed" not in res and cen["dispatches"] > 100:
            res["printed"] = 1
            print("draft node types:", {k: v for k, v in list(cen["by_type"].items())[:a.types]})
        t = time.perf_counter()
        mx.async_eval(*xs); mark["e"] = time.perf_counter() - t
        real_eval(*xs)
    mx.eval = hook
    t = time.perf_counter(); drafter.draft(int(rng.integers(16, tc["vocab_size"] - 16)), p0, 3)
    mx.eval = real_eval
    return mark["b"] - t, mark["e"], mark["c"]


def timed_append(n, p0):
    tp = [mx.random.normal((n, H)).astype(mx.float16) for _ in drafter.tap_ids]
    mx.eval(tp)
    t = time.perf_counter(); drafter.append(tp, p0); b = time.perf_counter() - t
    out = [r for r in drafter.ctx.rows]
    c = graph_census(out)["dispatches"]
    t = time.perf_counter(); mx.async_eval(*out); e = time.perf_counter() - t
    real_eval(*out)
    return b, e, c


p = a.len
for kind in ["draft"] + [f"append{n}" for n in (1, 2, 3, 4)]:
    rows = []
    for rep in range(a.reps + 3):
        if kind == "draft":
            r = timed_draft(p)
        else:
            r = timed_append(int(kind[6:]), p)
        p += 1
        if rep >= 3:
            rows.append(r)
    b, e, c = zip(*rows)
    res[kind] = dict(build_min=1e3 * min(b), enc_min=1e3 * min(e), cpu_min=1e3 * min(x + y for x, y in rows if True) if False else 1e3 * min(x + y for x, y, _ in rows),
                     cpu_med=1e3 * statistics.median(x + y for x, y, _ in rows), launches=statistics.median(c))
    print(f"{kind:8s} launches {res[kind]['launches']:6.0f} | build {res[kind]['build_min']:.2f} + enc {res[kind]['enc_min']:.2f} ms | cpu min {res[kind]['cpu_min']:.2f} med {res[kind]['cpu_med']:.2f} ms")
print("dspark.STATS", D.STATS, "compile", D.COMPILE)
if a.json:
    json.dump(res, open(a.json, "w"), indent=1)
