"""Launches (graph dispatches) and CPU (python build + async_eval encode) of the chained verify of 1..N rows and of the solo step, with an optional cProfile
of the python build of one verify depth.
usage: verify_census.py PACK [--len 135] [--reps 25] [--rows 5] [--prof N] [--json F]     (M10_LIGHT=F as in spec_cpu: GPU-light custom kernels)
Warm census: the first call of a fused block runs the stock code, so every depth is run twice before it is counted."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("M10_LIGHT"):
    import light
os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "100000"); os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "1000000")

import mlx.core as mx
import numpy as np

from kyojin_mlx import patch as P, rows as R, spec as S
from count_launches import graph_census

ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("--len", type=int, default=135); ap.add_argument("--reps", type=int, default=25); ap.add_argument("--rows", type=int, default=5)
ap.add_argument("--prof", type=int, default=0, help="cProfile the build of this verify depth"); ap.add_argument("--types", type=int, default=0, help="print the node types of the graph of depth TYPES - 1 (1 = solo)")
ap.add_argument("--advance", action="store_true", help="every rep continues from the previous one of the same depth (new positions); default: fixed position")
ap.add_argument("--diff", action="store_true", help="print the node types each extra row adds (depth n minus depth n - 1)")
ap.add_argument("--json"); ap.add_argument("--set", nargs="*", default=[], help="text_config overrides key=json (a layer-kind slice: compress_ratios=[0,0] kv_source_layer_ids=[] ...)")
ap.add_argument("--first", type=int, default=0, help="keep only the first N layers (layer 0 alone = a full layer without the reuse layer behind it)")
ap.add_argument("--variant", default="full", choices=["full", "reindex"], help="reindex: after the prefill the compressor is switched off (the indexer of a reindex layer reads the shared compressed KV)")
a = ap.parse_args()
import json as _json
model, spec, cfg = P.build_model(a.pack, {k: _json.loads(v) for k, v in (x.split("=", 1) for x in a.set)})
P.patch_model(model, a.pack, spec, cfg)
R.enable(model, spec)
S.install(model, spec)
if a.first:
    model.language_model.layers = model.language_model.layers[:a.first]
tc = cfg["text_config"]
rng = np.random.default_rng(1)
prompt = rng.integers(16, tc["vocab_size"] - 16, a.len).tolist()
cache, lg, _, _ = S.prefill(model, prompt)
if a.variant == "reindex":
    import importlib
    importlib.import_module("mlx_vlm.models.deepseek_v41.language").Compressor.__call__ = lambda self, x, start_pos, cache: None
tok = [int(mx.argmax(lg[0]).item())] + rng.integers(16, tc["vocab_size"] - 16, a.rows).tolist()


STATS = {}
try:
    from kyojin_mlx import fuse_layer as FL
    STATS = getattr(FL, "STATS", {})
except ImportError:
    pass
cur = {}


def run(n, advance=False):
    """One solo step (n = 0) or one chained verify of n rows. advance: continue from the previous call of the same depth (every rep at a new position, so a
    per-position retrace or cache miss shows up); else every call starts from the same cache (fixed position)."""
    if not advance:
        return R.step(model, R.fork(cache), mx.array([tok[0]])) if n == 0 else S.verify(model, cache, tok[:n])[0]
    c = cur.setdefault(n, R.fork(cache))
    if n == 0:
        return R.step(model, c, mx.array([int(rng.integers(16, tc["vocab_size"] - 16))]))
    lg, _, chain = S.verify(model, c, [int(t) for t in rng.integers(16, tc["vocab_size"] - 16, n)])
    cur[n] = S.commit(chain, n - 1)[0]
    return lg


res = {}
BT = {}
for n in range(0, a.rows + 1):
    for _ in range(3):
        mx.eval(run(n, a.advance))
    cens = []
    for _ in range(4 if a.advance else 1):                               # advancing: the average over consecutive positions (a closing step has more launches)
        o = run(n, a.advance); cens.append(graph_census(o)); mx.eval(o)
    BT[n] = cens[0]["by_type"]
    cen = dict(cens[0]); cen["dispatches"] = sum(c["dispatches"] for c in cens) / len(cens); cen["nodes"] = sum(c["nodes"] for c in cens) / len(cens)
    if a.types and a.types - 1 == n:
        print(f"-- node types of depth {n}:", {k: v for k, v in cen["by_type"].items()})
    snap = dict(STATS)
    ts = []
    for _ in range(a.reps):
        t0 = time.perf_counter(); out = run(n, a.advance); t1 = time.perf_counter(); mx.async_eval(out); t2 = time.perf_counter(); mx.eval(out)
        ts.append((t1 - t0, t2 - t1))
    cpu = [b + e for b, e in ts]
    name = "solo" if n == 0 else f"verify{n}"
    res[name] = dict(launches=cen["dispatches"], nodes=cen["nodes"], build_min=1e3 * min(b for b, _ in ts), enc_min=1e3 * min(e for _, e in ts), cpu_min=1e3 * min(cpu), cpu_med=1e3 * sorted(cpu)[len(cpu) // 2],
                     stats_during_reps={k: STATS[k] - snap.get(k, 0) for k in STATS})
for k, v in res.items():
    v["x_solo"] = v["cpu_min"] / res["solo"]["cpu_min"]
    print(f"{k:8s} launches {v['launches']:6.1f} nodes {v['nodes']:6.1f} | build {v['build_min']:6.2f} enc {v['enc_min']:6.2f} cpu min {v['cpu_min']:6.2f} med {v['cpu_med']:6.2f} ms | x solo {v['x_solo']:.2f} {v['stats_during_reps'] or ''}")
if a.diff:
    from count_launches import VIEWS
    for n in range(2, a.rows + 1):
        d = {k: BT[n].get(k, 0) - BT[n - 1].get(k, 0) for k in set(BT[n]) | set(BT[n - 1])}
        d = {k: v for k, v in sorted(d.items(), key=lambda kv: -abs(kv[1])) if v and k not in VIEWS}
        print(f"-- row {n} adds (dispatching node types):", d, "total", sum(d.values()))
if a.prof:
    import cProfile, pstats
    for _ in range(3):
        run(a.prof)
    pr = cProfile.Profile(); pr.enable()
    for _ in range(20):
        run(a.prof)
    pr.disable()
    pstats.Stats(pr).sort_stats("tottime").print_stats(28)
if a.json:
    json.dump(res, open(a.json, "w"), indent=1)
