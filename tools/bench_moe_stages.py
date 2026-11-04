"""GPU time of each stage of one MoeGather call (synced stage by stage, per-eval floor subtracted), plain vs union, at R rows and a given routing.
usage: bench_moe_stages.py PACK [--layer 1] [--rows 4] [--uniq all|disjoint|N] [--reps 30]"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mlx.core as mx
from kyojin_mlx import patch as P, rows as R, moe as M

ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("--layer", type=int, default=1); ap.add_argument("--rows", type=int, default=4); ap.add_argument("--uniq", default="all"); ap.add_argument("--reps", type=int, default=30)
a = ap.parse_args()
model, spec, cfg = P.build_model(a.pack)
P.patch_model(model, a.pack, spec, cfg)
R.enable(model, spec)
moe = model.language_model.layers[a.layer].ffn.switch_mlp.moe
E, topk = moe.g.experts, moe.topk
rng = np.random.default_rng(0)
rows = a.rows
if a.uniq == "disjoint":
    perm = rng.permutation(E); ids = np.stack([perm[r * topk:(r + 1) * topk] for r in range(rows)])
else:
    pool = rng.permutation(E)[:E if a.uniq == "all" else int(a.uniq)]
    ids = np.stack([rng.choice(pool, topk, replace=False) for _ in range(rows)])
ids = mx.array(ids.astype(np.int32)); wts = mx.array(rng.random((rows, topk)).astype(np.float32))
x = mx.array(rng.standard_normal((rows, moe.g.kin)).astype(np.float16))
mx.eval(ids, wts, x)
fl = []
x0 = mx.ones((1, 5120), mx.float16); mx.eval(x0)
for _ in range(100):
    t = time.perf_counter(); y = x0 * 1.0001; mx.eval(y); fl.append(time.perf_counter() - t)
floor = min(fl)
ids_f, wts_f = ids.reshape(-1), wts.reshape(-1)
print(f"rows {rows}, distinct experts {len(np.unique(np.array(ids)))}, floor {1e6 * floor:.0f} us")
for impl in ("plain", "union"):
    moe.union = impl == "union"
    best = {}
    for rep in range(a.reps + 2):
        t = {}
        s = time.perf_counter(); xh = moe.rot_gu(x, ids_f); mx.eval(xh); t["rot_gu"] = time.perf_counter() - s
        s = time.perf_counter(); part = moe.gemv_gu(xh, ids_f, rows); mx.eval(part); t["gemv_gu"] = time.perf_counter() - s
        s = time.perf_counter(); xh2 = moe.mid(part, ids_f, wts_f, rows); mx.eval(xh2); t["mid"] = time.perf_counter() - s
        s = time.perf_counter(); part2 = moe.gemv_dn(xh2, ids_f, rows); mx.eval(part2); t["gemv_dn"] = time.perf_counter() - s
        s = time.perf_counter(); y = moe.final(part2, ids_f, wts_f, rows); mx.eval(y); t["final"] = time.perf_counter() - s
        if rep >= 2:
            for k, v in t.items():
                best[k] = min(best.get(k, 9), v)
    print(f"{impl:6s} " + "  ".join(f"{k} {1e3 * (v - floor):6.3f}" for k, v in best.items()) + f"   total {1e3 * sum(v - floor for v in best.values()):6.3f} ms")
