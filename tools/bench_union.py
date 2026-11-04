"""GPU time of one MoE layer call (MoeGather) for R rows at a controlled number of distinct experts: plain gather vs union, dependent chain of K calls in one
command buffer (time / K). usage: bench_union.py PACK [--layer 1] [--rows 1,2,4,8,10] [--k 12] [--reps 7]"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "100000"); os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "1000000")
import numpy as np
import mlx.core as mx
from kyojin_mlx import patch as P, rows as R, moe as M

ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("--layer", type=int, default=1); ap.add_argument("--rows", default="1,2,4,8,10")
ap.add_argument("--k", type=int, default=12); ap.add_argument("--reps", type=int, default=7); ap.add_argument("--uniq", default="all,16,8,6")
ap.add_argument("--impl", default="plain,union"); ap.add_argument("--json")
a = ap.parse_args()
model, spec, cfg = P.build_model(a.pack)
P.patch_model(model, a.pack, spec, cfg)
R.enable(model, spec)
moe = model.language_model.layers[a.layer].ffn.switch_mlp.moe
E, topk = moe.g.experts, moe.topk
H = moe.g.kin
rng = np.random.default_rng(0)


def make_ids(rows, uniq):
    """[rows, topk] ids; every row picks topk distinct experts out of a pool of `uniq` experts (all = whole layer)."""
    if uniq == "disjoint":                                   # no two rows share an expert (rows * topk <= E)
        perm = rng.permutation(E)
        return np.stack([perm[r * topk:(r + 1) * topk] for r in range(rows)]).astype(np.int32)
    pool = rng.permutation(E)[:E if uniq == "all" else int(uniq)]
    return np.stack([rng.choice(pool, topk, replace=False) for _ in range(rows)]).astype(np.int32)


def chain(rows, ids, k, impl):
    moe.union = impl == "union"
    x = mx.array(rng.standard_normal((rows, H)).astype(np.float16))
    w = mx.array(rng.random((rows, topk)).astype(np.float32))
    mx.eval(x, w, ids)
    t0 = time.perf_counter()
    for _ in range(k):
        y = moe(x, ids, w)
        x = x + (y * 1e-3).astype(mx.float16)
    mx.async_eval(x); t1 = time.perf_counter(); mx.eval(x); t2 = time.perf_counter()
    return (t2 - t1) / k, int(len(np.unique(np.array(ids))))


out = {}
print(f"layer {a.layer}: {E} experts, top{topk}; ms per call (GPU, dependent chain of {a.k})")
print(f"{'rows':>5s} {'uniq':>5s} {'distinct':>8s} " + " ".join(f"{i:>8s}" for i in a.impl.split(",")))
impls = a.impl.split(",")
for rows in [int(r) for r in a.rows.split(",")]:
    for u in a.uniq.split(","):
        if u == "disjoint" and rows * topk > E:
            continue
        if u not in ("all", "disjoint") and int(u) < topk:
            continue
        ids = mx.array(make_ids(rows, u))
        res = {i: [] for i in impls}
        d = None
        for rep in range(a.reps + 1):
            for i in impls:
                t, d = chain(rows, ids, a.k, i)
                if rep:
                    res[i].append(t)
        out[f"{rows}_{u}"] = dict(distinct=d, **{i: 1e3 * min(v) for i, v in res.items()})
        print(f"{rows:5d} {u:>5s} {d:8d} " + " ".join(f"{1e3 * min(res[i]):8.3f}" for i in impls), flush=True)
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
