#!/usr/bin/env python3
"""Routed MoE decode with the mul1 affine folded out of the per-weight decode (KYOJIN_MOE_AFFOLD, metal_src.af_fold) vs the default, on a stacked
DS4.1 layer slice: same MoeGather, gather GEMV (rows 1) and grouped GEMV (rows >= 2) kernels with cb mul1 / mul1af, interleaved in one process.
Time per call: slope of chains of n and 4n calls with fresh inputs in one eval, min / median over --reps. Error: max |y_af - y| / max |y| per case.
usage: affold108.py [--dir models/ds_l10] [--layer 10] [--rows 1,2,4] [--reps 5]"""
import argparse, os, sys, time

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kyojin_mlx import patch as P, moe as M  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--dir", default=os.path.expanduser("~/kyojin-mlx/models/ds_l10"))
ap.add_argument("--layer", type=int, default=10)
ap.add_argument("--rows", default="1,2,4")
ap.add_argument("--topk", type=int, default=6)
ap.add_argument("--chain", type=int, default=8)
ap.add_argument("--reps", type=int, default=5)
a = ap.parse_args()
mx.set_cache_limit(256 << 20)
pre = f"layers.{a.layer}"
t = P.load_pack(a.dir, prefix=pre + ".")
st = [M.Stack.from_arrays(t, f"{pre}.ffn.experts.{w}", "mul1") for w in ("w1", "w3", "w2")]
mx.eval(*[x for s in st for x in (s.words, s.offset, s.k, s.suh, s.svh)])
E, H = st[0].experts, st[0].kin
print(f"{os.path.basename(a.dir)}: {E} experts, bitrates {dict(zip(*np.unique(st[0].kh, return_counts=True)))}", flush=True)
rng = np.random.default_rng(0)
ms = {}
for cbg in ("mul1", "mul1af"):
    m = M.MoeGather(*st, a.topk, limit=10.0, wpre=True, fq=True, grouped=True, hotk=0)
    m.cbg = cbg
    ms[cbg] = m


def timeit(f, ins):
    n = len(ins)
    def run(c):
        outs = [f(*ins[i % n]) for i in range(c)]
        t0 = time.perf_counter(); mx.eval(*outs); return time.perf_counter() - t0
    T = {n: [], 4 * n: []}
    for r in range(3):
        for c in T:
            dt = run(c)
            if r:
                T[c].append(dt)
    return (min(T[4 * n]) - min(T[n])) / (3 * n) * 1e6


for R in [int(v) for v in a.rows.split(",")]:
    ins = [(mx.array((rng.standard_normal((R, H)) * 0.5).astype(np.float16)),
            mx.array(np.stack([rng.choice(E, a.topk, replace=False) for _ in range(R)]).astype(np.int32)),
            mx.array(rng.random((R, a.topk)).astype(np.float32))) for _ in range(a.chain)]
    mx.eval(*[v for i in ins for v in i])
    y0, y1 = ms["mul1"](*ins[0]).astype(mx.float32), ms["mul1af"](*ins[0]).astype(mx.float32)
    err = (mx.abs(y1 - y0).max() / mx.abs(y0).max()).item()
    T = {k: [] for k in ms}
    for _ in range(a.reps):
        for k, m in ms.items():
            T[k].append(timeit(m, ins))
    b0, b1 = min(T["mul1"]), min(T["mul1af"])
    print(f"R{R}: mul1 {b0:.1f} us (med {np.median(T['mul1']):.1f}) -> affine folded {b1:.1f} us (med {np.median(T['mul1af']):.1f}) ({(b1 / b0 - 1) * 100:+.1f}%), "
          f"max rel err {err:.2e}", flush=True)
