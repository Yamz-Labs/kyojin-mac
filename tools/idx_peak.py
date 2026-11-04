#!/usr/bin/env python3
"""Peak MLX memory of one prefill indexer layer (ds_pf_idx._impl_indexer) on synthetic tensors: the blocked top-k against the single-pass form.

No pack, no weights of the real model: the score kernel and the selection are the served ones, so the numbers show the transient of the indexer alone.
    python tools/idx_peak.py --T 4096 --Tc 32768 --qblock 0,512
"""
import argparse
import os
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ap = argparse.ArgumentParser()
ap.add_argument("--T", type=int, default=4096, help="query rows of the chunk")
ap.add_argument("--Tc", type=int, default=32768, help="index keys")
ap.add_argument("--ratio", type=int, default=1)
ap.add_argument("--qblock", default="0,512")
ap.add_argument("--topk", type=int, default=512)
ap.add_argument("--cache-mb", type=int, default=256, help="MLX free-list limit (the served setting); 0 = unlimited")
a = ap.parse_args()

from kyojin_mlx import ds_pf_idx as I                                       # noqa: E402
from kyojin_mlx import idx_topk as IT                                       # noqa: E402

if a.cache_mb:
    mx.set_cache_limit(a.cache_mb << 20)

NH, D = 32, 128
rng = np.random.default_rng(0)
q = mx.array(rng.standard_normal((1, a.T, NH, D)).astype(np.float16))
keys = mx.array(rng.standard_normal((1, a.Tc, D)).astype(np.float16))
w = mx.array((rng.standard_normal((1, a.T, NH)) * 0.05).astype(np.float16))
GiB = 2 ** 30
base = mx.metal.get_active_memory()
print(f"T={a.T} Tc={a.Tc} ratio={a.ratio} topk={a.topk}  (inputs {(q.nbytes + keys.nbytes + w.nbytes) / GiB:.2f} GiB)")
for qb in [int(x) for x in a.qblock.split(",")]:
    mx.metal.reset_peak_memory()
    mx.clear_cache()
    it = IT.qblock(a.T, qb)
    out = IT.scan(lambda t0, n: I.scores_range(q, keys, w, 0, a.ratio, t0, n), a.T, it, lambda t0, s: IT.topk(s, min(a.topk, a.Tc)))
    mx.eval(out)
    peak = mx.metal.get_peak_memory() - base
    print(f"  qblock={qb:5d} (block {it or a.T:5d} rows): peak {peak / GiB:6.3f} GiB  active after {mx.metal.get_active_memory() / GiB:6.3f} GiB")
