#!/usr/bin/env python3
"""One DS4.1 MoE layer (real layer-3 experts, subset scaled like tools/bench_nax_layer.py) through the fused MoePrefill: the tile policy of
a0a740d (row block 16/32/64, bks 32) against the current `pick_cfg`, rounds interleaved in one process (the Air is shared).
   PYTHONPATH=$PWD:$PWD/bench:$PWD/tools python tools/bench_layer_cfg_ab.py [--T 4096,1024,256] [--esub 48] [--rounds 5]"""
import argparse, glob, os, time
import numpy as np
import mlx.core as mx
from kyojin_mlx import moe as M, moe_prefill as MP, metal_nax as NX
from nax_common import mini_stack

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.path.expanduser("~/kyojin-mlx/models/ds_l3"))
ap.add_argument("--esub", type=int, default=48); ap.add_argument("--T", default="4096,1024,256")
ap.add_argument("--experts", type=int, default=384); ap.add_argument("--topk", type=int, default=6)
ap.add_argument("--rounds", type=int, default=5); ap.add_argument("--fused-max", type=int, default=512)
a = ap.parse_args()
mx.set_cache_limit(512 << 20)
OLD = {16: 1.7, 32: 1.9, 64: 2.5}
old_pick = lambda counts: (min(OLD, key=lambda b: sum(-(-int(c) // b) for c in counts) * OLD[b]), 1, 32, 32)
new_pick = NX.pick_cfg
rng = np.random.default_rng(1)
exps = sorted(rng.choice(a.experts, a.esub, replace=False).tolist())
g, u, d = (mini_stack(a.dir, p, exps) for p in ("w1", "w3", "w2"))
mx.eval(g.words, u.words, d.words)
dec = M.MoeGather(g, u, d, a.topk, wpre=True, fq=False)
scale = a.experts / a.esub
print(f"layer subset {a.esub}/{a.experts} experts, K {g.ks}/{u.ks}/{d.ks}", flush=True)
FLOP_PER_ROW = 2.0 * 3 * g.kin * g.nout * a.topk          # three projections, topk experts per token (w2 is nout x kin: same product)
for T in [int(v) for v in a.T.split(",")]:
    rpe = T * a.topk / a.experts
    T_sub = max(1, round(rpe * a.esub / a.topk))
    ids = np.stack([rng.choice(a.esub, a.topk, replace=False) for _ in range(T_sub)]).astype(np.int32)
    wts = rng.uniform(0.1, 1.0, (T_sub, a.topk)).astype(np.float32); wts = wts / wts.sum(-1, keepdims=True) * 1.5
    x = mx.array((rng.standard_normal((T_sub, g.kin)) * 2).astype(np.float16))
    idm, wm = mx.array(ids), mx.array(wts)
    pf = MP.MoePrefill(dec, fused=True, fused_max=a.fused_max, group_gib=0.5)
    best = {"old": 1e9, "new": 1e9}
    ys = {}
    for r in range(a.rounds + 1):
        for name, fn in (("old", old_pick), ("new", new_pick)):
            NX.pick_cfg = fn
            t0 = time.perf_counter(); y = pf(x, idm, wm); mx.eval(y); dt = time.perf_counter() - t0
            if r:
                best[name] = min(best[name], dt)
            ys[name] = np.array(y)
    NX.pick_cfg = new_pick
    diff = np.linalg.norm(ys["new"] - ys["old"]) / np.linalg.norm(ys["old"])
    fl = T_sub * FLOP_PER_ROW
    print(f"T={T:5d} ({rpe:5.1f} rows/expert, {T_sub} tokens in the subset): old {best['old']*1e3:7.1f} ms ({fl/best['old']/1e12:4.1f} TFLOPS)  "
          f"new {best['new']*1e3:7.1f} ms ({fl/best['new']/1e12:4.1f} TFLOPS)  x{best['old']/best['new']:.2f}  full layer x{scale:.0f}: "
          f"{best['old']*scale*1e3:6.0f} > {best['new']*scale*1e3:6.0f} ms  rel diff {diff:.1e}", flush=True)
print(f"peak {mx.get_peak_memory()/1e9:.2f} GB")
