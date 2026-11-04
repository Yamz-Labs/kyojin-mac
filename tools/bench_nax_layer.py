#!/usr/bin/env python3
"""One DS4.1 MoE layer (real layer-3 experts) through MoePrefill: today's S3 group path (decode_pf + gather_mm) vs the fused EXL3 + NAX path.
A subset of E_SUB experts is read from the pack (the whole layer is 5 GB, the Air budget is 2 GB per agent) with T_SUB tokens routed inside the
subset so that every expert gets the rows it gets in a full 384-expert layer at T tokens (rows/expert = T * topk / 384). The time scales linearly in
the number of experts: full layer ~ measured * 384 / E_SUB.
   PYTHONPATH=$PWD:$PWD/bench:$PWD/tools python tools/bench_nax_layer.py [--dir models/ds_l3] [--esub 48] [--T 4096] [--skew 0] [--reps 3]"""
import argparse, glob, os, time
import numpy as np
import mlx.core as mx
from safetensors import safe_open
from kyojin_mlx import moe as M, moe_prefill as MP
from nax_common import mini_stack

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.path.expanduser("~/kyojin-mlx/models/ds_l3"))
ap.add_argument("--esub", type=int, default=48); ap.add_argument("--T", type=int, default=4096)
ap.add_argument("--experts", type=int, default=384); ap.add_argument("--topk", type=int, default=6)
ap.add_argument("--skew", type=float, default=0.0, help="Zipf exponent of the expert popularity (0 = uniform)")
ap.add_argument("--reps", type=int, default=3); ap.add_argument("--group-gib", type=float, default=0.5)
ap.add_argument("--fused-max", type=int, default=512); ap.add_argument("--fq", action="store_true")
ap.add_argument("--projs", default="w1,w3,w2", help="gate,up,down projection names of the pack (GLM: gate_proj,up_proj,down_proj)")
ap.add_argument("--wpre", type=int, default=1, help="routing weight before the down projection (DS4.1 1, GLM 0)")
ap.add_argument("--full-route", action="store_true", help="route T tokens over all --experts, keep esub of the ACTIVE experts with their true row counts, "
                "extrapolate by (active experts of the full layer) / esub: right for small T where only some experts are active")
a = ap.parse_args()
rows_per_expert = a.T * a.topk / a.experts
T_sub = max(1, round(rows_per_expert * a.esub / a.topk))
scale = a.experts / a.esub
if a.full_route:
    rf = np.random.default_rng(2)
    pf_ = 1.0 / np.arange(1, a.experts + 1) ** a.skew if a.skew else np.ones(a.experts)
    pf_ = pf_[rf.permutation(a.experts)]; pf_ /= pf_.sum()
    full = np.stack([rf.choice(a.experts, a.topk, replace=False, p=pf_) for _ in range(a.T)])
    cfull = np.bincount(full.reshape(-1), minlength=a.experts)
    active = np.nonzero(cfull)[0]
    sub = rf.choice(active, min(a.esub, len(active)), replace=False)
    a.esub = len(sub)
    pair_exp = np.repeat(np.arange(a.esub), cfull[sub])
    T_sub = len(pair_exp) // a.topk
    pair_exp = pair_exp[:T_sub * a.topk]
    scale = len(active) / a.esub
    print(f"full layer T={a.T}: {len(active)}/{a.experts} experts active, {cfull[active].mean():.1f} rows per active expert; subset {a.esub} experts, {T_sub} tokens")
f = sorted(glob.glob(os.path.join(a.dir, "stacked.*.safetensors")))[0]
rng = np.random.default_rng(1)
exps = sorted(rng.choice(a.experts, a.esub, replace=False).tolist())
g, u, d = (mini_stack(a.dir, p, exps) for p in a.projs.split(","))
mx.eval(g.words, u.words, d.words)
print(f"layer: {a.esub}/{a.experts} experts (K {g.ks}/{u.ks}/{d.ks}), T_sub {T_sub} tokens -> {rows_per_expert:.0f} rows/expert (T={a.T}), skew {a.skew}")
pop = 1.0 / np.arange(1, a.esub + 1) ** a.skew if a.skew else np.ones(a.esub)
pop = pop[rng.permutation(a.esub)]; pop /= pop.sum()
if a.full_route:
    ids = np.stack([pair_exp[t::T_sub][:a.topk] for t in range(T_sub)]).astype(np.int32)
    assert all(len(set(r)) == a.topk for r in ids.tolist()), "an expert has more rows than the subset has tokens"
else:
    ids = np.stack([rng.choice(a.esub, a.topk, replace=False, p=pop) for _ in range(T_sub)]).astype(np.int32)
wts = rng.uniform(0.1, 1.0, (T_sub, a.topk)).astype(np.float32); wts = wts / wts.sum(-1, keepdims=True) * 1.5
x = mx.array((rng.standard_normal((T_sub, g.kin)) * 2).astype(np.float16))
cnt = np.bincount(ids.reshape(-1), minlength=a.esub)
print(f"rows/expert: min {cnt.min()} median {int(np.median(cnt))} max {cnt.max()}; experts above {a.fused_max}: {(cnt > a.fused_max).sum()}")
try:
    dec = M.MoeGather(g, u, d, a.topk, wpre=bool(a.wpre), fq=a.fq)
except Exception as e:
    print("fq unavailable:", e); dec = M.MoeGather(g, u, d, a.topk, wpre=bool(a.wpre), fq=False)
idm, wm = mx.array(ids), mx.array(wts.astype(np.float32))
res = {}
for name, kw in (("today", dict(fused=False, group_gib=a.group_gib)), ("fused", dict(fused=True, fused_max=a.fused_max, group_gib=a.group_gib))):
    pf = MP.MoePrefill(dec, **kw)
    y = pf(x, idm, wm); mx.eval(y)
    ts = []
    for _ in range(a.reps):
        t0 = time.perf_counter(); y = pf(x, idm, wm); mx.eval(y); ts.append(time.perf_counter() - t0)
    res[name] = (min(ts), np.array(y))
    print(f"{name}: {min(ts)*1e3:8.1f} ms (min of {a.reps}), full layer x{scale:.1f} = {min(ts)*scale*1e3:8.0f} ms, {a.T/(min(ts)*scale):7.0f} tok/s in this layer  {pf.last}")
r = res["fused"][1] - res["today"][1]
print(f"x{res['today'][0]/res['fused'][0]:.2f} fused vs today; rel diff {np.linalg.norm(r)/np.linalg.norm(res['today'][1]):.2e}")
