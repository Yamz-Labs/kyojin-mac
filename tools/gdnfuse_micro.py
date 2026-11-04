#!/usr/bin/env python3
"""gdnfuse_micro: fused GDN decode step vs the chain (pre + step kernel + gated norm), synthetic real shapes, L layers per eval. Prints time and max abs diff."""
import sys, time
import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_kernel
from kyojin_mlx import gdn_pf, gdn_fuse
HK, HV, L = 16, 32, 30
C = 2 * HK * 128 + HV * 128
dt = mx.float16 if (len(sys.argv) > 2 and sys.argv[2] == "f16") else mx.bfloat16
def mk(S, seed):
    mx.random.seed(seed)
    d = dict(qkv=mx.random.normal((1, S, C)).astype(dt), z=mx.random.normal((1, S, HV, 128)).astype(dt), a=mx.random.normal((1, S, HV)).astype(dt),
             b=mx.random.normal((1, S, HV)).astype(dt), cs=mx.random.normal((1, 3, C)).astype(dt), state=mx.random.normal((1, HV, 128, 128)).astype(mx.float32) * 0.1)
    return d
mx.random.seed(1)
cw = (mx.random.normal((C, 4, 1)) * 0.3).astype(dt); Al = mx.random.normal((HV,)).astype(mx.float32) * 0.5; dtb = mx.random.normal((HV,)).astype(dt)
nw = (1 + 0.1 * mx.random.normal((128,))).astype(dt); eps = 1e-6
inv = mx.array(list(reversed(range(HV * 128))), dtype=mx.uint32)
def chain(d, perm):
    q, k, v, cn, g, beta = gdn_pf.pre(d["qkv"], d["cs"], cw, d["a"], d["b"], Al, dtb, HK, HV)
    y, st = gated_delta_kernel(q, k, v, g, beta, d["state"], None)
    return gdn_pf.gated_norm(y.astype(dt), d["z"], nw, eps, inv=inv if perm else None).reshape(1, d["qkv"].shape[1], -1), cn, st
def fused(d, perm, cap=False):
    return gdn_fuse.fused_step(d["qkv"], d["z"], d["a"], d["b"], d["cs"], d["state"], cw, Al, dtb, nw, eps, HK, HV, inv=inv if perm else None, capture=cap)
def run(fn, ds, perm):
    outs, prev = [], None            # layer i+1 reads layer i output (a dependent chain, like the model)
    for d in ds:
        if prev is not None:
            d = dict(d, a=prev[0][..., :HV].astype(dt))
        prev = fn(d, perm); outs.append(prev)
    mx.eval([o for t in outs for o in t[:3]])
    return outs
for S in (1, 2, 5, 8):
    ds = [mk(S, i) for i in range(L)]
    mx.eval([v for d in ds for v in d.values()])
    for perm in (False, True):
        a = run(chain, ds[:2], perm); b = run(fused, ds[:2], perm)
        diffs = [float(mx.max(mx.abs(x.astype(mx.float32) - y.astype(mx.float32)))) for ta, tb in zip(a, b) for x, y in zip(ta[:3], tb[:3])]
        cap = fused(ds[0], perm, True); ref = gdn_pf.pre(ds[0]["qkv"], ds[0]["cs"], cw, ds[0]["a"], ds[0]["b"], Al, dtb, HK, HV)
        mx.eval(cap[3:], ref)
        cd = [float(mx.max(mx.abs(cap[3 + i].astype(mx.float32) - ref[[0, 1, 2, 4, 5][i]].astype(mx.float32)))) for i in range(5)]
        res = {"chain": 1e9, "fused": 1e9}
        for fn in (chain, fused):
            run(fn, ds, perm)
        for _ in range(25):
            for name, fn in (("chain", chain), ("fused", fused)):
                t0 = time.perf_counter(); run(fn, ds, perm); res[name] = min(res[name], (time.perf_counter() - t0) / L * 1e6)
        print(f"S={S} perm={int(perm)} chain {res['chain']:.1f} us fused {res['fused']:.1f} us  -{100*(1-res['fused']/res['chain']):.0f}%  maxdiff {max(diffs):.3g}  capdiff {max(cd):.3g}", flush=True)
