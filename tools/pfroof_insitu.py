#!/usr/bin/env python3
"""pfroof: in-situ gate_up stage paired ratios (rot output as input, real
skewed counts, synced stage times), n=11 pairs."""
import os, sys, time
import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from bench_moe import synth_stack  # noqa: E402
from kyojin_mlx import moe as M, moe_prefill as MP  # noqa: E402

mx.set_cache_limit(256 << 20)
POOL, E, TOPK, H, I, K, T = 48, 384, 6, 5120, 2304, 3, 8192
g0 = synth_stack(H, I, [K] * POOL, "mul1", 1)
u0 = synth_stack(H, I, [K] * POOL, "mul1", 2)
d0 = synth_stack(I, H, [K] * POOL, "mul1", 3)


def virtual(st, V):
    pool = st.experts
    phys = np.arange(V) % pool
    off = np.array(st.offset)[:pool][phys]
    off = np.concatenate([off, off[-1:] + 1]).astype("<u4")
    return type(st)(st.words, mx.array(off), mx.array(np.array(st.k)[phys]),
                    mx.array(np.array(st.suh)[phys]), mx.array(np.array(st.svh)[phys]),
                    st.kt, st.nt, st.cb)


g, u, d = (virtual(s, E) for s in (g0, u0, d0))
dec = M.MoeGather(g, u, d, TOPK, wpre=True, fq=False)
rng = np.random.default_rng(7)
hot = rng.choice(E, 96, replace=False)
p = np.full(E, 0.25 / (E - 96)); p[hot] = 0.75 / 96
ids = np.stack([rng.choice(E, TOPK, replace=False, p=p) for _ in range(T)]).astype(np.int32)
w = rng.uniform(0.1, 1.0, (T, TOPK)).astype(np.float32)
w = (w / w.sum(-1, keepdims=True) * 1.5).astype(np.float32)
x = mx.array((rng.standard_normal((T, H)) * 2).astype(np.float16))
idsm, wm = mx.array(ids), mx.array(w)
mx.eval(x, idsm, wm)

ratios, gc, gs = [], [], []
for r in range(11):
    got = {}
    for split in (False, True) if rng.random() < 0.5 else (True, False):
        MP.SPLIT_GU = split
        pf = MP.MoePrefill(dec, chunk=8192)
        pf.prof = {}
        y = pf(x, idsm, wm); mx.eval(y)
        got[split] = pf.prof["gate_up"]
    gc.append(got[False]); gs.append(got[True])
    ratios.append(got[True] / got[False])
print(f"in-situ gate_up paired split/comb: med {np.median(ratios):.3f} "
      f"p10 {sorted(ratios)[1]:.3f} p90 {sorted(ratios)[-2]:.3f}", flush=True)
print(f"comb med {np.median(gc) * 1e3:.1f}ms | split med {np.median(gs) * 1e3:.1f}ms", flush=True)
print(f"comb reps: {[f'{v * 1e3:.0f}' for v in gc]}", flush=True)
print(f"split reps: {[f'{v * 1e3:.0f}' for v in gs]}", flush=True)
