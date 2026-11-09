#!/usr/bin/env python3
"""Micro gate of KYOJIN_MOE_CHAIN: one decode step of the routed chain (40 layers, rows 1 and 2) on synthetic stacks of DS4.1 shapes
(hidden 5120, inter 2048, top-6, mixed 2/3 bit, wpre + fp8 fake quant), each layer fed by the previous layer's output. Five-launch chain vs
three-launch chain, interleaved, bit-compared. Run under the GPU lock."""
import sys, time, os
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from kyojin_mlx import moe as M
from test_moe import make_stack

KD, NI, TOPK, NL, NS, NE = 5120, 2048, 6, 40, 6, 16
ks = [2, 3, 3, 2, 2, 3, 2, 2, 3, 2, 3, 3, 2, 3, 2, 3]
mods = []
for s in range(NS):
    g, _ = make_stack(KD, NI, ks, "mul1", 10 * s + 1); u, _ = make_stack(KD, NI, ks, "mul1", 10 * s + 2); d, _ = make_stack(NI, KD, ks, "mul1", 10 * s + 3)
    mods.append(M.MoeGather(g, u, d, TOPK, wpre=True, fq=True))
rng = np.random.default_rng(0)
print("stack bytes MB:", sum(int(m.g.words.nbytes + m.u.words.nbytes + m.d.words.nbytes) for m in mods) // 2**20)


def run(rows, chain, x0, routes):
    M.CHAIN = chain
    x = x0
    for l in range(NL):
        ids, w = routes[l]
        y = mods[l % NS](x, ids, w, fq8=True)
        x = (y * 0.001).astype(mx.float16)
    return y


for rows in (1, 2):
    x0 = mx.array(rng.standard_normal((rows, KD)).astype(np.float16))
    routes = [(mx.array(np.stack([rng.choice(NE, TOPK, replace=False) for _ in range(rows)]).astype(np.int32)),
               mx.array(rng.uniform(0.1, 1.0, (rows, TOPK)).astype(np.float32))) for _ in range(NL)]
    mx.eval(x0, [r for t in routes for r in t])
    a = np.array(run(rows, False, x0, routes)); b = np.array(run(rows, True, x0, routes))
    print(f"rows {rows}: bit-identical {np.array_equal(a.view(np.uint32), b.view(np.uint32))}")
    res = {False: [], True: []}
    for rep in range(40):
        for chain in ((False, True) if rep % 2 == 0 else (True, False)):
            t0 = time.perf_counter(); mx.eval(run(rows, chain, x0, routes)); res[chain].append(time.perf_counter() - t0)
    for chain in (False, True):
        r = sorted(res[chain])
        print(f"rows {rows} {'chain3' if chain else 'base5 '}: min {1e3 * r[0]:.3f} ms median {1e3 * r[20]:.3f} ms -> {1e6 * r[0] / NL:.1f} us/layer (min), {1e6 * r[20] / NL:.1f} (median)")
