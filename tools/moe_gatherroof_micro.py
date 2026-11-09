#!/usr/bin/env python3
"""Gate of KYOJIN_GATHER_ROOF (same harness as moe_routedfx_micro, NI = 2304 = the DS4.1 moe_intermediate) / stage table of KYOJIN_ROUTED_FX (routed-expert decode, 40 layers, rows 1 and 2, DS4.1 shapes). Stage costs by differencing chains that stop after rot /
gu / dn (each layer fed by the previous one through a one-kernel glue, measured by a glue-only chain). Then base vs fx, interleaved, bit-compared. Run under the GPU lock."""
import sys, time, os
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from kyojin_mlx import moe as M
from test_moe import make_stack

KD, NI, TOPK, NL, NS, NE = 5120, 2304, 6, 40, 6, 16
ks = [2, 3, 3, 2, 2, 3, 2, 2, 3, 2, 3, 3, 2, 3, 2, 3]
mods = []
for s in range(NS):
    g, _ = make_stack(KD, NI, ks, "mul1", 10 * s + 1); u, _ = make_stack(KD, NI, ks, "mul1", 10 * s + 2); d, _ = make_stack(NI, KD, ks, "mul1", 10 * s + 3)
    mods.append(M.MoeGather(g, u, d, TOPK, wpre=True, fq=True))
rng = np.random.default_rng(0)
FX = hasattr(M, "GATHER_ROOF")


def run(rows, stop, x0, routes, fx=False):
    if FX: M.GATHER_ROOF = fx
    x = x0
    for l in range(NL):
        ids, w = routes[l]
        m = mods[l % NS]
        if stop == "full":
            y = m(x, ids, w.astype(mx.float32), fq8=True) if False else m(x, ids, w, fq8=True)
        else:
            i2, w2 = ids.reshape(-1).astype(mx.int32), w.reshape(-1).astype(mx.float32)
            if stop == "glue":
                y = x.astype(mx.float32) * 1.0
            elif stop == "rot":
                y = m.rot_gu(x, i2, True)[:rows, :KD]
            elif stop == "gu":
                y = m.chain_gu(m.rot_gu(x, i2, True), i2, w2, rows)[1][:rows, :KD]
            elif stop == "gemvgu":
                y = m.gemv_gu(m.rot_gu(x, i2, True), i2, rows)[:rows, 0, :KD]
            elif stop == "gemvdn":
                xh2 = m.chain_gu(m.rot_gu(x, i2, True), i2, w2, rows)[1]
                y = m.gemv_dn(xh2, i2, rows)[:rows, 0, :KD]
            elif stop == "dn":
                y = m.chain_dn(m.chain_gu(m.rot_gu(x, i2, True), i2, w2, rows)[1], i2, w2, rows)
        x = (y * 0.001).astype(mx.float16)
    return y


def tm(f, n=30):
    r = []
    for _ in range(n):
        t0 = time.perf_counter(); mx.eval(f()); r.append(time.perf_counter() - t0)
    return sorted(r)[0] * 1e6 / NL


for rows in (1, 2):
    x0 = mx.array(rng.standard_normal((rows, KD)).astype(np.float16))
    routes = [(mx.array(np.stack([rng.choice(NE, TOPK, replace=False) for _ in range(rows)]).astype(np.int32)),
               mx.array(rng.uniform(0.1, 1.0, (rows, TOPK)).astype(np.float32))) for _ in range(NL)]
    mx.eval(x0, [r for t in routes for r in t])
    t = {s: tm(lambda s=s: run(rows, s, x0, routes)) for s in ("glue", "rot", "gu", "dn", "gemvgu", "gemvdn")}
    print(f"rows {rows} us/layer cumulative: " + " ".join(f"{k} {v:.1f}" for k, v in t.items()))
    print(f"rows {rows} stage: rot {t['rot']-t['glue']:.1f}  gu {t['gu']-t['rot']:.1f}  dn+combine {t['dn']-t['gu']:.1f}")
    if FX:
        a = np.array(run(rows, "full", x0, routes, False)); b = np.array(run(rows, "full", x0, routes, True))
        print(f"rows {rows}: bit-identical {np.array_equal(a.view(np.uint32), b.view(np.uint32))}")
        res = {False: [], True: []}
        for rep in range(40):
            for fx in ((False, True) if rep % 2 == 0 else (True, False)):
                t0 = time.perf_counter(); mx.eval(run(rows, "full", x0, routes, fx)); res[fx].append(time.perf_counter() - t0)
        for fx in (False, True):
            r = sorted(res[fx]); print(f"rows {rows} {'fx  ' if fx else 'base'}: min {1e3*r[0]:.3f} ms -> {1e6*r[0]/NL:.1f} us/layer, median {1e6*r[20]/NL:.1f}")
