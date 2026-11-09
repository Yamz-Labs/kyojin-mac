"""Micro gate for shared_glu: the shared-expert chain of a DS4.1 decode step (gate|up group launch, swiglu launch, fp8 round trip + down GEMV) vs the same chain
with the swiglu folded into the down GEMV prologue (Mul1Gemv.fused_glu). NL layers chained (each its own weights), `rows` decode rows, median of REPS steps.
Run: python tools/sharedglu_micro.py [--rows 1] [--layers 40] [--k 3]"""
import argparse
import statistics
import time

import mlx.core as mx
import numpy as np

from kyojin_mlx import exl3_linear as EL, fused_group as FG, kernels as KN, pf_glue as PG

ap = argparse.ArgumentParser()
ap.add_argument("--rows", type=int, default=1)
ap.add_argument("--layers", type=int, default=40)
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--hidden", type=int, default=5120)
ap.add_argument("--inter", type=int, default=2304)
ap.add_argument("--reps", type=int, default=30)
a = ap.parse_args()


def lin(kin, nout, seed):
    r = np.random.default_rng(seed)
    tr = r.integers(-32768, 32768, (kin // 16, nout // 16, 16 * a.k), dtype=np.int16)
    suh = (r.choice([-1.0, 1.0], kin) * r.uniform(0.5, 1.5, kin)).astype(np.float16)
    svh = (r.choice([-1.0, 1.0], nout) * r.uniform(0.3, 0.6, nout)).astype(np.float16)
    m = EL.Exl3Linear(mx.array(tr), mx.array(suh), mx.array(svh), cb="mcg")
    m._gemv(a.rows)
    return m


layers = []
for i in range(a.layers):
    g, u, d = lin(a.hidden, a.inter, 3 * i), lin(a.hidden, a.inter, 3 * i + 1), lin(a.inter, a.hidden, 3 * i + 2)
    layers.append((FG.Exl3Group([g, u]), d))
mx.eval([m.words for grp, d in layers for m in [grp]][0:0])
R, LIM = a.rows, 10.0
x0 = mx.array((np.random.default_rng(0).standard_normal((R, a.hidden)) * 2).astype(np.float16))
sw = PG._swiglu_h(LIM, mx.float16)


def chain(glu):
    x = x0
    for grp, d in layers:
        g, u = grp.run(x, True)
        gv = d._g.get(R) or d._gemv(R)
        x = gv.fused_glu(g, u, LIM) if glu else gv.fused(sw(g, u), fq8=True)
    return x


def t(glu):
    ts = []
    for _ in range(a.reps + 5):
        t0 = time.perf_counter()
        mx.eval(chain(glu))
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts[5:]), min(ts[5:])


ya, yb = chain(False), chain(True)
mx.eval(ya, yb)
print("finite", bool(mx.all(mx.isfinite(ya)).item()), "bit-identical", bool(np.array_equal(np.array(ya).view(np.uint16), np.array(yb).view(np.uint16))))
for _ in range(2):
    base, glu = t(False), t(True)
    print(f"rows={R} layers={a.layers}: base med {base[0]:.3f} min {base[1]:.3f} ms | glu med {glu[0]:.3f} min {glu[1]:.3f} ms | "
          f"step saved {base[0] - glu[0]:.3f} ms, per layer {(base[0] - glu[0]) / a.layers * 1e3:.1f} us")
