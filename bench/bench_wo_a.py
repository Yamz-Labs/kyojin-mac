"""wo_a (DS4.1 attn.wo_a: 8 slices 4096 -> 1024, K=5) per call: per-group path (8 x rot + GEMV + finalize + stack) vs grouped GEMV (3 launches).
Random trellis (timing does not depend on values). Prints us/call for 1-4 rows, synced (eval per call) and chained (40 calls, one eval = per-layer cost
inside a decode graph), bit-exactness, and the per-token saving over 40 layers."""
import argparse
import json
import time

import numpy as np
import mlx.core as mx

from kyojin_mlx.patch import Exl3MultiLinear


def make(G, kin, nout, K, seed=1):
    rng = np.random.default_rng(seed)
    tr = rng.integers(-32768, 32768, (G, kin // 16, nout // 16, 16 * K), dtype=np.int16)
    suh = (rng.choice([-1.0, 1.0], (G, kin)) * rng.uniform(0.5, 1.5, (G, kin))).astype(np.float16)
    svh = (rng.choice([-1.0, 1.0], (G, nout)) * rng.uniform(0.5, 1.5, (G, nout))).astype(np.float16)
    return mx.array(tr), mx.array(suh), mx.array(svh)


def per_group(ml, x):
    return mx.stack([ml.groups[g](x[:, g]) for g in range(ml.num_groups)], axis=1)


def synced(f, x, n):
    for _ in range(10):
        mx.eval(f(x))
    t = time.perf_counter()
    for _ in range(n):
        mx.eval(f(x))
    return (time.perf_counter() - t) / n * 1e6


def chained(f, xs, n=40, reps=20):
    for _ in range(3):
        mx.eval([f(x) for x in xs[:n]])
    t = time.perf_counter()
    for _ in range(reps):
        mx.eval([f(x) for x in xs[:n]])
    return (time.perf_counter() - t) / (reps * n) * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kin", type=int, default=4096)
    ap.add_argument("--nout", type=int, default=1024)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--cb", default="mul1")
    ap.add_argument("--json")
    a = ap.parse_args()
    ml = Exl3MultiLinear(*make(a.groups, a.kin, a.nout, a.K), a.cb)
    res = {"kin": a.kin, "nout": a.nout, "K": a.K, "groups": a.groups, "rows": {}}
    for rows in (1, 2, 3, 4):
        xs = [mx.array(np.random.default_rng(i).standard_normal((1, a.groups, rows, a.kin)).astype(np.float16)) for i in range(40)]
        mx.eval(xs)
        g, p = ml.grouped(xs[0]), per_group(ml, xs[0])
        exact = bool(np.array_equal(np.array(g).view(np.uint16), np.array(p).view(np.uint16)))
        r = dict(exact=exact, sync_pg=synced(lambda x: per_group(ml, x), xs[0], 100), sync_gr=synced(ml.grouped, xs[0], 100),
                 chain_pg=chained(lambda x: per_group(ml, x), xs), chain_gr=chained(ml.grouped, xs))
        r["saving_us_per_token"] = (r["chain_pg"] - r["chain_gr"]) * a.layers
        res["rows"][rows] = r
        print(f"rows {rows}: exact {exact} | synced per-group {r['sync_pg']:.0f} us, grouped {r['sync_gr']:.0f} us (x{r['sync_pg'] / r['sync_gr']:.1f}) | "
              f"chained per-group {r['chain_pg']:.0f} us, grouped {r['chain_gr']:.0f} us (x{r['chain_pg'] / r['chain_gr']:.1f}) | "
              f"saving {r['saving_us_per_token'] / 1000:.2f} ms/token over {a.layers} layers")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
