"""Dense multi-row GEMV: is row r of an N-row launch bit-identical to the 1-row launch, and what does a row cost?
   python bench/bench_rows.py [--shapes 2048x6144,6144x2048,2048x248320] [--cb mcg|mul1] [--K 2,3] [--iters 100] [--json out.json]
Time per launch (rot + partial + finalize, chained, one eval) for rows 1..4; ratio = t(rows) / t(1)."""
import argparse, json, time
import numpy as np
import mlx.core as mx
from kyojin_mlx import kernels as KN, ref_numpy as R


def timeit(fn, iters):
    for _ in range(10):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    mx.eval([fn() for _ in range(iters)])
    mx.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="2048x6144,6144x2048,2048x16384,2048x248320")
    ap.add_argument("--cb", default="mcg"); ap.add_argument("--K", default="2,3"); ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--json", default=None); a = ap.parse_args()
    res = []
    for shp in a.shapes.split(","):
        k, n = map(int, shp.split("x"))
        for K in map(int, a.K.split(",")):
            t, su, sv = R.random_linear(k, n, K, seed=K)
            if a.cb == "mcg":
                t, su, sv = R.random_linear(k, n, K, seed=K)      # layout is codebook independent; random words
            x4 = mx.array(np.random.default_rng(1).standard_normal((4, k)).astype(np.float16))
            lins = {r: KN.Mul1Gemv(t, su, sv, rows=r, out_dtype=mx.float16, cb=a.cb) for r in (1, 2, 3, 4)}
            y1 = [np.array(lins[1](x4[i:i + 1])) for i in range(4)]
            ok = {}
            for r in (2, 3, 4):
                y = np.array(lins[r](x4[:r]))
                ok[r] = all(np.array_equal(y[i].view(np.uint16), y1[i][0].view(np.uint16)) for i in range(r))
            ts = {r: timeit(lambda r=r: lins[r](x4[:r]), a.iters) for r in (1, 2, 3, 4)}
            row = dict(k=k, n=n, K=K, cb=a.cb, bitwise={str(r): v for r, v in ok.items()}, us={str(r): round(v * 1e6, 1) for r, v in ts.items()},
                       ratio={str(r): round(ts[r] / ts[1], 3) for r in (2, 3, 4)}, tgk=lins[1].tgk)
            res.append(row)
            print(json.dumps(row), flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


main()
