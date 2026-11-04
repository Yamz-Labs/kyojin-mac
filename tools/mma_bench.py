#!/usr/bin/env python3
"""MMA GEMV vs the union/row-group GEMV: correctness (rel err of the linear output) and us per linear at 1/2/4/8/16 rows, 4B shapes.
   PYTHONPATH=$PWD python tools/mma_bench.py [--shapes mlp_up,...] [--rows 1,2,4,8,16] [--check-only]"""
import argparse, time
import numpy as np
import mlx.core as mx
from kyojin_mlx import exl3_linear as EL, kernels as KN, mma_gemv as MM

SHAPES = {"mlp_up": (2560, 9216, 4), "mlp_down": (9216, 2560, 4), "qkv": (2560, 8192, 4), "o": (4096, 2560, 4),
          "up3": (2560, 9216, 3), "up2": (2560, 9216, 2), "up6": (2560, 9216, 6), "up5": (2560, 9216, 5), "up8": (2560, 4096, 8), "up1": (2560, 4096, 1)}


def synth(kin, nout, K, cb="mul1", seed=1):
    rng = np.random.default_rng(seed)
    kt, nt = kin // 16, nout // 16
    tr = mx.array(rng.integers(-32768, 32768, (kt, nt, 16 * K), dtype=np.int16))
    f16 = lambda n: mx.array((rng.choice([-1.0, 1.0], n) * rng.uniform(0.5, 1.5, n)).astype(np.float16))
    return EL.Exl3Linear(tr, f16(kin), f16(nout), cb=cb)


def mma_call(lin, x2):
    g = lin._g[1]
    R = x2.shape[0]
    return MM.finalize_rows(g, MM.partial_mma(g, KN.rot(x2, g.suh), R), R)


def gemv_call(lin, x2):
    EL.ROW_SPLIT_MAX = 32
    EL.ROWS_NAX = False
    return lin(x2)


def timeit(fn, xs, reps, chain):
    mx.eval(*[fn(x) for x in xs[:2]])
    best, allt = 1e9, []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(*[fn(x) for x in xs[:chain]])
        t = (time.perf_counter() - t0) / chain
        allt.append(t); best = min(best, t)
    return best * 1e6, sorted(allt)[len(allt) // 4] * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="mlp_up,mlp_down"); ap.add_argument("--rows", default="1,2,4,8,16")
    ap.add_argument("--reps", type=int, default=7); ap.add_argument("--chain", type=int, default=12); ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    mx.set_cache_limit(256 << 20)
    for name in a.shapes.split(","):
        kin, nout, K = SHAPES[name]
        lin = synth(kin, nout, K)
        mx.eval(lin._words)
        print(f"== {name} {kin}x{nout} K={K} tgk={lin._g[1].tgk}", flush=True)
        rng = np.random.default_rng(5)
        for R in [int(v) for v in a.rows.split(",")]:
            xs = [mx.array(rng.standard_normal((R, kin)).astype(np.float16)) for _ in range(a.chain)]
            mx.eval(*xs)
            ref = np.array(gemv_call(lin, xs[0])).astype(np.float64)
            if R >= 2:
                y = np.array(mma_call(lin, xs[0])).astype(np.float64)
                rel = float(np.linalg.norm(y - ref) / np.linalg.norm(ref)); mxe = float(np.abs(y - ref).max())
            else:
                rel = mxe = -1
            line = f"  R={R:2d} rel {rel:.2e} max {mxe:.2e}"
            if not a.check_only:
                tg, pg = timeit(lambda x: gemv_call(lin, x), xs, a.reps, a.chain)
                line += f" | GEMV min {tg:7.1f} p25 {pg:7.1f} us"
                if R >= 2:
                    tm, pm = timeit(lambda x: mma_call(lin, x), xs, a.reps, a.chain)
                    line += f" | MMA min {tm:7.1f} p25 {pm:7.1f} us  x{tg / tm:.2f}"
            print(line, flush=True)
        print(f"  peak {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)
        del lin
        mx.clear_cache()


if __name__ == "__main__":
    main()
