"""Microbenchmark: weights/s and GB/s of the mul1 GEMV (K = 2, 3) vs MLX fp16 and 4-bit quantized_matmul.
   python bench/bench_gemv.py [--iters 200] [--shapes 4096x4096,4096x12288]"""
import argparse, json, platform, subprocess, time

import numpy as np
import mlx.core as mx

from kyojin_mlx import kernels as KN, ref_numpy as R


def timeit(fn, iters):
    for _ in range(10):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    outs = [fn() for _ in range(iters)]
    mx.eval(outs)
    mx.synchronize()
    return (time.perf_counter() - t0) / iters


def row(label, dt, k, n, bits):
    w = k * n
    return {"kernel": label, "us": round(dt * 1e6, 1), "Gweights/s": round(w / dt / 1e9, 1),
            "GB/s": round(w * bits / 8 / dt / 1e9, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--shapes", default="2048x4096,4096x4096,4096x12288")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    print(f"# {chip or platform.platform()}  mlx {mx.__version__}  iters {a.iters}")
    results = []
    for shp in a.shapes.split(","):
        k, n = map(int, shp.split("x"))
        x = mx.array(np.random.default_rng(0).standard_normal((1, k)).astype(np.float16))
        rows = []
        for K in (2, 3):
            t, su, sv = R.random_linear(k, n, K, seed=K)
            for pf in (True, False):
                lin = KN.Mul1Gemv(t, su, sv, rows=1, prefetch=pf)
                xh = KN.rot(x, lin.suh)
                mx.eval(xh)
                rows.append(row(f"exl3 mul1 K={K} full pf={int(pf)} tgk={lin.tgk}", timeit(lambda: lin(x), a.iters), k, n, K))
                rows.append(row(f"exl3 mul1 K={K} partial-only pf={int(pf)}", timeit(lambda: lin.partial(xh), a.iters), k, n, K))
            for tgk in (1, 2, 4, 8):
                lin = KN.Mul1Gemv(t, su, sv, rows=1, tgk=tgk)
                rows.append(row(f"exl3 mul1 K={K} full tgk={tgk}", timeit(lambda: lin(x), a.iters), k, n, K))
        w16 = mx.random.normal((n, k)).astype(mx.float16)
        rows.append(row("mlx fp16 matmul", timeit(lambda: x @ w16.T, a.iters), k, n, 16))
        for bits in (4, 3, 2):
            wq, sc, bi = mx.quantize(w16, group_size=64, bits=bits)
            mx.eval(wq, sc, bi)
            rows.append(row(f"mlx affine {bits}-bit", timeit(
                lambda: mx.quantized_matmul(x, wq, sc, bi, transpose=True, group_size=64, bits=bits), a.iters),
                k, n, bits + 0.5 if bits else bits))
        print(f"\n## k={k} n={n}")
        for r in rows:
            print(f"{r['kernel']:<44} {r['us']:>9} us {r['Gweights/s']:>8} Gw/s {r['GB/s']:>7} GB/s")
        results.append({"shape": shp, "rows": rows})
    if a.json:
        json.dump({"chip": chip, "mlx": mx.__version__, "results": results}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
