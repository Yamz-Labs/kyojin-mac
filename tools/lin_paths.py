"""Cost (us, chained) of one EXL3 linear at R rows through each path: default (what the server runs), row-group GEMV, MMA, NAX fused. usage: lin_paths.py [--shapes mlp_up,mlp_down,qkv] [--rows 1,5,8]"""
import argparse, statistics, numpy as np, mlx.core as mx
from kyojin_mlx import exl3_linear as EL
from mma_bench import synth, SHAPES, timeit
ap = argparse.ArgumentParser(); ap.add_argument("--shapes", default="mlp_up,mlp_down,qkv"); ap.add_argument("--rows", default="1,5,8"); a = ap.parse_args()
def setp(m):
    EL.ROW_SPLIT_MAX, EL.MMA_ROWS, EL.ROWS_NAX = {"default": (0, 0, True), "rg": (32, 0, False), "mma": (0, 1, True), "nax": (0, 0, True)}[m]
for name in a.shapes.split(","):
    kin, nout, K = SHAPES[name]; lin = synth(kin, nout, K); mx.eval(lin._words); rng = np.random.default_rng(5)
    for R in [int(v) for v in a.rows.split(",")]:
        xs = [mx.array(rng.standard_normal((R, kin)).astype(np.float16)) for _ in range(12)]; mx.eval(*xs)
        T = {}
        for rep in range(5):
            for m in (["default"] if R == 1 else ["default", "rg", "mma"]):
                setp(m); EL.GEMV_MAX_ROWS = 4
                T.setdefault(m, []).append(timeit(lambda x: lin(x), xs, 5, 12)[1])
        print(name, R, " ".join(f"{m} {statistics.median(t):.0f}" for m, t in T.items()), flush=True)
