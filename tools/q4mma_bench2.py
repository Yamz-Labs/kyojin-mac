"""q4mma variants vs mx.quantized_matmul: error + time (distinct weights, 24 calls per eval, interleaved passes). usage: q4mma_bench2.py [--shapes gate,head] [--rows 1,4,8]"""
import argparse, time, statistics
import mlx.core as mx
from kyojin_mlx import q4mma as Q
ap = argparse.ArgumentParser(); ap.add_argument("--rows", default="1,4,8"); ap.add_argument("--shapes", default="gate,out,kv,head"); ap.add_argument("--passes", type=int, default=5)
ap.add_argument("--cfgs", default="v1:4:4,v2:4:1,v2:4:2,v2:2:1,v2:2:2"); a = ap.parse_args()
SH = {"qkv": (8192, 2560), "z": (4096, 2560), "out": (2560, 4096), "gate": (9216, 2560), "down": (2560, 9216), "kv": (1024, 2560), "head": (248320, 2560)}
def tm(fn, n):
    t0 = time.perf_counter(); mx.eval([fn(i) for i in range(n)]); return (time.perf_counter() - t0) / n * 1e6
for name in a.shapes.split(","):
    N, K = SH[name]; L = 3 if name == "head" else 24
    ws = [mx.random.randint(0, 2**31 - 1, (N, K // 8)).astype(mx.uint32) * mx.array(2654435761, dtype=mx.uint32) for _ in range(L)]
    scs = [mx.random.uniform(0.001, 0.02, (N, K // 64)).astype(mx.bfloat16) for _ in range(L)]
    bis = [mx.random.uniform(-0.1, 0.0, (N, K // 64)).astype(mx.bfloat16) for _ in range(L)]
    mx.eval(ws, scs, bis)
    for R in [int(r) for r in a.rows.split(",")]:
        x = mx.random.normal((R, K)).astype(mx.bfloat16); mx.eval(x)
        ref = mx.quantized_matmul(x, ws[0], scs[0], bis[0], transpose=True, group_size=64, bits=4).astype(mx.float32)
        px1 = Q.prep_x(x); px2 = Q.prep_x2(x); mx.eval(px1, px2)
        fns = {"mlx": lambda i: mx.quantized_matmul(x, ws[i % L], scs[i % L], bis[i % L], transpose=True, group_size=64, bits=4)}
        errs = {}
        for cfg in a.cfgs.split(","):
            v, nt, ks = cfg.split(":"); nt = int(nt); ks = int(ks)
            if v == "f": fns[cfg] = (lambda i: Q.q4fma(x, ws[i % L], scs[i % L], bis[i % L]))
            elif v.startswith("d"): fns[cfg] = (lambda nt, ks, dbg: lambda i: Q.q4mma2(px2, ws[i % L], scs[i % L], bis[i % L], R, nt, ks, dbg))(nt, ks, int(v[1:]))
            elif v == "v1": fns[cfg] = (lambda nt, ks: lambda i: Q.q4mma(px1, ws[i % L], scs[i % L], bis[i % L], R, nt, ks))(nt, ks)
            else: fns[cfg] = (lambda nt, ks: lambda i: Q.q4mma2(px2, ws[i % L], scs[i % L], bis[i % L], R, nt, ks))(nt, ks)
            errs[cfg] = float(mx.max(mx.abs(fns[cfg](0).astype(mx.float32) - ref)) / mx.max(mx.abs(ref)))
        res = {k: [] for k in fns}
        for p in range(a.passes + 1):
            for k, f in fns.items():
                t = tm(f, L)
                if p: res[k].append(t)
        gb = N * K / 2 / 1e3
        h = len(next(iter(res.values()))) // 3
        print(f"{name:5s} R={R}: first-third min | last-third median (us)  " + "  ".join(f"{k} {min(v[:h]):4.0f}|{statistics.median(v[-h:]):4.0f}" for k, v in res.items()), flush=True)
