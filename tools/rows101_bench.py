"""Few-row q4 linear table (hot regime, distinct weight copies per call, prep of x charged to every call): MLX quantized_matmul vs q4mma3 vs
q4nax variants, us per call (median of passes) and fraction of the read roof (weight + bf16 scale/bias bytes / 131.9 GB/s).
usage: rows101_bench.py NxK[,..] [ROWS=1,2,4,5,8,12,16,32] [IMPLS=mlx,q3,n:2:4:1] [PASSES=5]   (n:u:nb:ks)"""
import sys, time, statistics
import mlx.core as mx
from kyojin_mlx import q4mma as Q

BW = 131.9e9
shapes = [tuple(int(v) for v in s.split("x")) for s in sys.argv[1].split(",")]
Rs = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "1,2,4,5,8,12,16,32").split(",")]
impls = (sys.argv[3] if len(sys.argv) > 3 else "mlx,q3,n:2:4:1").split(",")
P = int(sys.argv[4]) if len(sys.argv) > 4 else 5
z = mx.random.normal((2048, 2048)).astype(mx.float16); t0 = time.perf_counter()
while time.perf_counter() - t0 < 4: mx.eval([z @ z for _ in range(8)])
for N, K in shapes:
    L = max(2, min(16, int(1.0e9 // (N * K // 2))))
    ws = [mx.random.randint(0, 2**31 - 1, (N, K // 8)).astype(mx.uint32) * mx.array(2654435761, dtype=mx.uint32) for _ in range(L)]
    scs = [mx.random.uniform(0.001, 0.02, (N, K // 64)).astype(mx.bfloat16) for _ in range(L)]
    bis = [mx.random.uniform(-0.1, 0.0, (N, K // 64)).astype(mx.bfloat16) for _ in range(L)]
    mx.eval(ws, scs, bis)
    roof = (N * K // 2 + 2 * 2 * N * K // 64) / BW * 1e6
    print(f"{N}x{K} roof {roof:.0f} us", flush=True)
    for R in Rs:
        x = mx.random.normal((R, K)).astype(mx.bfloat16); mx.eval(x)
        f = {}
        for im in impls:
            p = im.split(":")
            if im == "mlx": f[im] = lambda i: mx.quantized_matmul(x, ws[i], scs[i], bis[i], transpose=True, group_size=64, bits=4)
            elif im == "q3" and R <= 16: f[im] = lambda i: Q.q4mma3(Q.prep_x(x), ws[i], scs[i], bis[i], R, Q.NT, Q.KS)
            elif p[0] == "n" and R <= 32 and K % (128 * int(p[1])) == 0:
                u, nb, ks = int(p[1]), int(p[2]), int(p[3]); dbg = int(p[4]) if len(p) > 4 else 0; pf = int(p[5]) if len(p) > 5 else 1
                f[im] = (lambda u, nb, ks, dbg, pf: lambda i: Q.q4nax(Q.prep_xn(x, u), ws[i], scs[i], bis[i], R, u, nb, ks, dbg, pf))(u, nb, ks, dbg, pf)
        ref = f["mlx"](0).astype(mx.float32) if "mlx" in f else None
        err = {k: (mx.abs(fn(0).astype(mx.float32) - ref).max().item() / (mx.abs(ref).max().item() + 1e-9)) if ref is not None else 0 for k, fn in f.items()}
        T = {k: [] for k in f}
        for p in range(P + 1):
            for k, fn in f.items():
                t0 = time.perf_counter(); mx.eval([fn(i) for i in range(L)]); dt = (time.perf_counter() - t0) / L * 1e6
                if p: T[k].append(dt)
        print(f"  R{R:2d} " + "  ".join(f"{k} {statistics.median(v):6.0f}({roof / statistics.median(v):4.0%}{'' if err[k] < 1e-2 or ':' in k and int((k.split(':') + ['0'] * 5)[4]) else ' ERR%.1e' % err[k]})" for k, v in T.items()), flush=True)
    del ws, scs, bis
