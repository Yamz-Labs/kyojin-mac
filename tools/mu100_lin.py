"""Few-row 4-bit linear microbench (hot regime): MLX quantized_matmul vs q4mma (up to q4mma.MAX_ROWS rows) per shape and row count,
us per call and fraction of the read roof (weight + scale/bias bytes / BW). Distinct weight copies per call so nothing stays in the SLC.
usage: mu100_lin.py NxK[,NxK..] [ROWS=1,4,8,12,16] [PASSES=5]"""
import sys, time, statistics
import mlx.core as mx
from kyojin_mlx import q4mma as Q

BW = 139.3e9
shapes = [tuple(int(v) for v in s.split("x")) for s in sys.argv[1].split(",")]
Rs = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "1,4,8,12,16").split(",")]
P = int(sys.argv[3]) if len(sys.argv) > 3 else 5
z = mx.random.normal((2048, 2048)).astype(mx.float16); t0 = time.perf_counter()
while time.perf_counter() - t0 < 8: mx.eval([z @ z for _ in range(8)])
for N, K in shapes:
    L = max(2, min(16, int(1.2e9 // (N * K // 2))))
    ws = [mx.random.randint(0, 2**31 - 1, (N, K // 8)).astype(mx.uint32) * mx.array(2654435761, dtype=mx.uint32) for _ in range(L)]
    scs = [mx.random.uniform(0.001, 0.02, (N, K // 64)).astype(mx.bfloat16) for _ in range(L)]
    bis = [mx.random.uniform(-0.1, 0.0, (N, K // 64)).astype(mx.bfloat16) for _ in range(L)]
    mx.eval(ws, scs, bis)
    roof = (N * K // 2 + 2 * 2 * N * K // 64) / BW * 1e6
    line = [f"{N}x{K} roof {roof:.0f} us:"]
    for R in Rs:
        x = mx.random.normal((R, K)).astype(mx.bfloat16); mx.eval(x)
        f = {"mlx": lambda i: mx.quantized_matmul(x, ws[i], scs[i], bis[i], transpose=True, group_size=64, bits=4)}
        if R <= Q.MAX_ROWS: f["q"] = lambda i: Q.q4mma3(Q.prep_x(x), ws[i], scs[i], bis[i], R, Q.NT, Q.KS)
        ref = f["mlx"](0).astype(mx.float32)
        T = {k: [] for k in f}
        err = {}
        for k, fn in f.items():
            err[k] = mx.abs(fn(0).astype(mx.float32) - ref).max().item() / (mx.abs(ref).max().item() + 1e-9)
        for p in range(P + 1):
            for k, fn in f.items():
                t0 = time.perf_counter(); mx.eval([fn(i) for i in range(L)]); dt = (time.perf_counter() - t0) / L * 1e6
                if p: T[k].append(dt)
        line.append(f"R{R} " + " ".join(f"{k} {statistics.median(v):.0f}({roof / statistics.median(v):.0%}{'' if err[k] < 2e-2 else ' ERR%.1e' % err[k]})" for k, v in T.items()))
    print(" | ".join(line), flush=True)
    del ws, scs, bis
