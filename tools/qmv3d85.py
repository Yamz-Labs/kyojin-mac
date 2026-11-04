"""mlx4 few-row linear: x as [R, K] (M=R) vs [R, 1, K] (batch R, M=1, what the batched decode step feeds the quantized linears). Interleaved, us per call (24 distinct weights, hot).
usage: qmv3d85.py [shape ...]"""
import sys, time, statistics
import mlx.core as mx
SH = {"qkv": (8192, 2560), "z": (4096, 2560), "out": (2560, 4096), "gate": (9216, 2560), "down": (2560, 9216), "kv": (1024, 2560), "head": (248320, 2560)}
shapes = sys.argv[1:] or ["gate", "down", "qkv", "out", "head"]
for sh in shapes:
    N, K = SH[sh]; L = 3 if sh == "head" else 24
    ws = [mx.random.randint(0, 2**31 - 1, (N, K // 8)).astype(mx.uint32) * mx.array(2654435761, dtype=mx.uint32) for _ in range(L)]
    scs = [mx.random.uniform(0.001, 0.02, (N, K // 64)).astype(mx.bfloat16) for _ in range(L)]
    bis = [mx.random.uniform(-0.1, 0.0, (N, K // 64)).astype(mx.bfloat16) for _ in range(L)]
    mx.eval(ws, scs, bis)
    Rs = [1, 2, 3, 4]
    X = {R: mx.random.normal((R, K)).astype(mx.bfloat16) for R in Rs}
    X3 = {R: X[R].reshape(R, 1, K) for R in Rs}
    mx.eval(list(X.values()), list(X3.values()))
    def f(x):
        return lambda: [mx.quantized_matmul(x, ws[i], scs[i], bis[i], transpose=True, group_size=64, bits=4) for i in range(L)]
    fns = {(R, d): f(X[R] if d == "2d" else X3[R]) for R in Rs for d in ("2d", "3d")}
    res = {k: [] for k in fns}
    for k, fn in fns.items(): mx.eval(fn())
    for p in range(15):
        for k, fn in fns.items():
            t0 = time.perf_counter(); mx.eval(fn()); res[k].append((time.perf_counter() - t0) / L * 1e6)
    print(sh, "us/call (median): " + "  ".join(f"R{R} 2d {statistics.median(res[(R,'2d')]):.0f} 3d {statistics.median(res[(R,'3d')]):.0f}" for R in Rs), flush=True)
