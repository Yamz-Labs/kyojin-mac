"""Best-case GEMM throughput: single calls (eval each), N calls per shape, report min / p10 / median TF, interleaved fp16 matmul and MLX qmm4, qmm8-free. usage: gemm_peak84.py [calls=60]"""
import sys, time, mlx.core as mx
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
shapes = [(2048, 2560, 9216), (2048, 9216, 2560), (2048, 2560, 8192), (2048, 4096, 12288), (2048, 12288, 4096), (2048, 4096, 8192)]
ops = []
for M, K, Nn in shapes:
    x = mx.random.normal((M, K)).astype(mx.float16); w = mx.random.normal((Nn, K)).astype(mx.float16); wq, s, bz = mx.quantize(w, group_size=64, bits=4); mx.eval(x, w, wq, s, bz)
    ops.append((f"fp16 {M}x{K}x{Nn}", 2 * M * K * Nn, lambda x=x, w=w: x @ w.T)); ops.append((f"qmm4 {M}x{K}x{Nn}", 2 * M * K * Nn, lambda x=x, wq=wq, s=s, bz=bz: mx.quantized_matmul(x, wq, s, bz, transpose=True, group_size=64, bits=4)))
T = {n: [] for n, _, _ in ops}
for _ in range(3):
    for n, fl, f in ops: mx.eval(f())
for r in range(N):
    for n, fl, f in ops:
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(f()); T[n].append(time.perf_counter() - t0)
for n, fl, f in ops:
    v = sorted(T[n]); print(f"{n:26s} min {fl/v[0]/1e12:6.2f} TF  p10 {fl/v[len(v)//10]/1e12:6.2f}  med {fl/v[len(v)//2]/1e12:6.2f}  ({v[0]*1e3:.2f} ms)")
