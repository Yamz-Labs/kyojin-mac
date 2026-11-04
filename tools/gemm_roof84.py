"""Air GEMM roof: fp16/bf16 mx.matmul (NAX path on M5 in MLX 0.32), MLX 4-bit qmm, at prefill shapes. usage: gemm_roof84.py [reps]
Interleaved shapes, min over blocks, a cool-down is not applied (sustained numbers: blocks run back to back)."""
import sys, time, mlx.core as mx
R = int(sys.argv[1]) if len(sys.argv) > 1 else 6
def bench(fn, it):
    mx.eval(fn()); mx.synchronize(); t0 = time.perf_counter()
    ys = [fn() for _ in range(it)]
    mx.eval(ys); return (time.perf_counter() - t0) / it
rows = []
for dt in (mx.float16, mx.bfloat16):
    for M, K, N in [(2048, 2560, 9216), (2048, 2560, 4096), (2048, 4096, 12288), (2048, 4096, 4096), (512, 2560, 9216), (4096, 4096, 4096), (8192, 8192, 8192)]:
        a = mx.random.normal((M, K)).astype(dt); b = mx.random.normal((K, N)).astype(dt); mx.eval(a, b)
        it = min(24, max(4, int(2e12 / (2 * M * K * N))))
        ts = [bench(lambda: a @ b, it) for _ in range(R)]
        t = min(ts); print(f"matmul {str(dt)[9:]:8s} {M}x{K}x{N}: min {t*1e3:7.3f} ms {2*M*K*N/t/1e12:6.2f} TF  med {sorted(ts)[len(ts)//2]*1e3:7.3f} ms {2*M*K*N/sorted(ts)[len(ts)//2]/1e12:6.2f} TF", flush=True)
for M, K, N in [(2048, 2560, 9216), (2048, 4096, 12288), (512, 2560, 9216)]:
    x = mx.random.normal((M, K)).astype(mx.float16); w = mx.random.normal((N, K)).astype(mx.float16)
    wq, s, bz = mx.quantize(w, group_size=64, bits=4); mx.eval(x, wq, s, bz)
    it = min(24, max(4, int(2e12 / (2 * M * K * N))))
    ts = [bench(lambda: mx.quantized_matmul(x, wq, s, bz, transpose=True, group_size=64, bits=4), it) for _ in range(R)]
    t = min(ts); print(f"qmm4 {M}x{K}x{N}: min {t*1e3:7.3f} ms {2*M*K*N/t/1e12:6.2f} TF", flush=True)
