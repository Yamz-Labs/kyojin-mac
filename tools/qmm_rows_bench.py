"""MLX 4-bit affine quantized_matmul time per call at 1..16 rows, weights streamed from a pool larger than cache. usage: qmm_rows_bench.py [K N]"""
import sys, time
import mlx.core as mx
K, N = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (2560, 9216)
NW = max(2, int(400e6 // (K * N // 2)))
ws = []
for _ in range(NW):
    w = mx.random.normal((N, K)).astype(mx.float16)
    ws.append(mx.quantize(w, group_size=64, bits=4)); mx.eval(ws[-1])
for R in (1, 2, 3, 4, 5, 6, 8, 12, 16):
    x = mx.random.normal((R, K)).astype(mx.float16); mx.eval(x)
    def run():
        o = [mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=4) for (w, s, b) in ws]
        mx.eval(o)
    run(); run()
    t0 = time.perf_counter(); run(); dt = (time.perf_counter() - t0) / NW
    print(f"R={R:2d} {dt*1e6:7.1f} us  {K*N/2*1.125/dt/1e9:6.1f} GB/s")
