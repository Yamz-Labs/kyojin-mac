import time, numpy as np, mlx.core as mx
rng = np.random.default_rng(0)
A = mx.array(rng.standard_normal((2048, 1280)).astype(np.float16)); W = mx.array((rng.standard_normal((32768, 1280)) * 0.03).astype(np.float16)); mx.eval(A, W)
best = 1e9
for _ in range(15):
    t0 = time.perf_counter(); mx.eval(A @ W.T); best = min(best, time.perf_counter() - t0)
print(f"ruler fp16 matmul 2048x32768x1280: {best*1e3:.2f} ms = {2*2048*32768*1280/best/1e12:.1f} TF")
