import time, numpy as np, mlx.core as mx
rng = np.random.default_rng(0)
A = mx.array(rng.standard_normal((2048, 1280)).astype(np.float16)); W = mx.array((rng.standard_normal((32768, 1280)) * 0.03).astype(np.float16)); mx.eval(A, W)
def burst(n):
    res = []
    for _ in range(n):
        t0 = time.perf_counter(); mx.eval(A @ W.T); res.append(2 * 2048 * 32768 * 1280 / (time.perf_counter() - t0) / 1e12)
    return res
def sustain(secs):
    t_end = time.time() + secs
    while time.time() < t_end: mx.eval(A @ W.T)
for idle in (1, 2, 4, 8, 15, 30, 60):
    sustain(8)
    time.sleep(idle)
    r = burst(12)
    print(f"idle {idle:3d}s after 8s load: matmul TF " + " ".join(f"{v:4.1f}" for v in r), flush=True)
