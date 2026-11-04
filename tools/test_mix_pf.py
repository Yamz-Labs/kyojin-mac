import sys, time, numpy as np, mlx.core as mx
from kyojin_mlx import mhc_prefill as MP
rng = np.random.default_rng(0)
for T in (64, 100, 2048):
    D = 5120
    h = mx.array((rng.standard_normal((1, T, 4, D)) * 3).astype(np.float16)); fn = mx.array((rng.standard_normal((24, 4 * D)) * 0.02).astype(np.float32))
    ref = mx.fast.rms_norm(h.flatten(-2).astype(mx.float32), None, 1e-20) @ fn.T
    got = MP.mixes(h, fn, 1e-20); mx.eval(ref, got)
    tm = lambda f: min((lambda t0: (mx.eval(f()), time.perf_counter() - t0)[1])(time.perf_counter()) for _ in range(6)) * 1e3
    print(f"T={T}: rel {(mx.linalg.norm(got - ref) / mx.linalg.norm(ref)).item():.2e} max|d| {mx.abs(got-ref).max().item():.2e} | stock {tm(lambda: mx.fast.rms_norm(h.flatten(-2).astype(mx.float32), None, 1e-20) @ fn.T):.2f} ms kernel {tm(lambda: MP.mixes(h, fn, 1e-20)):.2f} ms")
