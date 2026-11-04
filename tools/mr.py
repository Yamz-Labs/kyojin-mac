import time, numpy as np, mlx.core as mx
from kyojin_mlx import kernels as KN, ref_numpy as R
k, n = 4096, 12288
for K in (2, 3):
    t, su, sv = R.random_linear(k, n, K, seed=K)
    for mr in (1, 2, 4):
        lin = KN.Mul1Gemv(t, su, sv, rows=mr)
        x = mx.array(np.random.default_rng(0).standard_normal((mr, k)).astype(np.float16))
        xh = KN.rot(x, lin.suh); mx.eval(xh)
        for _ in range(10): mx.eval(lin.partial(xh))
        mx.synchronize(); t0 = time.perf_counter(); o = [lin.partial(xh) for _ in range(200)]; mx.eval(o); mx.synchronize()
        dt = (time.perf_counter() - t0) / 200
        print(f"K={K} rows={mr} {dt*1e6:6.1f} us  {k*n/dt/1e9:6.1f} Gw/s  {mr*k*n/dt/1e9:6.1f} G(w*row)/s")
