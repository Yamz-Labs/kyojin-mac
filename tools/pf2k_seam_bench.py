import sys, time, mlx.core as mx, numpy as np
from kyojin_mlx import kernels as KN, layer_pf as LP
R = int(sys.argv[1]); K = int(sys.argv[2]) if len(sys.argv) > 2 else 2560
rng = np.random.default_rng(1)
h16 = lambda *s: mx.array((rng.standard_normal(s) * 0.5).astype(np.float16))
inner, x = h16(R, K), h16(R, K); svh, nw, sg, su = (h16(K) for _ in range(4)); mx.eval(inner, x, svh, nw, sg, su)
fp = mx.array([1, K, R], dtype=mx.uint32); grid, tg = KN.S.had_grid(K // 128, R)
def stock():
    y = KN.fin_kernel(mx.float16, mx.float16)(inputs=[inner.reshape(1, R, K), svh, fp], template=[("OT", mx.float16)], grid=grid, threadgroup=tg, output_shapes=[(R, K)], output_dtypes=[mx.float16])[0]
    h = x + y; n = mx.fast.rms_norm(h, nw, 1e-6)
    return [h, KN.rot(n, sg, mx.float16, perm=True), KN.rot(n, su, mx.float16, perm=False)]
fused = lambda: list((LP.seam1_split if len(sys.argv) > 3 else LP.seam1)(inner, svh, x, nw, 1e-6, sg, su, True, False))
a, b = stock(), fused(); mx.eval(a, b)
print("bitexact", [bool(mx.array_equal(p, q).item()) for p, q in zip(a, b)])
def T(f, n=12):
    mx.eval(f()); ts = []
    for _ in range(n):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3
print(f"R={R} K={K} stock {T(stock):.3f} ms fused {T(fused):.3f} ms")
