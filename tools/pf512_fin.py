import sys, time, numpy as np, mlx.core as mx
from kyojin_mlx import exl3_linear as EL, kernels as KN
R = int(sys.argv[1])
def T(f, n=6, c=16):
    mx.eval(f()); mx.synchronize(); ts = []
    for _ in range(n):
        t0 = time.perf_counter(); mx.eval([f() for _ in range(c)]); ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3 / c
for kin, nout in ((2560, 9216), (9216, 2560), (4096, 2560), (2560, 1024)):
    suh = mx.ones((kin,), dtype=mx.float16); svh = mx.ones((nout,), dtype=mx.float16)
    x = (mx.random.normal((R, kin)) * .5).astype(mx.float16)
    inner = (mx.random.normal((1, R, nout)) * .5).astype(mx.float16); fp = mx.array([1, nout, R], dtype=mx.uint32)
    grid, tg = KN.S.had_grid(nout // 128, R)
    mx.eval(x, inner)
    rot = T(lambda: KN.rot(x, suh, mx.float16, perm=True))
    fin = T(lambda: KN.fin_kernel(mx.float16, mx.float16)(inputs=[inner, svh, fp], template=[("OT", mx.float16)], grid=grid, threadgroup=tg, output_shapes=[(R, nout)], output_dtypes=[mx.float16])[0])
    cp = T(lambda: inner * mx.array(1.0, dtype=mx.float16))
    print(f"{kin}x{nout} R={R}: rot {rot:.3f} ms ({x.nbytes*2/rot/1e6:.0f} GB/s)  fin {fin:.3f} ms ({inner.nbytes*2/fin/1e6:.0f} GB/s)  elementwise copy {cp:.3f}")
