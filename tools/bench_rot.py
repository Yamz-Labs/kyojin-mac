"""Hadamard kernel latency: old barrier kernel vs simdgroup kernel, chained launches. usage: bench_rot.py"""
import time, numpy as np, mlx.core as mx
from kyojin_mlx import kernels as KN, metal_src as S
rot0 = mx.fast.metal_kernel(name="v0_rot", input_names=["x", "suh", "params"], output_names=["xh"], source=S.ROT_SOURCE_V0)
def t(fn, n=300):
    for _ in range(5): mx.eval(fn())
    mx.synchronize(); t0 = time.perf_counter(); o = [fn() for _ in range(n)]; mx.eval(o); mx.synchronize()
    return (time.perf_counter() - t0) / n * 1e6
for rows, k in ((1, 2048), (1, 6144), (4, 2048), (512, 2048)):
    x = mx.array(np.random.randn(rows, k).astype(np.float16)); suh = mx.array(np.ones(k, np.float16)); p = mx.array([k], dtype=mx.uint32)
    old = lambda: rot0(inputs=[x, suh, p], grid=(k, rows, 1), threadgroup=(128, 1, 1), output_shapes=[(rows, k)], output_dtypes=[mx.float32])[0]
    print(f"rows={rows:4d} k={k}: old {t(old):7.1f} us   new fp32-out {t(lambda: KN.rot(x, suh)):7.1f} us   new fp16-out {t(lambda: KN.rot(x, suh, mx.float16)):7.1f} us")
