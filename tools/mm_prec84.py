"""Is the fp32 NAX matmul full fp32? relative error vs float64 numpy, fp32 / bf16-rounded-input / fp16 inputs."""
import numpy as np, mlx.core as mx
rng = np.random.default_rng(0); M, K, N = 256, 1024, 256
a = rng.standard_normal((M, K)).astype(np.float32); b = rng.standard_normal((N, K)).astype(np.float32)
ref = a.astype(np.float64) @ b.astype(np.float64).T
for name, dt in (("fp32", mx.float32), ("fp16", mx.float16), ("bf16", mx.bfloat16)):
    A = mx.array(a).astype(dt); B = mx.array(b).astype(dt)
    r = (np.array(A.astype(mx.float32)).astype(np.float64) @ np.array(B.astype(mx.float32)).astype(np.float64).T)   # exact product of the rounded inputs
    y = np.array((A @ B.T).astype(mx.float32)).astype(np.float64)
    print(f"{name}: rel err vs exact-of-rounded-inputs {np.abs(y-r).max()/np.abs(r).max():.3g}   (vs fp32 truth {np.abs(y-ref).max()/np.abs(ref).max():.3g})")
# fp32 matmul small tile
A = mx.array(a[:64, :128]); B = mx.array(b[:64, :128]); r = a[:64, :128].astype(np.float64) @ b[:64, :128].astype(np.float64).T
print("fp32 64x128x64 rel err", np.abs(np.array(A @ B.T).astype(np.float64) - r).max() / np.abs(r).max())
