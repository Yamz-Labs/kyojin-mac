"""Residual add + RMSNorm: stock (add, mx.fast.rms_norm) vs one fused kernel, and silu(gate)*up: stock mlx_lm swiglu (compiled) vs one custom kernel. [2048,H] bf16, min of blocks, 64 add+norm and 32 swiglu per chunk."""
import time, mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu
def bench(fn, it=40, blocks=6):
    mx.eval(fn()); best = 1e9
    for _ in range(blocks):
        mx.synchronize(); t0 = time.perf_counter(); ys = [fn() for _ in range(it)]; mx.eval(ys); best = min(best, (time.perf_counter() - t0) / it)
    return best * 1e6
src = """
 uint row = threadgroup_position_in_grid.x; uint t = thread_position_in_threadgroup.x;
 float ss = 0; threadgroup float red[32];
 for (uint i = t; i < H; i += 256) { float v = float(x[row*H+i]) + float(r[row*H+i]); ss += v*v; }
 ss = simd_sum(ss); if (thread_index_in_simdgroup == 0) red[simdgroup_index_in_threadgroup] = ss; threadgroup_barrier(mem_flags::mem_threadgroup);
 float tot = 0; for (uint i = 0; i < 8; i++) tot += red[i];
 float inv = metal::precise::rsqrt(tot / H + eps[0]);
 for (uint i = t; i < H; i += 256) { T h = T(float(x[row*H+i]) + float(r[row*H+i])); hout[row*H+i] = h; nout[row*H+i] = T(float(h) * inv * float(w[i])); }
"""
k = mx.fast.metal_kernel(name="addnorm84", input_names=["x", "r", "w", "eps"], output_names=["hout", "nout"], source=src)
for H in (2560, 4096):
    x = mx.random.normal((2048, H)).astype(mx.bfloat16); r = mx.random.normal((2048, H)).astype(mx.bfloat16); w = (mx.random.normal((H,)) * .1 + 1).astype(mx.bfloat16); eps = mx.array([1e-6], dtype=mx.float32)
    stock = lambda: (lambda h: (h, mx.fast.rms_norm(h, w, 1e-6)))(x + r)
    fused = lambda: tuple(k(inputs=[x, r, w, eps], template=[("T", mx.bfloat16), ("H", H)], grid=(2048 * 256, 1, 1), threadgroup=(256, 1, 1), output_shapes=[x.shape, x.shape], output_dtypes=[mx.bfloat16] * 2))
    a, b = stock(), fused(); mx.eval(a, b)
    print(f"H={H}: add+norm stock {bench(stock):6.1f} us  fused {bench(fused):6.1f} us  max|d| h {float(mx.abs(a[0].astype(mx.float32)-b[0].astype(mx.float32)).max()):.2g} norm {float(mx.abs(a[1].astype(mx.float32)-b[1].astype(mx.float32)).max()):.2g}", flush=True)
    I = 9216 if H == 2560 else 12288
    g = mx.random.normal((2048, I)).astype(mx.bfloat16); u = mx.random.normal((2048, I)).astype(mx.bfloat16)
    print(f"H={H}: swiglu (mlx_lm) {bench(lambda: swiglu(g, u)):6.1f} us  eager silu*up {bench(lambda: nn.silu(g) * u):6.1f} us  (floor 3 passes {3*2048*I*2/139e9*1e6:.0f} us)", flush=True)
