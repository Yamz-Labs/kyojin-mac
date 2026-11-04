"""fp32 ceilings: fp32 matmul (simdgroup_matrix float) and pure fp32 FMA ALU rate. The accurate GDN core must run fp32 state math."""
import time, mlx.core as mx
def best(f, n=25):
    mx.eval(f()); t = []
    for _ in range(n):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(f()); t.append(time.perf_counter() - t0)
    t.sort(); return t[0], t[n // 4]
for M, K, N in [(2048, 2560, 9216), (4096, 4096, 4096)]:
    x = mx.random.normal((M, K)); w = mx.random.normal((N, K)); mx.eval(x, w)
    for dt in (mx.float32, mx.float16):
        a, b = x.astype(dt), w.astype(dt); mx.eval(a, b)
        m, p = best(lambda: a @ b.T); fl = 2 * M * K * N
        print(f"matmul {dt} {M}x{K}x{N}: min {fl/m/1e12:.2f} TF  p25 {fl/p/1e12:.2f}", flush=True)
src = """
 uint i = thread_position_in_grid.x; float a = x[i], b = 1.0001f, c = 0.5f, d = 0.25f, e = 0.125f, f = 0.0625f, g = 0.03f, h = 0.01f;
 for (int j = 0; j < 4096; ++j) { a = fma(a, b, c); c = fma(c, b, d); d = fma(d, b, e); e = fma(e, b, f); f = fma(f, b, g); g = fma(g, b, h); h = fma(h, b, a); b = fma(b, 0.99999f, 1e-6f); }
 y[i] = a + c + d + e + f + g + h + b;
"""
k = mx.fast.metal_kernel(name="fmab", input_names=["x"], output_names=["y"], source=src)
n = 1 << 20; x = mx.random.uniform(shape=(n,))
f = lambda: k(inputs=[x], output_shapes=[(n,)], output_dtypes=[mx.float32], grid=(n, 1, 1), threadgroup=(256, 1, 1))[0]
m, p = best(f); print(f"fp32 FMA ALU: min {n*4096*8*2/m/1e12:.2f} TF  p25 {n*4096*8*2/p/1e12:.2f}")
