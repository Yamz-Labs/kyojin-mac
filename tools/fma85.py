"""FMA throughput on this GPU: float vs half vs half2/half4 vs simdgroup_matrix(half) dependent-free chains. TFLOPS (2 flop per FMA)."""
import time, mlx.core as mx
SRC = """
    uint gid = thread_position_in_grid.x;
    TYPE a0 = (TYPE)(x[gid & 1023] * 0.001f), a1 = a0 + (TYPE)1, a2 = a0 + (TYPE)2, a3 = a0 + (TYPE)3, a4 = a0 + (TYPE)4, a5 = a0 + (TYPE)5, a6 = a0 + (TYPE)6, a7 = a0 + (TYPE)7;
    TYPE a8 = a0 + (TYPE)8, a9 = a0 + (TYPE)9, a10 = a0 + (TYPE)10, a11 = a0 + (TYPE)11, a12 = a0 + (TYPE)12, a13 = a0 + (TYPE)13, a14 = a0 + (TYPE)14, a15 = a0 + (TYPE)15;
    TYPE b = (TYPE)(x[(gid + 7) & 1023] * 0.001f + 0.999f), c = (TYPE)(x[(gid + 3) & 1023] * 0.001f);
    for (uint i = 0; i < ITERS; ++i) {
        a0 = fma(a0, b, c); a1 = fma(a1, b, c); a2 = fma(a2, b, c); a3 = fma(a3, b, c);
        a4 = fma(a4, b, c); a5 = fma(a5, b, c); a6 = fma(a6, b, c); a7 = fma(a7, b, c);
        a8 = fma(a8, b, c); a9 = fma(a9, b, c); a10 = fma(a10, b, c); a11 = fma(a11, b, c); a12 = fma(a12, b, c); a13 = fma(a13, b, c); a14 = fma(a14, b, c); a15 = fma(a15, b, c);
    }
    TYPE s = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
    out[gid] = (float)(s[0]);
"""
SRC1 = SRC.replace("(float)(s[0])", "(float)s")
x = mx.random.normal((1024,)).astype(mx.float32)
ITERS = 1024; NT = 1 << 20
for name, ty, lanes, src in (("float", "float", 1, SRC1), ("half", "half", 1, SRC1), ("half2", "half2", 2, SRC), ("float2", "float2", 2, SRC), ("half4", "half4", 4, SRC)):
    k = mx.fast.metal_kernel("fma_" + name, ["x"], ["out"], src.replace("TYPE", ty).replace("ITERS", str(ITERS)))
    f = lambda: k(inputs=[x], grid=(NT, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(NT,)], output_dtypes=[mx.float32])[0]
    mx.eval(f())
    ts = []
    for _ in range(7):
        t0 = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - t0)
    t = min(ts); flop = NT * ITERS * 16 * lanes * 2
    print(f"{name:7s} {flop / t / 1e12:6.2f} TFLOPS  ({t*1e3:.1f} ms)", flush=True)
