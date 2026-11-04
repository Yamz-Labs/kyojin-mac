"""CPU cost of building and encoding GPU work in MLX: per-op python build time, async_eval return time, wait time, for builtin ops and custom kernels.
Separates the CPU critical path (what bounds an Ultra whose GPU is 8x wider) from the GPU time."""
import statistics, time
import mlx.core as mx
import numpy as np

x0 = mx.ones((1, 5120), mx.float16); mx.eval(x0)
K = mx.fast.metal_kernel(name="m10_noop", input_names=["a"], output_names=["o"], source="o[thread_position_in_grid.x] = a[thread_position_in_grid.x] + half(1);")
def cust(x):
    return K(inputs=[x], grid=(5120, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(1, 5120)], output_dtypes=[mx.float16])[0]
cases = {"builtin mul": lambda x: x * 1.0001, "builtin mul+add+astype": lambda x: (x * 1.0001 + 0.5).astype(mx.float16), "custom metal_kernel": cust}
for name, f in cases.items():
    for n in (1000,):
        rows = []
        for rep in range(9):
            x = x0
            t0 = time.perf_counter()
            for _ in range(n):
                x = f(x)
            t1 = time.perf_counter()
            mx.async_eval(x)
            t2 = time.perf_counter()
            mx.eval(x)
            t3 = time.perf_counter()
            rows.append((t1 - t0, t2 - t1, t3 - t2))
        b, e, w = (1e6 * statistics.median(r[i] for r in rows) / n for i in range(3))
        print(f"{name:26s} n={n}: build {b:5.2f} us/op, async_eval return {e:5.2f} us/op, wait {w:5.2f} us/op, total {b + e + w:5.2f}")
