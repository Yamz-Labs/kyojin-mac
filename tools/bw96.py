"""bw96.py: Air GPU roofs for the decode table: streaming read GB/s (custom uint4 read kernel, several widths; MLX max-reduce), the GEMV of
MLX affine q4 at a few shapes, and the cost of one dependent / independent tiny dispatch inside one command buffer (the step's serial gap)."""
import time, statistics, json
import mlx.core as mx

N = 1 << 28                                   # 1 GiB of uint32
big = mx.random.randint(0, 2**31 - 1, (N,)).astype(mx.uint32); mx.eval(big)
SRC = """
    uint tid = thread_position_in_grid.x; uint nt = threads_per_grid.x;
    const device uint4* p = (const device uint4*)a;
    uint4 acc = 0;
    for (uint i = tid; i < NV; i += nt) acc ^= p[i];
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345u) out[0] = 1u;
"""
k = mx.fast.metal_kernel(name="bw96_read", input_names=["a"], output_names=["out"], source=SRC)


def tbest(f, n=7):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - t0)
    return min(ts[1:]), statistics.median(ts[1:])


res = {}
for tg, grid in ((256, 1 << 16), (256, 1 << 18), (1024, 1 << 18), (512, 1 << 20)):
    b, m = tbest(lambda: k(inputs=[big], template=[("NV", N // 4)], grid=(grid, 1, 1), threadgroup=(tg, 1, 1), output_shapes=[(1,)], output_dtypes=[mx.uint32])[0])
    res[f"read_tg{tg}_g{grid}"] = round(big.nbytes / b / 1e9, 1)
b, m = tbest(lambda: big.max()); res["max_reduce"] = round(big.nbytes / b / 1e9, 1)
for (n_, k_) in ((16384, 8192), (9216, 2560), (2560, 9216), (248320, 2560)):
    w = mx.random.normal((n_, k_)).astype(mx.float16)
    wq, s, bb = mx.quantize(w, group_size=64, bits=4); x = mx.random.normal((1, k_)).astype(mx.bfloat16); s = s.astype(mx.bfloat16); bb = bb.astype(mx.bfloat16)
    del w; mx.eval(wq, s, bb, x)
    by = wq.nbytes + s.nbytes + bb.nbytes
    reps = max(1, int(400e6 // by))
    b, m = tbest(lambda: [mx.quantized_matmul(x, wq, s, bb, transpose=True, group_size=64, bits=4) for _ in range(reps)])
    res[f"qmv_{n_}x{k_}"] = round(by * reps / b / 1e9, 1)
x0 = mx.zeros((256,), dtype=mx.float16); mx.eval(x0)
def dep(n):
    def f():
        y = x0
        for _ in range(n): y = y + 1
        return y
    return f
def ind(n):
    return lambda: [x0 + i for i in range(n)]
for n in (1, 801):
    res[f"dep{n}_ms"] = round(tbest(dep(n))[0] * 1e3, 3); res[f"ind{n}_ms"] = round(tbest(ind(n))[0] * 1e3, 3)
res["us_per_dependent_dispatch"] = round((res["dep801_ms"] - res["dep1_ms"]) / 800 * 1e3, 2)
res["us_per_independent_dispatch"] = round((res["ind801_ms"] - res["ind1_ms"]) / 800 * 1e3, 2)
print("BW96", json.dumps(res), flush=True)
