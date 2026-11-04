"""Measured rooflines of this Mac: memory read / copy bandwidth and ALU rates (fp32 fma, fp16 fma, int32 mul / add), plus the EXL3 GEMV weight rate.
usage: roofline.py [--gib 2] [--json F]
Memory: sum over a large fp16 array (read only), `a * 1` (read + write), fp16 GEMV through mx.matmul (the best a stock kernel does).
ALU: one Metal kernel per op class, 8 independent chains per thread, every thread of the GPU busy: ops per second."""
import argparse, json, statistics, time

import mlx.core as mx
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--gib", type=float, default=2.0); ap.add_argument("--json")
a = ap.parse_args()
res = {}


def bench(fn, reps=9, warm=2):
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t)
    return statistics.median(ts), min(ts)


n = int(a.gib * 2**30 / 2)
big = mx.random.uniform(shape=(n,)).astype(mx.float16); mx.eval(big)
nb = big.nbytes
t, tmin = bench(lambda: mx.sum(big.reshape(-1, 4096), axis=-1))
res["read_sum_GBps"] = nb / tmin / 1e9
t, tmin = bench(lambda: big * mx.array(1.0, dtype=mx.float16))
res["copy_GBps_rw"] = 2 * nb / tmin / 1e9
N, K = 32768, 8192
W = mx.random.uniform(shape=(N, K)).astype(mx.float16); x = mx.random.uniform(shape=(K,)).astype(mx.float16); mx.eval(W, x)
t, tmin = bench(lambda: W @ x)
res["gemv_fp16_GBps"] = W.nbytes / tmin / 1e9
t, tmin = bench(lambda: mx.sum(W.reshape(N, K).astype(mx.float32)[:, :1]))   # strided read touches full lines: sanity only
del big, W

ITERS = 2048
cases = {
    "fp32_fma": ("float", "a = fma(a, m, c);", " + ", 2),
    "fp16_fma": ("half", "a = fma(a, m, c);", " + ", 2),
    "int32_mul": ("uint", "a = a * m;", " + ", 1),
    "int32_add": ("uint", "a = a + c;", " + ", 1),
    "int32_and_shift_add": ("uint", "a = ((a & 0x00ff00ffu) + ((a >> 8) & 0x00ff00ffu));", " + ", 4),
}
threads = 1 << 20
for name, (ty, op, join, flops) in cases.items():
    stmts = []
    for r in range(8):
        s = op.replace("a = ", f"a{r} = ").replace("fma(a,", f"fma(a{r},").replace("a * m", f"a{r} * m").replace("a + c", f"a{r} + c")
        s = s.replace("((a &", f"((a{r} &").replace("((a >>", f"((a{r} >>").replace("(a >>", f"(a{r} >>")
        stmts.append(s)
    src = f"""
    uint tid = thread_position_in_grid.x;
    {ty} a0 = {ty}(tid), a1 = {ty}(tid + 1), a2 = {ty}(tid + 2), a3 = {ty}(tid + 3), a4 = {ty}(tid + 4), a5 = {ty}(tid + 5), a6 = {ty}(tid + 6), a7 = {ty}(tid + 7);
    {ty} m = {ty}(par[0]); {ty} c = {ty}(par[1]);
    for (uint i = 0; i < {ITERS}; ++i) {{
        {' '.join(stmts)}
    }}
    out[tid] = (float)(a0 {join} a1 {join} a2 {join} a3 {join} a4 {join} a5 {join} a6 {join} a7);
    """
    kern = mx.fast.metal_kernel(name=f"roof_{name}", input_names=["par"], output_names=["out"], source=src)
    par = mx.array([3 if ty != "float" and ty != "half" else 1, 1], dtype=mx.uint32)
    if ty in ("float", "half"):
        par = mx.array([1, 1], dtype=mx.uint32)
    f = lambda: kern(inputs=[par], grid=(threads, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(threads,)], output_dtypes=[mx.float32])[0]
    t, tmin = bench(f, reps=7)
    ops = threads * ITERS * 8
    res[name + "_Gops"] = ops / tmin / 1e9
    res[name + "_Gflops" if flops > 1 else name + "_Gops_"] = ops * flops / tmin / 1e9

# streaming read: every thread sums uint4 loads of a big buffer (the best a custom kernel gets)
rd = mx.fast.metal_kernel(name="roof_read", input_names=["buf", "par"], output_names=["out"], source="""
    uint tid = thread_position_in_grid.x; uint nth = threads_per_grid.x;
    uint n4 = par[0];
    uint4 acc = uint4(0);
    for (uint i = tid; i < n4; i += nth) { acc ^= ((const device uint4*)buf)[i]; }
    out[tid] = acc.x ^ acc.y ^ acc.z ^ acc.w;
""")
words = mx.random.randint(0, 2**31 - 1, shape=(int(1.0 * 2**30) // 4,)).astype(mx.uint32); mx.eval(words)
n4 = mx.array([words.size // 4], dtype=mx.uint32)
for nth in (1 << 17, 1 << 19):
    f = lambda: rd(inputs=[words, n4], grid=(nth, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(nth,)], output_dtypes=[mx.uint32])[0]
    t, tmin = bench(f, reps=9)
    res[f"read_kernel_{nth >> 10}k_threads_GBps"] = words.nbytes / tmin / 1e9
del words

# EXL3 GEMV weight rate, 24 independent launches per eval (launch floor amortised), synthetic mul1 trellis, real DS4.1 shapes
from kyojin_mlx import kernels as KN
rng = np.random.default_rng(0)
shapes = {"wq_b 1280x32768": (1280, 32768), "wo_b 8192x5120": (8192, 5120), "gate 5120x2304": (5120, 2304)}
for Kb in (2, 3, 4, 5, 6):
    for nm, (kin, nout) in shapes.items():
        kt, nt = kin // 16, nout // 16
        tr = rng.integers(-32768, 32767, size=(kt, nt, 16 * Kb), dtype=np.int16)
        g = KN.Mul1Gemv(tr, np.ones(kin, np.float16), np.ones(nout, np.float16), rows=1, out_dtype=mx.float16, cb="mul1")
        xh = mx.array(rng.standard_normal((1, kin)).astype(np.float32)); mx.eval(xh)
        R = 24
        def f():
            return mx.concatenate([g.partial(xh).reshape(-1)[:8] + i for i in range(R)])
        for _ in range(3): mx.eval(f())
        ts = []
        for _ in range(9):
            t = time.perf_counter(); mx.eval(f()); ts.append((time.perf_counter() - t) / R)
        tm = min(ts)
        res[f"gemv_k{Kb}_{nm}_Gw_s"] = kin * nout / tm / 1e9
        res[f"gemv_k{Kb}_{nm}_GBps"] = tr.nbytes / tm / 1e9
        res[f"gemv_k{Kb}_{nm}_tgk"] = g.tgk
for k, v in res.items():
    print(f"{k:28s} {v:10.1f}")
if a.json:
    json.dump(res, open(a.json, "w"), indent=1)
