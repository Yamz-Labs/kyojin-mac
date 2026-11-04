"""Gate: FP8 (E4M3) read as stored (the shipped foreign_metal.gemv8_source kernel) against the same matrix decoded to fp16, per row count. Synthetic 8 x 8192^2, bounded,
GPU only, no model. Arms: fp16 matmul (resident fp16), fp8 GEMV, fp8 dequant kernel + matmul (decoded per call, no fp16 copy resident). Run under /tmp/lane_ab.lock."""
import os, sys, time, statistics as st
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mlx.core as mx
from kyojin_mlx import foreign_linear as FL, foreign_metal as FM

KIN, N, COUNT, REPS = 8192, 8192, 8, 7
ROWS = tuple(int(a) for a in sys.argv[1:]) or (1, 2, 4, 8, 16)
TAB = np.array(FM.E4M3_TABLE, np.float32)
rng = np.random.default_rng(1)
order = np.argsort(TAB[:127])
idx = np.clip(np.searchsorted(np.sort(TAB[:127]), np.abs(rng.normal(0, 0.05, (N, KIN))).astype(np.float32)), 0, 126)
codes = (order[idx] | (rng.integers(0, 2, (N, KIN)) << 7)).astype(np.uint8)
W8s, W16s = [], []
for m in range(COUNT):
    c = np.ascontiguousarray(np.roll(codes, m * 37, axis=0))
    W8s.append(mx.array(c.view(np.uint32).reshape(N, KIN // 4)))
    W16s.append(mx.array(TAB[c].astype(np.float16)))
mx.eval(W8s, W16s)
gs = mx.ones((N,), mx.float32)
print("resident GiB fp8", round(COUNT * N * KIN / 2 ** 30, 2), "fp16", round(COUNT * N * KIN * 2 / 2 ** 30, 2), flush=True)


def gemv(x, q, R):
    return FL._gemv8_kern("fp8", 2)(inputs=[x, q, gs], template=[("R", R), ("KIN", KIN), ("N", N), ("NBR", 2)], grid=(N // 16 * 128, 1, 1), threadgroup=(128, 1, 1),
                                    output_shapes=[(R, N)], output_dtypes=[mx.float16])[0]


def deq(q):
    return FL._deq8_kern("fp8")(inputs=[q, gs], template=[("KIN", KIN), ("N", N)], grid=(N * KIN // 4, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(N, KIN)],
                                output_dtypes=[mx.float16])[0]


for R in ROWS:
    x = mx.array(rng.normal(0, 1, (R, KIN)).astype(np.float16))
    ref = x.astype(mx.float32) @ mx.array(TAB[codes]).T
    mx.eval(x, ref)
    q0 = mx.array(np.ascontiguousarray(codes).view(np.uint32).reshape(N, KIN // 4))
    for name, e in (("gemv", gemv(x, q0, R)), ("deq+mm", x @ deq(q0).T)):
        mx.eval(e)
        print(f"R{R} {name} rel err {float(mx.max(mx.abs(e.astype(mx.float32) - ref))) / float(mx.max(mx.abs(ref))):.5f}", flush=True)
    arms = {"fp16": lambda m: (x @ W16s[m].T).astype(mx.float16), "fp8_gemv": lambda m: gemv(x, W8s[m], R), "fp8_deq_mm": lambda m: (x @ deq(W8s[m]).T).astype(mx.float16)}
    res = {a: [] for a in arms}
    for rep in range(REPS):
        for a, f in arms.items():
            t = time.perf_counter()
            mx.eval([f(m) for m in range(COUNT)])
            res[a].append(time.perf_counter() - t)
    w = COUNT * N * KIN * R
    print(f"R{R} " + " | ".join(f"{a} {st.median(v[1:]) * 1e3:.1f} ms {w / st.median(v[1:]) / 1e9:.0f} Gw/s" for a, v in res.items()), flush=True)
