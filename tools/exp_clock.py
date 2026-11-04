"""Clock-state curve: time per 4096^3 fp16 matmul (137 GF) as a function of how long the GPU has been busy, after an idle gap. Same for the dense wq_b path.
usage: exp_clock.py PACK"""
import sys, time, numpy as np, mlx.core as mx
from kyojin_mlx import patch as P
A = mx.array(np.random.default_rng(0).standard_normal((4096, 4096)).astype(np.float16)); B = A.T * 1; mx.eval(A, B)
model, spec, cfg = P.build_model(sys.argv[1]); P.patch_model(model, sys.argv[1], spec, cfg)
m = model.language_model.layers[0].attn.wq_b
x = mx.array(np.random.default_rng(1).standard_normal((1, 4096, 1280)).astype(np.float16)); mx.eval(x, m(x))
def curve(fn, n=14, flops=0):
    out = []
    for rep in range(3):
        time.sleep(3.0); ts = []
        for _ in range(n):
            t0 = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t0)
        out.append(ts)
    return [min(o[i] for o in out) for i in range(n)]
mm = curve(lambda: A @ B); dn = curve(lambda: m(x))
print("matmul 4096^3 ms per op, call index 0..13:", [round(v * 1e3, 1) for v in mm], " TF:", [round(2 * 4096**3 / v / 1e12, 1) for v in mm])
print("wq_b T=4096 ms per call:", [round(v * 1e3, 1) for v in dn], " TF:", [round(2 * 4096 * 1280 * 32768 / v / 1e12, 1) for v in dn])
