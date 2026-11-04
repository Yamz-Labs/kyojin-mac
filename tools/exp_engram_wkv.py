import sys, time, numpy as np, mlx.core as mx
from kyojin_mlx import patch as P
model, spec, cfg = P.build_model(sys.argv[1]); P.patch_model(model, sys.argv[1], spec, cfg)
lm = model.language_model
eng = [l.engram for l in lm.layers if getattr(l, "engram", None) is not None][0]
w = eng.wkv.weight; print("wkv weight", w.shape, w.dtype)
T = 2048
def tm(f, n=6):
    best = 1e9
    for _ in range(n):
        t0 = time.perf_counter(); o = f(); mx.eval(o); best = min(best, time.perf_counter() - t0)
    return best * 1e3
fl = 2 * T * w.shape[1] * w.shape[0]
for dt in (mx.float16, mx.bfloat16):
    x = mx.array(np.random.default_rng(0).standard_normal((1, T, w.shape[1])).astype(np.float32)).astype(dt); wd = w.astype(dt); mx.eval(x, wd)
    t = tm(lambda: x @ wd.T); print(f"{dt}: x @ w.T {t:.1f} ms = {fl/t/1e9:.1f} TF")
x = mx.array(np.random.default_rng(0).standard_normal((1, T, w.shape[1])).astype(np.float32)).astype(w.dtype)
print("module call:", tm(lambda: eng.wkv(x)), "ms")
