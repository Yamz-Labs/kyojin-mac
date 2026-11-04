"""A/B interleaved: module __call__ (3D input) vs _dense (2D), burst state. usage: exp_dense_call.py PACK T"""
import sys, time, numpy as np, mlx.core as mx
from kyojin_mlx import patch as P
pack, T = sys.argv[1], int(sys.argv[2])
model, spec, cfg = P.build_model(pack); P.patch_model(model, pack, spec, cfg)
L = model.language_model.layers[0]; a = L.attn
mods = {"wq_b": a.wq_b, "wo_b": a.wo_b, "wq_a": a.wq_a, "wkv": a.wkv}
rng = np.random.default_rng(0)
def mk(m): return mx.array(rng.standard_normal((1, T, m.in_features)).astype(np.float16))
for name, m in mods.items():
    x = mk(m); mx.eval(x); x2 = x.reshape(T, -1)
    print(type(m).__name__, name, flush=True)
    arms = {"call": lambda: m(x), "dense": lambda: m._dense(x2)}
    best = {k: 1e9 for k in arms}
    for r in range(6):
        time.sleep(1.5)
        order = list(arms) if r % 2 == 0 else list(arms)[::-1]
        for k in order:
            for _ in range(2):
                t0 = time.perf_counter(); mx.eval(arms[k]()); best[k] = min(best[k], time.perf_counter() - t0)
    print(name, {k: round(v * 1e3, 2) for k, v in best.items()}, flush=True)
