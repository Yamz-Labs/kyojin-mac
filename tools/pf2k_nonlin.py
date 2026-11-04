"""Time of everything except the dense linears: Exl3Linear.__call__ replaced by a cheap constant-shape op. usage: pf2k_nonlin.py MODEL_DIR rows"""
import sys, time, mlx.core as mx
from kyojin_mlx import exl3_linear as EL
from kyojin_mlx.serve.backends import load_backend
be, _ = load_backend(sys.argv[1]); R = int(sys.argv[2]); mx.set_cache_limit(256 * 2 ** 20)
ids = [(i * 7919) % 5000 + 100 for i in range(R)]
orig = EL.Exl3Linear.__call__
def go():
    c = be.make_cache(); mx.synchronize(); t0 = time.perf_counter(); be.prefill_chunk(c, ids); return time.perf_counter() - t0
_z = {}
def fake(self, x):
    if self.out_features > 100000:
        return orig(self, x)
    k = (*x.shape[:-1], self.out_features)
    z = _z.get(k)
    if z is None:
        z = _z[k] = (mx.random.normal(k) * 0.1).astype(mx.float16); mx.eval(z)
    return z
res = {"real": [], "linears-free": []}
for mode in res: 
    EL.Exl3Linear.__call__ = orig if mode == "real" else fake; go()
for _ in range(4):
    for mode in res:
        EL.Exl3Linear.__call__ = orig if mode == "real" else fake; res[mode].append(go())
for n, v in res.items(): print(f"{n:14s} {min(v)*1e3:7.1f} ms {R/min(v):7.0f} tok/s")
