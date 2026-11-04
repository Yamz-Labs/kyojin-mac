"""Op-class profile of one served prefill (in process): every module class of the first blocks is wrapped, eval-synchronized, time summed by class.
usage: pf_roof_prof.py MODEL T [--chunk 2048]   (the sync adds gaps: read it as shares, not absolute ms)"""
import argparse, collections, time
import numpy as np, mlx.core as mx
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("T", type=int); ap.add_argument("--chunk", type=int, default=2048); a = ap.parse_args()
from kyojin_mlx.serve.backends import load_backend
be, cfg = load_backend(a.model)
model = getattr(be, "model", None)
TT = collections.defaultdict(float); CN = collections.Counter(); depth = [0]
def wrap(cls):
    o = cls.__call__
    def w(self, *x, **k):
        if depth[0]: return o(self, *x, **k)           # only the outermost wrapped module is timed (no double count)
        depth[0] += 1; mx.eval([v for v in x if hasattr(v, "shape")]); t = time.perf_counter()
        r = o(self, *x, **k); mx.eval(r); TT[cls.__name__.replace('Fused','')] += time.perf_counter() - t; CN[cls.__name__.replace('Fused','')] += 1; depth[0] -= 1; return r
    cls.__call__ = w
wrapped = set()
for n, m in model.named_modules():
    c = type(m); nm = c.__name__
    if c in wrapped: continue
    if "Attention" in nm or "GatedDeltaNet" in nm or "SparseMoeBlock" in nm or (nm.endswith("MLP") and ".mlp" in n):
        wrapped.add(c); wrap(c)
ids = np.random.default_rng(1).integers(1000, 100000, a.T).tolist()
cache = be.make_cache(); mx.eval(be.prefill_chunk(cache, ids[:a.chunk]))   # warm
TT.clear(); CN.clear(); cache = be.make_cache(); t0 = time.perf_counter()
for s in range(0, a.T, a.chunk): be.prefill_chunk(cache, ids[s:s + a.chunk])
tot = time.perf_counter() - t0
print(f"total {tot:.2f} s {a.T/tot:.0f} tok/s")
for k, v in sorted(TT.items(), key=lambda kv: -kv[1]): print(f"{k:34s} {v:7.2f} s {100*v/tot:5.1f} % calls {CN[k]}")
print(f"unwrapped (norms, elementwise, host gaps) {tot-sum(TT.values()):.2f} s")
