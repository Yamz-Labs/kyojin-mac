"""Prefill time of a whole (shrunk) model slice, N chunks of T tokens: python tools/fwd_time.py PACK T [reps]  (env KYOJIN_PF_NAX, KYOJIN_PF_MIN_ROWS)"""
import sys, time, numpy as np, mlx.core as mx
from kyojin_mlx import patch as P
pack, T = sys.argv[1], int(sys.argv[2]); reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
model, spec, cfg = P.build_model(pack); P.patch_model(model, pack, spec, cfg)
lm = model.language_model
ids = mx.array([np.random.default_rng(0).integers(100, 20000, T).tolist()])
ts = []
for i in range(reps + 1):
    cache = lm.make_cache(); t0 = time.perf_counter(); mx.eval(lm(ids, cache=cache).logits); ts.append(time.perf_counter() - t0)
print(f"{pack.split('/')[-1]} T={T} forward {1e3 * min(ts[1:]):.1f} ms")
