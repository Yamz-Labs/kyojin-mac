"""interleaved A/B: old (impl 1, table) vs v2 configs. usage: bench_ab.py K tiles"""
import sys, time, numpy as np, mlx.core as mx
sys.path.insert(0, ".")
from kyojin_mlx import conv_viterbi as V
K = int(sys.argv[1]); n = int(sys.argv[2]); cfgs = [eval(a) for a in sys.argv[3:]] or [{"impl": 1}, {"impl": 2}]
x = np.random.default_rng(0).standard_normal((n, 256)).astype(np.float32)
base = dict(V.CFG)
res = {i: [] for i in range(len(cfgs))}
for i, c in enumerate(cfgs):
    V.CFG.update(base); V.CFG.update(c); V.quantize_tiles(x[:8], K)
for r in range(4):
    for i, c in enumerate(cfgs):
        V.CFG.update(base); V.CFG.update(c)
        t0 = time.perf_counter(); w, ix = V.quantize_tiles(x, K, chunk=c.get("chunk")); mx.eval(w, ix); res[i].append(n / (time.perf_counter() - t0))
for i, c in enumerate(cfgs):
    v = sorted(res[i], reverse=True); print(f"K={K} {c}: best {v[0]:.0f} p25(2nd) {v[1]:.0f} tiles/s", flush=True)
