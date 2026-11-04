"""Burst timing of the sparse attention core, flag arms interleaved (min / p25): usage bench_core.py T [reps] ATTR=a,b  (module attr of attn_prefill toggled)"""
import sys, time, numpy as np, mlx.core as mx
from kyojin_mlx import attn_prefill as AP
T = int(sys.argv[1]); reps = int(sys.argv[2]); attr = sys.argv[3]
rng = np.random.default_rng(0); H, D, W, topk = 64, 512, 128, 512
def rnd(*shape, s=0.5): return mx.array((rng.standard_normal(shape) * s).astype(np.float16))
q = rnd(1, H, T, D); win = rnd(1, T, D); Np = T // 2; pool = rnd(1, Np, D); sink = mx.array(rng.standard_normal(H).astype(np.float32))
idx = np.full((1, T, topk), -1, np.int32)
for t in range(T):
    n = (t + 1) // 2; k = min(topk, n)
    if k: idx[0, t, :k] = rng.choice(n, k, replace=False)
idxs = mx.array(idx); mx.eval(q, win, pool, idxs)
def run(v):
    setattr(AP, attr, v); return AP.sparse_attention(q, win, pool, idxs, D ** -0.5, sink, W)
a = run(False); b = run(True); mx.eval(a, b)
print("rel diff arms:", float(mx.linalg.norm((a - b).astype(mx.float32)).item() / mx.linalg.norm(a.astype(mx.float32)).item()))
res = {False: [], True: []}
for r in range(reps):
    for v in ((False, True) if r % 2 == 0 else (True, False)):
        time.sleep(1.5); t0 = time.perf_counter(); mx.eval(run(v)); res[v].append((time.perf_counter() - t0) * 1e3)
for v in (False, True): print(f"{attr}={v}: min {min(res[v]):.1f} p25 {np.percentile(res[v],25):.1f} ms")
