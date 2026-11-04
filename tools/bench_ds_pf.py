"""Time of the DS4.1 prefill sparse attention alone: python tools/bench_ds_pf.py stock|new T [ratio=2] [reps=3]"""
import sys, time, importlib, numpy as np, mlx.core as mx
from kyojin_mlx import ds_pf_attn as A
V4 = importlib.import_module("mlx_vlm.models.deepseek_v4.language")
mode, T = sys.argv[1], int(sys.argv[2]); ratio = int(sys.argv[3]) if len(sys.argv) > 3 else 2; reps = int(sys.argv[4]) if len(sys.argv) > 4 else 3
Tc = T // ratio; K = min(512, Tc); H, D, win = 64, 512, 128
rng = np.random.default_rng(1)
f = lambda *s: mx.array(rng.standard_normal(s).astype(np.float16))
q = f(1, H, T, D); pool = f(1, Tc, D); window = f(1, T, D); sink = mx.array(rng.standard_normal(H).astype(np.float32))
idx = np.full((1, T, K), -1, np.int32)
for t in range(T):
    n = min((t + 1) // ratio, Tc)
    c = np.sort(rng.choice(n, size=min(K, n), replace=False)) if n else []
    idx[0, t, :len(c)] = c
idxs = mx.array(idx)
pos = mx.arange(T)
wm = ((pos[None] <= pos[:, None]) & (pos[None] > pos[:, None] - win))[None, None]
mx.eval(q, pool, window, idxs, wm)
scale = D ** -0.5
fns = {"stock": lambda: V4._sparse_pooled_attention(q.astype(mx.float32), window[:, None], pool, idxs, wm, (idxs != -1)[:, None], scale, sink).astype(mx.float16),
       "new": lambda: A.sparse_pf(q, window, pool, idxs, wm, scale, sink, win)}
modes = ["stock", "new"] if mode == "both" else [mode]
ts = {m: [] for m in modes}
for i in range(reps + 1):
    for m in modes:
        t0 = time.perf_counter(); mx.eval(fns[m]()); ts[m].append(time.perf_counter() - t0)
for m in modes:
    print(f"{m} T={T} ratio={ratio} K={K}: {1e3 * min(ts[m][1:]):.1f} ms")
print(f"peak {mx.get_peak_memory() / 2**30:.2f} GiB")
