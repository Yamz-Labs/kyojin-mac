"""Correctness of ds_pf_attn.sparse_pf against the stock _sparse_pooled_attention (random data). python tests/test_ds_pf.py [T] [Tc] [K] [start]"""
import sys, importlib, numpy as np, mlx.core as mx
from kyojin_mlx import ds_pf_attn as A
V4 = importlib.import_module("mlx_vlm.models.deepseek_v4.language")
T = int(sys.argv[1]) if len(sys.argv) > 1 else 300
ratio = 2; start = int(sys.argv[4]) if len(sys.argv) > 4 else 0
Tc = int(sys.argv[2]) if len(sys.argv) > 2 else (start + T) // ratio
K = min(int(sys.argv[3]) if len(sys.argv) > 3 else 512, Tc)
H, D, win = 64, 512, 128
rng = np.random.default_rng(1)
f = lambda *s, sc=1.0: mx.array((rng.standard_normal(s) * sc).astype(np.float16))
q = f(1, H, T, D); pool = f(1, Tc, D); kv = f(1, T, D)
base = max(0, start - win + 1)
window = f(1, start - base + T, D)
sink = mx.array(rng.standard_normal(H).astype(np.float32))
# top-k: causal valid positions, sorted, -1 padded
idx = np.full((1, T, K), -1, np.int32)
for t in range(T):
    n = min((start + t + 1) // ratio, Tc)
    c = np.sort(rng.choice(n, size=min(K, n), replace=False)) if n else []
    idx[0, t, :len(c)] = c
idxs = mx.array(idx)
pos = mx.arange(base, start + T); qq = mx.arange(start, start + T)
wm = ((pos[None] <= qq[:, None]) & (pos[None] > qq[:, None] - win))[None, None]
scale = D ** -0.5
ref = V4._sparse_pooled_attention(q.astype(mx.float32), window[:, None], pool, idxs, wm, (idxs != -1)[:, None], scale, sink)                  # fp32 truth (not rounded)
stock = V4._sparse_pooled_attention(q, window[:, None], pool, idxs, wm, (idxs != -1)[:, None], scale, sink)   # the stock fp16 path
out = A.sparse_pf(q, window, pool, idxs, wm, scale, sink, win)
from kyojin_mlx import attn_prefill as N
nax = N.sparse_attention(q, window, pool, idxs, scale, sink, win) if N.supported(q, window, pool) else None
mx.eval(ref, stock, out, *([nax] if nax is not None else []))
r = np.array(ref.astype(mx.float32))
for tag, x in (("stock16", stock), ("ds_pf", out), ("attn_prefill", nax)):
    if x is None:
        print(f"{tag}: not supported"); continue
    o = np.array(x.astype(mx.float32))
    print(f"{tag} T={T} Tc={Tc} K={K} start={start} maxabs {np.abs(r - o).max():.4g} rel-rms {np.sqrt(((r - o) ** 2).mean() / (r ** 2).mean()):.3g} nan {np.isnan(o).sum()}")
