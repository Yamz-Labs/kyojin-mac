"""Sparse pooled attention (ratio-2 layer, H 64, D 512, top-k 512 of T/2 pool, window 128): stock mlx-vlm function vs attn_prefill, interleaved, min of rounds.
usage: bench_attn_ab.py T,T,.. [rounds]   (stock is O(T^2): T <= 1024 keeps it under 4 GiB)"""
import sys, time, numpy as np, mlx.core as mx
from mlx_vlm.models.deepseek_v4.language import _sparse_pooled_attention
from kyojin_mlx import attn_prefill as AP
H, D, W, topk = 64, 512, 128, 512
rng = np.random.default_rng(0); rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 8
for T in map(int, sys.argv[1].split(",")):
    Np = T // 2
    idx = np.full((1, T, topk), -1, np.int32)
    for t in range(T):
        n = (t + 1) // 2; k = min(topk, n)
        if k: idx[0, t, :k] = np.sort(rng.choice(n, k, replace=False))
    q = mx.array((rng.standard_normal((1, H, T, D)) * 0.5).astype(np.float16)); win = mx.array((rng.standard_normal((1, T, D)) * 0.5).astype(np.float16))
    pool = mx.array((rng.standard_normal((1, Np, D)) * 0.5).astype(np.float16)); idxs = mx.array(idx); sink = mx.array(rng.standard_normal(H).astype(np.float32)); scale = D ** -0.5
    pos = mx.arange(T); mask = ((pos[None, :] <= pos[:, None]) & (pos[None, :] > pos[:, None] - W))[None, None]
    arms = {"stock": lambda: _sparse_pooled_attention(q.astype(mx.float32), win, pool, idxs, mask, (idxs != -1)[:, None], scale, sink).astype(mx.float16),
            "kernel": lambda: AP.sparse_attention(q, win, pool, idxs, scale, sink, W)}
    names = list(arms); res = {n: [] for n in names}; outs = {}
    for n in names: outs[n] = arms[n](); mx.eval(outs[n])
    for r in range(rounds):
        for n in (names[r % 2:] + names[:r % 2]):
            t0 = time.perf_counter(); mx.eval(arms[n]()); res[n].append(time.perf_counter() - t0)
    a, b = outs["kernel"].astype(mx.float32), outs["stock"].astype(mx.float32)
    print(f"T={T:5d}: stock min {min(res['stock'])*1e3:7.1f} ms ({T/min(res['stock']):6.0f} tok/s)  kernel min {min(res['kernel'])*1e3:6.1f} ms ({T/min(res['kernel']):7.0f} tok/s)  x{min(res['stock'])/min(res['kernel']):.1f}  rel {(mx.linalg.norm(a-b)/mx.linalg.norm(b)).item():.1e}  peak {mx.get_peak_memory()/2**30:.2f} GiB", flush=True)
    del outs, arms, q, win, pool; mx.clear_cache()
