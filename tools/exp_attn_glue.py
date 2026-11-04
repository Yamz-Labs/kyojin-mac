"""Glue around the attention kernels at T: q transpose, key_ids, table concat, output concat; whole sparse_attention vs sum of K stages. usage: exp_attn_glue.py T"""
import sys, time, numpy as np, mlx.core as mx
from kyojin_mlx import attn_prefill as AP
T = int(sys.argv[1]); H, D, W, topk = 64, 512, 128, 512
rng = np.random.default_rng(0)
def rnd(*s, sc=0.5): return mx.array((rng.standard_normal(s) * sc).astype(np.float32)).astype(mx.float16)
q = rnd(1, H, T, D); win = rnd(1, T, D); pool = rnd(1, T // 2, D); sink = mx.array(rng.standard_normal(H).astype(np.float32))
idx = np.full((1, T, topk), -1, np.int32)
for t in range(T):
    n = (t + 1) // 2; k = min(topk, n)
    if k: idx[0, t, :k] = rng.choice(n, k, replace=False)
idxs = mx.array(idx); mx.eval(q, win, pool, idxs)
outs8 = [rnd(H, T // 8, D) for _ in range(8)]; mx.eval(outs8)
arms = {"qt": lambda: mx.contiguous(q[0].transpose(1, 0, 2)), "key_ids": lambda: AP.key_ids(T, T, idxs, W), "tab": lambda: mx.concatenate([win[0], pool[0]], axis=0),
        "outcat": lambda: mx.concatenate(outs8, axis=1), "full": lambda: AP.sparse_attention(q, win, pool, idxs, D ** -0.5, sink, W)}
best = {k: 1e9 for k in arms}
for k, f in arms.items(): mx.eval(f())
for r in range(6):
    time.sleep(1.5)
    for k in arms:
        for _ in range(3):
            t0 = time.perf_counter(); mx.eval(arms[k]()); best[k] = min(best[k], time.perf_counter() - t0)
print(" ".join(f"{k} {v*1e3:.2f}" for k, v in best.items()))
