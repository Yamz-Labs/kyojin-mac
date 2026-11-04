"""sparse_attention total vs CHUNK (queries per launch group). usage: exp_attn_chunk.py T chunks"""
import sys, time, numpy as np, mlx.core as mx
from kyojin_mlx import attn_prefill as AP
T = int(sys.argv[1]); chunks = [int(c) for c in sys.argv[2].split(",")]; H, D, W, topk = 64, 512, 128, 512
rng = np.random.default_rng(0)
def rnd(*s, sc=0.5): return mx.array((rng.standard_normal(s) * sc).astype(np.float32)).astype(mx.float16)
q = rnd(1, H, T, D); win = rnd(1, T, D); pool = rnd(1, T // 2, D); sink = mx.array(rng.standard_normal(H).astype(np.float32))
idx = np.full((1, T, topk), -1, np.int32)
for t in range(T):
    n = (t + 1) // 2; k = min(topk, n)
    if k: idx[0, t, :k] = rng.choice(n, k, replace=False)
idxs = mx.array(idx); mx.eval(q, win, pool, idxs)
def run(c):
    AP.CHUNK = c; mx.eval(AP.sparse_attention(q, win, pool, idxs, D ** -0.5, sink, W))
def go(c):
    AP.CHUNK = c; t0 = time.perf_counter(); mx.eval(AP.sparse_attention(q, win, pool, idxs, D ** -0.5, sink, W)); return time.perf_counter() - t0
for c in chunks: go(c)
best = {c: [] for c in chunks}
for r in range(8):
    time.sleep(1.5)
    for c in chunks[r % len(chunks):] + chunks[:r % len(chunks)]:
        for _ in range(2): best[c].append(go(c))
mx.reset_peak_memory(); go(chunks[-1])
for c in chunks: v = sorted(best[c]); print(f"chunk {c}: min {v[0]*1e3:.1f} p25 {v[len(v)//4]*1e3:.1f} ms")
print("peak GiB (last chunk)", mx.get_peak_memory() / 2**30)
