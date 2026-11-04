"""A/B shipped (HEAD, tools/ap_old.py) vs working-tree kyojin_mlx.attn_prefill: bit equality + interleaved burst times (min, p25). usage: ab_attn.py T [window]"""
import sys, time, numpy as np, mlx.core as mx
from kyojin_mlx import attn_prefill as NEW
import ap_old as OLD
T = int(sys.argv[1]); win_only = len(sys.argv) > 2; H, D, W, topk = 64, 512, 128, 512
rng = np.random.default_rng(0)
def rnd(*s, sc=0.5): return mx.array((rng.standard_normal(s) * sc).astype(np.float32)).astype(mx.float16)
q = rnd(1, H, T, D); win = rnd(1, T, D); pool = rnd(1, max(T // 2, 1), D); sink = mx.array(rng.standard_normal(H).astype(np.float32))
idx = np.full((1, T, 0 if win_only else topk), -1, np.int32)
for t in range(T if not win_only else 0):
    n = (t + 1) // 2; k = min(topk, n)
    if k: idx[0, t, :k] = rng.choice(n, k, replace=False)
idxs = mx.array(idx); pool = win[:, :1] if win_only else pool; mx.eval(q, win, pool, idxs)
arms = {"old": lambda: OLD.sparse_attention(q, win, pool, idxs, D ** -0.5, sink, W), "new": lambda: NEW.sparse_attention(q, win, pool, idxs, D ** -0.5, sink, W)}
o, n = arms["old"](), arms["new"](); mx.eval(o, n)
print("bit-identical:", bool(mx.array_equal(o, n).item()), "maxdiff", float(mx.max(mx.abs(o.astype(mx.float32) - n.astype(mx.float32))).item()))
ts = {k: [] for k in arms}
for r in range(20):
    time.sleep(1.5)
    for k in (list(arms) if r % 2 == 0 else list(arms)[::-1]):
        for _ in range(3):
            t0 = time.perf_counter(); mx.eval(arms[k]()); ts[k].append(time.perf_counter() - t0)
for k, v in ts.items(): v.sort(); print(f"{k}: min {v[0]*1e3:.1f} p25 {v[len(v)//4]*1e3:.1f} ms")
print("ratio new/old min %.3f p25 %.3f" % (ts["new"][0] / ts["old"][0], ts["new"][len(ts["new"])//4] / ts["old"][len(ts["old"])//4]))
