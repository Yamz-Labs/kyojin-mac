"""Gate: attn_prefill.sparse_attention vs the stock _sparse_pooled_attention on captured DS4.1 inputs. usage: test_attn_pf.py NPZ [reps]"""
import sys, time, numpy as np, mlx.core as mx
from mlx_vlm.models.deepseek_v4.language import _sparse_pooled_attention
from kyojin_mlx import attn_prefill as AP
d = np.load(sys.argv[1]); reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
q = mx.array(d["q"]).astype(mx.float16); win = mx.array(d["window"]).astype(mx.float16); pool = mx.array(d["pool"]).astype(mx.float16)
idx = mx.array(d["idxs"]); mask = mx.array(d["mask"]); sink = mx.array(d["sink"]); scale = float(d["scale"])
def tm(f, n):
    best = 1e9
    for _ in range(n):
        t0 = time.perf_counter(); o = f(); mx.eval(o); best = min(best, time.perf_counter() - t0)
    return best * 1e3, o
ts, o_s = tm(lambda: _sparse_pooled_attention(q.astype(mx.float32), win, pool, idx, mask, (idx != -1)[:, None], scale, sink).astype(mx.float16), 2)
tk, o_k = tm(lambda: AP.sparse_attention(q, win, pool, idx, scale, sink, 128), reps)
a, b = o_k.astype(mx.float32), o_s.astype(mx.float32)
print(f"T={q.shape[2]} stock {ts:.1f} ms  kernel {tk:.1f} ms  x{ts/tk:.1f}  rel L2 {(mx.linalg.norm(a-b)/mx.linalg.norm(b)).item():.2e}  max|d|/max|ref| {(mx.abs(a-b).max()/mx.abs(b).max()).item():.2e}  peak {mx.get_peak_memory()/2**30:.2f} GiB")
