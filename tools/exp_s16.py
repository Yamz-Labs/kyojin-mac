import sys, numpy as np, mlx.core as mx, time
from kyojin_mlx import attn_prefill as AP
z = np.load(sys.argv[1], allow_pickle=True)
q = mx.array(z["q"]).astype(mx.float16); w = mx.array(z["window"]).astype(mx.float16); p = mx.array(z["pool"]).astype(mx.float16); idx = mx.array(z["idxs"]); sink = mx.array(z["sink"]); sc = float(z["scale"])
print("q", q.shape, "pool", p.shape, "idxs", idx.shape)
def run(s16):
    AP.S16 = s16; AP._kernels.cache_clear()
    return AP.sparse_attention(q, w, p, idx, sc, sink, 128)
a = run(False); b = run(True); mx.eval(a, b)
a32, b32 = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
print("rel diff S16 vs S32 (fro):", np.linalg.norm(a32 - b32) / np.linalg.norm(a32), "max abs", np.abs(a32 - b32).max(), "max |o|", np.abs(a32).max())
# fp64 reference on a few queries
qq = np.array(q.astype(mx.float32))[0]; W = np.array(w.astype(mx.float32))[0]; P_ = np.array(p.astype(mx.float32))[0]; I = np.array(idx)[0]; sk = np.array(sink)
T = qq.shape[1]; hist = W.shape[0] - T
ref = np.zeros((qq.shape[0], 16, qq.shape[2]))
for j, t in enumerate(np.linspace(T // 2, T - 1, 16).astype(int)):
    rows = [r for r in range(t + hist - 127, t + hist + 1) if r >= 0]
    keys = np.concatenate([W[rows], P_[[i for i in I[t] if i >= 0]]], 0).astype(np.float64)
    s = sc * (qq[:, t].astype(np.float64) @ keys.T); m = np.maximum(s.max(-1), sk); e = np.exp(s - m[:, None])
    ref[:, j] = (e / (e.sum(-1) + np.exp(sk - m))[:, None]) @ keys
sel = np.linspace(T // 2, T - 1, 16).astype(int)
for nm, o in (("S32", a32), ("S16", b32)):
    r = o[0][:, sel]; print(nm, "rel err vs fp64 ref:", np.linalg.norm(r - ref) / np.linalg.norm(ref))
print("max |score|*scale sample:", float(np.abs(sc * np.einsum('htd,td->ht', qq[:, :64], W[hist:hist+64])).max()))
# score magnitude over all queries / heads (numpy, window keys + pool keys of every 8th query)
mx_s = 0.0; allv = []
for t in range(0, T, 8):
    rows = [r for r in range(t + hist - 127, t + hist + 1) if r >= 0]
    keys = np.concatenate([W[rows], P_[[i for i in I[t] if i >= 0]]], 0)
    s = sc * (qq[:, t] @ keys.T); allv.append(s.max(-1))
allv = np.concatenate(allv); print("row max scores: p50", np.percentile(allv, 50), "p99", np.percentile(allv, 99), "max", allv.max())
