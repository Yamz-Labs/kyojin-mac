"""Per-kernel burst times of the gathered-key attention core (K1 scores, K2 softmax, K3 values) at T, one 512-query chunk each. usage: exp_attn_split.py T"""
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
idxs = mx.array(idx)
ids = AP.key_ids(T, T, idxs, W); NK = ids.shape[-1]
k1, k2, k3 = AP._kernels(H, D, NK, D ** -0.5)
qt = mx.contiguous(q[0].transpose(1, 0, 2)); tab = mx.concatenate([win[0], pool[0]], axis=0); sk = sink
tc = AP.CHUNK; n_ch = T // tc
ic = ids[:tc]
def f1(): return k1(inputs=[qt[:tc], tab, ic], grid=(-(-NK // 128) * 128, tc, 1), threadgroup=(128, 1, 1), output_shapes=[(tc * H, NK)], output_dtypes=[mx.float32])[0]
S = f1(); mx.eval(S)
def f2(): return k2(inputs=[S, ic, sk], grid=(tc * H * 32, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(tc * H, NK)], output_dtypes=[mx.float16])[0]
P = f2(); mx.eval(P)
def f3(): return k3(inputs=[P, tab, ic, mx.array([tc], dtype=mx.uint32)], grid=(D // 128 * 128, tc, 1), threadgroup=(128, 1, 1), output_shapes=[(H, tc, D)], output_dtypes=[mx.float16])[0]
def full(): return AP.sparse_attention(q, win, pool, idxs, D ** -0.5, sink, W)
best = {}
for name, f in (("K1", f1), ("K2", f2), ("K3", f3), ("full", full)):
    mx.eval(f()); b = 1e9
    for _ in range(6):
        time.sleep(1.5)
        for _ in range(3):
            t0 = time.perf_counter(); mx.eval(f()); b = min(b, time.perf_counter() - t0)
    best[name] = b * 1e3
print(f"T={T} NK={NK} chunk={tc}: " + " ".join(f"{k} {v:.2f}ms" for k, v in best.items()) + f" | x{n_ch}: K1 {best['K1']*n_ch:.1f} K2 {best['K2']*n_ch:.1f} K3 {best['K3']*n_ch:.1f} sum {sum(best[k] for k in ('K1','K2','K3'))*n_ch:.1f}")
fl = 2 * 2 * H * NK * D * T
print("full TF %.1f" % (fl / best["full"] / 1e9))
