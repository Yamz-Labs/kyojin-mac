"""Stage times of attn_prefill at synthetic T (random valid top-k like the DS4.1 indexer). usage: bench_attn_pf.py T [ratio=2] [H=64]"""
import sys, time, numpy as np, mlx.core as mx
from kyojin_mlx import attn_prefill as AP
T = int(sys.argv[1]); ratio = int(sys.argv[2]) if len(sys.argv) > 2 else 2; H = int(sys.argv[3]) if len(sys.argv) > 3 else 64; D = 512; W = 128; topk = int(sys.argv[4]) if len(sys.argv) > 4 else 512
rng = np.random.default_rng(0)
Np = int(sys.argv[5]) if len(sys.argv) > 5 else T // ratio
idx = np.full((1, T, topk), -1, np.int32)
for t in range(T):
    n = Np if len(sys.argv) > 5 else (t + 1) // ratio
    k = min(topk, n)
    if k: idx[0, t, :k] = rng.choice(n, k, replace=False)
q = mx.array((rng.standard_normal((1, H, T, D)) * 0.5).astype(np.float16)); win = mx.array((rng.standard_normal((1, T, D)) * 0.5).astype(np.float16))
pool = mx.array((rng.standard_normal((1, Np, D)) * 0.5).astype(np.float16)); idxs = mx.array(idx); sink = mx.array(rng.standard_normal(H).astype(np.float32))
scale = D ** -0.5
def tm(f, n=5, chain=1):
    best = 1e9
    for _ in range(n):
        t0 = time.perf_counter(); o = f(); mx.eval(o); best = min(best, time.perf_counter() - t0)
    return best * 1e3
whole = tm(lambda: AP.sparse_attention(q, win, pool, idxs, scale, sink, W))
ids = AP.key_ids(T, T, idxs, W); NK = ids.shape[-1]; k1, k2, k3 = AP._kernels(H, D, NK, scale)
qt = mx.contiguous(q[0].transpose(1, 0, 2)); tab = mx.concatenate([win[0], pool[0]], 0); sk = sink; tc = min(AP.CHUNK, T)
ic = ids[:tc]; mx.eval(qt, tab, ic)
f1 = lambda: k1(inputs=[qt[:tc], tab, ic], grid=(-(-NK // 128) * 128, tc, 1), threadgroup=(128, 1, 1), output_shapes=[(tc * H, NK)], output_dtypes=[mx.float32])[0]
S = f1(); mx.eval(S)
f2 = lambda: k2(inputs=[S, ic, sk], grid=(tc * H * 32, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(tc * H, NK)], output_dtypes=[mx.float16])[0]
P = f2(); mx.eval(P)
f3 = lambda: k3(inputs=[P, tab, ic, mx.array([tc], dtype=mx.uint32)], grid=(D // 128 * 128, tc, 1), threadgroup=(128, 1, 1), output_shapes=[(H, tc, D)], output_dtypes=[mx.float16])[0]
nch = -(-T // tc)
t1, t2, t3 = tm(f1) * nch, tm(f2) * nch, tm(f3) * nch
fl = 2 * 2 * T * H * NK * D
print(f"T={T} ratio={ratio} H={H} NK={NK}: whole {whole:.1f} ms ({fl/whole/1e9:.1f} TF eff) | K1 {t1:.1f} K2 {t2:.1f} K3 {t3:.1f} (x{nch} chunks of {tc}) | peak {mx.get_peak_memory()/2**30:.2f} GiB")
