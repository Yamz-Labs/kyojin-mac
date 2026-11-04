import sys, time, numpy as np, mlx.core as mx
from kyojin_mlx import attn_prefill as AP
T = 4096; rng = np.random.default_rng(0); H, D, W, topk = 64, 512, 128, 512
def rnd(*shape, s=0.5): return mx.array((rng.standard_normal(shape) * s).astype(np.float16))
q = rnd(1, H, T, D); win = rnd(1, T, D); pool = rnd(1, T // 2, D); sink = mx.array(rng.standard_normal(H).astype(np.float32))
idx = np.full((1, T, topk), -1, np.int32)
for t in range(T):
    n = (t + 1) // 2; k = min(topk, n)
    if k: idx[0, t, :k] = rng.choice(n, k, replace=False)
idxs = mx.array(idx); ids = AP.key_ids(T, T, idxs, W); NK = ids.shape[-1]; sc = D ** -0.5
q0 = mx.contiguous(q[0]); tab = mx.concatenate([win[0], pool[0]], axis=0); sk = sink; mx.eval(q0, tab, ids)
k1, k2, k3 = AP._kernels(H, D, NK, sc); k1b, k3b = AP._kernels_sm(H, D, NK, sc)
a, tc = 2048, 512; ic = ids[a:a + tc]; mx.eval(ic)
p1 = mx.array([a, T], dtype=mx.uint32); p3 = mx.array([tc], dtype=mx.uint32)
f1 = lambda: k1(inputs=[q0, tab, ic, p1], grid=(-(-NK // 128) * 128, tc, 1), threadgroup=(128, 1, 1), output_shapes=[(tc * H, NK)], output_dtypes=[mx.float32])[0]
S = f1(); mx.eval(S)
f2 = lambda: k2(inputs=[S, ic, sk], grid=(tc * H * 32, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(tc * H, NK)], output_dtypes=[mx.float16])[0]
P = f2(); mx.eval(P)
f3 = lambda: k3(inputs=[P, tab, ic, p3], grid=(D // 128 * 128, tc, 1), threadgroup=(128, 1, 1), output_shapes=[(H, tc, D)], output_dtypes=[mx.float16])[0]
g1 = lambda: k1b(inputs=[q0, tab, ic, p1], grid=(-(-NK // 128) * 128, tc, 1), threadgroup=(128, 1, 1), output_shapes=[(tc * H, NK), (tc * H * (NK // 32) * 2,)], output_dtypes=[mx.float16, mx.float32])
Sb, ST = g1(); mx.eval(Sb, ST)
g3 = lambda: k3b(inputs=[Sb, ST, tab, ic, sk, p3], grid=(D // 128 * 128, tc, 1), threadgroup=(128, 1, 1), output_shapes=[(H, tc, D)], output_dtypes=[mx.float16])[0]
def tm(f, n=15):
    mx.eval(f()); b = 1e9
    for _ in range(n):
        time.sleep(0.6); t = time.perf_counter(); mx.eval(f()); b = min(b, time.perf_counter() - t)
    return b * 1e3
for nm, f in (("K1 scores (fp32 S)", f1), ("K2 softmax", f2), ("K3 values", f3), ("K1B scores+stats", g1), ("K3B values+P", g3)):
    print(f"{nm:22s} {tm(f):7.2f} ms per 512-query group  (x8 = {8*tm(f, 7):6.1f})")
