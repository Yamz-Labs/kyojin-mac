"""Does alternating the launch groups of the sparse attention over 2 GPU streams overlap K2 (memory) with K1/K3 (matmul)? min over interleaved reps."""
import sys, time, numpy as np, mlx.core as mx
from kyojin_mlx import attn_prefill as AP
T = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
rng = np.random.default_rng(0); H, D, W, topk = 64, 512, 128, 512
def rnd(*shape, s=0.5): return mx.array((rng.standard_normal(shape) * s).astype(np.float16))
q = rnd(1, H, T, D); win = rnd(1, T, D); Np = T // 2; pool = rnd(1, Np, D); sink = mx.array(rng.standard_normal(H).astype(np.float32))
idx = np.full((1, T, topk), -1, np.int32)
for t in range(T):
    n = (t + 1) // 2; k = min(topk, n)
    if k: idx[0, t, :k] = rng.choice(n, k, replace=False)
idxs = mx.array(idx); mx.eval(q, win, pool, idxs)
sc = D ** -0.5
streams = [mx.new_stream(mx.gpu) for _ in range(2)]
_, H, T, D = q.shape
Tw = win.shape[1]; ids = AP.key_ids(T, Tw, idxs, W); NK = ids.shape[-1]
k1, k2, k3 = AP._kernels(H, D, NK, sc)
q0 = mx.contiguous(q[0]); tab = mx.concatenate([win[0], pool[0]], axis=0); sk = sink.astype(mx.float32); mx.eval(ids, q0, tab)
def run(nstreams, cs=512):
    outs = []
    for i, a in enumerate(range(0, T, cs)):
        tc = min(cs, T - a)
        with mx.stream(streams[i % nstreams] if nstreams > 1 else mx.default_stream(mx.gpu)):
            ic = ids[a:a + tc]
            S = k1(inputs=[q0, tab, ic, mx.array([a, T], dtype=mx.uint32)], grid=(-(-NK // 128) * 128, tc, 1), threadgroup=(128, 1, 1), output_shapes=[(tc * H, NK)], output_dtypes=[mx.float32])[0]
            P = k2(inputs=[S, ic, sk], grid=(tc * H * 32, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(tc * H, NK)], output_dtypes=[mx.float16])[0]
            outs.append(k3(inputs=[P, tab, ic, mx.array([tc], dtype=mx.uint32)], grid=(D // 128 * 128, tc, 1), threadgroup=(128, 1, 1), output_shapes=[(H, tc, D)], output_dtypes=[mx.float16])[0])
    return outs
ref = mx.concatenate(run(1), axis=1); mx.eval(ref)
res = {1: [], 2: []}
for r in range(14):
    for n in ((1, 2) if r % 2 == 0 else (2, 1)):
        time.sleep(1.5); t0 = time.perf_counter(); o = run(n); mx.eval(o); res[n].append((time.perf_counter() - t0) * 1e3)
        if r == 0: assert bool(mx.array_equal(mx.concatenate(o, axis=1), ref).item())
for n in (1, 2): print(f"{n} stream(s): min {min(res[n]):.1f} p25 {np.percentile(res[n],25):.1f} ms")
