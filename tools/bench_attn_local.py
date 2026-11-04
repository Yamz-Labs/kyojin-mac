"""Window-only attention (ratio-0 layers): stock sdpa with a T x T mask vs the gathered-key kernels with no pooled key. usage: bench_attn_local.py T,T,.."""
import sys, time, numpy as np, mlx.core as mx
from kyojin_mlx import attn_prefill as AP
H, D, W = 64, 512, 128
rng = np.random.default_rng(0)
for T in map(int, sys.argv[1].split(",")):
    q = mx.array((rng.standard_normal((1, H, T, D)) * 0.5).astype(np.float16)); kv = mx.array((rng.standard_normal((1, T, D)) * 0.5).astype(np.float16))
    sink = mx.array(rng.standard_normal(H).astype(np.float32)); scale = D ** -0.5
    pos = mx.arange(T); mask = ((pos[None, :] <= pos[:, None]) & (pos[None, :] > pos[:, None] - W))[None, None]
    pool = mx.zeros((1, 1, D), dtype=mx.float16); idx = mx.zeros((1, T, 0), dtype=mx.int32)
    f_s = lambda: mx.fast.scaled_dot_product_attention(q, kv[:, None], kv[:, None], scale=scale, mask=mask, sinks=sink.astype(mx.float16))
    f_k = lambda: AP.sparse_attention(q, kv, pool, idx, scale, sink, W)
    res = {}
    for name, f in (("stock", f_s), ("kernel", f_k)):
        o = f(); mx.eval(o); best = 1e9
        for _ in range(5):
            t0 = time.perf_counter(); o = f(); mx.eval(o); best = min(best, time.perf_counter() - t0)
        res[name] = (best * 1e3, o)
    a, b = res["kernel"][1].astype(mx.float32), res["stock"][1].astype(mx.float32)
    print(f"T={T}: stock sdpa {res['stock'][0]:.1f} ms | kernel {res['kernel'][0]:.1f} ms | rel {(mx.linalg.norm(a-b)/mx.linalg.norm(b)).item():.1e} | peak {mx.get_peak_memory()/2**30:.2f} GiB", flush=True)
