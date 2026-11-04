"""verify87.py N [T]: 8 layers at depth N: attention for a T-row verify chunk (update_and_fetch(T) + attend, generic dequant path) vs T x (update 1 + fused decode), ms."""
import sys, time, json
import mlx.core as mx
from kyojin_mlx import tq, kvwin as KW
KW.install()
N = int(sys.argv[1]); Ts = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "2,4,8").split(",")]
Hk, Hq, D, NL = 4, 16, 256, 8
mx.random.seed(3)
caches = []
for i in range(NL):
    c = tq.TQKVCache(6, 6, 4, 256); P0 = 1024
    c.update_and_fetch(mx.random.normal((1, Hk, P0, D)).astype(mx.float16), mx.random.normal((1, Hk, P0, D)).astype(mx.float16))
    st = c.state; r = -(-N // P0)
    c.state = [mx.tile(x[:, :, :P0], (1, 1, r) + (1,) * (x.ndim - 3))[:, :, :N] for x in st[:4]] + list(st[4:])
    caches.append(c)
mx.eval([c.state for c in caches]); mx.clear_cache()
for T in Ts:
    k = mx.random.normal((1, Hk, T, D)).astype(mx.float16); q = mx.random.normal((1, Hq, T, D)).astype(mx.float16); mx.eval(k, q)
    def chunk():
        outs = []
        for c in caches:
            kv = c.update_and_fetch(k, k)
            outs.append(c.attend(q, D ** -0.5))
        mx.eval(outs)
        for c in caches: c.trim(T)
    def rows():
        for t in range(T):
            outs = []
            for c in caches:
                c.update_and_fetch(k[:, :, t:t + 1], k[:, :, t:t + 1]); outs.append(c.attend(q[:, :, t:t + 1], D ** -0.5))
            mx.eval(outs)
        for c in caches: c.trim(T)
    res = {}
    for name, f in (("chunk_T", chunk), ("T_x_fused", rows)):
        ts = []
        for _ in range(7):
            t0 = time.perf_counter(); f(); ts.append((time.perf_counter() - t0) * 1e3)
        res[name] = round(sorted(ts[2:])[0], 2)
    print(json.dumps(dict(N=N, T=T, **res)), flush=True)
