"""bmicro87.py N [Bs]: TQ attend, 8 layers, batch B (4B shape): fused kernel vs generic dequant path, ms."""
import sys, time, json
import mlx.core as mx
from kyojin_mlx import tq
N = int(sys.argv[1]); Bs = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1,2,4").split(",")]
Hk, Hq, D, NL = 4, 16, 256, 8
mx.random.seed(2)
base = []
for i in range(NL):
    c = tq.TQKVCache(6, 6, 4, 256); P0 = 1024
    c.update_and_fetch(mx.random.normal((1, Hk, P0, D)).astype(mx.float16), mx.random.normal((1, Hk, P0, D)).astype(mx.float16))
    st = c.state; r = -(-N // P0)
    base.append([mx.tile(x[:, :, :P0], (1, 1, r) + (1,) * (x.ndim - 3))[:, :, :N] for x in st[:4]] + list(st[4:]))
for B in Bs:
    caches = []
    for st in base:
        c = tq.TQKVCache(6, 6, 4, 256)
        c.state = [mx.concatenate([x] * B, axis=0) for x in st]
        caches.append(c)
    mx.eval([c.state for c in caches]); mx.clear_cache()
    q = mx.random.normal((B, Hq, 1, D)).astype(mx.float16); mx.eval(q)
    res = {}
    for name, fused in (("fused", True),):
        for c in caches: c.fused = fused
        ts = []
        for _ in range(8):
            t0 = time.perf_counter(); mx.eval([c.attend(q, D ** -0.5) for c in caches]); ts.append((time.perf_counter() - t0) * 1e3)
        res[name] = round(sorted(ts[2:])[0], 2)
    for c in caches: c.fused = True
    print(json.dumps(dict(N=N, B=B, **res)), flush=True)
    del caches; mx.clear_cache()
