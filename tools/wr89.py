"""wr89.py N: TQ KV append (update_and_fetch of one token) for 8 layers (4B shape), ms per 8-layer pass; compares with attend. best + median of reps."""
import sys, time, statistics
import mlx.core as mx
from kyojin_mlx import tq
N = int(sys.argv[1]); Hk, Hq, D, NL = 4, 16, 256, 8
mx.random.seed(2)
base = tq.TQKVCache(6, 6, 4, 256); P0 = 1024
base.update_and_fetch(mx.random.normal((1, Hk, P0, D)).astype(mx.float16), mx.random.normal((1, Hk, P0, D)).astype(mx.float16))
st = base.state; r = -(-N // P0); caches = []
for i in range(NL):
    c = tq.TQKVCache(6, 6, 4, 256)
    c.state = [mx.tile(x[:, :, :P0], (1, 1, r) + (1,) * (x.ndim - 3))[:, :, :N] for x in st[:4]] + list(st[4:])
    caches.append(c)
mx.eval([c.state for c in caches]); mx.clear_cache()
k = mx.random.normal((1, Hk, 1, D)).astype(mx.float16); v = mx.random.normal((1, Hk, 1, D)).astype(mx.float16); q = mx.random.normal((1, Hq, 1, D)).astype(mx.float16); mx.eval(k, v, q)
def app():
    outs = []
    for c in caches: outs += list(c.update_and_fetch(k, v))
    mx.eval(outs)
def att():
    mx.eval([c.attend(q, D ** -0.5) for c in caches])
def both():
    o = []
    for c in caches: c.update_and_fetch(k, v); o.append(c.attend(q, D ** -0.5))
    mx.eval(o)
res = {"append": [], "attend": [], "both": []}
for f in (app, att, both): f(); f()
for _ in range(10):
    for n, f in (("append", app), ("attend", att), ("both", both)):
        t0 = time.perf_counter(); f(); res[n].append((time.perf_counter() - t0) * 1e3)
print(N, {n: (round(min(x), 2), round(statistics.median(x), 2)) for n, x in res.items()})
