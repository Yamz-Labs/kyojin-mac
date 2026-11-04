"""bchain88.py N [VARIANT,...]: TQ attend, 8 layers (4B shape), chained (layer i+1 query = layer i output, like the model) vs independent, ms per 8-layer pass, best + median of interleaved reps."""
import sys, time, json, statistics
import mlx.core as mx
from kyojin_mlx import tq
N = int(sys.argv[1])
Hk, Hq, D, NL = 4, 16, 256, 8
mx.random.seed(2)
caches = []
base = tq.TQKVCache(6, 6, 4, 256); P0 = 1024
base.update_and_fetch(mx.random.normal((1, Hk, P0, D)).astype(mx.float16), mx.random.normal((1, Hk, P0, D)).astype(mx.float16))
st = base.state; r = -(-N // P0)
for i in range(NL):
    c = tq.TQKVCache(6, 6, 4, 256)
    c.state = [mx.tile(x[:, :, :P0], (1, 1, r) + (1,) * (x.ndim - 3))[:, :, :N] + (0 if x.dtype != mx.float16 else 0) for x in st[:4]] + list(st[4:])
    caches.append(c)
mx.eval([c.state for c in caches]); mx.clear_cache()
q = mx.random.normal((1, Hq, 1, D)).astype(mx.float16); mx.eval(q)
def chained():
    x = q
    for c in caches: x = c.attend(x, D ** -0.5)
    mx.eval(x)
def indep():
    mx.eval([c.attend(q, D ** -0.5) for c in caches])
VUS = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "1,2").split(",")]
res = {k + str(v): [] for k in ("chain", "indep") for v in VUS}
for v in VUS:
    tq.VU = v; chained(); indep()
for _ in range(12):
    for v in VUS:
        tq.VU = v
        for n, f in (("chain", chained), ("indep", indep)):
            t0 = time.perf_counter(); f(); res[n + str(v)].append((time.perf_counter() - t0) * 1e3)
print(json.dumps(dict(N=N, **{k: dict(best=round(min(v), 2), med=round(statistics.median(v), 2)) for k, v in res.items()})), flush=True)
