"""bchain88.py N [VARIANT,...]: TQ attend, 8 layers (4B shape), chained (layer i+1 query = layer i output, like the model) vs independent, ms per 8-layer pass, best + median of interleaved reps."""
import os, sys, time, json, statistics
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
VUS = [v for v in (sys.argv[2] if len(sys.argv) > 2 else "2:1,2:2,4:2").split(",")]   # VU:VH variants
KS = {}
def setv(v):
    f = [int(x) for x in v.split(":")]
    tq.VU, tq.VH = f[0], f[1]; tq.OV4 = f[3] if len(f) > 3 else 1; tq.OVU = f[4] if len(f) > 4 else 4; ab = f[2] if len(f) > 2 else 0
    os.environ["KYOJIN_TQ_ABL"] = str(ab); tq._kernels = KS.setdefault(ab, {})
def outv(v):
    setv(v); x = caches[0].attend(q, D ** -0.5); mx.eval(x); return x
ref = outv(VUS[0])
for v in [x for x in VUS[1:] if x.count(":") < 2]: print(v, "bit-identical", bool(mx.array_equal(ref, outv(v))), flush=True)
res = {k + v: [] for k in ("chain", "indep") for v in VUS}
for v in VUS:
    setv(v); chained(); indep()
for _ in range(12):
    for v in VUS:
        setv(v)
        for n, f in (("chain", chained), ("indep", indep)):
            t0 = time.perf_counter(); f(); res[n + v].append((time.perf_counter() - t0) * 1e3)
print(json.dumps(dict(N=N, **{k: dict(best=round(min(v), 2), med=round(statistics.median(v), 2)) for k, v in res.items()})), flush=True)
