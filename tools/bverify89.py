"""bverify89.py N T VARIANTS: TQ attend of a T-row chunk (T=1 decode, 5 = MTP verify), 8 layers (4B shape), ms per pass, best + median of interleaved reps. Variants CF:RC (CF = chunk-fastest grid, RC = rows cap, 0 = auto)."""
import os, sys, time, json, statistics
import mlx.core as mx
from kyojin_mlx import tq
N, T = int(sys.argv[1]), int(sys.argv[2]); Hk, Hq, D, NL = 4, 16, 256, 8
mx.random.seed(2)
base = tq.TQKVCache(6, 6, 4, 256); P0 = 1024
base.update_and_fetch(mx.random.normal((1, Hk, P0, D)).astype(mx.float16), mx.random.normal((1, Hk, P0, D)).astype(mx.float16))
st = base.state; r = -(-N // P0); caches = []
for i in range(NL):
    c = tq.TQKVCache(6, 6, 4, 256)
    c.state = [mx.tile(x[:, :, :P0], (1, 1, r) + (1,) * (x.ndim - 3))[:, :, :N] for x in st[:4]] + list(st[4:])
    # verify chunks read the last T keys as fp16 ring rows: append T tokens so the kernel sees a real chunk
    caches.append(c)
mx.eval([c.state for c in caches]); mx.clear_cache()
q = mx.random.normal((1, Hq, T, D)).astype(mx.float16)
if T > 1:
    for c in caches:
        c.update_and_fetch(mx.random.normal((1, Hk, T, D)).astype(mx.float16), mx.random.normal((1, Hk, T, D)).astype(mx.float16))
    mx.eval([c.state for c in caches])
VS = sys.argv[3].split(",")
def setv(v):
    f = [int(x) for x in v.split(":")]; tq.CF = f[0]; tq.RC_MAX = f[1] if len(f) > 1 else 0
def run():
    x = q
    outs = [c.attend(q, D ** -0.5) for c in caches]
    mx.eval(outs); return outs
setv(VS[0]); ref = run()[0]
for v in VS[1:]:
    setv(v); o = run()[0]; print(v, "bit-identical", bool(mx.array_equal(ref, o)), flush=True)
res = {v: [] for v in VS}
for v in VS: setv(v); run()
for _ in range(10):
    for v in VS:
        setv(v); t0 = time.perf_counter(); run(); res[v].append((time.perf_counter() - t0) * 1e3)
print(json.dumps(dict(N=N, T=T, **{k: dict(best=round(min(x), 2), med=round(statistics.median(x), 2)) for k, x in res.items()})), flush=True)
