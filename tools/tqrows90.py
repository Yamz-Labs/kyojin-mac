"""tqrows90.py N T VARIANTS [D]: TQ attend of a T-row chunk, 8 layers; variants ROWS:RCV[:NAX[:WN:NSX:PAD]] (ROWS 0 = per-row-chunk plain kernel; NAX 1 = the matrix-unit kernel _DECN_SRC, absent = 0;
WN / NSX / PAD = tq.NAX_WN / NAX_NSX / NAX_PAD, the per-chip knobs, default 0 / 1 / 8). Max abs diff vs the first variant + ms (best, median)."""
import os, sys, time, json, statistics
import mlx.core as mx
from kyojin_mlx import tq
N, T = int(sys.argv[1]), int(sys.argv[2]); D = int(sys.argv[4]) if len(sys.argv) > 4 else 256
Hk, Hq, NL = 4, (16 if D == 256 else 32), 8
mx.random.seed(2)
P0 = 1024
def mk(): return mx.random.normal((1, Hk, P0, D)).astype(mx.float16)
base = tq.TQKVCache(6, 6, 4, 256); base.update_and_fetch(mk(), mk())
st = base.state; r = -(-N // P0); caches = []
for i in range(NL):
    c = tq.TQKVCache(6, 6, 4, 256)
    c.state = [mx.tile(x[:, :, :P0], (1, 1, r) + (1,) * (x.ndim - 3))[:, :, :N] for x in st[:4]] + list(st[4:])
    caches.append(c)
mx.eval([c.state for c in caches]); mx.clear_cache()
q = (mx.random.normal((1, Hq, T, D)) * 2).astype(mx.float16)
if T > 1:
    for c in caches:
        c.update_and_fetch(mk()[:, :, :T], mk()[:, :, :T])
    mx.eval([c.state for c in caches])
VS = sys.argv[3].split(",")
def setv(v):
    f = [int(x) for x in v.split(":")] + [0, 0, 0, 1, 8][len(v.split(":")) - 1:]
    tq.ROWS, tq.RCV_MAX, tq.NAX, tq.NAX_WN, tq.NAX_NSX, tq.NAX_PAD = f[:6]
def run():
    outs = [c.attend(q, D ** -0.5) for c in caches]
    mx.eval(outs); return outs
setv(VS[0]); ref = run()
for v in VS[1:]:
    setv(v); o = run()
    print(v, "maxdiff", max(float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)))) for a, b in zip(ref, o)), "ref absmax", float(mx.max(mx.abs(ref[0].astype(mx.float32)))), flush=True)
res = {v: [] for v in VS}
for v in VS: setv(v); run()
for _ in range(10):
    for v in VS:
        setv(v); t0 = time.perf_counter(); run(); res[v].append((time.perf_counter() - t0) * 1e3)
print(json.dumps(dict(N=N, T=T, D=D, **{k: dict(best=round(min(x), 2), med=round(statistics.median(x), 2)) for k, x in res.items()})), flush=True)
