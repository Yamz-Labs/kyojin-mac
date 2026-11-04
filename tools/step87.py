"""step87.py N [steps] : 8 layer caches at depth N, per decode step: update_and_fetch (1 token) + attend for every layer, one mx.eval per step. Time per step, with and without the write."""
import sys, os, time, json
import mlx.core as mx
from kyojin_mlx import tq, kvwin as KW
N = int(sys.argv[1]); steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
Hk, Hq, D, NL = 4, 16, 256, 8
mx.random.seed(1)
caches = []
for i in range(NL):
    c = tq.TQKVCache(6, 6, 4, 256); P0 = 1024
    c.update_and_fetch(mx.random.normal((1, Hk, P0, D)).astype(mx.float16), mx.random.normal((1, Hk, P0, D)).astype(mx.float16))
    st = c.state; r = -(-N // P0)
    c.state = [mx.tile(x[:, :, :P0], (1, 1, r) + (1,) * (x.ndim - 3))[:, :, :N] for x in st[:4]] + list(st[4:])
    caches.append(c)
mx.eval([c.state for c in caches]); mx.clear_cache()
q = mx.random.normal((1, Hq, 1, D)).astype(mx.float16); k1 = mx.random.normal((1, Hk, 1, D)).astype(mx.float16); v1 = k1 * 0.5
def run(write, sync_each):
    ts = []
    for _ in range(steps):
        t0 = time.perf_counter()
        outs = []
        for c in caches:
            if write: c.update_and_fetch(k1, v1)
            outs.append(c.attend(q, D ** -0.5))
        mx.eval(outs); ts.append((time.perf_counter() - t0) * 1e3)
    ts = sorted(ts[3:]); return round(ts[0], 2), round(ts[len(ts) // 4], 2)
for rnd in range(2):
    print(json.dumps(dict(N=N, attend_only=run(False, 1), write_attend=run(True, 1))), flush=True)
