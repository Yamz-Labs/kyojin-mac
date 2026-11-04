"""TQ decode attention alone, 8 distinct layer caches (like the 4B), at depth N: correctness vs the generic dequant path + ms/8 layers per (NSPLIT, KP) combo.
usage: curve83_micro.py N [cfg 6|5] [combos nsplit:kp,..] [reps]"""
import sys, time, json
import mlx.core as mx
from kyojin_mlx import tq
N = int(sys.argv[1]); bits = int(sys.argv[2]) if len(sys.argv) > 2 else 6
combos = [tuple(map(int, c.split(":"))) for c in (sys.argv[3] if len(sys.argv) > 3 else "32:1").split(",")]
reps = int(sys.argv[4]) if len(sys.argv) > 4 else 15
import os
Hk, Hq, D, NL = int(os.environ.get("HK", 4)), int(os.environ.get("HQ", 16)), int(os.environ.get("HD", 256)), 8
mx.random.seed(1)
caches = []
for i in range(NL):
    c = tq.TQKVCache(bits, bits, 4, 256)
    P0 = 1024
    k = mx.random.normal((1, Hk, P0, D)).astype(mx.float16); v = mx.random.normal((1, Hk, P0, D)).astype(mx.float16)
    c.update_and_fetch(k, v)
    st = c.state; r = -(-N // P0)
    c.state = [mx.tile(x[:, :, :P0], (1, 1, r) + (1,) * (x.ndim - 3))[:, :, :N] for x in st[:4]] + list(st[4:])
    caches.append(c)
mx.eval([c.state for c in caches]); mx.clear_cache()
q = mx.random.normal((1, Hq, 1, D)).astype(mx.float16); scale = D ** -0.5
mx.eval(q)
def reference(c):
    L = c.offset; S = min(c.sink, L); T0 = max(S, L - c.window); NR = c.rk.shape[2]
    kd, vd = c._deq_rows(0, L)
    kd, vd = kd.astype(mx.float32), vd.astype(mx.float32)
    kd = mx.concatenate([c.sk[:, :, :S].astype(mx.float32), kd[:, :, S:T0], c.rk[:, :, NR - (L - T0):].astype(mx.float32)], axis=2)
    vd = mx.concatenate([c.sv[:, :, :S].astype(mx.float32), vd[:, :, S:T0], c.rv[:, :, NR - (L - T0):].astype(mx.float32)], axis=2)
    qr = tq.rotate(q, D).astype(mx.float32).reshape(1, Hk, Hq // Hk, D)
    sc = (qr * scale) @ kd.swapaxes(-1, -2)
    o = (mx.softmax(sc, axis=-1) @ vd).reshape(1, Hq, 1, D)
    return tq.unrotate(o, D)
c0 = caches[0]; ref = reference(c0); mx.eval(ref)
for ns, kp in combos:
    tq.NSPLIT = ns; tq.ABL = kp

    o = c0.attend(q, scale); mx.eval(o)
    err = mx.abs(o.astype(mx.float32) - ref.astype(mx.float32)).max().item(); rel = err / mx.abs(ref.astype(mx.float32)).max().item()
    ts = []
    for _ in range(reps + 2):
        t0 = time.perf_counter(); outs = [c.attend(q, scale) for c in caches]; mx.eval(outs); ts.append((time.perf_counter() - t0) * 1e3)
    ts = sorted(ts[2:])
    nbytes = sum(x.nbytes for c in caches for x in c.state[:4])
    print(json.dumps(dict(N=N, bits=bits, nsplit=ns, kp=kp, ms8_min=round(ts[0], 3), ms8_p25=round(ts[len(ts) // 4], 3), rel_err=rel, pool_MB=nbytes / 1e6, roof_ms=nbytes / 139.3e9 * 1e3)), flush=True)
