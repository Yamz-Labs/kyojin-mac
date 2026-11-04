"""Proxy for a user-batched TQ decode attention: N users' pools stacked along the head axis (Hk' = N Hk, same rep) and ONE two-dispatch decode call, vs N separate
per-user calls (what MU does now: RowsKV -> one attend per user). One layer, Qwen3.5-4B shapes (Hq 16, Hk 4, D 256, tq6, sink 4, window 256). Interleaved, per-layer ms.
usage: sdpa_batch85.py CTX [USERS=1,2,3,4] [PASSES=21] [LAYERS=8]"""
import sys, time, statistics
import mlx.core as mx
from kyojin_mlx import tq, kvwin as KW
CTX = int(sys.argv[1]); Ns = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "1,2,3,4").split(",")]
PASSES = int(sys.argv[3]) if len(sys.argv) > 3 else 21; LAY = int(sys.argv[4]) if len(sys.argv) > 4 else 8
Hq, Hk, D = 16, 4, 256
KW.install()
mx.random.seed(0)
def mk():
    c = tq.TQKVCache(6, 6, 4, 256)
    for s in range(0, CTX, 1024):
        n = min(1024, CTX - s)
        c.update_and_fetch(mx.random.normal((1, Hk, n, D)).astype(mx.float16), mx.random.normal((1, Hk, n, D)).astype(mx.float16))
    mx.eval(c.state)
    return c
cat1 = lambda ts: tuple(mx.concatenate([t[i] for t in ts], axis=1) for i in range(len(ts[0]))) if isinstance(ts[0], tuple) else mx.concatenate(ts, axis=1)
caches = {}
for N in Ns:
    # LAY distinct layers per user so the KV does not sit in a cache
    users = [[mk() for _ in range(LAY)] for _ in range(N)]
    stacked = []
    for l in range(LAY):
        cs = [users[u][l] for u in range(N)]
        b = tq.TQKVCache(6, 6, 4, 256)
        b.kq, b.vq = cat1([c.kq for c in cs]), cat1([c.vq for c in cs])
        b.sk, b.sv, b.rk, b.rv = (cat1([getattr(c, a) for c in cs]) for a in ("sk", "sv", "rk", "rv"))
        b.offset = cs[0].offset
        stacked.append(b)
    mx.eval([x for b in stacked for x in (*b.kq, *b.vq, b.sk, b.sv, b.rk, b.rv)])
    caches[N] = (users, stacked)
    mx.clear_cache()
scale = D ** -0.5
def per_user(N):
    users, _ = caches[N]
    qs = [mx.random.normal((1, Hq, 1, D)).astype(mx.float16) for _ in range(N)]
    outs = []
    for l in range(LAY):
        outs += [users[u][l].attend(qs[u], scale) for u in range(N)]
    return outs
def batched(N):
    _, st = caches[N]
    q = mx.random.normal((1, N * Hq, 1, D)).astype(mx.float16)
    return [st[l]._decode(q, scale) for l in range(LAY)]
def t(fn, N, reps=3):
    r = []
    for _ in range(reps):
        t0 = time.perf_counter(); mx.eval(fn(N)); r.append((time.perf_counter() - t0) * 1e3)
    return min(r)
for N in Ns: t(per_user, N); t(batched, N)
res = {(N, k): [] for N in Ns for k in ("per_user", "batched")}
for p in range(PASSES):
    for N in (Ns if p % 2 == 0 else Ns[::-1]):
        res[(N, "per_user")].append(t(per_user, N)); res[(N, "batched")].append(t(batched, N))
kvb = None
print(f"ctx {CTX}, {LAY} layers: ms per layer-step (median / min)")
for N in Ns:
    a, b = res[(N, "per_user")], res[(N, "batched")]
    print(f"N={N}: per-user launches {statistics.median(a)/LAY:.3f} / {min(a)/LAY:.3f}   batched {statistics.median(b)/LAY:.3f} / {min(b)/LAY:.3f}   ratio {statistics.median(a)/statistics.median(b):.2f}")
