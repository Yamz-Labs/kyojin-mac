"""bk_micro.py N [nsplit list] [reps]: new byte-layout decode vs current TQ6 decode, 8 layers, interleaved. env HK HQ HD."""
import sys, os, time, json
import numpy as np, mlx.core as mx
from kyojin_mlx import tq
import tq_bk
N = int(sys.argv[1]); nss = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "32").split(",")]; reps = int(sys.argv[3]) if len(sys.argv) > 3 else 12
Hk, Hq, D, NL, bits = int(os.environ.get("HK", 4)), int(os.environ.get("HQ", 16)), int(os.environ.get("HD", 256)), 8, 6
REP = Hq // Hk
mx.random.seed(1)
caches, news = [], []
for i in range(NL):
    c = tq.TQKVCache(bits, bits, 4, 256); P0 = 1024
    k = mx.random.normal((1, Hk, P0, D)).astype(mx.float16); v = mx.random.normal((1, Hk, P0, D)).astype(mx.float16)
    c.update_and_fetch(k, v)
    st = c.state; r = -(-N // P0)
    c.state = [mx.tile(x[:, :, :P0], (1, 1, r) + (1,) * (x.ndim - 3))[:, :, :N] for x in st[:4]] + list(st[4:])
    # new layout from the first 1024 rows, tiled (codes only differ by layout)
    kw = np.array(st[0][0, :, :P0]); vw = np.array(st[2][0, :, :P0])
    kcodes, vcodes = tq_bk.relayout_np(kw, bits, D), tq_bk.relayout_np(vw, bits, D)
    kcodes = np.tile(kcodes, (1, r, 1))[:, :N]; vcodes = np.tile(vcodes, (1, r, 1))[:, :N]
    cap = -(-N // 256) * 256
    pad = lambda a: np.concatenate([a, np.zeros((a.shape[0], cap - N, a.shape[2]), np.uint8)], 1)
    KM, VM = int(os.environ.get("KM", 0)), int(os.environ.get("VM", 0))
    padw = lambda a: np.concatenate([a, np.zeros((a.shape[0], cap - N, a.shape[2]), a.dtype)], 1)
    kc = mx.array(tq_bk.k_tiles(pad(kcodes))) if KM == 0 else mx.array(tq_bk.k_tiles_planes(padw(np.tile(kw, (1, r, 1))[:, :N]), D, bits))
    vc = mx.array(pad(vcodes)) if VM == 0 else mx.array(padw(np.tile(vw, (1, r, 1))[:, :N]))
    pads = lambda s: mx.concatenate([s[0, :, :N][None], mx.zeros((1, Hk, cap - N), mx.float16)], axis=2) if cap > N else s
    ks, vs = pads(c.kq[1][:, :, :N][None] if False else mx.tile(st[1][:, :, :P0], (1, 1, r))[:, :, :N]), pads(mx.tile(st[3][:, :, :P0], (1, 1, r))[:, :, :N])
    news.append((kc, ks[0], vc, vs[0]))
    caches.append(c)
mx.eval([c.state for c in caches]); mx.eval([x for n in news for x in n]); mx.clear_cache()
q = mx.random.normal((1, Hq, 1, D)).astype(mx.float16); scale = D ** -0.5; mx.eval(q)
def newdec(c, n, ns):
    L = c.offset; S = min(c.sink, L); T0 = max(S, L - c.window)
    NR = c.rk.shape[2]; NS = c.sk.shape[2]
    cap = n[0].shape[1] * 32
    info = mx.array([L, S, T0, NR, cap, NS], dtype=mx.int32)
    return tq_bk.decode(q, n[0], n[1], n[2], n[3], c.sk, c.sv, c.rk, c.rv, info, D, bits, bits, REP, Hk, ns, scale, q.dtype, KM=int(os.environ.get('KM', 0)), VM=int(os.environ.get('VM', 0)))
ref = caches[0].attend(q, scale).astype(mx.float32); mx.eval(ref)
for ns in nss:
    o = newdec(caches[0], news[0], ns).astype(mx.float32); mx.eval(o)
    rel = (mx.abs(o - ref).max() / mx.abs(ref).max()).item()
    t_old, t_new = [], []
    for _ in range(reps + 2):
        t0 = time.perf_counter(); mx.eval([c.attend(q, scale) for c in caches]); t_old.append((time.perf_counter() - t0) * 1e3)
        t0 = time.perf_counter(); mx.eval([newdec(c, n, ns) for c, n in zip(caches, news)]); t_new.append((time.perf_counter() - t0) * 1e3)
    f = lambda t: (round(sorted(t[2:])[0], 3), round(sorted(t[2:])[len(t[2:]) // 4], 3))
    pool_old = sum(x.nbytes for c in caches for x in c.state[:4]); pool_new = sum(x.nbytes for n in news for x in n)
    print(json.dumps(dict(N=N, ns=ns, old_min_p25=f(t_old), new_min_p25=f(t_new), rel_vs_old=rel, roof_old=round(pool_old / 139.3e9 * 1e3, 2), roof_new=round(pool_new / 139.3e9 * 1e3, 2))), flush=True)
