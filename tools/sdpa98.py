"""MLA / MQA decode attention over gathered top-k rows (GLM-5 DSA layer shape), GB/s of the K/V read vs the read roof.
usage: sdpa98.py [N,N,..] [K] [Hq:D] [variants]     N = latent cache rows (depth), K = top-k width (GLM 2048), default 64:512.
The query attends to K latent rows picked by an index (some -1 = padding, masked: VALID = valid fraction); K = V = the same gathered rows (MLA absorbed form).
variants: stock (mx.fast.sdpa with the bool mask, the mlx-vlm decode call), nax (split-K kernel, sdpa_dec.py, array mask), n<k> (k splits).
Columns: gather ms (take_along_axis), core ms, GB/s of the core read (K rows read once: K = V) and of the gather + core (read + write + read).
Env: L independent layers per eval (IT passes side by side: launches overlap, an optimistic stock time; the shape is compute-bound, so the
sign holds), IT, REP, DT (f16 / bf16), VALID."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx.core as mx
import numpy as np

Ns = [int(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "4096,32768,65536").split(",")]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
Hq, D = (int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "64:512").split(":"))
VAR = (sys.argv[4] if len(sys.argv) > 4 else "stock,nax").split(",")
L = int(os.environ.get("L", "4")); IT = int(os.environ.get("IT", "3")); REP = int(os.environ.get("REP", "5"))
DT = mx.bfloat16 if os.environ.get("DT", "bf16") == "bf16" else mx.float16
VALID = float(os.environ.get("VALID", "0.9"))
S = mx.fast.scaled_dot_product_attention


def bench(fn):
    mx.eval(fn()); best = 1e9
    for _ in range(REP):
        mx.synchronize(); t0 = time.perf_counter(); ys = [fn() for _ in range(IT)]; mx.eval(ys); best = min(best, (time.perf_counter() - t0) / IT)
    return best


def roof(n_bytes):
    x = mx.random.normal((n_bytes // 2,)).astype(mx.float16); mx.eval(x)
    t = bench(lambda: mx.sum(x.reshape(-1, 1024), axis=1))
    del x; return n_bytes / t / 1e9


print(f"read roof (mx.sum fp16 1 GiB): {roof(1 << 30):.1f} GB/s", flush=True)
rng = np.random.default_rng(0)
for N in Ns:
    lat = [mx.random.normal((1, 1, N, D)).astype(DT) for _ in range(L)]
    idx = []
    for _ in range(L):
        ii = np.sort(rng.choice(N, size=min(K, N), replace=False)).astype(np.int32)
        ii = np.concatenate([ii, -np.ones(K - ii.size, np.int32)])
        ii[rng.random(K) > VALID] = -1
        idx.append(mx.array(ii.reshape(1, 1, K)))
    q = mx.random.normal((1, Hq, 1, D)).astype(DT)
    mx.eval(lat, idx, q)
    sc = D ** -0.5

    def gather(l, i):
        safe = mx.clip(i, 0, N - 1)
        return mx.take_along_axis(l, safe[:, None, 0, :, None], axis=2), (i >= 0)[:, None]

    sel = [gather(l, i) for l, i in zip(lat, idx)]
    mx.eval(sel)
    tg = bench(lambda: [gather(l, i)[0] for l, i in zip(lat, idx)])
    nb = L * K * D * 2
    ref = None
    for var in VAR:
        if var == "stock":
            f = lambda s, m: S(q, s, s, scale=sc, mask=m)
        elif var == "nax" or var.startswith("n"):
            from kyojin_mlx.sdpa_dec import sdpa_decode
            ns = int(var[1:]) if var != "nax" else None
            f = lambda s, m, ns=ns: sdpa_decode(q, s, s, sc, ns=ns, mask=m)
        else:
            raise ValueError(var)
        t = bench(lambda: [f(s, m) for s, m in sel])
        o = f(*sel[0]).astype(mx.float32)
        r32 = S(q.astype(mx.float32), sel[0][0].astype(mx.float32), sel[0][0].astype(mx.float32), scale=sc, mask=sel[0][1])
        if ref is None:
            ref = o
        e32 = mx.abs(o - r32).max().item()
        print(f"Hq{Hq} Hk1 D{D} N={N:6d} K={K} {str(DT).split('.')[-1]} {var:6s}: gather {tg / L * 1e3:6.3f} core {t / L * 1e3:6.3f} ms/layer  "
              f"core {nb / t / 1e9:6.1f} GB/s  gather+core {3 * nb / (t + tg) / 1e9:6.1f} GB/s  maxerr vs fp32 {e32:.2e}", flush=True)
    del lat, sel
