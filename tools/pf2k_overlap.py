"""Does MLX overlap the decode of linear i+1 with the matmul of linear i when the graph orders [mm_i, dec_i+1, depends]? usage: pf2k_overlap.py k,K R n"""
import sys, time, mlx.core as mx, numpy as np
from kyojin_mlx import exl3_linear as EL, kernels as KN
d, K = map(int, sys.argv[1].split(",")); R = int(sys.argv[2]); n = int(sys.argv[3])
rng = np.random.default_rng(1)
def lin(seed):
    kt = nt = d // 16
    tr = mx.array(np.random.default_rng(seed).integers(-32768, 32768, (kt, nt, 16 * K), dtype=np.int16))
    one = mx.ones((d,), dtype=mx.float16)
    return EL.Exl3Linear(tr, one, one, cb="mul1")
lins = [lin(i) for i in range(n)]
x0 = (mx.random.normal((R, d)) * 0.1).astype(mx.float16); mx.eval(x0)
dec = lambda l: KN.decode_wr(l._words, l.K, l._kt, l._nt, l.cb)
def chain(mode):
    x = x0; nxt = dec(lins[0])
    for i, l in enumerate(lins):
        wr = nxt
        y = (x @ wr.T)
        if mode == "ovl" and i + 1 < n:
            nxt = dec(lins[i + 1]); y = mx.depends([y], [nxt])[0]
        elif i + 1 < n:
            nxt = dec(lins[i + 1])
        x = y * mx.array(0.5, dtype=mx.float16)
    return x
def lazy(mode):
    x = x0
    for i, l in enumerate(lins):
        y = x @ dec(l).T
        x = y * mx.array(0.5, dtype=mx.float16)
    return x
arms = {"serial": lambda: lazy("s"), "ovl": lambda: chain("ovl"), "mm-only": None}
wrs = [dec(l) for l in lins]; mx.eval(wrs)
def mmonly():
    x = x0
    for w in wrs: x = (x @ w.T) * mx.array(0.5, dtype=mx.float16)
    return x
arms["mm-only"] = mmonly
res = {k: [] for k in arms}
for k, f in arms.items(): mx.eval(f())
for _ in range(7):
    for k, f in arms.items():
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(f()); res[k].append((time.perf_counter() - t0) * 1e3)
for k, v in res.items(): print(f"{k:8s} min {min(v):.2f} ms ({min(v)/n:.3f}/linear)")
