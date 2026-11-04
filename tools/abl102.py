"""abl102.py N T ABLS: the TQ verify-rows kernel (dec2) of a T-row chunk over 8 layers at N keys (Qwen3.5 4B/9B attention shape Hk 4, Hq 16,
D 256), ms best / median per ablation (comma list of ABL bits, 0 = the real kernel; 16 K codebook gather off, 32 K dot on 1 row, 1024 P.V loop off,
2048 softmax off, 4096 K dot off; nB = the NAX kernel _DECN_SRC with its ABL bits B: 1 K decode, 2 V decode, 4 S MMA, 8 PV MMA, 16 softmax, 32 loop off),
and the FLOP rate of the first item (2 * T * Hq * N * D * 2 FLOP per layer)."""
import os, sys, time, json, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx.core as mx
from kyojin_mlx import tq
N, T = int(sys.argv[1]), int(sys.argv[2]); ABLS = sys.argv[3].split(",")
D, Hk, Hq, NL, P0 = 256, 4, 16, 8, 1024
mx.random.seed(2)
mk = lambda n=P0: mx.random.normal((1, Hk, n, D)).astype(mx.float16)
base = tq.TQKVCache(6, 6, 4, 256); base.update_and_fetch(mk(), mk())
st = base.state; r = -(-N // P0); caches = []
for i in range(NL):
    c = tq.TQKVCache(6, 6, 4, 256)
    c.state = [mx.tile(x[:, :, :P0], (1, 1, r) + (1,) * (x.ndim - 3))[:, :, :N] for x in st[:4]] + list(st[4:])
    c.update_and_fetch(mk(T), mk(T)); caches.append(c)
mx.eval([c.state for c in caches]); mx.clear_cache()
q = (mx.random.normal((1, Hq, T, D)) * 2).astype(mx.float16)
def run():
    outs = [c.attend(q, D ** -0.5) for c in caches]; mx.eval(outs); return outs
res = {a: [] for a in ABLS}
def seta(a): tq.NAX = int(a.startswith("n")); tq.ABL2 = int(a.lstrip("n"))
for a in ABLS: seta(a); run()
for _ in range(12):
    for a in ABLS:
        seta(a); t0 = time.perf_counter(); run(); res[a].append((time.perf_counter() - t0) * 1e3)
tq.ABL2 = 0; tq.NAX = 1
b = min(res[ABLS[0]]); fl = 2 * T * Hq * N * D * 2 * NL
print(json.dumps(dict(N=N, T=T, tflops_real=round(fl / b * 1e-9, 2), **{str(k): dict(best=round(min(x), 2), med=round(statistics.median(x), 2)) for k, x in res.items()})), flush=True)
