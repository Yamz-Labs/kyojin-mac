"""Stock indexer chain vs dsa_decode kernels, ms per layer per token, synthetic fq4-like keys.  python tools/bench_dsa_new.py [--ratio 1]"""
import argparse, os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.expanduser("~/kyojin-mlx/vlm_pkg"))
from kyojin_mlx import dsa_decode as D
from mlx_vlm.models.deepseek_v41 import language as L
CH = 16
def t(fn, reps=7):
    mx.eval(fn()); ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); mx.eval([fn() for _ in range(CH)]); ts.append((time.perf_counter() - t0) / CH)
    return min(ts) * 1e3
ap = argparse.ArgumentParser(); ap.add_argument("--ratio", type=int, default=1); ap.add_argument("--depths", default="4096,16384,65536,131072,262144")
a = ap.parse_args(); rng = np.random.default_rng(0)
print(f"ratio {a.ratio}   ms/layer/token       stock: dense  cand-layer  cand-src |  new: dense  cand-layer  cand-src")
for depth in [int(x) for x in a.depths.split(",")]:
    n = depth // a.ratio; dt = mx.float16
    q = mx.array(rng.standard_normal((1, 1, 32, 128)).astype(np.float16)); ik = mx.array(rng.standard_normal((1, n + 8, 128)).astype(np.float16))
    w = mx.array((rng.standard_normal((1, 1, 32)) * .05).astype(np.float16)); nn = mx.array([n], dtype=mx.int32)
    s0 = mx.array(rng.standard_normal((1, n)).astype(np.float16)); cand = L.select_candidate_blocks(s0, n, 2048, 8); mx.eval(q, ik, w, cand)
    def stock(c=None, src=False):
        def f():
            tile = mx.einsum("bsnd,btd->bsnt", q, ik[:, :n]); sc = (mx.maximum(tile, 0) * w[..., None]).sum(axis=2)
            if c is not None: sc = mx.where(c, sc, -mx.inf)
            outs = [mx.sort(mx.argsort(-sc, axis=-1)[..., :512], axis=-1)]
            if src: outs.append(L.select_candidate_blocks(sc, n, 2048, 8))
            return outs
        return f
    def new(c=None, src=False):
        def f():
            sc = D.scores(q.reshape(32, 128), w.reshape(32), ik.reshape(-1, 128), nn, n, c.reshape(-1) if c is not None else None)
            outs = [D.select(sc, nn, n, 512)]
            if src: outs.append(D.candidate_mask(sc, nn, n, 2048))
            return outs
        return f
    r = [t(stock()), t(stock(cand)), t(stock(None, True)), t(new()), t(new(cand)), t(new(None, True))]
    print(f"{depth:7d} n={n:7d}   {r[0]:6.2f} {r[1]:6.2f} {r[2]:6.2f}   |   {r[3]:6.2f} {r[4]:6.2f} {r[5]:6.2f}", flush=True)
