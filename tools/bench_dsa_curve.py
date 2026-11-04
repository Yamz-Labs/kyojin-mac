"""Per-token DS4.1 decode attention cost vs depth (synthetic KV), stock chain vs dsa_decode kernels, model of the 43-layer net:
index sources 2, 8, 14 (ratio 2, dense), 20 (ratio 1, dense + candidate source), 24 28 32 36 (ratio 1, candidate consumers); 38 compressing layers run the
sparse attend (top-k 512 + window 128).  Prints ms per layer for each piece and ms per token.   python tools/bench_dsa_curve.py"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.expanduser("~/kyojin-mlx/vlm_pkg"))
from kyojin_mlx import dsa_decode as D, fuse_attn as FA
from mlx_vlm.models.deepseek_v41 import language as L
CH = 16
def t(fn, reps=7):
    mx.eval(fn()); ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); mx.eval([fn() for _ in range(CH)]); ts.append((time.perf_counter() - t0) / CH)
    return min(ts) * 1e3
rng = np.random.default_rng(0)
depths = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "4096,16384,65536,131072").split(",")]
rows = []
for depth in depths:
    P = {}
    for ratio in (2, 1):
        n = depth // ratio
        q = mx.array(rng.standard_normal((1, 1, 32, 128)).astype(np.float16)); ik = mx.array(rng.standard_normal((1, n + 8, 128)).astype(np.float16))
        w = mx.array((rng.standard_normal((1, 1, 32)) * .05).astype(np.float16)); nn = mx.array([n], dtype=mx.int32)
        cand = L.select_candidate_blocks(mx.array(rng.standard_normal((1, n)).astype(np.float16)), n, 2048, 8); mx.eval(q, ik, w, cand)
        def stock(c=None, src=False):
            def f():
                tile = mx.einsum("bsnd,btd->bsnt", q, ik[:, :n]); sc = (mx.maximum(tile, 0) * w[..., None]).sum(axis=2)
                if c is not None: sc = mx.where(c, sc, -mx.inf)
                o = [mx.sort(mx.argsort(-sc, axis=-1)[..., :512], axis=-1)]
                if src: o.append(L.select_candidate_blocks(sc, n, 2048, 8))
                return o
            return f
        def new(c=None, src=False):
            def f():
                sc = D.scores(q.reshape(32, 128), w.reshape(32), ik[0, :n], nn, n, c.reshape(-1) if c is not None else None)
                o = [D.select(sc, nn, n, 512)]
                if src: o.append(D.candidate_mask(sc, nn, n, 2048))
                return o
            return f
        P[ratio] = dict(sd=t(stock()), sc=t(stock(cand)), ss=t(stock(None, True)), nd=t(new()), nc=t(new(cand)), ns=t(new(None, True)))
        pool = mx.array(rng.standard_normal((1, n, 512)).astype(np.float16)); ring = mx.array(rng.standard_normal((1, 128, 512)).astype(np.float16))
        qa = mx.array(rng.standard_normal((1, 1, 64 * 512)).astype(np.float16)); sink = mx.zeros((64,), dtype=mx.float32)
        rt = mx.array(np.tile([1.0, 0.0], 32).astype(np.float32)); pos = mx.array([depth], dtype=mx.int32)
        idx = mx.array(np.sort(rng.choice(n, 512, replace=False)).astype(np.int32)); mx.eval(pool, ring, qa, idx)
        FA.MMA, FA.SPLIT_PV = False, False
        P[ratio]["att0"] = t(lambda: FA.attend(qa, ring, pool, idx, sink, rt, rt, pos, 64, 64, 0.04))
        FA.MMA, FA.SPLIT_PV = True, True
        P[ratio]["att"] = t(lambda: FA.attend(qa, ring, pool, idx, sink, rt, rt, pos, 64, 64, 0.04))
    att = 18 * P[2]["att"] + 20 * P[1]["att"]
    att0 = 18 * P[2]["att0"] + 20 * P[1]["att0"]
    stock_i = 3 * P[2]["sd"] + P[1]["ss"] + 4 * P[1]["sc"]
    new_i = 3 * P[2]["nd"] + P[1]["ns"] + 4 * P[1]["nc"]
    rows.append((depth, stock_i, new_i, att, att0))
    print(f"depth {depth:7d}: indexer/token stock {stock_i:6.2f} ms  new {new_i:5.2f} ms | attend 38 layers old {att0:6.2f} new {att:6.2f} | attention total before {stock_i + att0:6.2f}  after {new_i + att:6.2f}", flush=True)
b = rows[0]
print("ratio vs own 4K (indexer + attend):", "  ".join(f"{r[0] // 1024}K before x{(r[1] + r[4]) / (b[1] + b[4]):.2f} after x{(r[2] + r[3]) / (b[2] + b[3]):.2f}" for r in rows))
print("ratio vs 4K (indexer only):   ", "  ".join(f"{r[0] // 1024}K stock x{r[1] / b[1]:.1f} new x{r[2] / b[2]:.1f}" for r in rows))
