"""Per-layer us of the decode indexer chain (scores + top-k [+ candidate mask]) at DS4.1 dims: stock rt chain vs idxfuse, interleaved, medians.
Dependent chain of LAYERS layers (the next lens depends on the previous idx, as in a decode step), minus a null chain with the same glue ops.
  flock /tmp/ultra_bench.lock python tools/idxfuse_bench.py [--reps 11] [--out file.json]"""
import argparse, json, time
import numpy as np
import mlx.core as mx
from kyojin_mlx import dsv4_core as C, idxfuse as IF

NH, DI, K, KB, BS = 32, 128, 512, 2048, 8
LAYERS = 12


def stock(q, keys, w, ln, n, cs):
    sc = C.index_scores(q, keys, w, ln, n, rt=True)
    idx = C.index_topk_rt(sc, ln, K)
    return idx, (C.candidate_mask_rt(sc, ln, KB, BS) if cs else None)


def fuse(q, keys, w, ln, n, cs):
    idx, m, _ = IF.fused(q, keys, w, ln, n, K, topk_blocks=KB if cs else 0, block_size=BS)
    return idx, m


def chain(fn, q, keys, w, ln0, n, cs, tb=None):
    ln = ln0
    outs = []
    for l in range(LAYERS):
        idx, m = fn(q, keys[l % len(keys)], w, ln, n, cs)
        ln = ln0 + (idx[:, 0] * 0).astype(mx.int32)
        outs.append(idx if m is None else (idx, m))
    return outs


def null(q, keys, w, ln0, n, cs):
    ln = ln0
    idx = mx.zeros((ln0.shape[0], K), dtype=mx.int32)
    for l in range(LAYERS):
        ln = ln0 + (idx[:, 0] * 0).astype(mx.int32)
        idx = idx + ln[:, None]
    return [idx]


def timeit(f):
    t = time.perf_counter(); mx.eval(f()); return (time.perf_counter() - t) * 1e6


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=11); ap.add_argument("--out"); a = ap.parse_args()
    rng = np.random.default_rng(3)
    res = []
    for R in (1, 4):
        for n in (1024, 4096, 16384, 32768):
            q = mx.array(rng.standard_normal((R, NH, DI)).astype(np.float16))
            w = mx.array((rng.standard_normal((R, NH)) * 0.05).astype(np.float16))
            keys = [mx.array((rng.standard_normal((R, n, DI)) * 0.5).astype(np.float16)) for _ in range(4)]
            ln = mx.array([n - i * 7 for i in range(R)], dtype=mx.int32)
            mx.eval(q, w, ln, *keys)
            for cs in (False, True):
                arms = {"null": lambda: null(q, keys, w, ln, n, cs), "stock": lambda: chain(stock, q, keys, w, ln, n, cs), "fuse": lambda: chain(fuse, q, keys, w, ln, n, cs)}
                r0, r1 = chain(stock, q, keys, w, ln, n, cs), chain(fuse, q, keys, w, ln, n, cs)
                mx.eval(r0, r1)
                for x, y in zip(r0, r1):
                    xs, ys = (x, y) if isinstance(x, tuple) else ((x,), (y,))
                    assert np.array_equal(np.array(xs[0]), np.array(ys[0]))
                for _ in range(3):
                    for f in arms.values(): timeit(f)
                t = {k: [] for k in arms}
                for _ in range(a.reps):
                    for k, f in arms.items(): t[k].append(timeit(f))
                m = {k: float(np.median(v)) for k, v in t.items()}
                per = {k: (m[k] - m["null"]) / LAYERS for k in ("stock", "fuse")}
                row = dict(R=R, n=n, cand=cs, stock_us=round(per["stock"], 1), fuse_us=round(per["fuse"], 1), speedup=round(per["stock"] / max(per["fuse"], 1e-9), 2))
                print(row, flush=True); res.append(row)
    if a.out: json.dump(res, open(a.out, "w"), indent=1)


main()
