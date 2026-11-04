"""Decode indexer selection at DS4.1 dims: stock mx chain (full argsort + sort, select_candidate_blocks) vs idxfast
(topk_mt / select_fused), one session, n = indexed keys. Sets the KYOJIN_IDX_FAST_MIN crossover.
Run: PYTHONPATH=.:vlm_pkg python tools/longctx_topk_bench.py [--json out.json]   (light: < 1 GB)"""
import argparse, json, time
import numpy as np
import mlx.core as mx
from kyojin_mlx import dsv4_core as C
from kyojin_mlx import idxfast as IX

import os
NH, DI, K, BLK, BS, REPS = 32, 128, 512, 2048, 8, 7
FAN = int(os.environ.get("BENCH_FAN", "8"))     # 1 = serial latency (eval per op), 8 = throughput


def med(fn):
    mx.eval(fn()); mx.eval(fn())
    ts = []
    for _ in range(REPS):
        o = fn(); t = time.perf_counter(); mx.eval(o); ts.append((time.perf_counter() - t) * 1e3 / FAN)
    return sorted(ts)[REPS // 2]


def stock_topk(sc, lens):
    s = sc[:, None, :]
    ids = mx.sort(mx.argsort(-s, axis=-1)[..., :min(K, s.shape[-1])], axis=-1).astype(mx.int32)
    return mx.where(ids < lens[:, None, None], ids, -1)


def stock_cand(sc, lens):
    from mlx_vlm.models.deepseek_v41 import language as L
    return L.select_candidate_blocks(sc[:, None, :], lens[:, None, None], BLK, BS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--ns", default="4096,16384,32768,65536,131072,262144,524288,1048576")
    a = ap.parse_args()
    rng = np.random.default_rng(5)
    out = []
    print(f"{'n':>8s} {'stock_topk':>10s} {'fast_topk':>9s} {'stock_t+c':>9s} {'fast_fused':>10s}  (ms/op)")
    for n in map(int, a.ns.split(",")):
        q = mx.array(rng.standard_normal((1, NH, DI)).astype(np.float16))
        w = mx.array((rng.standard_normal((1, NH)) * 0.05).astype(np.float16))
        kk = mx.array((rng.standard_normal((1, n, DI)) * 0.5).astype(np.float16))
        lens = mx.array([n], dtype=mx.int32)
        sc = C.index_scores(q, kk, w, lens, n, rt=True); mx.eval(sc)
        r = dict(n=n,
                 stock_topk=med(lambda: [stock_topk(sc, lens) for _ in range(FAN)]),
                 fast_topk=med(lambda: [IX.topk_mt(sc, lens, K) for _ in range(FAN)]),
                 stock_tc=med(lambda: [(stock_topk(sc, lens), stock_cand(sc, lens)) for _ in range(FAN)]),
                 fast_fused=med(lambda: [IX.select_fused(sc, lens, K, BLK, BS) for _ in range(FAN)]))
        print(f"{n:8d} {r['stock_topk']:10.3f} {r['fast_topk']:9.3f} {r['stock_tc']:9.3f} {r['fast_fused']:10.3f}", flush=True)
        out.append(r)
        del kk, sc
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
