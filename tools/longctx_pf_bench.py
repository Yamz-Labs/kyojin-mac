"""One prefill indexer layer at DS4.1 dims (T query rows x Tc past keys, 32 x 128, top-k 512): stock (scores_mm [T, Tc] + full argsort)
vs longctx.pf_topk (key tiles, exact per-tile top-k, stable merge). Checks bit equality, prints ms and the stock peak memory.
Run: PYTHONPATH=.:vlm_pkg python tools/longctx_pf_bench.py --T 8192 --tcs 16384,65536,131072   (Tc 131072: ~10 GB peak for stock)"""
import argparse, time
import numpy as np
import mlx.core as mx
from kyojin_mlx import ds_pf_idx as PI, longctx as LC


def run(fn, reps):
    mx.eval(fn())
    ts = []
    for _ in range(reps):
        mx.synchronize(); t = time.perf_counter(); mx.eval(fn()); ts.append((time.perf_counter() - t) * 1e3)
    return sorted(ts)[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=8192)
    ap.add_argument("--tcs", default="16384,32768,65536")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tile", type=int, default=LC.PF_TILE)
    a = ap.parse_args()
    rng = np.random.default_rng(1)
    T = a.T
    q = mx.array(rng.standard_normal((1, T, 32, 128)).astype(np.float16))
    w = mx.array((rng.standard_normal((1, T, 32)) * 0.05).astype(np.float16))
    print(f"{'T':>5s} {'Tc':>7s} {'stock_ms':>9s} {'tiled_ms':>9s} {'stock_GB':>8s} {'tiled_GB':>8s} same")
    for Tc in map(int, a.tcs.split(",")):
        keys = mx.array((rng.standard_normal((1, Tc, 128)) * 0.5).astype(np.float16))
        start = 2 * Tc - T                                   # ratio 2: the chunk ends at the last key
        stock = lambda: mx.sort(mx.argsort(-PI.scores_mm(q, keys, w, start, 2), axis=-1)[..., :512], axis=-1)
        tiled = lambda: LC.pf_topk(q, keys, w, start, 2, 512, None, tile=a.tile)
        same = bool(mx.array_equal(stock().astype(mx.int32), tiled()))
        mx.reset_peak_memory(); ts = run(stock, a.reps); ps = mx.get_peak_memory() / 2**30
        mx.reset_peak_memory(); tt = run(tiled, a.reps); pt = mx.get_peak_memory() / 2**30
        print(f"{T:5d} {Tc:7d} {ts:9.1f} {tt:9.1f} {ps:8.2f} {pt:8.2f} {same}", flush=True)
        del keys


if __name__ == "__main__":
    main()
