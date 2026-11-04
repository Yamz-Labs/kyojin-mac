"""pfroof2 synthetic bench, Metal only: one prefill-indexer layer, stock full width vs causal width (radix top-k and argsort top-k), indices compared.
usage: python tools/bench_pfroof2.py [T=8192] [ratio=1] [start=0] [reps=3]   (T <= 8192: the hard chunk cap; run T=4096 first)
Prints ms per layer (min of reps, scores + top-k, blocks of KYOJIN_IDX_QBLOCK rows), peak GiB, and whether the three index arrays are equal."""
import os, sys, time, numpy as np, mlx.core as mx
from kyojin_mlx import ds_pf_idx as I, idx_topk as IT, pfroof2 as PR
T = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
ratio = int(sys.argv[2]) if len(sys.argv) > 2 else 1
start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
reps = int(sys.argv[4]) if len(sys.argv) > 4 else 3
assert T <= 8192, "hard cap 8192 rows per chunk"
NH, K = 32, 512
Tc = (start + T) // ratio
rng = np.random.default_rng(0)
q = mx.array(rng.standard_normal((1, T, NH, 128)).astype(np.float16))
keys = mx.array(rng.standard_normal((1, Tc, 128)).astype(np.float16))
w = mx.array((rng.standard_normal((1, T, NH)) * 0.05).astype(np.float16))
mx.eval(q, keys, w)


def stock():
    return IT.scan(lambda t0, n: I.scores_range(q, keys, w, start, ratio, t0, n), T, IT.qblock(T), lambda t0, s: IT.topk(s, K))


def narrow(radix):
    os.environ[PR.ENV_TOPK] = "1" if radix else "0"
    return IT.scan(lambda t0, n: I.scores_range(q, keys[:, :PR.causal_width(start, t0, n, ratio, Tc, K)], w, start, ratio, t0, n), T, IT.qblock(T),
                   lambda t0, s: PR.select_topk(s, K))


res = {}
for name, fn in (("stock", stock), ("narrow+radix", lambda: narrow(True)), ("narrow+argsort", lambda: narrow(False))):
    mx.reset_peak_memory()
    best, out = 9e9, None
    for _ in range(reps + 1):
        t = time.perf_counter(); out = fn(); mx.eval(out); best = min(best, time.perf_counter() - t)
    res[name] = np.array(out)
    print(f"{name:16s} T={T} ratio={ratio} start={start} Tc={Tc}: {1e3 * best:8.1f} ms/layer   peak {mx.get_peak_memory() / 2**30:.2f} GiB")
print("indices equal:", np.array_equal(res["stock"], res["narrow+radix"]), np.array_equal(res["stock"], res["narrow+argsort"]))
