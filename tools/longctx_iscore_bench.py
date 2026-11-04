"""Decode index-score kernel: stock _ISCORE vs register-blocked _ISCORE_KB (bit-identical), ms per call vs the key-byte roof.
Usage: PYTHONPATH=. python tools/longctx_iscore_bench.py [n_keys ...]   (one row, 32 heads x 128, fp16 keys; 819 GB/s roof)"""
import sys
import time

import mlx.core as mx
import numpy as np

from kyojin_mlx import dsv4_core as C


def bench(fn, it=50):
    for _ in range(5):
        mx.eval(fn())
    t = time.perf_counter()
    for _ in range(it):
        mx.eval(fn())
    return (time.perf_counter() - t) / it * 1e3


rng = np.random.default_rng(0)
for n in [int(a) for a in sys.argv[1:]] or [4096, 32768, 131072, 524288, 1048576]:
    q = mx.array(rng.standard_normal((1, 32, 128)).astype(np.float16))
    keys = mx.array((rng.standard_normal((1, n, 128)) * 0.5).astype(np.float16))
    w = mx.array((rng.standard_normal((1, 32)) * 0.1).astype(np.float16))
    lens = mx.array([n], dtype=mx.int32)
    res = {}
    for kb in (False, True):
        C.ISCORE_KB = kb
        res[kb] = (bench(lambda: C.index_scores(q, keys, w, lens, n, None, None, True)), np.array(C.index_scores(q, keys, w, lens, n).astype(mx.float32)))
    roof = n * 128 * 2 / 819e9 * 1e3
    print(f"n={n:8d}  stock {res[False][0]:.3f} ms  kb {res[True][0]:.3f} ms  roof {roof:.3f} ms  equal {np.array_equal(res[False][1], res[True][1])}")
