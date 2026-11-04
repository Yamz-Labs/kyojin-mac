"""ms per call of idxfast.topk_mt (old) vs idxblk.topk_blocks (new) on synthetic fp16 scores. Memory stays under ~1.5 GiB.
usage: python tools/bench_idxblk.py [k]"""
import sys
import time

import mlx.core as mx

from kyojin_mlx import idxblk as B
from kyojin_mlx import idxfast as F

k = int(sys.argv[1]) if len(sys.argv) > 1 else 512


def bench(fn, n=20):
    for _ in range(3):
        mx.eval(fn())
    t = time.perf_counter()
    for _ in range(n):
        mx.eval(fn())
    return (time.perf_counter() - t) / n * 1e3


mx.random.seed(0)
for N, tag in ((1, "decode N=1"), (4, "verify N=4"), (128, "prefill N=128")):
    for Tm in (128 * 1024, 512 * 1024, 1024 * 1024):
        s = (mx.random.normal((N, Tm)) * 3).astype(mx.float16)
        ln = mx.full((N,), Tm, dtype=mx.int32)
        mx.eval(s, ln)
        old = bench(lambda: F._run(s, ln, k, 2, 8, False)[0])
        new = bench(lambda: B.topk_blocks(s, ln, k))
        same = bool(mx.array_equal(F._run(s, ln, k, 2, 8, False)[0], B.topk_blocks(s, ln, k)).item())
        print(f"{tag:14s} rows(keys)={Tm:8d} k={k}: old {old:8.3f} ms  new {new:8.3f} ms  x{old / new:5.2f}  same={same}", flush=True)
        del s
