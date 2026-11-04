"""Synced GPU+host wall of the DS4.1 spec round parts at one context: plain step, draft chain of depth 1..8 (be.draft, one call), verify of
R = 1..9 rows (spec_verify_bc + commit of 0 drafts, so the cache stays at the same length). Medians of N calls after 3 warm calls.
usage: spec_cost_ds41.py [P=32768] [N=12]"""
import sys, os, time, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mlx.core as mx
from kyojin_mlx.serve import backends as B

P = int(sys.argv[1]) if len(sys.argv) > 1 else 32768
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
be, cfg = B.load_backend(os.path.expanduser("~/ds41/model"), mtp=True, ced=True, log=lambda *a: None)      # wires the buffers first, as served
ids = np.random.default_rng(0).integers(16, 120000, P + 64).tolist()
cache, d = be.make_cache(), be.new_draft()
t = time.perf_counter()
for i in range(0, P, 8192):
    be.prefill_chunk(cache, ids[i:min(P, i + 8192)], draft=d)
mx.synchronize()
print(f"prefill {P} {P / (time.perf_counter() - t):.0f} tok/s", flush=True)
bc = be.batch([cache])


def med(fn):
    for _ in range(3):
        fn()
    ts = []
    for _ in range(N):
        t = time.perf_counter(); fn(); mx.synchronize(); ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)


def verify(r):
    def f():
        ps = be.spec_verify_bc(bc, [ids[P:P + r]])
        mx.eval(ps.logits)
        ps.commit([0])
        be.settle(bc, [d])
    return f


def plain():
    lg = be.step(bc, mx.array([ids[P]], dtype=mx.int32))
    mx.eval(lg)


for r in range(1, 10):
    print(f"verify R={r} {med(verify(r)):.2f} ms", flush=True)
for k in range(1, 9):
    print(f"draft depth={k} {med(lambda: be.draft(d, ids[P], P + 1, k)):.2f} ms", flush=True)
print(f"plain step {med(plain):.2f} ms (cache grows by {N + 3})", flush=True)
