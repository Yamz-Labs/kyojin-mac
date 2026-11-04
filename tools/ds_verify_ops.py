"""Per-module GPU ms of the served DS4.1 decode: plain step (1 row) and verify round (1 + D rows), real pack, served backend (load_vlm + CED + MTP).
Phase 1 times bare rounds (no sync inside); phase 2 wraps the class __call__ of every child module of a decoder layer and of its ffn with a sync
before and after, so each class gets its own ms (nested classes count inside their parent too; the sum of the wraps is an upper bound).
usage: ds_verify_ops.py [PROMPT=1024] [D=3] [ROUNDS=24]"""
import sys, os, time, collections, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
from kyojin_mlx.serve import backends as B

P = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
D = int(sys.argv[2]) if len(sys.argv) > 2 else 3
N = int(sys.argv[3]) if len(sys.argv) > 3 else 24
B.wire_memory()
be, cfg = B.load_vlm(os.path.expanduser("~/ds41/model"), mtp=True, ced=True)
lm = be.lm
rng = np.random.default_rng(0)
ids = rng.integers(16, 120000, P + max(4096, int(os.environ.get("PF", "8192")))).tolist()   # the prefill profile reads ids[:PF]: short ids timed fewer rows
cache, d = be.make_cache(), be.new_draft()
be.prefill_chunk(cache, ids[:P], draft=d)
bc = be.batch([cache])
pos = [P]


def plain():
    t = ids[pos[0]]
    lg = be.step(bc, mx.array([t], dtype=mx.int32))
    mx.eval(lg, *[a for c in bc if hasattr(c, "stacked_arrays") for a in c.stacked_arrays()])
    pos[0] += 1


def verify():
    toks = ids[pos[0]:pos[0] + D + 1]
    ps = be.spec_verify_bc(bc, [toks])
    mx.eval(ps.logits)
    ps.commit([min(2, D)])
    be.settle(bc)
    mx.synchronize()
    pos[0] += min(2, D) + 1


def bare(fn, n):
    for _ in range(4):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)


print(f"BARE plain {bare(plain, N):.2f} ms  verify{D + 1} {bare(verify, N):.2f} ms", flush=True)

TT, CN = collections.defaultdict(float), collections.Counter()
blk = lm.layers[3]
kinds = {}
for name, m in list(blk.children().items()) + [(f"ffn.{k}", v) for k, v in getattr(blk, "ffn", blk).children().items()]:
    if hasattr(m, "__call__") and hasattr(type(m), "__call__") and type(m).__module__ != "builtins" and not isinstance(m, (list, dict)):
        kinds.setdefault(type(m), name)
kinds.setdefault(type(blk), "layer")
def _arrs(r):
    return [x for _, x in tree_flatten(r if isinstance(r, (list, tuple, dict)) else [r]) if isinstance(x, mx.array)]


for cls, name in kinds.items():
    o = cls.__call__
    def w(self, *a, _o=o, _n=f"{name}:{cls.__name__}", **k):
        mx.eval([x for x in a if isinstance(x, mx.array)])
        t = time.perf_counter(); r = _o(self, *a, **k); mx.eval(_arrs(r)); TT[_n] += time.perf_counter() - t; CN[_n] += 1
        return r
    cls.__call__ = w
PF = int(os.environ.get("PF", "8192"))                              # prompt rows of the prefill profile (one chunk; never >= 16384)
assert PF < 16384 and len(ids) >= PF


def prefill():
    c, dd = be.make_cache(), be.new_draft()
    t = time.perf_counter(); be.prefill_chunk(c, ids[:PF], draft=dd); mx.synchronize()
    return (time.perf_counter() - t) * 1e3


print(f"BARE prefill {PF} rows: {[round(prefill()) for _ in range(3)]} ms (first = new shape)", flush=True)
for label, fn in ((f"verify{D + 1}", verify), ("plain", plain), ("prefill", None)):
    if fn is None:
        try:
            prefill(); TT.clear(); CN.clear()
            wall = prefill()
        except Exception as e:
            print(f"WRAPPED prefill skipped: {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        print(f"WRAPPED prefill {PF} rows {wall:.0f} ms", flush=True)
        for k, v in sorted(TT.items(), key=lambda kv: -kv[1]):
            print(f"  {k:48s} {v * 1e3:8.1f} ms  calls {CN[k]}", flush=True)
        continue
    try:
        for _ in range(3):
            fn()
    except Exception as e:                                          # a compiled region traced the wrapped modules
        print(f"WRAPPED {label} skipped: {type(e).__name__}: {str(e)[:120]}", flush=True)
        continue
    TT.clear(); CN.clear()
    t0 = time.perf_counter()
    for _ in range(N):
        fn()
    wall = (time.perf_counter() - t0) * 1e3 / N
    print(f"WRAPPED {label} {wall:.2f} ms/round", flush=True)
    for k, v in sorted(TT.items(), key=lambda kv: -kv[1]):
        print(f"  {k:48s} {v * 1e3 / N:8.2f} ms/round  calls/round {CN[k] / N:.0f}", flush=True)
