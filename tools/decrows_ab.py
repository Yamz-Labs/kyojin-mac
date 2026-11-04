"""decrows_ab.py PACK [BLOCKS]: decrows (1..8 rows add+norm / fused projections / attention front end) vs the stock forward, same process.
Bit check: hidden states and every cache array equal, chained (1 sequence x S rows) and batched (B users x 1 row), rows 1..8. Speed: synced ms per forward, interleaved blocks.
Prints DECROWS_AB json + table."""
import os as _os; _os.environ.setdefault("KYOJIN_DECROWS", "1")
import json, os, statistics, sys, time
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), s) for s in ("..", ".")]
import mlx.core as mx
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx import decrows as DR
from count_launches import graph_census

PACK = sys.argv[1]
BLOCKS = int(sys.argv[2]) if len(sys.argv) > 2 else 7
be, _ = load_backend(PACK, mtp=False, log=lambda *a: None)
tm = be.lm.model
assert getattr(tm, "_kr_stock", None) is not None, "decrows not installed"
base = be.make_cache()
be.prefill_chunk(base, list(range(1000, 1161)))


def _flat(st):
    if st is None:
        return []
    if isinstance(st, (list, tuple)):
        return [a for s in st for a in _flat(s)]
    if hasattr(st, "caches"):
        return _flat([c.state for c in st.caches])
    return [st] if isinstance(st, mx.array) else []


def arrays(cache):
    out = []
    for c in cache:
        out += _flat(c) if hasattr(c, "caches") else _flat(c.state if hasattr(c, "state") else c.cache)
    return out


def run(fn, mode, n, toks):
    cs = [be.fork(base) for _ in range(n if mode == "batched" else 1)]
    cache = be.batch(cs) if mode == "batched" else cs[0]
    t = mx.array([toks]) if mode == "chained" else mx.array(toks)[:, None]
    h = fn(t, cache)
    return h, cache


def ref(t, c):
    """the stock forward with every decrows piece off (the class-level attention / MoE patches included)"""
    DR.ENABLED = False
    try:
        return tm._kr_stock(t, c)
    finally:
        DR.ENABLED = True


res = {"bit": {}, "ms": {}, "dispatches": {}}
for mode in os.environ.get("AB_MODES", "chained,batched").split(","):
    for n in [int(r) for r in os.environ.get("AB_ROWS", "1,2,3,4,5,6,7,8").split(",")]:
        toks = [2000 + 7 * i for i in range(n)]
        h0, c0 = run(ref, mode, n, toks)
        h1, c1 = run(lambda t, c: tm(t, c), mode, n, toks)
        mx.eval(h0, h1, arrays(c0), arrays(c1))
        a0, a1 = arrays(c0), arrays(c1)
        ok = mx.array_equal(h0, h1).item() and len(a0) == len(a1) and all(x.shape == y.shape and mx.array_equal(x, y).item() for x, y in zip(a0, a1))
        res["bit"][f"{mode}{n}"] = bool(ok)
        # speed
        ts = {"stock": [], "decrows": []}
        for blk in range(BLOCKS):
            for name, fn in (("stock", ref), ("decrows", lambda t, c: tm(t, c))):
                cs = [be.fork(base) for _ in range(n if mode == "batched" else 1)]
                cache = be.batch(cs) if mode == "batched" else cs[0]
                t = mx.array([toks]) if mode == "chained" else mx.array(toks)[:, None]
                mx.eval(arrays(cache))
                o = be._head(fn(t, cache)[:, -1:]) if mode == "chained" else be._head(fn(t, cache))   # graph is built lazily: DR.ENABLED only matters while it is built
                t0 = time.perf_counter(); mx.eval(o); ts[name].append((time.perf_counter() - t0) * 1e3)
        res["ms"][f"{mode}{n}"] = {k: round(min(v), 3) for k, v in ts.items()}
        cs = [be.fork(base) for _ in range(n if mode == "batched" else 1)]
        cache = be.batch(cs) if mode == "batched" else cs[0]
        t = mx.array([toks]) if mode == "chained" else mx.array(toks)[:, None]
        res["dispatches"][f"{mode}{n}"] = {"stock": graph_census(be._head(ref(t, be.fork(base) if mode == "chained" else be.batch([be.fork(base) for _ in range(n)]))), n)["dispatches"],
                                           "decrows": graph_census(be._head(tm(t, cache)), n)["dispatches"]}
print("DECROWS_AB " + json.dumps(res | {"stats": DR.STATS}))
print("| case | bit-equal | stock ms | decrows ms | delta ms | dispatches stock -> decrows |\n|---|---|---|---|---|---|")
for k in res["bit"]:
    m = res["ms"][k]; d = res["dispatches"][k]
    print(f"| {k} | {res['bit'][k]} | {m['stock']} | {m['decrows']} | {m['decrows'] - m['stock']:+.3f} | {d['stock']} -> {d['decrows']} |")
