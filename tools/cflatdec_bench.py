"""Decode cost against context, one op at a time, synthetic data, no model (cflatdec). Bounded: the process memory limit is 3.5 GiB (mx.set_memory_limit), the peak
(mx.get_peak_memory) is checked after every group and the run aborts above the limit; every group frees its arrays and clears the cache before the next L.

For every L (tokens of context) and R (rows of the pass) it prints ms per op (median / min of --reps, after --warm warm-ups):
  scores_stock   dsv4_core.index_scores (_ISCORE) over the keys of a ratio-2 layer (L/2 keys) and of the candidate source (L keys), R rows        [term T1]
  scores_kb      the same with ISCORE_KB on (register-blocked), printed with bit equality against the stock scores
  topk_radix / topk_tree   idxfast.topk_mt over those scores with topk_tree off / on, bit equality of the indices                               [term T2]
  filter_*       decfilter: approx (matmul + reduce), block topk, gather, exact rescore, total select, total + the certificate read (what core pays),
                 the fraction of rows whose certificate holds, equality with the stock selection on the rows that hold, and the bound check
                 (`bound slack` = max |approx - exact stored score| / (delta + ulp): must stay below 1, it is the assumption the matmul accumulates in fp32)
  put            dsv4_core._put (the real cache append) into a [1, cap, 512] fp16 pool, synchronous and pipelined (the previous step still in flight)
                 -> flat in L means MLX donates the buffer, linear means a full pool copy per append                                              [term T7]
  copy_baseline  one full pool copy (what an O(L) copy costs at this L)
  gather         the 512 selected pool rows of each of R rows (bounded read)                                                                      [term T5]
  rope_build / rope_gather   longctx index rope table at length L (first call builds, second is the cached slice) and the gather of R positions   [term T6]
Context for the arithmetic: docs/reports/cflatdec_REPORT.md. Hold /tmp/ultra_heavy.lock while it runs on the Ultra GPU (taken here unless --no-lock).

usage: PYTHONPATH=. python tools/cflatdec_bench.py [--lens 131072,524288,1048576] [--rows 1,6] [--dist normal,cluster] [--reps 5] [--ratio 2] [--json out.json]
       PYTHONPATH=. python tools/cflatdec_bench.py --smoke      # tiny sizes, any device, runs the parts that do not need Metal (used to check the script itself)"""
import argparse
import gc
import json
import os
import statistics
import sys
import time
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

LIMIT = int(3.5 * 2 ** 30)
ap = argparse.ArgumentParser()
ap.add_argument("--lens", default="131072,524288,1048576", help="context lengths in tokens")
ap.add_argument("--rows", default="1,6")
ap.add_argument("--dist", default="normal,cluster", help="key distributions for the filter: normal (worst case for any pruning), cluster")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--warm", type=int, default=2)
ap.add_argument("--ratio", type=int, default=2, help="compress ratio of the layer whose pool/keys are built (2: layers 2, 8, 14; 1: layer 20)")
ap.add_argument("--heads", type=int, default=32)
ap.add_argument("--dim", type=int, default=128)
ap.add_argument("--topk", type=int, default=512)
ap.add_argument("--pool-dim", type=int, default=512)
ap.add_argument("--cand-blocks", type=int, default=2048, help="candidate_topk_blocks of the source layer")
ap.add_argument("--cand-bs", type=int, default=8)
ap.add_argument("--no-lock", action="store_true")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--json", default="")
a = ap.parse_args()

METAL = mx.metal.is_available()
if a.smoke:
    a.lens, a.rows, a.reps, a.warm, a.topk = "4096,8192", "1,2", 2, 1, 32
mx.set_memory_limit(LIMIT)
RESULTS = []
LOCK = None
if METAL and not a.no_lock and not a.smoke:
    import fcntl
    LOCK = open("/tmp/ultra_heavy.lock", "w")
    fcntl.flock(LOCK, fcntl.LOCK_EX)


def peak_check(tag):
    p = mx.get_peak_memory()
    if p > LIMIT:
        print(f"ABORT {tag}: peak {p / 2 ** 30:.2f} GiB above the 3.5 GiB bound", flush=True)
        sys.exit(2)
    return p


def clean():
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def bench(fn, reps=None, warm=None):
    """(median ms, min ms) of fn(); fn returns what to evaluate."""
    for _ in range(a.warm if warm is None else warm):
        mx.eval(fn())
    ts = []
    for _ in range(a.reps if reps is None else reps):
        t = time.perf_counter()
        mx.eval(fn())
        ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts), min(ts)


def emit(op, L, R, med, mn, note="", **extra):
    print(f"{op:<22} L={L:>8} R={R} {med:9.3f} ms (min {mn:8.3f}) {note}", flush=True)
    RESULTS.append(dict(op=op, L=L, R=R, ms=med, ms_min=mn, note=note, **extra))


def make_keys(n, dist, seed):
    rng = np.random.default_rng(seed)
    if dist == "cluster":
        c = rng.standard_normal((64, a.dim)).astype(np.float32)
        k = c[rng.integers(0, 64, size=n)] + 0.15 * rng.standard_normal((n, a.dim)).astype(np.float32)
    else:
        k = rng.standard_normal((n, a.dim)).astype(np.float32)
    return mx.array(k.astype(np.float16))[None]


def make_q(R, seed):
    rng = np.random.default_rng(seed)
    q = mx.array(rng.standard_normal((R, a.heads, a.dim)).astype(np.float16))
    w = mx.array(np.abs(rng.standard_normal((R, a.heads))).astype(np.float16) * 0.05)
    return q, w


# ------------------------------------------------------------------------------------------------------------------- scores / top-k (Metal)
def group_scores(L, R, Lk, tag):
    from kyojin_mlx import dsv4_core as DC, idxfast as IX
    keys = make_keys(Lk, "normal", 1)
    q, w = make_q(R, 2)
    lens = mx.array([Lk - R + 1 + u for u in range(R)], dtype=mx.int32)
    mx.eval(keys, q, w, lens)
    sess = DC._identity(R)
    DC.ISCORE_KB = False
    f = lambda: DC.index_scores(q, keys, w, lens, Lk, None, sess, rt=False)
    med, mn = bench(f)
    ref = f()
    emit(f"scores_stock[{tag}]", L, R, med, mn, f"{Lk} keys")
    DC.ISCORE_KB, DC.ISCORE_KB_MIN = True, 0
    g = lambda: DC.index_scores(q, keys, w, lens, Lk, None, sess, rt=False)
    med, mn = bench(g)
    eq = bool(mx.array_equal(g(), ref).item())
    emit(f"scores_kb[{tag}]", L, R, med, mn, f"bit-equal to stock: {eq}")
    DC.ISCORE_KB = False
    k = min(a.topk, Lk // 4)
    IX.TOPK_TREE = False
    med, mn = bench(lambda: IX.topk_mt(ref, lens, k))
    base = IX.topk_mt(ref, lens, k)
    emit(f"topk_radix[{tag}]", L, R, med, mn, f"k={k}")
    IX.TOPK_TREE = True
    try:
        med, mn = bench(lambda: IX.topk_mt(ref, lens, k))
        eq = bool(mx.array_equal(IX.topk_mt(ref, lens, k), base).item())
        emit(f"topk_tree[{tag}]", L, R, med, mn, f"bit-equal to radix: {eq}")
    except Exception as e:                                   # the tree has its own gates (rows, width)
        print(f"topk_tree[{tag}] L={L} R={R}: not run ({e})", flush=True)
    IX.TOPK_TREE = False
    return keys, q, w, lens, ref


# ------------------------------------------------------------------------------------------------------------------- the filter
def group_filter(L, R, Lk, dist, tag):
    from kyojin_mlx import decfilter as DF, dsv4_core as DC, idxfast as IX
    k = min(a.topk, Lk // 4)
    B = DF.BLOCK
    keys = make_keys(Lk + B, dist, 3)
    q, w = make_q(R, 4)
    if dist == "cluster":                                  # the first query aligned with a cluster direction so that there is something to find
        q = q * 0.5
    lens_l = [Lk - R + 1 + u for u in range(R)]
    T = -(-max(lens_l) // B) * B
    lens = mx.array(lens_l, dtype=mx.int32)
    mx.eval(keys, q, w)
    norms = DF.Norms(1).get(keys, [0], T)
    mx.eval(norms)
    sess = [0] * R
    tile = DF.TILE
    med, mn = bench(lambda: DF.approx_all(q, keys, w, sess, lens_l, T, norms, tile))
    emit(f"filter_approx[{tag},{dist}]", L, R, med, mn, f"{T} keys, tile {tile}")
    Mb = min(T // B, DF.BUDGET * k)
    A = mx.random.uniform(shape=(R, T // B))
    nbv = (lens + (B - 1)) // B
    med, mn = bench(lambda: IX.topk_mt(A, nbv, Mb))
    emit(f"filter_topk_blocks[{tag},{dist}]", L, R, med, mn, f"fp32 {T // B} blocks, k={Mb}")
    gk = mx.zeros((R, Mb * B, a.dim), dtype=mx.float16)
    med, mn = bench(lambda: DC.index_scores(q, gk, w, mx.full((R,), Mb * B, dtype=mx.int32), Mb * B, None, None, rt=False))
    emit(f"filter_rescore[{tag},{dist}]", L, R, med, mn, f"{Mb * B} keys per row")
    f = lambda: DF.select(q, keys, w, sess, lens_l, T, k, DF.default_exact, DF.default_topk, norms)
    med, mn = bench(lambda: f()[0])
    emit(f"filter_select[{tag},{dist}]", L, R, med, mn, f"blocks {Mb} of {T // B}, no host read")
    t0 = []
    for _ in range(a.reps):
        t = time.perf_counter()
        idx, ok = f()
        okv = bool(mx.all(ok).item())
        mx.eval(idx)
        t0.append((time.perf_counter() - t) * 1e3)
    emit(f"filter_select+cert[{tag},{dist}]", L, R, statistics.median(t0), min(t0), "what core pays per filtered layer (certificate read = a sync)")
    # exactness against the stock chain, on the rows whose certificate holds; bound check on the first row
    idx, ok = f()
    sc = DC.index_scores(q, keys[:, :Lk], w, lens, Lk, None, DC._identity(R), rt=False)
    ref = IX.topk_mt(sc, lens, k)
    okn = np.array(ok)
    eq = [bool(mx.array_equal(idx[u], ref[u]).item()) for u in range(R) if okn[u]]
    a0, d0 = DF.approx_session(q[0:1], keys[0, :T], w[0:1], norms[0, :T])
    s0 = sc[0, :lens_l[0]].astype(mx.float32)
    ulp = 2.0 ** -10 * mx.maximum(mx.abs(s0), 1e-4)
    slack = float(mx.max(mx.abs(a0[0, :lens_l[0]] - s0) / (d0[0, :lens_l[0]] + ulp)).item())
    print(f"filter_check[{tag},{dist}] L={L} R={R}: certificate holds on {int(okn.sum())}/{R} rows, selection equals the stock chain on all of them: {all(eq)}, "
          f"bound slack {slack:.4f} (< 1 required)", flush=True)
    RESULTS.append(dict(op=f"filter_check[{tag},{dist}]", L=L, R=R, ok_rows=int(okn.sum()), equal=all(eq), bound_slack=slack))


def group_source(L, R, dist):
    """The candidate source (L keys): filter with the block mask against idxfast.select_fused over the stock scores."""
    from kyojin_mlx import decfilter as DF, dsv4_core as DC, idxfast as IX
    k, kb, B = min(a.topk, L // 4), a.cand_blocks, a.cand_bs
    keys = make_keys(L + B, dist, 6)
    q, w = make_q(R, 7)
    lens_l = [L - R + 1 + u for u in range(R)]
    T = -(-max(lens_l) // B) * B
    lens = mx.array(lens_l, dtype=mx.int32)
    norms = DF.Norms(1).get(keys, [0], T)
    mx.eval(keys, q, w, norms)
    sess = [0] * R
    f = lambda: DF.select(q, keys, w, sess, lens_l, T, k, DF.default_exact, DF.default_topk, norms, block=B, kb=kb, cap=T)
    med, mn = bench(lambda: f()[0])
    emit(f"filter_select[src,{dist}]", L, R, med, mn, f"blocks {min(T // B, DF.BUDGET * max(k, kb))} of {T // B}, mask included")
    t0 = []
    for _ in range(a.reps):
        t = time.perf_counter()
        idx, mask, ok = f()
        okv = bool(mx.all(ok).item())
        mx.eval(idx, mask)
        t0.append((time.perf_counter() - t) * 1e3)
    emit(f"filter_select+cert[src,{dist}]", L, R, statistics.median(t0), min(t0), "with the certificate read")
    sc = DC.index_scores(q, keys[:, :T], w, lens, T, None, DC._identity(R), rt=False)
    ri, rm = IX.select_fused(sc, lens, k, kb, B)
    idx, mask, ok = f()
    okn = np.array(ok)
    eq = [bool(mx.array_equal(idx[u], ri[u]).item()) and bool(mx.array_equal(mask[u, :lens_l[u]], rm[u, :lens_l[u]]).item()) for u in range(R) if okn[u]]
    print(f"filter_check[src,{dist}] L={L} R={R}: certificate holds on {int(okn.sum())}/{R} rows, idx and mask equal select_fused on all of them: {all(eq)}", flush=True)
    RESULTS.append(dict(op=f"filter_check[src,{dist}]", L=L, R=R, ok_rows=int(okn.sum()), equal=all(eq)))


# ------------------------------------------------------------------------------------------------------------------- cache terms (any device)
def group_put(L, R, Lp):
    from kyojin_mlx import dsv4_core as DC
    D = a.pool_dim
    while (max(a.reps, 4) + 4) * (Lp + 4096) * D * 2 > 0.9 * LIMIT and D > 32:       # a full-copy arm can hold the live pool + one copy per queued append: shrink the row width to stay under the bound (bytes per row scale linearly)
        D //= 2
    cap0 = -(-Lp // 256) * 256 + 4096
    pool = [mx.zeros((1, cap0, D), dtype=mx.float16)]
    mx.eval(pool[0])
    vals = mx.ones((R, D), dtype=mx.float16)

    def em(n):
        return SimpleNamespace(need=n + R, sess_arr=mx.zeros((R,), dtype=mx.int32), widx_arr=mx.arange(n, n + R, dtype=mx.int32))

    n = Lp
    for _ in range(a.warm):
        DC._put(pool, 0, 1, (D,), mx.float16, em(n), vals, 256)
        mx.eval(pool[0])
    ts = []
    for _ in range(a.reps):                                  # synchronous: nothing else holds the buffer when the scatter is built
        n += R
        t = time.perf_counter()
        DC._put(pool, 0, 1, (D,), mx.float16, em(n), vals, 256)
        mx.eval(pool[0])
        ts.append((time.perf_counter() - t) * 1e3)
    emit("put_sync", L, R, statistics.median(ts), min(ts), f"pool {Lp} rows x {D} fp16 = {Lp * D * 2 / 2 ** 20:.0f} MiB" + ("" if D == a.pool_dim else f" (width scaled {a.pool_dim}->{D} for the 3.5 GiB bound; x{a.pool_dim // D} for the real row)"))
    K = max(a.reps, 4)
    inflight = None
    t = time.perf_counter()
    for _ in range(K):                                       # pipelined: back to back, each step's reader of the OLD buffer still queued when the next append is built
        n += R
        old = pool[0]
        mx.async_eval(old[:, n - 512:n].sum())
        DC._put(pool, 0, 1, (D,), mx.float16, em(n), vals, 256)
        mx.async_eval(pool[0])
        if inflight is not None:                             # at most two appends in flight: bounds the pools a non-donating copy keeps alive
            mx.eval(inflight)
        inflight = pool[0]
    mx.eval(pool[0])
    per = (time.perf_counter() - t) * 1e3 / K
    emit("put_pipelined", L, R, per, per, f"mean of {K} back-to-back appends with the previous reader in flight")
    med, mn = bench(lambda: pool[0][mx.arange(1)])
    emit("copy_baseline", L, R, med, mn, "full pool copy (the gather copy of _CANON_POOLS)")
    idx = mx.array(np.sort(np.random.default_rng(5).choice(Lp, size=(R, min(a.topk, Lp)), replace=True), axis=-1).astype(np.int32))
    med, mn = bench(lambda: mx.take(pool[0], idx, axis=1))
    emit("gather", L, R, med, mn, f"{idx.shape[1]} rows per row (bounded read)")


def group_rope(L, R):
    """The vendor index rope table is `cos(float32(i) * inv_freq)` in float64 rounded to float32 (longctx.py module doc); this times that numpy work at length L (a
    stand-in for the vendor build, which needs the vendor module) and the gather of R positions from the finished table."""
    rd = 64
    inv = (1.0 / (160000.0 ** (np.arange(0, rd, 2, dtype=np.float64) / rd))).astype(np.float32)
    t = time.perf_counter()
    ang = np.arange(L, dtype=np.float32)[:, None].astype(np.float64) * inv[None, :].astype(np.float64)
    cos, sin = mx.array(np.cos(ang).astype(np.float32)), mx.array(np.sin(ang).astype(np.float32))
    mx.eval(cos, sin)
    ms = (time.perf_counter() - t) * 1e3
    emit("rope_build", L, R, ms, ms, f"numpy float64 table of {L} x {rd // 2} (once per doubling; table {2 * L * (rd // 2) * 4 / 2 ** 20:.0f} MiB)")
    pa = mx.array([L - 1 - u for u in range(R)], dtype=mx.int32)
    med, mn = bench(lambda: (cos[pa], sin[pa]))
    emit("rope_gather", L, R, med, mn, "R positions")
    del ang, cos, sin


def main():
    print(f"device {mx.default_device()} metal {METAL}  limit 3.5 GiB  reps {a.reps}", flush=True)
    for L in [int(x) for x in a.lens.split(",")]:
        Lk = L // a.ratio
        for R in [int(x) for x in a.rows.split(",")]:
            tag = f"r{a.ratio}"
            if METAL:
                for kind, n in ((tag, Lk), ("src", L)):
                    group_scores(L, R, n, kind)
                    peak_check(f"scores {kind} L={L} R={R}")
                    clean()
                for dist in a.dist.split(","):
                    group_filter(L, R, Lk, dist, tag)
                    peak_check(f"filter {dist} L={L} R={R}")
                    clean()
                    group_source(L, R, dist)
                    peak_check(f"filter source {dist} L={L} R={R}")
                    clean()
            group_put(L, R, Lk)
            peak_check(f"put L={L} R={R}")
            clean()
        try:
            group_rope(L, max(int(x) for x in a.rows.split(",")))
        except Exception as e:
            print(f"rope L={L}: {e}", flush=True)
        clean()
    if not METAL:
        print("no Metal here: scores / top-k / filter groups skipped (they need the kernels)", flush=True)
    print(f"peak memory of the run: see per-group checks (bound {LIMIT / 2 ** 30:.1f} GiB)", flush=True)
    if a.json:
        json.dump(RESULTS, open(a.json, "w"), indent=1)


main()
