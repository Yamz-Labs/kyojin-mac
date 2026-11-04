#!/usr/bin/env python3
"""cpftrellis: the routed prefill expert GEMM (gate+up, one launch, 384 virtual experts) in its stock form against two new forms, on skewed routing.

One kernel (metal_nax.expert_gemm, the register path), synthetic weights, interleaved arms, burn-in, median of --reps, md5 of every output.

  stock    today's kernel at pick_cfg's 64-row block (32, 2, 32, 64)
  u2       pf_unroll2: k loop by two steps, word sets wa / wb alternate (no copy)      -- must be bit-identical to stock
  b16      pf_bmu16: the same 64-row block as four 16-row matmul runs (16, 4, 32, 64)   -- md5 is REPORTED, not assumed (tensor-op tile M changes)
  u2b16    both

Routing: --skew 0.75 (default) = the hottest quarter of the experts takes 75 % of the pairs, drawn multinomial (seeded) at --m mean rows per expert, the production
4096-row pass (E 384, top-6, mean 64). --skew 0 = uniform. TF = useful flops (rows actually routed, 4 H I per row for gate+up) / time, so padding costs TF.

A compile failure of one arm (the new forms were written without a Metal compiler) is printed and the other arms still run.

usage: tools/cpftrellis_gemm.py [--e 384] [--m 64] [--k 3] [--pool 24] [--reps 9] [--burn 4] [--skew 0.75] [--json F] [--arms stock,u2,b16,u2b16]
bound: pool x 2 stacks x 4.4 MB (K=3) of weights + A (2 n x 5120 half, 0.50 GB at the defaults) + Y (2 n x 2304 half, 0.45 GB) + one np copy of Y:
~1.7 GiB at the defaults; the tool computes the bound before allocating and refuses above --cap-gib (3.5).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "bench"))

import mlx.core as mx  # noqa: E402
from bench_moe import synth_stack  # noqa: E402
from kyojin_mlx import metal_nax as NX  # noqa: E402

H, I = 5120, 2304
ARMS = {"stock": ((32, 2, 32, 64), False), "u2": ((32, 2, 32, 64), True), "b16": ((16, 4, 32, 64), False), "u2b16": ((16, 4, 32, 64), True)}


def virtual(st, V, pool):
    """A stack of V experts aliased onto the pool's real expert slabs (same words, same offsets layout)."""
    phys = np.arange(V) % pool
    off = np.array(st.offset)[:pool][phys]
    off = np.concatenate([off, off[-1:] + 1]).astype("<u4")
    return type(st)(st.words, mx.array(off), mx.array(np.array(st.k)[phys]), mx.array(np.array(st.suh)[phys]), mx.array(np.array(st.svh)[phys]), st.kt, st.nt, st.cb)


def routing(E, M, skew, seed):
    rng = np.random.default_rng(seed)
    P = E * M
    if not skew:
        return np.full(E, M)
    nh = E // 4
    pr = np.r_[np.full(nh, skew / nh), np.full(E - nh, (1 - skew) / (E - nh))]
    return rng.multinomial(P, pr)


def padded_rows(counts, bmu, nmb):
    t = 0
    for c in counts:
        c = int(c)
        for mg in range(0, c, bmu * nmb):
            t += sum(bmu for mb in range(nmb) if mb == 0 or mg + bmu * mb < c)
    return t


def md5(y) -> str:
    return hashlib.md5(np.ascontiguousarray(y).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e", type=int, default=384)
    ap.add_argument("--m", type=int, default=64, help="mean rows per expert")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--pool", type=int, default=24)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--burn", type=int, default=4)
    ap.add_argument("--skew", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--arms", default="stock,u2,b16,u2b16")
    ap.add_argument("--cap-gib", type=float, default=3.5)
    ap.add_argument("--json")
    a = ap.parse_args()
    E, pool = a.e, min(a.pool, a.e)
    counts = routing(E, a.m, a.skew, a.seed)
    n = int(counts.sum())
    wbytes = 2 * pool * (H * I * a.k // 8)
    bound = wbytes + 2 * n * H * 2 + 2 * n * I * 2 * 2 + 2 * E * 4096
    print(f"E={E} mean rows {a.m} K={a.k} pool={pool} skew={a.skew} rows {n} (min/mean/max {counts.min()}/{counts.mean():.0f}/{counts.max()})  "
          f"padded matmul rows: 32-row runs {padded_rows(counts, 32, 2)}, 16-row runs {padded_rows(counts, 16, 4)}  bound {bound / 2**30:.2f} GiB", flush=True)
    if bound > a.cap_gib * 2**30:
        print(f"refused: bound {bound / 2**30:.2f} GiB > cap {a.cap_gib} GiB; lower --m / --e")
        return 2
    if not mx.metal.is_available():
        print("Metal not available (run this on the Mac)")
        return 1
    mx.set_cache_limit(1 << 30)
    g, u = (synth_stack(H, I, [a.k] * pool, "mul1", s) for s in (1, 2))
    mx.eval(g.words, u.words)
    g, u = virtual(g, E, pool), virtual(u, E, pool)
    rng = np.random.default_rng(0)
    eids = mx.array(np.arange(E, dtype=np.int32))
    A = mx.array(NX.permute_k(rng.standard_normal((2 * n, H)).astype(np.float16)))
    cl = np.concatenate([[0], np.cumsum(counts)]).astype(np.uint32)
    ro = mx.array(np.concatenate([cl[:-1], n + cl]))
    mx.eval(A)
    fl = 4.0 * n * H * I
    arms = [x for x in a.arms.split(",") if x]

    def run(arm):
        cfg, un = ARMS[arm]
        old = NX.UNROLL2
        NX.UNROLL2 = un
        try:
            return NX.expert_gemm((g, u), eids, ro, A, 2 * n, int(counts.max()), cfg)
        finally:
            NX.UNROLL2 = old

    res = {"geometry": dict(E=E, m=a.m, K=a.k, pool=pool, skew=a.skew, rows=n, counts_max=int(counts.max())), "arms": {}}
    ref = None
    ok = []
    for arm in arms:
        try:
            y = np.array(run(arm))
        except Exception as ex:                                      # a kernel that does not compile is a result too
            print(f"  {arm:6s} FAILED: {type(ex).__name__}: {str(ex)[:300]}", flush=True)
            res["arms"][arm] = dict(error=str(ex)[:500])
            continue
        h = md5(y.view(np.uint16))
        if ref is None:
            ref = y
            print(f"  {arm:6s} md5 {h}  (reference)", flush=True)
            res["arms"][arm] = dict(md5=h)
        else:
            eq = bool(np.array_equal(y.view(np.uint16), ref.view(np.uint16)))
            d = float(np.abs(y.astype(np.float32) - ref.astype(np.float32)).max())
            print(f"  {arm:6s} md5 {h}  bit-identical to {arms[0]}: {eq}  max abs diff {d:.3g}", flush=True)
            res["arms"][arm] = dict(md5=h, bit_identical=eq, max_abs_diff=d)
        ok.append(arm)
        del y
    if len(ok) < 1:
        return 3
    for _ in range(a.burn):
        for arm in ok:
            mx.eval(run(arm))
    order = np.random.default_rng(7)
    T = {arm: [] for arm in ok}
    for r in range(a.reps + 1):
        seq = ok if r == 0 else [ok[i] for i in order.permutation(len(ok))]
        for arm in seq:
            y = run(arm)
            mx.synchronize()
            t0 = time.perf_counter()
            mx.eval(y)
            dt = time.perf_counter() - t0
            if r:
                T[arm].append(dt)
    base = np.array(T[ok[0]])
    print(f"{'arm':7s} {'ms':>8s} {'TF':>6s} {'TFmin':>6s} {'ratio':>7s} {'se':>6s}   (ratio = paired median vs {ok[0]}; < 1 is faster)")
    for arm in ok:
        v = np.array(T[arm])
        med = float(np.median(v))
        ratio = float(np.median(v / base))
        se = float((v / base).std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        res["arms"][arm].update(ms=med * 1e3, tf=fl / med / 1e12, tf_min=fl / v.min() / 1e12, ratio=ratio, ratio_se=se)
        print(f"{arm:7s} {med * 1e3:8.2f} {fl / med / 1e12:6.1f} {fl / v.min() / 1e12:6.1f} {ratio:7.3f} {se:6.3f}")
    peak = mx.get_peak_memory() / 2**30
    res["peak_gib"] = peak
    print(f"peak {peak:.2f} GiB")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
        print(f"json -> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
