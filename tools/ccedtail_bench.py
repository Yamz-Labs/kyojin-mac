#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ccedtail: kernel timing of the routed MoE of ONE prefill-tail layer, three paths, at the served tail shape, on the Mac (Metal).

Arms (one layer, `rows` rows x top-k, real H / I, random valid trellises, as tools/moe_k_bench.py):
  c4       the four-row chunks (patch._moe_rows' fallback: ceil(rows / step) calls), what a band below the S3 floor runs today
  grouped  ONE grouped call (pf_tail_c4 / pf_tail_grp: MoeGather.__call__(max_pairs=TAIL_PAIRS)), one trellis decode per distinct expert
  s3       the fused prefill path (MoeGather.prefill), what a band at or above pf_min_rows runs
The row count is a shape: --rows takes a list. The lever's band on the lane is (ROW_SPLIT_MAX, pf_min_rows): above the verify / multi-user rows
(--row-split, 24 = the served mtp_rows) and below the S3 floor; the default list is that band (just above the split, middle, floor - 1) plus the floor
and the CED tail (--tail) as the S3 references. Every row count prints which path the live gate (patch.pf_min_rows / tail_band_takes) serves it with.
The layer is built with grouped=True, hotk=0 (the served default, KYOJIN_ROWS_GROUPED=1); it prints both, and the gate check asserts the lever's fired counter. Arms are interleaved A B C ... C B A over --rounds; per arm the min
and median of chains of n and 4n calls in one eval (slope = ms per call). Values: grouped vs c4 and s3 vs c4, bit-equal / rel L2 / max abs, with the fp32
rows (HALF_ROWS=0, where grouped must equal c4 bit for bit) and as served (HALF_ROWS as set in the environment, where it need not: a pair's body depends
on how many rows of the call share its expert). Memory: the three stacks are sized before they are built and the peak is asserted under 3.5 GiB.
Nothing here is a claim until you run it: it prints numbers, it does not know a verdict.

usage (from the repo root, on the Mac, nothing else loaded):
  python tools/ccedtail_bench.py --experts 192 --rows 17,37,64,112,128,170 --rounds 5
"""
import argparse
import os
import statistics
import sys
import time

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kyojin_mlx import fired as FD, flags as FL, metal_src as S, moe as M, patch as P  # noqa: E402
from moe_fuse_bench import make_stack  # noqa: E402

LIMIT_GIB = 3.5


def stack_gib(h, i, ks):
    """Bytes of gate + up + down for the experts `ks` (trellis words: kin * nout * K / 8 per projection) + scales, in GiB."""
    words = sum(3 * h * i * k // 8 for k in ks)
    return (words + len(ks) * 3 * (h + i) * 2) / 2 ** 30


def timed(f, ins, n):
    def run(c):
        outs = [f(*ins[j % len(ins)]) for j in range(c)]
        t0 = time.perf_counter()
        mx.eval(*outs)
        return time.perf_counter() - t0
    a, b = [], []
    for r in range(4):
        da, db = run(n), run(4 * n)
        if r:
            a.append(da)
            b.append(db)
    return (min(b) - min(a)) / (3 * n) * 1e3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", default="", help="comma list of row counts (shapes); default: below / at / above the live S3 floor and --tail")
    ap.add_argument("--tail", type=int, default=128, help="the served CED tail length (spec_defaults 'tail'), always added to the list")
    ap.add_argument("--row-split", type=int, default=24, help="verify / multi-user rows (EL.ROW_SPLIT_MAX as enable_mtp raises it: mtp_rows, 24); the band starts above it")
    ap.add_argument("--experts", type=int, default=192, help="192 keeps the stacks near 2.1 GiB; the pack has 384 (distinct experts per call differ, see the report)")
    ap.add_argument("--pack-experts", type=int, default=384, help="expert count of the served pack: the S3 floor (pf_min_rows) is taken for it, not for the smaller bench layer")
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--inter", type=int, default=2304)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--chain", type=int, default=4)
    ap.add_argument("--step", type=int, default=0, help="chunk rows of the c4 arm (0 = the live patch.MOE_MAX_ROWS)")
    a = ap.parse_args()
    H, I, TOPK, E = a.hidden, a.inter, a.topk, a.experts
    rng = np.random.default_rng(7)
    ks = [3 if r < 141 / 384 else 2 for r in rng.random(E)]
    est = stack_gib(H, I, ks)
    print(f"estimated stacks {est:.2f} GiB (bound {LIMIT_GIB})", flush=True)
    assert est < LIMIT_GIB - 0.5, "refusing: lower --experts"
    mx.set_cache_limit(256 << 20)
    t0 = time.time()
    g, u, d = make_stack(H, I, ks, 1), make_stack(H, I, ks, 2), make_stack(I, H, ks, 3)
    mx.eval(g.words, u.words, d.words)
    print(f"stacks built {time.time() - t0:.1f} s, peak {mx.get_peak_memory() / 2 ** 30:.2f} GiB", flush=True)
    assert mx.get_peak_memory() / 2 ** 30 < LIMIT_GIB, "bench memory bound"
    m = M.MoeGather(g, u, d, TOPK, limit=10.0, wpre=True, fq=False, grouped=True, hotk=0)
    P.EL.ROW_SPLIT_MAX = max(P.EL.ROW_SPLIT_MAX, a.row_split)
    FL.set("fired", 1)
    print(f"layer grouped={m.grouped} hotk={m.hotk} ROW_SPLIT_MAX={P.EL.ROW_SPLIT_MAX} UGROUPED={M.UGROUPED} HALF_ROWS={S.HALF_ROWS}", flush=True)
    step = a.step or max(1, min(P.MOE_MAX_ROWS, M.MAX_PAIRS // TOPK))
    stub = type("Pack", (), {"g": type("G", (), {"experts": a.pack_experts})(), "topk": TOPK})()
    floor = P.pf_min_rows(stub)                                   # the floor of the served pack, not of the E-expert bench layer
    P.PF_MIN_ROWS = str(floor)                                    # so the gate check below sees it too (patch.pf_min_rows honours KYOJIN_PF_MIN_ROWS)
    lo = max(step, P.EL.ROW_SPLIT_MAX)
    rows_l = sorted({int(r) for r in a.rows.split(",") if r} or {lo + 1, (lo + floor) // 2, floor - 1, floor, a.tail})
    rows_l = [r for r in rows_l if r > 0 and r * TOPK <= M.TAIL_PAIRS]
    print(f"bench experts {E}, pack experts {a.pack_experts}, top-k {TOPK}, chunk step {step}, pack S3 floor pf_min_rows = {floor}, pairs cap {M.TAIL_PAIRS}", flush=True)
    for r in rows_l:
        gate = P.tail_band_takes(r, TOPK, step, floor, bool(m.grouped), bool(m.hotk), M.TAIL_PAIRS)
        serves = "S3" if r >= floor else ("decode" if r <= step else "chunks")
        print(f"  rows {r:4d}: served by {serves}; pf_tail_c4 would take it: {gate}; c4 launches per layer {-(-r // step)}")

    def c4(x, ids, w):
        return mx.concatenate([m(x[i:i + step], ids[i:i + step], w[i:i + step]) for i in range(0, x.shape[0], step)], axis=0)

    def grouped(x, ids, w):
        return m(x, ids, w, max_pairs=M.TAIL_PAIRS)

    def s3(x, ids, w):
        return m.prefill(x, ids, w)

    arms = {"c4": c4, "grouped": grouped, "s3": s3}
    served_half = S.HALF_ROWS
    results = []
    for rows in rows_l:
        ids = np.stack([rng.choice(E, TOPK, replace=False) for _ in range(rows)]).astype(np.int32)
        nd = len(np.unique(ids))
        ins = [(mx.array((rng.standard_normal((rows, H)) * 0.5).astype(np.float16)), mx.array(ids),
                mx.array(rng.uniform(0.1, 1.0, (rows, TOPK)).astype(np.float32)))]
        mx.eval(*ins[0])
        val = {}
        for hr in (0, served_half):
            S.HALF_ROWS = hr
            o = {k: np.array(f(*ins[0])) for k, f in arms.items()}
            val[hr] = o
        S.HALF_ROWS = served_half
        us = {k: [] for k in arms}
        order = list(arms)
        for r in range(a.rounds):
            for name in (order if r % 2 == 0 else order[::-1]):
                us[name].append(timed(arms[name], ins, a.chain))
        mn = {k: min(v) for k, v in us.items()}
        md = {k: statistics.median(v) for k, v in us.items()}
        print(f"\nrows {rows} ({nd} distinct experts of {E}): ms/layer min [median]  " + "  ".join(f"{k} {mn[k]:.3f} [{md[k]:.3f}]" for k in arms))
        print(f"   grouped vs c4 {100 * (mn['grouped'] / mn['c4'] - 1):+.1f} %   s3 vs c4 {100 * (mn['s3'] / mn['c4'] - 1):+.1f} %   grouped vs s3 {100 * (mn['grouped'] / mn['s3'] - 1):+.1f} %"
              f"   us/row/layer c4 {1e3 * mn['c4'] / rows:.1f} grouped {1e3 * mn['grouped'] / rows:.1f} s3 {1e3 * mn['s3'] / rows:.1f}")
        for hr in (0, served_half):
            o = val[hr]
            for k in ("grouped", "s3"):
                eq = np.array_equal(o[k].view(np.uint32), o["c4"].view(np.uint32))
                den = np.linalg.norm(o["c4"]) + 1e-30
                print(f"   HALF_ROWS={hr}: {k} vs c4 bit-equal {eq}  relL2 {np.linalg.norm(o[k] - o['c4']) / den:.2e}  maxabs {np.abs(o[k] - o['c4']).max():.3e}")
        results.append((rows, mn))
    print(f"\npeak {mx.get_peak_memory() / 2 ** 30:.2f} GiB")
    assert mx.get_peak_memory() / 2 ** 30 < LIMIT_GIB, "bench memory bound"
    # the live gate end to end, one call per row count: flag off -> the old path, flag on -> grouped (or unchanged where the gate refuses)
    m_old = M.TAIL_C4
    for rows in rows_l:
        x = mx.array((rng.standard_normal((rows, H)) * 0.5).astype(np.float16))
        ids = mx.array(np.stack([rng.choice(E, TOPK, replace=False) for _ in range(rows)]).astype(np.int32))
        w = mx.array(rng.uniform(0.1, 1.0, (rows, TOPK)).astype(np.float32))
        outs, hits = {}, {}
        for fl in (0, 1):
            FL.set("pf_tail_c4", fl)
            FD.reset()
            outs[fl] = np.array(P._moe_rows(m, x, ids, w))
            hits[fl] = FD.HITS.get("pf_tail_c4", 0)
        same = np.array_equal(outs[0].view(np.uint32), outs[1].view(np.uint32))
        gate = P.tail_band_takes(rows, TOPK, step, floor, bool(m.grouped), bool(m.hotk), M.TAIL_PAIRS)
        assert hits[0] == 0 and hits[1] == (1 if gate else 0), f"rows {rows}: fired counter {hits} but the gate says {gate}"
        print(f"gate check rows {rows}: pf_tail_c4 fired {hits[1]} (gate {gate}); output flag on == off bit for bit: {same}")
    FL.set("pf_tail_c4", 1 if m_old else 0)

if __name__ == "__main__":
    main()
