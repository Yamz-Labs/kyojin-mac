#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Kernel bench for the grouped routed-MoE GEMV prefetch depth (KYOJIN_MOE_GPF, flag moe_gpf), at the real DS4.1 layer geometry.

One layer: H 5120, I 2304, topk 6, E experts of the pack's K 2/3 mix (the per-call read is topk x rows experts, so E only has to hold the
distinct experts of one call for the bytes to be the served ones). Same stacks as tools/moe_k_bench.py, same trellis generators.

Arms: gpf 0 (stock), 1 (one tile ahead), 2 (two tiles ahead) - loads reordered only, so every arm must be bit-identical to stock.
Timed per stage (gemv_gu, gemv_dn) and for the whole grouped layer call, as the slope of a chain of n and 4n calls in one eval, arms
interleaved A B C C B A over --rounds, median of the paired slopes. Quality: bits of the layer output against the stock arm (must be equal).

usage: python tools/bench_moe_gpf.py [--rows 2,4,6] [--rounds 5] [--experts 96] [--route random|identical|half]
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
from kyojin_mlx import flags as FL, moe as M  # noqa: E402
from moe_fuse_bench import make_stack  # noqa: E402

H, I, TOPK, LAYERS = 5120, 2304, 6, 40
KS = (2, 3, 3)                    # gate, up, down: the pack's 243 : 141 mix, worst case for the wide class
LIMIT_GIB = 3.5


def route(rng, rows, mode):
    """(rows, topk) expert ids: random = independent, identical = all rows on the same six, half = the first three shared."""
    if mode == "identical":
        base = rng.choice(np.arange(96), TOPK, replace=False)
        ids = np.stack([base for _ in range(rows)])
    elif mode == "half":
        base = rng.choice(np.arange(96), TOPK, replace=False)
        ids = np.stack([np.concatenate([base[:3], rng.choice(np.arange(96), TOPK - 3, replace=False)]) for _ in range(rows)])
    else:
        ids = np.stack([rng.choice(np.arange(96), TOPK, replace=False) for _ in range(rows)])
    w = rng.uniform(0.1, 1.0, (rows, TOPK)).astype(np.float32)
    return mx.array(ids.astype(np.int32)), mx.array((w / w.sum(-1, keepdims=True) * 1.5).astype(np.float32))


def timed(f, ins, n):
    """Slope (s/call) from chains of n and 4n calls in one eval (the eval boundary is not the kernel)."""
    def run(c):
        outs = [f(*ins[i % len(ins)]) for i in range(c)]
        t0 = time.perf_counter()
        mx.eval(*outs)
        return time.perf_counter() - t0
    a, b = [], []
    for r in range(4):
        da, db = run(n), run(4 * n)
        if r:
            a.append(da)
            b.append(db)
    return (min(b) - min(a)) / (3 * n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", default="2,4,6")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--experts", type=int, default=96)
    ap.add_argument("--route", default="random")
    ap.add_argument("--json")
    a = ap.parse_args()
    rows_l = [int(x) for x in a.rows.split(",")]
    g, gb = make_stack(H, I, KS, "mul1", 1)
    u, ub = make_stack(H, I, KS, "mul1", 2)
    d, db = make_stack(I, H, KS, "mul1", 3)
    tot = (gb + ub + db) / 2**30
    print(f"stacks {gb / 2**20:.0f} + {ub / 2**20:.0f} + {db / 2**20:.0f} MiB = {tot:.2f} GiB (limit {LIMIT_GIB})", flush=True)
    assert tot < LIMIT_GIB

    arms = [0, 1, 2]
    res = {}
    out = {}
    for rows in rows_l:
        rng = np.random.default_rng(rows)
        ids, w = route(rng, rows, a.route)
        x = mx.random.normal((rows, H)).astype(mx.float16)
        per = {}
        for gpf in arms:
            FL.set_many({"moe_gpf": str(gpf)})
            m = M.MoeGather(g, u, d, TOPK, wpre=False, tgk_gu=2, tgk_dn=2, grouped=True, hotk=0)
            y = m(x, ids, w)
            mx.eval(y)
            ref = per.setdefault("ref", np.array(y))
            eq = bool(np.array_equal(np.array(y), ref)) if gpf else True
            idsf, wf = ids.reshape(-1).astype(mx.int32), w.reshape(-1).astype(mx.float32)
            fl = idsf
            P = m._params(rows)
            plan = m.plan(fl, rows)
            xh = m.rot_gu(x, fl)
            part = m.ggemv((m.g, m.u), xh, fl, plan, P["ggemv_gu"], P["tgu"], m.ks_gu)
            xh2 = m.mid(part, fl, wf, rows)
            mx.eval(xh2)
            t = {}
            t["gemv_gu"] = timed(lambda: m.ggemv((m.g, m.u), xh, fl, plan, P["ggemv_gu"], P["tgu"], m.ks_gu), [(0,)], 2)
            t["gemv_dn"] = timed(lambda: m.ggemv((m.d,), xh2, fl, plan, P["ggemv_dn"], P["tdn"], m.ks_dn), [(0,)], 2)
            t["layer"] = timed(lambda: m(x, ids, w), [(0,)], 2)
            per[gpf] = dict(t, eq=eq)
            print(f"rows {rows} gpf {gpf}: gu {t['gemv_gu'] * 1e6:8.1f} us  dn {t['gemv_dn'] * 1e6:8.1f} us  layer {t['layer'] * 1e6:8.1f} us  bits_equal={eq}", flush=True)
        res[rows] = per
    FL.set_many({"moe_gpf": "0"})

    print(f"\n{'rows':>4} {'stage':>8} " + " ".join(f"{'gpf' + str(k):>10}" for k in arms) + "   ratio 1/0   2/0")
    for rows in rows_l:
        for st in ("gemv_gu", "gemv_dn", "layer"):
            v = [res[rows][k][st] for k in arms]
            print(f"{rows:4d} {st:>8} " + " ".join(f"{x * 1e6:10.1f}" for x in v) + f"   {v[1] / v[0]:8.3f} {v[2] / v[0]:8.3f}")
    out = {str(r): {str(k): {s: res[r][k][s] for s in ("gemv_gu", "gemv_dn", "layer", "eq")} for k in arms} for r in rows_l}
    if a.json:
        import json
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
