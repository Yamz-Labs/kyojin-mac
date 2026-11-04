#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-stage us of the NAX hot path (MoeGather._call_hot) on a layer slice, same routing as moe_decode_bench (same experts for all rows).
usage: moe_hot_stages.py [--arch ds|glm] [--rows 4,8] [--route same|distinct]"""
import argparse, os, sys
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe_decode_bench as B
from kyojin_mlx import moe as M

ap = argparse.ArgumentParser()
ap.add_argument("--arch", default="ds"); ap.add_argument("--rows", default="4,8"); ap.add_argument("--route", default="same"); ap.add_argument("--hotk", type=int, default=2)
a = ap.parse_args()
c, st = B.load(a.arch, os.path.expanduser("~/kyojin-mlx"))
E, H, topk = st[0].experts, st[0].kin, c["topk"]
m = M.MoeGather(*st, topk, limit=10.0, wpre=c["wpre"], fq=c["fq"], grouped=True, hotk=a.hotk)
rng = np.random.default_rng(0)
for R in [int(v) for v in a.rows.split(",")]:
    x, ids, wts, _ = B.make_inputs(rng, E, H, topk, R, a.route, 1)[0]
    ids, wts = ids.reshape(-1), wts.reshape(-1)
    P, npairs, cfg = m._params(R), R * topk, m.nax_cfg(R)
    plan = m.plan(ids, R); hrow, heids, roff_gu, roff_d, hinfo = plan[3:]
    xh, A = m.rot_hot(x, ids, hrow, hinfo, R)
    part = m.ggemv((m.g, m.u), xh, ids, plan, P["ggemv_gu"], P["tgu"], m.ks_gu)
    yn = m._hot_gemm((m.g, m.u), heids, roff_gu, A, 2 * npairs, R, cfg, m.ks_gu)
    xh2, A2 = m.mid_hot(part, yn, ids, wts, hrow, hinfo, R)
    part2 = m.ggemv((m.d,), xh2, ids, plan, P["ggemv_dn"], P["tdn"], m.ks_dn)
    yd = m._hot_gemm((m.d,), heids, roff_d, A2, npairs, R, cfg, m.ks_dn)
    mx.eval(plan, xh, A, part, yn, xh2, A2, part2, yd)
    S = (("plan", lambda: m.plan(ids, R)), ("rot_hot", lambda: m.rot_hot(x, ids, hrow, hinfo, R)),
         ("ggemv_gu", lambda: m.ggemv((m.g, m.u), xh, ids, plan, P["ggemv_gu"], P["tgu"], m.ks_gu)),
         ("nax_gu", lambda: m._hot_gemm((m.g, m.u), heids, roff_gu, A, 2 * npairs, R, cfg, m.ks_gu)),
         ("mid_hot", lambda: m.mid_hot(part, yn, ids, wts, hrow, hinfo, R)),
         ("ggemv_dn", lambda: m.ggemv((m.d,), xh2, ids, plan, P["ggemv_dn"], P["tdn"], m.ks_dn)),
         ("nax_dn", lambda: m._hot_gemm((m.d,), heids, roff_d, A2, npairs, R, cfg, m.ks_dn)),
         ("final_hot", lambda: m.final_hot(part2, yd, ids, wts, hrow, R)))
    tot = 0; line = []
    for n, f in S:
        us = min(B.timeit(lambda *_: f(), [(0,)] * 8) for _ in range(3)); tot += us; line.append(f"{n} {us:.1f}")
    print(f"{a.route} R{R} cfg {cfg}: " + "  ".join(line) + f"  sum {tot:.1f}", flush=True)
