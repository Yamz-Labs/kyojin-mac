#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Error of the packed-fp16 grouped MoE GEMV (S.HALF_ROWS) against the fp32 grouped GEMV on a layer slice: relative L2 error of the routed sum per row.
usage: moe_half_err.py [--arch ds|glm] [--rows 2,4,8] [--route same,distinct]"""
import argparse, os, sys
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe_decode_bench as B
from kyojin_mlx import moe as M, metal_src as S

ap = argparse.ArgumentParser()
ap.add_argument("--arch", default="ds"); ap.add_argument("--rows", default="2,4,8"); ap.add_argument("--route", default="same,distinct"); ap.add_argument("--cmp", default="half", help="half = packed-fp16 grouped GEMV, nax = NAX hot path (hotk 2)")
a = ap.parse_args()
c, st = B.load(a.arch, os.path.expanduser("~/kyojin-mlx"))
E, H, topk = st[0].experts, st[0].kin, c["topk"]
m = M.MoeGather(*st, topk, limit=10.0, wpre=c["wpre"], fq=c["fq"], grouped=True, hotk=0)
mn = M.MoeGather(*st, topk, limit=10.0, wpre=c["wpre"], fq=c["fq"], grouped=True, hotk=2)
rng = np.random.default_rng(1)
for route in a.route.split(","):
    for R in [int(v) for v in a.rows.split(",")]:
        worst = 0.0
        for _ in range(6):
            x, ids, wts, _i = B.make_inputs(rng, E, H, topk, R, route, 1)[0]
            S.HALF_ROWS = 0; ref = m(x, ids, wts, fq8=c["fq"]); mx.eval(ref)
            S.HALF_ROWS = 2 if a.cmp == "half" else 0; out = (m if a.cmp == "half" else mn)(x, ids, wts, fq8=c["fq"]); mx.eval(out)
            d = np.linalg.norm(np.array(out) - np.array(ref), axis=1) / np.linalg.norm(np.array(ref), axis=1)
            worst = max(worst, float(d.max()))
        print(f"{a.arch} {route} R{R}: worst rel L2 {worst:.2e}", flush=True)
