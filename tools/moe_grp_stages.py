#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-stage us of the grouped GEMV path (MoeGather._call_grouped) on a layer slice, same routing as moe_decode_bench.
usage: moe_grp_stages.py [--arch ds|glm] [--rows 1,8] [--route same|distinct]"""
import argparse, os, sys
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe_decode_bench as B
from kyojin_mlx import moe as M, metal_src as S

ap = argparse.ArgumentParser()
ap.add_argument("--arch", default="ds"); ap.add_argument("--rows", default="1,8"); ap.add_argument("--route", default="same")
a = ap.parse_args()
c, st = B.load(a.arch, os.path.expanduser("~/kyojin-mlx"))
E, H, topk = st[0].experts, st[0].kin, c["topk"]
m = M.MoeGather(*st, topk, limit=10.0, wpre=c["wpre"], fq=c["fq"], grouped=True, hotk=0)
rng = np.random.default_rng(0)
for R in [int(v) for v in a.rows.split(",")]:
    x, ids, wts, _ = B.make_inputs(rng, E, H, topk, R, a.route, 1)[0]
    ids, wts = ids.reshape(-1), wts.reshape(-1)
    half = S.half_on(R) and not S.af_fold(m.cbg)
    P, ent = m._gstatic(R, c["fq"], x.dtype == mx.float16, half)
    xh = m.rot_gu(x, ids, c["fq"], half)
    plan = lambda: ent["plan"](inputs=[ids, P["plan"]], grid=(256, 1, 1), threadgroup=(256, 1, 1), output_shapes=ent["shapes"], output_dtypes=[mx.uint32] * 3)
    lead, cnt, mem = plan()
    part = m._run_passes(ent["gu"], [xh, lead, cnt, mem, ids])
    xh2 = m.mid(part, ids, wts, R, half)
    part2 = m._run_passes(ent["dn"], [xh2, lead, cnt, mem, ids])
    mx.eval(lead, cnt, mem, xh, part, xh2, part2)
    S_ = (("plan", plan), ("rot_gu", lambda: m.rot_gu(x, ids, c["fq"], half)),
          ("gemv_gu", lambda: m._run_passes(ent["gu"], [xh, lead, cnt, mem, ids])),
          ("mid", lambda: m.mid(part, ids, wts, R, half)),
          ("gemv_dn", lambda: m._run_passes(ent["dn"], [xh2, lead, cnt, mem, ids])),
          ("final", lambda: m.final(part2, ids, wts, R)))
    tot = 0; line = []
    for n, f in S_:
        us = min(B.timeit(lambda *_: f(), [(0,)] * 8) for _ in range(3)); tot += us; line.append(f"{n} {us:.1f}")
    print(f"{a.route} R{R}: " + "  ".join(line) + f"  sum {tot:.1f}", flush=True)
