#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""NAX hot GEMM ablation (full / nodec / nomm) on the gate+up and down stacks of a layer slice, same routing as moe_hot_stages.
usage: moe_nax_abl.py [--arch ds|glm] [--rows 8] [--modes full,nodec,nomm]"""
import argparse, os, sys
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe_decode_bench as B
from kyojin_mlx import moe as M, metal_nax as NX

ap = argparse.ArgumentParser()
ap.add_argument("--arch", default="ds"); ap.add_argument("--rows", default="8"); ap.add_argument("--modes", default="full,nodec,nomm")
a = ap.parse_args()
c, st = B.load(a.arch, os.path.expanduser("~/kyojin-mlx"))
E, H, topk = st[0].experts, st[0].kin, c["topk"]
m = M.MoeGather(*st, topk, limit=10.0, wpre=c["wpre"], fq=c["fq"], grouped=True, hotk=2)
rng = np.random.default_rng(0)
for R in [int(v) for v in a.rows.split(",")]:
    x, ids, wts, _ = B.make_inputs(rng, E, H, topk, R, "same", 1)[0]
    ids, wts = ids.reshape(-1), wts.reshape(-1)
    npairs, cfg = R * topk, m.nax_cfg(R)
    plan = m.plan(ids, R); hrow, heids, roff_gu, roff_d, hinfo = plan[3:]
    xh, A = m.rot_hot(x, ids, hrow, hinfo, R)
    for mode in a.modes.split(","):
        f = lambda: NX.expert_gemm((m.g, m.u), heids, roff_gu, A, 2 * npairs, R, cfg, out_half=False, mode=mode)
        mx.eval(f())
        us = min(B.timeit(lambda *_: f(), [(0,)] * 8) for _ in range(3))
        print(f"R{R} gu {mode} {us:.1f}", flush=True)
