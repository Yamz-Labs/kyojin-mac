#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Routed MoE decode per layer on a stacked layer slice (DS4.1 / GLM): us per call at rows 1,2,4,8 against the read roof.
Per call: slope of chains of n and 4n calls with fresh inputs in one eval (no per-eval floor), min over --reps. Launches: counted
from the path taken (plan + rot + one GEMV per bitrate class for gate+up and for down + mid + final). Bytes: trellis words of the
distinct selected experts (gate, up, down) + their suh / svh. Roof = bytes / --gbs (measured Air read ceiling 125 GB/s).
Routing: `distinct` = every row picks its own experts, `same` = all rows pick the same experts (best case for tile sharing).
usage: moe_decode_bench.py [--arch ds|glm] [--rows 1,2,4,8] [--route distinct,same] [--impl cur|fused] [--reps 5]"""
import argparse, os, sys, time, json

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kyojin_mlx import patch as P, moe as M  # noqa: E402

ARCH = {
    "ds": dict(dir="models/ds_l10", pre="layers.10", names=("ffn.experts.w1", "ffn.experts.w3", "ffn.experts.w2"), topk=6, wpre=True, fq=True),
    "glm": dict(dir="models/glm_l10", pre="model.language_model.layers.10", names=("mlp.experts.gate_proj", "mlp.experts.up_proj", "mlp.experts.down_proj"), topk=8, wpre=False, fq=False),
}


def load(arch, root):
    c = ARCH[arch]
    t = P.load_pack(os.path.join(root, c["dir"]), prefix=c["pre"] + ".")
    cb = P._cb(t, f"{c['pre']}.{c['names'][0]}", "mul1")
    st = [M.Stack.from_arrays(t, f"{c['pre']}.{n}", cb) for n in c["names"]]
    mx.eval(*[x for s in st for x in (s.words, s.offset, s.k, s.suh, s.svh)])
    return c, st


def expert_bytes(st):
    """bytes read per expert id: trellis words of the three projections + sign vectors."""
    per = np.zeros(st[0].experts, dtype=np.int64)
    for s in st:
        off = np.array(s.offset).astype(np.int64)
        per += (off[1:] - off[:-1]) * 4 + (s.kin + s.nout) * 2
    return per


def launches(m, rows):
    nks = lambda ks: len(ks) if M.VB_SPLIT_K else 1
    if rows == 1:
        return 5
    return 1 + 1 + nks(m.ks_gu) + 1 + nks(m.ks_dn) + 1          # plan, rot, gu passes, mid, dn passes, final


def timeit(f, ins):
    n = len(ins)
    def run(c):
        outs = [f(*ins[i % n]) for i in range(c)]
        t0 = time.perf_counter(); mx.eval(*outs); return time.perf_counter() - t0
    T = {n: [], 4 * n: []}
    for r in range(4):
        for c in T:
            dt = run(c)
            if r:
                T[c].append(dt)
    return (min(T[4 * n]) - min(T[n])) / (3 * n) * 1e6


def stages(m, x, ids, wts, rows, fq8):
    """per-stage us of the rows == 1 / grouped path: chains of one stage over the same intermediates."""
    ids_f, wts_f = ids.reshape(-1).astype(mx.int32), wts.reshape(-1).astype(mx.float32)
    xh = m.rot_gu(x, ids_f, fq8); part = m.gemv_gu(xh, ids_f, rows); xh2 = m.mid(part, ids_f, wts_f, rows); part2 = m.gemv_dn(xh2, ids_f, rows)
    mx.eval(xh, part, xh2, part2)
    res = {}
    for name, f in (("rot_gu", lambda: m.rot_gu(x, ids_f, fq8)), ("gemv_gu", lambda: m.gemv_gu(xh, ids_f, rows)), ("mid", lambda: m.mid(part, ids_f, wts_f, rows)),
                    ("gemv_dn", lambda: m.gemv_dn(xh2, ids_f, rows)), ("final", lambda: m.final(part2, ids_f, wts_f, rows))):
        res[name] = min(timeit(lambda *_: f(), [(0,)] * 8) for _ in range(3))
    return res


def make_inputs(rng, E, H, topk, R, route, chain):
    ins = []
    for _ in range(chain):
        if route == "same":
            one = rng.choice(E, topk, replace=False); ids = np.stack([one] * R)
        else:
            perm = rng.permutation(E)
            ids = np.stack([perm[(r * topk) % E:(r * topk) % E + topk] for r in range(R)])
        ins.append((mx.array((rng.standard_normal((R, H)) * 0.5).astype(np.float16)), mx.array(ids.astype(np.int32)),
                    mx.array(rng.random((R, topk)).astype(np.float32)), ids))
    mx.eval(*[v for i in ins for v in i[:3]])
    return ins


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="ds"); ap.add_argument("--root", default=os.path.expanduser("~/kyojin-mlx"))
    ap.add_argument("--rows", default="1,2,4,8"); ap.add_argument("--route", default="distinct,same")
    ap.add_argument("--impl", default="cur"); ap.add_argument("--gbs", type=float, default=125.0)
    ap.add_argument("--hotk", type=int, default=0, help="groups of >= hotk rows take the NAX GEMM (0 = GEMV only)"); ap.add_argument("--stages", action="store_true"); ap.add_argument("--chain", type=int, default=8); ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    mx.set_cache_limit(256 << 20)
    c, st = load(a.arch, a.root)
    E, H, topk = st[0].experts, st[0].kin, c["topk"]
    per = expert_bytes(st)
    kw = dict(limit=10.0, wpre=c["wpre"], fq=c["fq"], grouped=True, hotk=a.hotk)
    m = M.MoeGather(*st, topk, **kw)
    if a.impl == "fused":
        from kyojin_mlx import moe_fused as MF
        m = MF.wrap(m)
    rng = np.random.default_rng(0)
    print(f"{a.arch}: {E} experts, topk {topk}, H {H}, bitrates {dict(zip(*np.unique(st[0].kh, return_counts=True)))}, mean {per.mean() / 1e6:.2f} MB/expert, impl {a.impl}", flush=True)
    out = {}
    for route in a.route.split(","):
        for R in [int(v) for v in a.rows.split(",")]:
            if route == "same" and R == 1:
                continue
            ins = make_inputs(rng, E, H, topk, R, route, a.chain)
            fq8 = c["fq"]
            f = lambda x, i, w, _ids: m(x, i, w, fq8=fq8)
            us = min(timeit(f, ins) for _ in range(a.reps))
            nb = float(np.mean([per[np.unique(i[3])].sum() for i in ins]))
            roof = nb / (a.gbs * 1e3)
            if a.stages and R == 1:
                x0, i0, w0, _ = ins[0]
                sg = stages(m, x0, i0, w0, R, fq8); print("   stages us: " + "  ".join(f"{k} {v:.1f}" for k, v in sg.items()) + f"  sum {sum(sg.values()):.1f}")
            out[f"{route}_r{R}"] = dict(us=round(us, 1), launches=launches(m, R), MB=round(nb / 1e6, 1), roof_us=round(roof, 1), x_roof=round(us / roof, 2))
            print(f"{route:8s} R{R}: {us:8.1f} us  launches {launches(m, R):2d}  read {nb / 1e6:7.1f} MB  roof {roof:7.1f} us  ({us / roof:.2f}x roof)", flush=True)
    print("BENCH " + json.dumps(dict(arch=a.arch, impl=a.impl, res=out)))
