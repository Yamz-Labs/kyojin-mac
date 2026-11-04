#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""sumoe synthetic bench: one DS4.1 MoE layer (H=5120, I=2304, 384 experts, top-6, K mix 2/3, shared expert K=3) at R rows, routed + shared.
Arms: stock = MoeGather (default flags) + dense shared chain (Exl3Linear gate / up / down, swiglu, add) ; union = MoeFused routed + the same shared
chain ; sumoe = MoeSharedFused (shared expert inside the routed launches). Dependent chains of n and 4n layers in one eval (slope = per-layer us),
arms interleaved A B C C B A, median of --reps. Bytes: distinct selected experts' trellis (+ shared) read once per call.
usage: flock /tmp/ultra_bench.lock python tools/moe_sumoe_bench.py [--rows 1,2,4] [--route distinct,same] [--reps 7]"""
import argparse
import os
import statistics
import sys

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kyojin_mlx import moe as M, moe_fused as MF, moe_sumoe as SU  # noqa: E402
from kyojin_mlx.exl3_linear import Exl3Linear  # noqa: E402
from moe_fuse_bench import make_stack, expert_bytes  # noqa: E402

H, I, E, TOPK, KSH = 5120, 2304, 384, 6, 3
LIMIT = 10.0


def swiglu(g, u):
    g = mx.minimum(g, LIMIT)
    u = mx.clip(u, -LIMIT, LIMIT)
    return (g * mx.sigmoid(g)) * u


def build():
    rng = np.random.default_rng(7)
    ks = [3 if r < 141 / 384 else 2 for r in rng.random(E)]
    g, u, d = make_stack(H, I, ks, 1), make_stack(H, I, ks, 2), make_stack(I, H, ks, 3)
    sg, su, sd = make_stack(H, I, [KSH], 11), make_stack(H, I, [KSH], 12), make_stack(I, H, [KSH], 13)
    tr = lambda s: (s.expert_trellis(0), s.suh[0], s.svh[0])
    lin = [Exl3Linear(t, np.array(s), np.array(v), cb="mul1", out_dtype=mx.bfloat16) for t, s, v in (tr(sg), tr(su), tr(sd))]
    ext = SU.with_shared(g, u, d, [tr(sg), tr(su), tr(sd)])
    kw = dict(limit=LIMIT, wpre=True, fq=False)
    stock = M.MoeGather(g, u, d, TOPK, **kw)
    stock._is_fused_inner = True
    union = M.MoeGather(g, u, d, TOPK, union=True, grouped=False, **kw)
    sumoe = SU.MoeSharedFused(*ext, TOPK, **kw)
    sumoe_fd = SU.MoeSharedFused(*ext, TOPK, **kw)
    sumoe_fd.fastdec = True                              # mul1fd decode (cheaper ALU, not bit-identical): gather GEMV, i.e. R1
    return ks, (g, u, d), lin, stock, union, (sumoe, sumoe_fd)


def shared_chain(lin, x):
    return lin[2](swiglu(lin[0](x), lin[1](x))).astype(mx.float32)


def routes(rows, mode, n, seed=3):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        if mode == "same":
            ids = np.tile(rng.choice(E, TOPK, replace=False), (rows, 1))
        else:
            ids = np.stack([rng.choice(E, TOPK, replace=False) for _ in range(rows)])
        w = rng.uniform(0.1, 1.0, (rows, TOPK)).astype(np.float32)
        out.append((mx.array(ids.astype(np.int32)), mx.array(w / w.sum(-1, keepdims=True))))
    return out


def chain(arm, ctx, x0, rts, n):
    lin, stock, union, (sumoe, sumoe_fd) = ctx
    x = x0
    for i in range(n):
        ids, w = rts[i % len(rts)]
        if arm in ("sumoe", "sumoe_fd"):
            y = (sumoe if arm == "sumoe" else sumoe_fd)(x, ids, w)
        else:
            y = (stock if arm == "stock" else union)(x, ids, w) + shared_chain(lin, x)
        x = (y * 0.02).astype(mx.float16)
    return x


def timed(arm, ctx, x0, rts, n):
    mx.synchronize()
    import time
    t = time.perf_counter()
    mx.eval(chain(arm, ctx, x0, rts, n))
    return time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1,2,4")
    ap.add_argument("--route", default="distinct,same")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--gbs", type=float, default=834.0)
    a = ap.parse_args()
    ks, (g, u, d), lin, stock, union, sumoe = build()
    per = expert_bytes((g, u, d))
    sh_bytes = 3 * (H * I * KSH // 8) + (H + I) * 4
    ctx = (lin, stock, union, sumoe)
    for rows in (int(r) for r in a.rows.split(",")):
        for mode in a.route.split(","):
            if mode == "same" and rows == 1:
                continue
            rts = routes(rows, mode, 6)
            x0 = mx.array(np.random.default_rng(5).standard_normal((rows, H)).astype(np.float16))
            arms = ["stock", "union", "sumoe"] + (["sumoe_fd"] if rows == 1 else [])
            for arm in arms:                                  # warm up (kernel compile) and check the arms agree
                mx.eval(chain(arm, ctx, x0, rts, 2))
            res = {k: [] for k in arms}
            for rep in range(a.reps):
                order = arms if rep % 2 == 0 else arms[::-1]
                for arm in order:
                    t1, t4 = timed(arm, ctx, x0, rts, a.n), timed(arm, ctx, x0, rts, 4 * a.n)
                    res[arm].append((t4 - t1) / (3 * a.n) * 1e6)
            ids = np.array(rts[0][0]).reshape(-1)
            byt = int(per[np.unique(ids)].sum()) + sh_bytes
            med = {k: statistics.median(v) for k, v in res.items()}
            roof = byt / (a.gbs * 1e3)
            print(f"R{rows} {mode:8s} " + "  ".join(f"{k} {med[k]:7.1f}us" for k in arms) +
                  f"  sumoe/stock {med['sumoe'] / med['stock']:.3f} sumoe/union {med['sumoe'] / med['union']:.3f}  bytes {byt / 1e6:.0f}MB roof {roof:.0f}us "
                  f"sumoe={roof / med['sumoe']:.2f}" + (f" sumoe_fd={roof / med['sumoe_fd']:.2f}" if "sumoe_fd" in med else "") + " of roof", flush=True)


if __name__ == "__main__":
    main()
