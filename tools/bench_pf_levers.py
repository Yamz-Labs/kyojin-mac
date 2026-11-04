# SPDX-License-Identifier: Apache-2.0
"""Bounded bench of the oppf4k prefill levers on one synthetic MoE prefill layer at real geometry (run on the Mac, not in CI).

Levers (all default 0): pf_dec 1 / 2 (cheaper exact trellis decode in the NAX GEMMs), pf_blk128 (128-row block), and the MoE pass size
(pf_moe_chunk 4096 vs 8192 rows, optionally with pf_moe_hot_fused). The geometry is the DS4.1 layer (hidden 5120, inter 2304, 6 experts per token) with
E real experts (default 48, ~0.65 GiB of weights at K = 3, peak under 3.5 GiB): the rows per expert are what the GEMM cost depends on, so the token count
is set from --rows (rows per expert; 64 = a 4096-token pass of 384 experts, 128 = an 8192-token pass) and the times scale by 384 / E to a full layer.
Arms run interleaved in random order, one warm-up round dropped, medians of --n rounds. Every arm is compared bit for bit with the first.

  python tools/bench_pf_levers.py [--rows 64] [--e 48] [--n 7] [--skew] [--passes]

Prints ms per layer for old and new (and the stage split of the first and the last arm from the profiling mode, which adds a sync per stage)."""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import mlx.core as mx  # noqa: E402

from bench_pf_moe import make_stack  # noqa: E402
from kyojin_mlx import metal_nax as NX, moe as M, moe_prefill as MP  # noqa: E402


def routing(T, E, topk, skew, seed=0):
    """ids [T, topk]: uniform, or skewed (a quarter of the experts take 75 % of the pairs, the pfroof mixture)."""
    rng = np.random.default_rng(seed)
    p = np.ones(E)
    if skew:
        p[: max(1, E // 4)] *= 9.0
    p /= p.sum()
    return np.stack([rng.choice(E, topk, replace=False, p=p) for _ in range(T)]).astype(np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=64, help="mean rows per expert (64 = a 4096-token pass over 384 experts, 128 = 8192 tokens)")
    ap.add_argument("--e", type=int, default=48, help="real experts (the layer time is scaled by 384 / e)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--inter", type=int, default=2304)
    ap.add_argument("--cb", default="mul1")
    ap.add_argument("--skew", action="store_true", help="skewed routing (hot experts)")
    ap.add_argument("--passes", action="store_true", help="also compare the pass size: 4096-row passes vs one 8192-row pass (+ hot_fused) over the 8192-token chunk")
    a = ap.parse_args()
    if not mx.metal.is_available():
        print("Metal not available")
        return 1
    mx.set_cache_limit(1 << 30)
    E, topk, kin, inter = a.e, a.topk, a.hidden, a.inter
    T = a.rows * E // topk
    scale = 384 / E
    ks = [a.k] * E
    moe = M.MoeGather(make_stack(kin, inter, ks, a.cb, 1), make_stack(kin, inter, ks, a.cb, 2), make_stack(inter, kin, ks, a.cb, 3), topk)
    rng = np.random.default_rng(0)
    w = rng.uniform(0.1, 1.0, (T, topk)).astype(np.float32)
    ids = mx.array(routing(T, E, topk, a.skew))
    wts = mx.array((w / w.sum(-1, keepdims=True) * 1.5).astype(np.float32))
    x = mx.array(rng.standard_normal((T, kin)).astype(np.float16) * np.float16(2.0))
    mx.eval(ids, wts, x)
    print(f"layer T={T} topk={topk} hidden={kin} inter={inter} E={E} (x{scale:.0f} to 384) rows/expert {a.rows} K={a.k} cb={a.cb} skew={a.skew}")

    # (name, PF_DEC, BLK128, pass rows or None, HOT_FUSED)
    arms = [("base", 0, False, None, False), ("pf_dec=1", 1, False, None, False), ("pf_dec=2", 2, False, None, False),
            ("pf_blk128", 0, True, None, False), ("pf_dec=2+blk128", 2, True, None, False)]
    if a.passes:
        half, full = a.rows * E // topk // 2, a.rows * E // topk
        arms += [("pass-half (2 passes)", 0, False, half, False), ("pass-full (1 pass)", 0, False, full, False), ("pass-full+hot_fused", 0, False, full, True)]
        # the pass arms use the production pass size scaled the same way as T: rows per expert of one pass = 64 (half) and 128 (full)
    pfs = {name: MP.MoePrefill(moe, chunk=ch) for name, _, _, ch, _ in arms}
    times = {name: [] for name, *_ in arms}
    ref, order_rng = None, np.random.default_rng(7)
    for r in range(a.n + 1):
        order = arms if r == 0 else [arms[i] for i in order_rng.permutation(len(arms))]
        for name, dec, blk, ch, hot in order:
            NX.PF_DEC, NX.BLK128, MP.HOT_FUSED = dec, blk, hot
            t0 = time.perf_counter()
            y = pfs[name](x, ids, wts)
            mx.eval(y)
            dt = (time.perf_counter() - t0) * 1e3
            if r == 0:
                arr = np.array(y)
                ref = arr if ref is None else ref
                if name != "base":
                    print(f"  {name:22s} vs base: bit-equal {np.array_equal(ref.view(np.uint32), arr.view(np.uint32))} rel {np.linalg.norm(arr - ref) / np.linalg.norm(ref):.2e}")
                continue
            times[name].append(dt)
    base = np.array(times["base"])
    print(f"{'arm':24s} {'median ms':>10s} {'x384/E ms':>10s} {'paired ratio':>14s}")
    for name, *_ in arms:
        v = np.array(times[name])
        ratio = v / base
        print(f"{name:24s} {np.median(v):10.2f} {np.median(v) * scale:10.1f} {np.median(ratio):8.3f} +- {ratio.std(ddof=1) / np.sqrt(len(ratio)):.3f}")
    for name, dec, blk, _, _ in (arms[0], arms[2]):                    # stage split: base and pf_dec=2
        NX.PF_DEC, NX.BLK128 = dec, blk
        pf = MP.MoePrefill(moe)
        mx.eval(pf(x, ids, wts))
        pf.prof = {}
        mx.eval(pf(x, ids, wts))
        print(f"stages {name:10s} " + "  ".join(f"{k} {v * 1e3:.2f}" for k, v in pf.prof.items()) + "  (ms, synced)")
    print(f"peak {mx.get_peak_memory() / 2**30:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
