#!/usr/bin/env python3
"""Union shape A (the union of the probes' own top-M keys) against shape B (the C blocks with the highest score over the probes), at equal coverage.
numpy only, no Metal, no pack: this is the experiment that chose A for `pf_cluster` (see docs/reports/pf1m_REPORT.md section 2).

usage: tools/pf1m_union_shape.py [--topk 32] [--probes 16]. Bounded: 16384 keys, under 1 GiB, seconds."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx                                            # noqa: E402

from kyojin_mlx import pfroof2 as PR                             # noqa: E402
from pf1m_recall import fp4_keys, queries                        # noqa: E402

BS = 8


def union_recall(s, probes, topk, shape, budget_blocks):
    """Mean recall over the rows that are not probes, for one union shape at `budget_blocks` blocks of coverage."""
    Tc = s.shape[1]
    nblk = -(-Tc // BS)
    if shape == "A":
        m = max(topk, budget_blocks * BS // len(probes))
        blk = set()
        for p in probes:
            blk |= set((np.argsort(-s[p])[:m] // BS).tolist())
    else:
        sp = np.pad(s[probes], [(0, 0), (0, nblk * BS - Tc)], constant_values=-np.inf)
        bm = sp.reshape(len(probes), nblk, BS).max(-1).max(0)
        blk = set(np.argsort(-bm)[:budget_blocks].tolist())
    cols = np.array(sorted(b * BS + j for b in blk for j in range(BS) if b * BS + j < Tc))
    rows = [t for t in range(s.shape[0]) if t not in probes]
    ref = np.argsort(-s[rows])[:, :topk]
    got = cols[np.argsort(-s[np.ix_(rows, cols)])[:, :topk]]
    return np.mean([len(set(a) & set(b)) / len(b) for a, b in zip(got, ref)]), len(blk) / nblk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--probes", type=int, default=16)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    T, Tc = 64, 16384
    start = 2 * (Tc - T)
    for tag, templates, qspread in (("near-identical rows", 1, 0.02), ("smooth 2 groups", 2, 0.25), ("8 groups", 8, 0.25)):
        rng = np.random.default_rng(args.seed)
        q, w = queries(rng, T, templates, qspread)
        k = fp4_keys(rng, Tc)
        s = np.array(PR.ref_scores(np.array(q[0]), np.array(k[0]), np.array(w[0]), start, 2))
        probes = sorted({(i * T) // args.probes for i in range(args.probes)} | {T - 1})
        out = [tag, f"topk {args.topk}", f"probes {len(probes)}"]
        for cov in (0.02, 0.05, 0.10, 0.25):
            budget = max(1, int(cov * (-(-Tc // BS))))
            ra, ca = union_recall(s, probes, args.topk, "A", budget)
            rb, _ = union_recall(s, probes, args.topk, "B", budget)
            out.append(f"cov {ca:.2f}: A {ra:.2f} / B {rb:.2f}")
        print("  ".join(out), flush=True)


if __name__ == "__main__":
    main()
