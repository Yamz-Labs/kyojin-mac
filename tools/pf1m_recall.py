#!/usr/bin/env python3
"""pf_cluster recall against the exact full-width top-k, on the CPU (numpy score mirror, no Metal, no pack). Prints the table quoted in
docs/reports/pf1m_REPORT.md section 5: recall of the rows between the probes and the fraction of the keys the union actually scored, for the shipped
defaults and for a wider arm.

usage: tools/pf1m_recall.py [--seed 4] [--topk 32]. Bounded: key counts <= 16384, under 1 GiB, under a minute."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import mlx.core as mx                                            # noqa: E402  (MLX CPU only: no Metal kernel is called)

from kyojin_mlx import idx_topk as IT                            # noqa: E402
from kyojin_mlx import pf_cluster as PC                          # noqa: E402
from kyojin_mlx import pf_flat as PF                             # noqa: E402
from kyojin_mlx import pfroof2 as PR                             # noqa: E402

NH, D, BS = 32, 128, 8
CODES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)      # the fp4 ue8m0 grid of the index keys


def fp4_keys(rng, Tc, centroids=24, spread=0.35):
    """Keys shaped like the pack's: every value a code of the fp4 grid times a power-of-two scale per 32-value block, drawn around `centroids` centres."""
    c = rng.standard_normal((centroids, D)).astype(np.float32)
    own = rng.integers(0, centroids, Tc)
    k = np.array(mx.array(c[own] + spread * rng.standard_normal((Tc, D)).astype(np.float32)).astype(mx.float16), dtype=np.float32)
    blk = k.reshape(Tc, D // 32, 32)
    scale = np.exp2(np.ceil(np.log2(np.abs(blk).max(-1, keepdims=True) + 1e-9)) - 2.0)
    q = np.clip(blk / scale, -6, 6)
    code = np.abs(CODES)[np.abs(np.abs(q)[..., None] - np.abs(CODES)).argmin(-1)]
    return mx.array((np.sign(q) * code * scale).reshape(1, Tc, D).astype(np.float16))


def queries(rng, T, templates, spread):
    """Query rows in `templates` groups (a prompt block whose neighbours ask nearly the same thing), plus the head weights."""
    c = rng.standard_normal((templates, NH, D)).astype(np.float32)
    q = c[np.arange(T) * templates // T] + spread * rng.standard_normal((T, NH, D)).astype(np.float32)
    return mx.array(q[None].astype(np.float16)), mx.array((rng.standard_normal((T, NH)) * 0.05).astype(np.float16)[None])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--topk", type=int, default=32)
    args = ap.parse_args()
    PF.METAL_OPT = False                                          # numpy mirrors for the scores and the rescoring
    PC.ENABLED, PC.MIN_KEYS, PC.MAXPCT, PC.CAP = True, 0, 100, 8192

    cases = [("fp4, 6 query groups", 48, 8192, 6, 0.25, "fp4", 16, 4),
             ("fp4, 12 query groups", 64, 16384, 12, 0.25, "fp4", 16, 4),
             ("fp4, one query group", 32, 8192, 1, 0.05, "fp4", 16, 4),
             ("gaussian, one group", 32, 8192, 1, 0.05, "gauss", 16, 4),
             ("fp4, 12 query groups", 64, 16384, 12, 0.25, "fp4", 32, 16)]
    print("| keys | rows x keys | probes x mult | recall | keys scored / keys |")
    print("|---|---|---|---|---|")
    for tag, T, Tc, templates, spread, kind, probes, mult in cases:
        rng = np.random.default_rng(args.seed)
        q, w = queries(rng, T, templates, spread)
        k = fp4_keys(rng, Tc) if kind == "fp4" else mx.array(rng.standard_normal((1, Tc, D)).astype(np.float16))
        start = 4 * (Tc - T)
        ref = np.array(IT.topk(mx.array(PR.ref_scores(np.array(q[0]), np.array(k[0]), np.array(w[0]), start, 4)), args.topk))
        PC.PROBES, PC.MULT = probes, mult
        for kk in PC.HITS:
            PC.HITS[kk] = 0
        os.environ[IT.QBLOCK_ENV] = str(T)
        try:
            got = np.array(PC.plain_topk(q, k, w, start, 4, args.topk, T))
        finally:
            os.environ.pop(IT.QBLOCK_ENV, None)
        rec = float(np.mean([len(set(a) & set(b)) / len(b) for a, b in zip(got, ref)]))
        cov = PC.HITS["union_keys"] / max(PC.HITS["blocks"], 1) / Tc
        print(f"| {tag} | {T} x {Tc} | {probes} x {mult} | **{rec:.2f}** | {cov:.2f} |", flush=True)
    print("\nAt these key counts the union covers 65-100 % of the keys, so the recall tracks the coverage: these rows check the machinery, they do not "
          "predict the served recall at 1M (see pf_idx_cluster_recall, docs/reports/pf1m_REPORT.md).")


if __name__ == "__main__":
    main()
