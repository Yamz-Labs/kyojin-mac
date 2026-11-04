#!/usr/bin/env python3
"""opbw: prefill routed-MoE cost vs the row-block that decodes each weight tile, on the real layer geometry.

Why: every row block of an expert decodes its trellis weights again (metal_nax.coop_source: the decode sits inside the block's k loop), so the
weight traffic of one prefill chunk is sum_e ceil(cnt_e / rows_per_block) * expert_bytes. A 4096-row pass and an 8192-row pass of the default
tiling both end at 2 decodes per expert; the 128-row block (pf_blk128) makes the 8192-row pass decode each expert once.

Arms (all on the same routing, one variable each, interleaved, first round dropped):
  two-passes-4096   today's default: two 4096-row passes, default tiling (64-row blocks)
  one-pass-8192     sfw: one 8192-row pass, default tiling (64-row blocks, 2 per expert)
  one-pass-8192-b128  one 8192-row pass, pf_blk128 (128-row blocks, 1 per expert)
  one-pass-8192-d1    one 8192-row pass, pf_blk128 + pf_dec 1
  two-passes-4096-b128  two 4096-row passes, pf_blk128
Prints median ms scaled to the full 384-expert layer, the weight bytes actually decoded (from the row block each expert lands in), effective
GB/s, and bit equality with the first arm.

usage: opbw_pfmoe.py [--e 48] [--n 7] [--skew] [--rows 128] [--json F]
Bounds: E real experts (default 48) x 3 projections at K=3, ~0.7 GiB of weights, peak < 3.5 GiB.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import mlx.core as mx  # noqa: E402
from bench_pf_moe import make_stack  # noqa: E402
from bench_pf_levers import routing  # noqa: E402
from kyojin_mlx import metal_nax as NX, moe as M, moe_prefill as MP  # noqa: E402

# (name, BLK128, PF_DEC, passes, pass_rows_per_expert)
ARMS = [("two-passes-4096", False, 0, 2, 64), ("one-pass-8192", False, 0, 1, 128),
        ("one-pass-8192-b128", True, 0, 1, 128), ("one-pass-8192-d1", True, 1, 1, 128),
        ("two-passes-4096-b128", True, 0, 2, 64)]


def expert_bytes(kin, inter, k, e):
    """Packed bytes of one expert's gate + up + down at bitrate k."""
    return (2 * kin * inter + inter * kin) * k / 8.0 * e


def blocks(counts, rows, nmb):
    """Row blocks an expert of `counts` rows pays with a block of `rows` rows (a tail that fits the first sub-block costs one block)."""
    full, tail = divmod(int(counts), rows)
    return full + (1 if tail else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e", type=int, default=48)
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--rows", type=int, default=128, help="rows per expert of ONE 8192-row pass")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--inter", type=int, default=2304)
    ap.add_argument("--cb", default="mul1")
    ap.add_argument("--skew", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    if not mx.metal.is_available():
        print("Metal not available")
        return 1
    mx.set_cache_limit(1 << 30)
    E, topk, kin, inter = a.e, a.topk, a.hidden, a.inter
    T = a.rows * E // topk                                   # tokens of one full pass
    scale = 384 / E
    ks = [a.k] * E
    moe = M.MoeGather(make_stack(kin, inter, ks, a.cb, 1), make_stack(kin, inter, ks, a.cb, 2),
                      make_stack(inter, kin, ks, a.cb, 3), topk)
    rng = np.random.default_rng(0)
    w = rng.uniform(0.1, 1.0, (T, topk)).astype(np.float32)
    ids = mx.array(routing(T, E, topk, a.skew))
    wts = mx.array((w / w.sum(-1, keepdims=True) * 1.5).astype(np.float32))
    x = mx.array(rng.standard_normal((T, kin)).astype(np.float16) * np.float16(2.0))
    mx.eval(ids, wts, x)
    counts = np.bincount(np.array(ids).reshape(-1), minlength=E)
    eb = expert_bytes(kin, inter, a.k, 1)
    print(f"layer E={E} (x{scale:.0f} -> 384) rows/expert mean {counts.mean():.0f} max {counts.max()} "
          f"kin {kin} inter {inter} K={a.k} skew={a.skew} one pass = {T} tokens")
    # weight bytes each arm decodes: blocks of the block size that arm uses, over one sweep of the bank
    res = {"geometry": dict(E=E, rows=int(counts.mean()), max_rows=int(counts.max()), kin=kin, inter=inter, k=a.k,
                            bytes_per_expert=eb, tokens_full_pass=T), "arms": {}}
    for name, blk, dec, passes, _ in ARMS:
        rows = 64 if not blk else 128
        # a pass of the chunk carries counts/passes rows per expert (the routing is the same text), and every block decodes its weights again
        nb = int(sum(blocks(c, rows, 1) for c in counts)) if passes == 1 else \
            passes * int(sum(blocks(-(-c // passes), rows, 1) for c in counts))
        res["arms"].setdefault(name, {})["blocks"] = nb
        res["arms"][name]["weight_gb"] = nb * eb * scale / 1e9

    pfs = {name: MP.MoePrefill(moe, chunk=T if passes == 1 else T // 2) for name, _, _, passes, _ in ARMS}
    times = {name: [] for name, *_ in ARMS}
    ref = None
    order_rng = np.random.default_rng(7)
    for r in range(a.n + 1):
        order = ARMS if r == 0 else [ARMS[i] for i in order_rng.permutation(len(ARMS))]
        for name, blk, dec, passes, _ in order:
            NX.BLK128, NX.PF_DEC = blk, dec
            pf = pfs[name]
            halves = [(x, ids, wts)] if passes == 1 else [(x[: T // 2], ids[: T // 2], wts[: T // 2]),
                                                          (x[T // 2:], ids[T // 2:], wts[T // 2:])]
            t0 = time.perf_counter()
            for xa, ida, wa in halves:                        # every pass evaluated: an unevaluated mlx graph is dropped, not run
                y = pf(xa, ida, wa)
                mx.eval(y)
            dt = (time.perf_counter() - t0) * 1e3
            arr = np.array(y)
            if r == 0:
                ref = arr
                print(f"  {name:22s} bit-equal {np.array_equal(ref.view(np.uint32), arr.view(np.uint32))}")
                continue
            times[name].append(dt)
    print(f"{'arm':24s} {'blocks':>7s} {'GB':>6s} {'ms':>8s} {'x384/E':>8s} {'GB/s':>7s} {'ratio':>7s}")
    for name, *_ in ARMS:
        v = np.array(times[name])
        base = np.array(times[ARMS[0][0]])
        med = float(np.median(v))
        gb = res["arms"][name]["weight_gb"]
        ratio = v / base
        res["arms"][name].update(ms=med, ms_full=med * scale, gbs=gb / (med * scale * 1e-3),
                                 ratio=float(np.median(ratio)), ratio_se=float(ratio.std(ddof=1) / np.sqrt(len(ratio))))
        print(f"{name:24s} {res['arms'][name]['blocks']:7d} {gb:6.1f} {med:8.2f} {med * scale:8.1f} {gb / (med * 1e-3):7.0f} "
              f"{res['arms'][name]['ratio']:7.3f} +- {res['arms'][name]['ratio_se']:.3f}")
    # ---- row-block sweep at a fixed pass structure: does a 128-row block that keeps bns = 32 (two 32-column tiles per simdgroup) pay?
    orig_pick = NX.pick_cfg
    one = pfs["one-pass-8192"]
    cfgs = {"64 (32,2,32)": (32, 2, 32, 64), "128 (32,4,16)": (32, 4, 16, 64), "128 (32,4,32)": (32, 4, 32, 64),
            "64 (32,2,16)": (32, 2, 16, 64)}
    res["cfg_sweep"] = {}
    print(f"\n{'block cfg':16s} {'ms':>8s} {'x384/E':>8s} {'ratio':>7s}  (one 8192-row pass, forced cfg)")
    base_cfg_ms = None
    for name, cfg in cfgs.items():
        NX.pick_cfg = lambda counts, allow128=False, _c=cfg: _c
        pf = MP.MoePrefill(moe, chunk=T)
        try:
            t0 = time.perf_counter()
            mx.eval(pf(x, ids, wts))
            t0 = time.perf_counter()
            vs = []
            for _ in range(a.n):
                mx.eval(pf(x, ids, wts))
                vs.append((time.perf_counter() - t0) * 1e3)
                t0 = time.perf_counter()
        finally:
            NX.pick_cfg = orig_pick
        med = float(np.median(vs))
        base_cfg_ms = base_cfg_ms or med
        res["cfg_sweep"][name] = dict(ms=med, ratio=med / base_cfg_ms)
        print(f"{name:16s} {med:8.2f} {med * scale:8.1f} {med / base_cfg_ms:7.3f}")
    print(f"peak {mx.get_peak_memory() / 2**30:.2f} GiB   (spread over {a.n} rounds, first dropped)")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
        print(f"json -> {a.json}")
    NX.BLK128, NX.PF_DEC = False, 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
