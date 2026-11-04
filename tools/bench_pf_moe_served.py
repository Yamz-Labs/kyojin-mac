# SPDX-License-Identifier: Apache-2.0
"""Prefill MoE at the SERVED shape on Metal, one arm per process (cmoe8k). Not runnable without Metal (the CPU counts are tools/pf_moe_count.py).

What it builds (bounded: the memory bound is checked BEFORE anything is allocated, default 3.5 GiB, `--budget-gib`):
  * 384 logical experts, hidden 5120, inter 2304, top-6, bitrates dealt 2,3,3,4 (2-4 classes = one pass per class). The trellis words exist for only
    `--distinct` physical experts (default 16, ~0.2 GiB); logical expert e reads physical bank e % distinct through the offset table (the kernels read
    `off[e]` only). This keeps every launch the size of the served one but FLATTERS the read rate (a 16-bank working set sits in the system cache): read
    arms against each other, never against the lane. `--distinct 64` costs ~0.85 GiB.
  * the served rows-per-expert histogram (tools/pf_moe_count.served_popularity: per 4096-row pass 355 of 384 experts <= 64 rows holding ~54.5 % of the pairs,
    29 hot experts, busiest ~1400), a different set of hot experts at every layer. An 8192-row call is two independent 4096-row draws. `--pop FILE.npy`
    replaces the popularity with a measured one (rows per expert of a served pass, any scale, normalised here).
  * `--sets N` routing sets cycled over the layers: 20 = buffer sizes change layer to layer as in a prefill chunk, 1 = the same sizes every layer (the MLX buffer
    cache always hits): the pair separates an allocator effect from a kernel effect.

One run = warm-up chunk (discarded) + `--reps` timed chunks of `--layers` layers. Each layer is one `MoePrefill.__call__` on the full call (`--rows`) followed by an
eval of its output (so memory stays bounded and every arm is timed the same way). Output: one JSON line (ms per chunk, tokens per second of the MoE alone, a
digest of the first two layers' outputs for bit comparison between arms, FD hits, peak memory against the bound). The exact command of an arm is the first line printed.

Arms: base4096 chunk8192 nosync4096 nosync8192 nosync_rowend4096 nosync_rowend8192 rowmap epi epi_cap rowmap+epi rowmap_rm64 (tools/pf_moe_count.ARMS). An arm whose conservative
memory estimate (sum of every launch output of one layer + the weights) exceeds the bound is REFUSED with its estimate; nosync arms need `--rows 4096` (one pass) or
`--budget-gib 7`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx  # noqa: E402

import pf_moe_count as C  # noqa: E402
from kyojin_mlx import fired as FD, metal_nax as NX, moe as M, moe_prefill as MP  # noqa: E402

GIB = 2 ** 30


def phys_ks(distinct: int, ks) -> list[int]:
    return [ks[p % len(ks)] for p in range(distinct)]


def weight_bytes(distinct: int, ks, hidden=C.HIDDEN, inter=C.INTER, experts=C.E_SERVED) -> int:
    kt, nt = hidden // 16, inter // 16
    words = sum(8 * k * kt * nt * 3 for k in phys_ks(distinct, ks)) * 4                    # gate + up + down, K bits per weight
    side = experts * (hidden + inter + inter + hidden + hidden + inter) * 2                  # suh / svh of the three stacks (fp16, per logical expert)
    return words + side


def make_stack(kin, nout, ks, E, distinct, cb, seed):
    """Random valid ragged stack of E logical experts over `distinct` physical banks (offset table aliasing)."""
    rng = np.random.default_rng(seed)
    kt, nt = kin // 16, nout // 16
    pk = phys_ks(distinct, ks)
    words, off = [], [0]
    for K in pk:
        w = np.ascontiguousarray(rng.integers(-32768, 32768, (kt, nt, 16 * K), dtype=np.int16)).view("<u4").reshape(-1)
        words.append(w)
        off.append(off[-1] + len(w))
    offs = np.array([off[e % distinct] for e in range(E)] + [off[-1]], "<u4")
    kk = np.array([pk[e % distinct] for e in range(E)], "u1")
    suh = (rng.choice([-1.0, 1.0], (E, kin)) * rng.uniform(0.5, 1.5, (E, kin))).astype(np.float16)
    svh = (rng.choice([-1.0, 1.0], (E, nout)) * rng.uniform(0.5, 1.5, (E, nout))).astype(np.float16)
    return M.Stack(mx.array(np.concatenate(words)), mx.array(offs), mx.array(kk), mx.array(suh), mx.array(svh), kt, nt, cb)


def estimate_gib(arm: str, rows: int, ks, distinct: int, queue: int, base_pass: int = 4096) -> tuple[float, float]:
    """(weights GiB, conservative activation GiB of `queue` layers): the CPU counter run on one layer, sum of every launch output + concatenation."""
    r = C.run_arm(arm, rows=rows, layers=1, ks=tuple(ks), seed=0, base_pass=base_pass)
    return weight_bytes(distinct, ks) / GIB, r["alloc_bytes"] * queue / GIB


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True, choices=sorted(C.ARMS))
    ap.add_argument("--rows", type=int, default=8192, help="rows of the MoE call (a CED chunk is 8192; the 4096-row arms then run two passes)")
    ap.add_argument("--layers", type=int, default=20, help="layers per chunk (the CED encoder half)")
    ap.add_argument("--sets", type=int, default=20, help="distinct routing sets cycled over the layers (1 = identical buffer sizes every layer)")
    ap.add_argument("--reps", type=int, default=5, help="timed chunks after one discarded warm-up")
    ap.add_argument("--queue", type=int, default=1, help="layers evaluated together (memory scales with it)")
    ap.add_argument("--distinct", type=int, default=16, help="physical expert banks behind the 384 logical experts")
    ap.add_argument("--ks", default="2,3,3,4")
    ap.add_argument("--budget-gib", type=float, default=3.5)
    ap.add_argument("--cache-limit-mb", type=int, default=16384, help="MLX buffer cache limit (the lane's: weights / 10, 16 GiB cap)")
    ap.add_argument("--pop", default="", help=".npy of rows per expert (or popularity) of a measured served pass")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prof", action="store_true", help="stage timers of the fused path (adds a sync per stage; for a breakdown, not for the A/B)")
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="run the whole script on the CPU with the recording kernels of tools/pf_moe_count.py (checks the Python, times nothing real)")
    a = ap.parse_args()
    ks = tuple(int(v) for v in a.ks.split(","))
    print("# command: python tools/bench_pf_moe_served.py " + " ".join(sys.argv[1:]), flush=True)
    w_gib, act_gib = estimate_gib(a.arm, a.rows, ks, a.distinct, a.queue)
    print(f"# memory bound: weights {w_gib:.2f} GiB + activations (upper bound, every launch output of {a.queue} layer(s)) {act_gib:.2f} GiB = {w_gib + act_gib:.2f} GiB "
          f"vs budget {a.budget_gib:.2f} GiB", flush=True)
    if w_gib + act_gib > a.budget_gib:
        print(json.dumps(dict(arm=a.arm, refused=True, est_gib=round(w_gib + act_gib, 2), budget_gib=a.budget_gib)))
        return 2
    if a.estimate_only:
        return 0
    restore = []
    if a.dry_run:                                       # CPU: the real host path against recording kernels; the timings printed are meaningless
        rec = C.Rec()
        restore = [(NX, "site_on", NX.site_on), (NX, "nax_available", NX.nax_available), (C.MS, "plan_device", C.MS.plan_device),
                   (mx.fast, "metal_kernel", mx.fast.metal_kernel), (mx.metal, "is_available", mx.metal.is_available)]
        NX.site_on, NX.nax_available, C.MS.plan_device = (lambda s: True), (lambda: True), C.plan_twin
        mx.fast.metal_kernel, mx.metal.is_available = rec.kernel_factory, (lambda: True)
    if not mx.metal.is_available():
        print("Metal not available")
        return 1
    if os.environ.get("KYOJIN_NAX") == "0":
        print("KYOJIN_NAX=0: the fused path is off")
        return 1
    mx.set_cache_limit(a.cache_limit_mb << 20)
    E, kin, inter = C.E_SERVED, C.HIDDEN, C.INTER
    g = make_stack(kin, inter, ks, E, a.distinct, "mul1", 1)
    u = make_stack(kin, inter, ks, E, a.distinct, "mul1", 2)
    d = make_stack(inter, kin, ks, E, a.distinct, "mul1", 3)
    moe = M.MoeGather(g, u, d, C.TOPK)
    pop = C.served_popularity(E, C.TOPK, 4096, a.seed)
    if a.pop:
        p = np.load(a.pop).astype(np.float64).reshape(-1)
        assert len(p) == E, f"--pop needs {E} values"
        pop = p / p.sum()
    sets = [mx.array(C.route(pop, a.rows, C.TOPK, a.seed, s)) for s in range(a.sets)]
    cnt = np.bincount(np.array(sets[0])[: min(a.rows, 4096)].reshape(-1), minlength=E)
    print("# routing, first 4096 rows of set 0:", C.hist_stats(cnt), flush=True)
    rng = np.random.default_rng(a.seed + 7)
    x = mx.array(rng.standard_normal((a.rows, kin)).astype(np.float16) * np.float16(2.0))
    w = rng.uniform(0.1, 1.0, (a.rows, C.TOPK)).astype(np.float32)
    wts = mx.array((w / w.sum(-1, keepdims=True) * 1.5).astype(np.float32))
    mx.eval(x, wts, *sets)
    chunk, sets_cfg = C.ARMS[a.arm]
    saved = {}
    for k, v in {**dict(MOE_ROWMAP=False, MOE_EPI=False, MOE_NOSYNC=False, NOSYNC_ROWEND=False, RM_ROWS=0, HOIST=False, ONE_PASS=False, EPI_CAP=False), **sets_cfg}.items():
        saved[k] = getattr(MP, k)
        setattr(MP, k, v)
    pf = MP.MoePrefill(moe, chunk=chunk)
    if a.prof:
        pf.prof = {}
    if not pf.fused:
        print("fused path not available on this GPU")
        return 1
    FD.ON = True
    FD.reset()

    def one_chunk(digest=False):
        outs, ys = [], []
        for l in range(a.layers):
            y = pf(x, sets[l % a.sets], wts)
            ys.append(y)
            if len(ys) == a.queue:
                mx.eval(*ys)
                if digest and len(outs) < 2:
                    outs.append(hashlib.sha1(np.array(ys[0]).tobytes()).hexdigest()[:12])
                ys = []
        if ys:
            mx.eval(*ys)
        return outs

    dig = one_chunk(digest=True)                        # warm-up (kernel builds, clocks) and the bit digest, outside the timed region
    mx.reset_peak_memory()
    ts = []
    for _ in range(a.reps):
        t0 = time.perf_counter()
        one_chunk()
        ts.append((time.perf_counter() - t0) * 1e3)
    peak = mx.get_peak_memory() / GIB
    med = float(np.median(ts))
    out = dict(arm=a.arm, rows=a.rows, layers=a.layers, sets=a.sets, queue=a.queue, distinct=a.distinct, cache_limit_mb=a.cache_limit_mb, ms=[round(t, 1) for t in ts],
               median_ms=round(med, 1), moe_tok_s=round(a.rows * 1e3 / med, 0), digest=dig, peak_gib=round(peak, 2), over_budget=bool(peak > a.budget_gib),
               fd={k: v for k, v in FD.HITS.items() if k.startswith(("pf_moe", "pf_nosync", "pf_glu", "pf_epi"))}, bailed=pf.bailed, prof=pf.prof or None)
    out["dry_run"] = a.dry_run
    print(json.dumps(out))
    for k, v in saved.items():
        setattr(MP, k, v)
    for mod, name, val in restore:
        setattr(mod, name, val)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
