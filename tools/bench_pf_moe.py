# SPDX-License-Identifier: Apache-2.0
"""Bounded microbench of the prefill MoE copy levers (pf_moe_rowmap, pf_moe_epi) on synthetic ragged stacks.

Synthetic only: random valid trellis words of one bitrate, a hidden of 5120 and an intermediate of 2304 (the DS4.1
shape) with 64 experts, so the weighted banks stay around 0.9 GiB and the whole run sits well under 4 GiB. Arms run
interleaved (one variable, the two flags), median of n >= 3; `--bytes` prints what the row map stops writing/reading.

  python tools/bench_pf_moe.py [--t 1024] [--e 64] [--k 3] [--n 7] [--cb mul1]

In-situ geometry (the served pass: 384 experts, a mixed bitrate pack, skewed routing, 4096 / 8192-row passes, many layers queued per eval):
  python tools/bench_pf_moe.py --e 384 --ks 2,3,4 --skew 1.0 --t 4096 --queue 4 --arms current,rowmap,epi,nosync
(weights: 3 x E x ~9 MiB at K=3, ~10 GiB at E=384; use --e 128 on a busy box.)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import mlx.core as mx  # noqa: E402

from kyojin_mlx import moe as M, moe_prefill as MP  # noqa: E402


def make_stack(kin, nout, ks, cb, seed):
    """Random valid ragged stack (same layout as tests/test_moe.make_stack)."""
    rng = np.random.default_rng(seed)
    kt, nt = kin // 16, nout // 16
    words, off, suhs, svhs = [], [0], [], []
    for K in ks:
        tr = rng.integers(-32768, 32768, (kt, nt, 16 * K), dtype=np.int16)
        suhs.append((rng.choice([-1.0, 1.0], kin) * rng.uniform(0.5, 1.5, kin)).astype(np.float16))
        svhs.append((rng.choice([-1.0, 1.0], nout) * rng.uniform(0.5, 1.5, nout)).astype(np.float16))
        w = np.ascontiguousarray(tr).view("<u4").reshape(-1)
        words.append(w)
        off.append(off[-1] + len(w))
    return M.Stack(mx.array(np.concatenate(words)), mx.array(np.array(off, "<u4")), mx.array(np.array(ks, "u1")),
                   mx.array(np.stack(suhs)), mx.array(np.stack(svhs)), kt, nt, cb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=int, default=1024, help="prompt tokens per chunk")
    ap.add_argument("--e", type=int, default=64, help="experts")
    ap.add_argument("--k", type=int, default=3, help="one bitrate for every expert")
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--n", type=int, default=7, help="timed rounds per arm")
    ap.add_argument("--cb", default="mul1")
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--inter", type=int, default=2304)
    ap.add_argument("--ks", default="", help="comma list of bitrates dealt to the experts in turn (mixed pack: one pass per class); default: --k for all")
    ap.add_argument("--skew", type=float, default=0.0, help="routing popularity ~ rank^-skew (0 = uniform); real traffic has hot experts")
    ap.add_argument("--queue", type=int, default=1, help="calls queued per mx.eval (the served pass queues every layer's work before one eval)")
    ap.add_argument("--arms", default="current,rowmap,epi,rowmap+epi", help="subset of current,rowmap,epi,rowmap+epi,nosync,nosync+y32")
    a = ap.parse_args()
    if not mx.metal.is_available():
        print("Metal not available")
        return 1
    if os.environ.get("KYOJIN_NAX") == "0":
        print("KYOJIN_NAX=0: the fused path is off, these levers need the NAX GEMM")
        return 1
    mx.set_cache_limit(256 << 20)                       # the box also serves: small cache, sized buffers, paired rounds
    T, E, topk = a.t, a.e, a.topk
    kin, inter = a.hidden, a.inter
    ks = [int(v) for v in a.ks.split(",")] if a.ks else [a.k]
    ks = [ks[i % len(ks)] for i in range(E)]
    g = make_stack(kin, inter, ks, a.cb, 1)
    u = make_stack(kin, inter, ks, a.cb, 2)
    d = make_stack(inter, kin, ks, a.cb, 3)
    moe = M.MoeGather(g, u, d, topk)
    rng = np.random.default_rng(0)
    pop = None
    if a.skew > 0:
        pop = np.arange(1, E + 1, dtype=np.float64) ** -a.skew
        pop = rng.permutation(pop / pop.sum())
    ids_np = np.stack([rng.choice(E, topk, replace=False, p=pop) for _ in range(T)]).astype(np.int32)
    ids = mx.array(ids_np)
    cnt = np.bincount(ids_np.reshape(-1), minlength=E)
    print(f"rows per expert: mean {cnt[cnt > 0].mean():.0f} max {cnt.max()} active {(cnt > 0).sum()}; bitrates {sorted(set(ks))}")
    w = rng.uniform(0.1, 1.0, (T, topk)).astype(np.float32)
    wts = mx.array((w / w.sum(-1, keepdims=True) * 1.5).astype(np.float32))
    x = mx.array(rng.standard_normal((T, kin)).astype(np.float16) * np.float16(2.0))
    mx.eval(ids, wts, x)
    P = T * topk
    print(f"shape T={T} topk={topk} hidden={kin} inter={inter} E={E} K={a.k} cb={a.cb}; P={P}")
    print(f"rot copy per layer: 2P*K*2 = {2 * P * kin * 2 / 2**20:.0f} MiB; gate/up product: 2P*inter*2 = {2 * P * inter * 2 / 2**20:.0f} MiB")
    allarms = {"current": (False, False, False, 0), "rowmap": (True, False, False, 0), "epi": (False, True, False, 0), "rowmap+epi": (True, True, False, 0),
               "nosync": (False, False, True, 0), "nosync+y32": (False, False, True, 32)}
    arms = [(n, *allarms[n]) for n in a.arms.split(",")]
    pfs = {arm[0]: MP.MoePrefill(moe) for arm in arms}
    times = {arm[0]: [] for arm in arms}
    from kyojin_mlx import metal_nax as NX
    ref = None
    rng_order = np.random.default_rng(7)
    for r in range(a.n + 1):
        order = arms if r == 0 else [arms[i] for i in rng_order.permutation(len(arms))]
        for name, rm, epi, ns, ny in order:
            MP.MOE_ROWMAP, MP.MOE_EPI, MP.MOE_NOSYNC, NX.NOSYNC_Y = rm, epi, ns, ny
            y = pfs[name](x, ids, wts)
            mx.eval(y)
            if r == 0:
                arr = np.array(y)
                if ref is None:
                    ref = arr
                else:
                    print(f"{name:12s} vs current: bit-equal {np.array_equal(ref.view(np.uint32), arr.view(np.uint32))} "
                          f"rel {np.linalg.norm(arr - ref) / np.linalg.norm(ref):.2e}")
                continue
            import time
            t0 = time.perf_counter()
            mx.eval([pfs[name](x, ids, wts) for _ in range(a.queue)])
            times[name].append((time.perf_counter() - t0) * 1e3 / a.queue)
    base = np.array(times["current"])
    for name, *_ in arms:
        v = np.array(times[name])
        ratio = v / base
        print(f"{name:12s} median {np.median(v):8.2f} ms  mean {v.mean():8.2f} +- {v.std(ddof=1) / np.sqrt(len(v)):.2f}  "
              f"min {v.min():8.2f}  paired {ratio.mean():.3f} +- {ratio.std(ddof=1) / np.sqrt(len(ratio)):.3f}x  "
              f"min-pair {np.sort(ratio)[len(ratio) // 2]:.3f}x")
    print(f"peak {mx.get_peak_memory() / 2**30:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
