"""Synthetic bench of the pfsmall levers (never the real model, small tensors, no server).

- pf_moe_nosync: one prefill-MoE layer, sync (per-layer `cum` readback) against nosync, same synthetic EXL3 stacks (tests/test_moe.make_stack).
- pf_route: the router of one prompt chunk: the stock chain (cast, matmul, softplus, sqrt, bias, neg, argpartition, take, sum, div, mul) against the
  flag's two launches (the same fp32 gate matmul + one selection kernel).
- pf_rdadvise: what the advisory costs on a scattered row list (the page-cache gain itself is not measurable this way).
- pf_moe_nosync launch census: prefill-MoE launches per layer, both paths.

Arms interleaved, n >= 5, mean +- SE.  PYTHONPATH=.:<vlm_pkg> python tools/bench_pf_small.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
import mlx.core as mx  # noqa: E402

from test_moe import make_stack  # noqa: E402
from kyojin_mlx import moe as M, moe_prefill as MP  # noqa: E402


def _mean_se(v):
    v = np.asarray(v, np.float64)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0


def bench_moe(E, topk, T, kin, ninter, ks_gate, cb="mul1", n=7):
    g, _ = make_stack(kin, ninter, ks_gate, cb, 1)
    u, _ = make_stack(kin, ninter, ks_gate, cb, 2)
    d, _ = make_stack(ninter, kin, ks_gate, cb, 3)
    moe = M.MoeGather(g, u, d, topk)
    rng = np.random.default_rng(11)
    ids = np.stack([rng.choice(E, topk, replace=False) for _ in range(T)]).astype(np.int32)
    w = rng.uniform(0.1, 1.0, (T, topk)).astype(np.float32)
    w = (w / w.sum(-1, keepdims=True) * 1.5).astype(np.float32)
    x = mx.array(rng.standard_normal((T, kin)).astype(np.float16))
    pf = MP.MoePrefill(moe)
    arms = {"sync": False, "nosync": True}
    times = {k: [] for k in arms}
    outs = {}
    for it in range(-2, n):                                     # two warm-up rounds (buffer cache + kernels), then interleaved arms
        for k, on in arms.items():
            MP.MOE_NOSYNC = on
            t0 = time.perf_counter()
            y = pf(x, mx.array(ids), mx.array(w))
            mx.eval(y)
            if it >= 0:
                times[k].append(time.perf_counter() - t0)
            elif it == -1:
                outs[k] = np.array(y)
    same = np.array_equal(outs["sync"].view(np.uint16), outs["nosync"].view(np.uint16))
    print(f"MoE layer T={T} E={E} topk={topk} K={kin}/{ninter} pairs={T * topk}  bit-identical={same}")
    for k in arms:
        m, se = _mean_se(times[k])
        print(f"  {k:7s} mean {m * 1e3:8.3f} +- {se * 1e3:6.3f} ms   median {np.median(times[k]) * 1e3:8.3f} ms   n={len(times[k])}")
    return times


def bench_route(T=1024, D=2048, E=384, topk=6, n=9):
    import mlx.nn as nn
    from kyojin_mlx import fuse as F

    rng = np.random.default_rng(3)
    w = mx.array((rng.standard_normal((E, D)) * 0.02).astype(np.float32))
    bias = mx.array(rng.standard_normal(E).astype(np.float32) * 0.1)
    x = mx.array(rng.standard_normal((1, T, D)).astype(np.float16))

    def stock():
        scores = mx.sqrt(nn.softplus(x.astype(mx.float32) @ w.T))
        inds = mx.argpartition(-(scores + bias), kth=topk - 1, axis=-1)[..., :topk].astype(mx.int32)
        inds = mx.stop_gradient(inds)
        wts = mx.take_along_axis(scores, inds, axis=-1)
        wts = wts / (wts.sum(axis=-1, keepdims=True) + 1e-20)
        return inds, wts * 2.5

    def fused():
        lg = (x.astype(mx.float32) @ w.T).reshape(-1, E)
        return F.router_select(lg, bias, topk, True, 2.5)

    times = {"stock": [], "pf_route": []}
    outs = {}
    for it in range(-1, n):
        for k, fn in (("stock", stock), ("pf_route", fused)):
            t0 = time.perf_counter()
            o = fn()
            mx.eval(*o)
            if it >= 0:
                times[k].append(time.perf_counter() - t0)
            else:
                outs[k] = o
    mi, mw = (np.array(v).reshape(T, topk) for v in outs["stock"])
    ai, aw = (np.array(v).reshape(T, topk) for v in outs["pf_route"])
    print(f"router chunk T={T} D={D} E={E}  ids_equal={np.array_equal(mi, ai)} weights_equal={np.array_equal(mw, aw)}")
    for k in times:
        m, se = _mean_se(times[k])
        print(f"  {k:9s} mean {m * 1e3:8.3f} +- {se * 1e3:6.3f} ms   median {np.median(times[k]) * 1e3:8.3f} ms   n={len(times[k])}")
    return times


def bench_advise(rows=10000, width=264):
    from kyojin_mlx import engram_rows as ER

    import tempfile
    d = tempfile.mkdtemp()
    path = os.path.join(d, "t.bin")
    with open(path, "wb") as fh:
        fh.write(b"\0" * (rows * width + 16))
    fd = os.open(path, os.O_RDONLY)
    r = np.random.default_rng(5).choice(rows * 8, rows, replace=False).astype(np.int64)
    os.environ["KYOJIN_PF_RDADVISE"] = "1"
    t0 = time.perf_counter()
    n_runs = ER.advise_rows(fd, 16, r, width)
    dt = time.perf_counter() - t0
    os.close(fd)
    print(f"rdadvise {rows} scattered rows -> {n_runs} advisory syscalls (skipped={ER.ADV['skipped']}) in {dt * 1e3:.2f} ms")


def census(E=64, topk=6, T=512, kin=512, ninter=768):
    """Launches per prefill-MoE layer, both paths (counted on the metal_kernel calls of kyojin_mlx.moe_prefill)."""
    g, _ = make_stack(kin, ninter, [2] * E, "mul1", 1)
    u, _ = make_stack(kin, ninter, [3] * E, "mul1", 2)
    d, _ = make_stack(ninter, kin, [2] * E, "mul1", 3)
    moe = M.MoeGather(g, u, d, topk)
    rng = np.random.default_rng(7)
    ids = mx.array(np.stack([rng.choice(E, topk, replace=False) for _ in range(T)]).astype(np.int32))
    w = mx.array(rng.uniform(0.1, 1.0, (T, topk)).astype(np.float32))
    x = mx.array(rng.standard_normal((T, kin)).astype(np.float16))
    pf = MP.MoePrefill(moe)
    for k, on in (("sync", False), ("nosync", True)):
        MP.MOE_NOSYNC = on
        y = pf(x, ids, w)
        mx.eval(y)
        print(f"  census {k}: pairs={pf.last.get('pairs')} active={pf.last.get('active')} nosync={pf.last.get('nosync', False)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=7)
    a = ap.parse_args()
    if not mx.metal.is_available():
        raise SystemExit("Metal not available")
    bench_moe(E=48, topk=6, T=512, kin=1024, ninter=1536, ks_gate=[2, 3, 2, 3, 2, 3] * 8, n=a.n)
    bench_route(n=max(5, a.n))
    bench_advise()
    census()
