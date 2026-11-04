"""pffuse task 1: time each stage of one MoE prefill chunk at DS4.1 shapes on random tensors.

Shapes: hidden 5120, inter 2304 (gate/up 5120->2304, down 2304->5120), E=384, topk=6,
chunk T=4096 rows (P=24576 pairs). Two arms: group path (S3 decode+gather_mm) and
fused EXL3+NAX path (production). Median of reps, synced per stage.
usage: flock /tmp/ultra_bench.lock python tools/bench_pffuse_stages.py [--reps 7]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "100000")
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "1000000")

import numpy as np

import mlx.core as mx

from kyojin_mlx import moe as M
from kyojin_mlx import moe_prefill as MP
from kyojin_mlx import metal_nax as NX
from kyojin_mlx import metal_pf as PF
from kyojin_mlx.moe_prefill import _had

KIN, NINTER, E, TOPK, T = 5120, 2304, 384, 6, 4096


def make_stack(kin, nout, ks, cb, seed):
    rng = np.random.default_rng(seed)
    kt, nt = kin // 16, nout // 16
    words, off = [], [0]
    suh_l, svh_l = [], []
    for K in ks:
        tr = rng.integers(-32768, 32768, (kt, nt, 16 * K), dtype=np.int16)
        suh_l.append((rng.choice([-1.0, 1.0], kin) * rng.uniform(0.5, 1.5, kin)).astype(np.float16))
        svh_l.append((rng.choice([-1.0, 1.0], nout) * rng.uniform(0.5, 1.5, nout)).astype(np.float16))
        w = np.ascontiguousarray(tr).view("<u4").reshape(-1)
        words.append(w)
        off.append(off[-1] + len(w))
    return M.Stack(mx.array(np.concatenate(words)), mx.array(np.array(off, "<u4")),
                   mx.array(np.array(ks, "u1")),
                   mx.array(np.stack(suh_l)), mx.array(np.stack(svh_l)), kt, nt, cb)


def med(x):
    return float(np.median(np.array(x, dtype=np.float64)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--cb", default="mul1")
    a = ap.parse_args()
    print(f"nax_available={NX.nax_available()} {NX.nax_status().get('reason')}", flush=True)
    ks = [2, 3] * (E // 2)
    g = make_stack(KIN, NINTER, ks, a.cb, 1)
    u = make_stack(KIN, NINTER, ks, a.cb, 2)
    d = make_stack(NINTER, KIN, ks, a.cb, 3)
    mx.eval(g.words, u.words, d.words)
    print(f"stacks built: E={E} kin={KIN} ninter={NINTER}", flush=True)

    rng = np.random.default_rng(0)
    ids_np = np.stack([rng.choice(E, TOPK, replace=False) for _ in range(T)]).astype(np.int32)
    w = rng.uniform(0.1, 1.0, (T, TOPK)).astype(np.float32)
    w = (w / w.sum(-1, keepdims=True) * 1.5).astype(np.float32)
    x_np = (rng.standard_normal((T, KIN)).astype(np.float16) * np.float16(2.0))
    ids, wts, x = mx.array(ids_np), mx.array(w), mx.array(x_np)
    mx.eval(ids, wts, x)

    x0 = mx.ones((1, 5120), mx.float16)
    mx.eval(x0)
    fl = []
    for _ in range(100):
        t = time.perf_counter()
        mx.eval(x0 * 1.0001)
        fl.append(time.perf_counter() - t)
    print(f"sync floor {min(fl) * 1e6:.0f} us", flush=True)

    dec = M.MoeGather(g, u, d, TOPK, wpre=True, fq=False)

    # ---- host plan pieces (today's _plan), timed on the side
    plan_sync, plan_cpu, plan_up = [], [], []
    pf0 = MP.MoePrefill(dec, fused=False)
    for _ in range(a.reps):
        t = time.perf_counter()
        ids_h = np.array(ids)
        mx.synchronize()
        plan_sync.append(time.perf_counter() - t)
        t = time.perf_counter()
        order, proc, cnt, cum, inv, nhot = pf0._plan(ids_h)
        plan_cpu.append(time.perf_counter() - t)
        flat_e = ids_h.reshape(-1)[order].astype(np.int32)
        t = time.perf_counter()
        mx.eval(mx.array(flat_e), mx.array((order // TOPK).astype(np.uint32)), mx.array(inv.astype(np.uint32)))
        plan_up.append(time.perf_counter() - t)
    print(f"host plan: device->host sync {med(plan_sync) * 1e3:.2f} ms | numpy bincount/argsort {med(plan_cpu) * 1e3:.2f} ms | uploads {med(plan_up) * 1e3:.2f} ms", flush=True)

    for name, kw in (("group(S3)", dict(fused=False)), ("fused(NAX)", dict(fused=True, fused_max=10 ** 9))):
        pf = MP.MoePrefill(dec, **kw)
        y = pf(x, ids, wts)
        mx.eval(y)
        acc: dict = {}
        if not kw.get("fused"):
            for _ in range(a.reps):
                ids_h = np.array(ids)
                wts_h = np.array(wts).astype(np.float32)
                order, proc, cnt, cum, inv, nhot = pf._plan(ids_h)
                G = pf.group_size()
                flat_e = ids_h.reshape(-1)[order].astype(np.int32)
                flat_w = wts_h.reshape(-1)[order]
                tok = (order // TOPK).astype(np.uint32)
                T_, topk = ids.shape
                ih, oh = x.dtype == mx.float16, pf.h
                outs = []
                tot = {"rot": 0.0, "dec_gu": 0.0, "gate_up_mm": 0.0, "act_rot": 0.0, "dec_d": 0.0, "down_mm": 0.0}
                for aa in range(0, nhot, G):
                    ex = proc[aa:min(aa + G, nhot)]
                    s0, s1 = int(cum[aa]), int(cum[aa + len(ex)])
                    n = s1 - s0
                    lidx_np = np.repeat(np.arange(len(ex), dtype=np.uint32), cnt[aa:aa + len(ex)])
                    ids_s, wts_s, tok_s = mx.array(flat_e[s0:s1]), mx.array(flat_w[s0:s1]), mx.array(tok[s0:s1])
                    lidx, eids = mx.array(lidx_np), mx.array(ex.astype(np.int32))
                    t = time.perf_counter()
                    xh = pf._rot(x, tok_s, ids_s, n, ih, oh, False)
                    mx.eval(xh)
                    tot["rot"] += time.perf_counter() - t
                    t = time.perf_counter()
                    wgu = MP.decode_group((g, u), eids, dec.cb)
                    mx.eval(wgu)
                    tot["dec_gu"] += time.perf_counter() - t
                    t = time.perf_counter()
                    inner = pf._mm(xh.reshape(2 * n, 1, g.kin), wgu, mx.concatenate([lidx, lidx + len(ex)])).reshape(2 * n, g.nout)
                    mx.eval(inner)
                    tot["gate_up_mm"] += time.perf_counter() - t
                    t = time.perf_counter()
                    xh2 = pf._mid(inner, ids_s, wts_s, n, oh, False)
                    mx.eval(xh2)
                    tot["act_rot"] += time.perf_counter() - t
                    t = time.perf_counter()
                    wd = MP.decode_group((d,), eids, dec.cb)
                    mx.eval(wd)
                    tot["dec_d"] += time.perf_counter() - t
                    t = time.perf_counter()
                    out = pf._mm(xh2.reshape(n, 1, d.kin), wd, lidx).reshape(n, d.nout)
                    mx.eval(out)
                    tot["down_mm"] += time.perf_counter() - t
                    outs.append(out)
                    del wgu, wd
                t = time.perf_counter()
                part = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
                kf = MP._kernel(("fin", True, oh), "kyojin_finpf",
                                input_names=["part", "inv", "ids", "wts", "svh", "params"],
                                output_names=["y"], source=PF.final_pf_source(True, oh))
                grid, tgp = _had(d.blocks, T_)
                y = kf(inputs=[part, mx.array(inv.astype(np.uint32)), mx.array(ids_np.reshape(-1).astype(np.int32)),
                               mx.array(wts.reshape(-1)), d.svh, mx.array([1, d.nout, TOPK], dtype=mx.uint32)],
                       grid=grid, threadgroup=tgp, output_shapes=[(T_, d.nout)], output_dtypes=[mx.float32])[0]
                mx.eval(y)
                acc.setdefault("final", []).append(time.perf_counter() - t)
                for k, v in tot.items():
                    acc.setdefault(k, []).append(v)
        else:
            for _ in range(a.reps):
                pf.prof = {}
                t = time.perf_counter()
                y = pf(x, ids, wts)
                mx.eval(y)
                acc.setdefault("e2e", []).append(time.perf_counter() - t)
                for k, v in pf.prof.items():
                    acc.setdefault("prof_" + k, []).append(v)
        keys = [k for k in acc if acc[k]]
        line = "  ".join(f"{k} {med(acc[k]) * 1e3:8.2f}" for k in keys)
        print(f"{name:12s} ms/chunk (median of {a.reps}): {line}", flush=True)
        print(f"  last={pf.last}", flush=True)


main()
