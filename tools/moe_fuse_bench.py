#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""OC_moefuse step 1 (+2 gate): routed MoE decode at real DS4.1 dims on synthetic EXL3 stacks.

Builds gate/up/down stacks at the DS4.1 trunk dims (H=5120, I=2304, 384 experts, per-expert
K taken from the real pack's layers.10.ffn expert_bits: 243x K=2, 141x K=3) with random valid
trellises, then times MoeGather (default) or the election-sharing variant (``--impl fused``,
KYOJIN_MOEFUSE=1: grouped=False, union=True) at R=1 and R=4.

Per call: slope of chains of n and 4n calls with fresh inputs in one eval (no per-eval floor),
min over --reps. Launches: counted from the path taken. Bytes: trellis words of the distinct
selected experts (gate, up, down) + their suh/svh. Roof = bytes / --gbs (default: the measured
Ultra stream-read median, 834 GB/s).
usage: flock /tmp/ultra_bench.lock python tools/moe_fuse_bench.py [--rows 1,4] [--route distinct,same] [--impl cur|fused] [--reps 5]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kyojin_mlx import moe as M  # noqa: E402

H, I, E, TOPK = 5120, 2304, 384, 6
# real pack layers.10.ffn expert_bits (K per expert, same list for gate/up/down)
KS = [3, 3, 3, 3, 2, 3, 2, 2] * 48  # placeholder, replaced below by the exact 384-list when available
_KS_REAL = None


def real_ks():
    global _KS_REAL
    if _KS_REAL is None:
        try:
            cfg = json.load(open(os.path.expanduser("~/ds41/model/config.json")))
            _KS_REAL = list(cfg["quantization_config"]["expert_bits"]["layers.10.ffn"])
            assert len(_KS_REAL) == 384 and set(_KS_REAL) <= {2, 3}
        except Exception:
            rng = np.random.default_rng(7)
            _KS_REAL = [3 if r < 141 / 384 else 2 for r in rng.random(384)]
    return _KS_REAL


def make_stack(kin, nout, ks, seed):
    rng = np.random.default_rng(seed)
    kt, nt = kin // 16, nout // 16
    words, off, suhs, svhs = [], [0], [], []
    for K in ks:
        tr = rng.integers(-32768, 32768, (kt, nt, 16 * K)).astype(np.int16)
        w = np.ascontiguousarray(tr).view("<u4").reshape(-1)
        words.append(w)
        off.append(off[-1] + len(w))
        suhs.append((rng.choice([-1.0, 1.0], kin) * rng.uniform(0.5, 1.5, kin)).astype(np.float16))
        svhs.append((rng.choice([-1.0, 1.0], nout) * rng.uniform(0.5, 1.5, nout)).astype(np.float16))
    return M.Stack(mx.array(np.concatenate(words)), mx.array(np.array(off, "<u4")),
                   mx.array(np.array(ks, "u1")), mx.array(np.stack(suhs)), mx.array(np.stack(svhs)),
                   kt, nt, "mul1")


def expert_bytes(st):
    per = np.zeros(st[0].experts, dtype=np.int64)
    for s in st:
        off = np.array(s.offset).astype(np.int64)
        per += (off[1:] - off[:-1]) * 4 + (s.kin + s.nout) * 2
    return per


def launches(m, rows):
    nks = lambda ks: len(ks) if M.VB_SPLIT_K else 1
    if getattr(m, "is_fused", False) or m.__class__.__name__ == "MoeFused":
        return 5  # rot + moefuse_gu + mid + moefuse_dn + final (no plan/split passes at R>1)
    if rows == 1:
        return 5
    return 1 + 1 + nks(m.ks_gu) + 1 + nks(m.ks_dn) + 1


def burn_in(f, R, topk):
    rng = np.random.default_rng(999)
    ids = mx.array(rng.integers(0, E, (R, topk)).astype(np.int32))
    x = mx.array((rng.standard_normal((R, H)) * 0.5).astype(np.float16))
    w = mx.array(rng.random((R, topk)).astype(np.float32))
    mx.eval(x, ids, w)
    for _ in range(30):
        mx.eval(f(x, ids, w, None))
    mx.synchronize()


def timeit(f, ins):
    n = len(ins)

    def run(c):
        outs = [f(*ins[i % n]) for i in range(c)]
        t0 = time.perf_counter()
        mx.eval(*outs)
        return time.perf_counter() - t0

    T = {n: [], 4 * n: []}
    for r in range(4):
        for c in T:
            dt = run(c)
            if r:
                T[c].append(dt)
    return (min(T[4 * n]) - min(T[n])) / (3 * n) * 1e6


def make_inputs(rng, R, route, chain):
    ins = []
    for _ in range(chain):
        if route == "same":
            one = rng.choice(E, TOPK, replace=False)
            ids = np.stack([one] * R)
        else:
            ids = np.stack([rng.choice(E, TOPK, replace=False) for _ in range(R)])
        ins.append((mx.array((rng.standard_normal((R, H)) * 0.5).astype(np.float16)),
                    mx.array(ids.astype(np.int32)),
                    mx.array((rng.random((R, TOPK))).astype(np.float32)), ids))
    mx.eval(*[v for i in ins for v in i[:3]])
    return ins


def stages(m, x, ids, wts, rows):
    ids_f, wts_f = ids.reshape(-1).astype(mx.int32), wts.reshape(-1).astype(mx.float32)
    xh = m.rot_gu(x, ids_f, True)
    part = m.gemv_gu(xh, ids_f, rows)
    xh2 = m.mid(part, ids_f, wts_f, rows)
    part2 = m.gemv_dn(xh2, ids_f, rows)
    mx.eval(xh, part, xh2, part2)
    res = {}
    for name, f in (("rot_gu", lambda: m.rot_gu(x, ids_f, True)),
                    ("gemv_gu", lambda: m.gemv_gu(xh, ids_f, rows)),
                    ("mid", lambda: m.mid(part, ids_f, wts_f, rows)),
                    ("gemv_dn", lambda: m.gemv_dn(xh2, ids_f, rows)),
                    ("final", lambda: m.final(part2, ids_f, wts_f, rows))):
        res[name] = min(timeit(lambda *_: f(), [(0,)] * 8) for _ in range(3))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", default="1,4")
    ap.add_argument("--route", default="distinct,same")
    ap.add_argument("--impl", default="cur", choices=["cur", "fused"])
    ap.add_argument("--gbs", type=float, default=834.0)
    ap.add_argument("--chain", type=int, default=8)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--stages", action="store_true")
    ap.add_argument("--gate", action="store_true", help="MoeUnfused reference check (bit-identical / rel L2)")
    a = ap.parse_args()
    mx.set_cache_limit(256 << 20)
    ks = real_ks()
    t0 = time.time()
    g, u, d = make_stack(H, I, ks, 1), make_stack(H, I, ks, 2), make_stack(I, H, ks, 3)
    mx.eval(g.words, u.words, d.words)
    print(f"stacks built in {time.time() - t0:.1f} s, peak {mx.get_peak_memory() / 2 ** 30:.2f} GiB", flush=True)
    m = M.MoeGather(g, u, d, TOPK, limit=10.0, wpre=True, fq=True)
    if a.impl == "fused":
        from kyojin_mlx import moe_fused as MF
        m = MF.wrap(m)
    per = expert_bytes((g, u, d))
    print(f"ds41-dims: {E} experts topk {TOPK} H {H} I {I}, K2x{(np.array(ks) == 2).sum()} K3x{(np.array(ks) == 3).sum()}, "
          f"mean {per.mean() / 1e6:.2f} MB/expert, impl {a.impl}", flush=True)
    rng = np.random.default_rng(0)
    out = {}
    for route in a.route.split(","):
        for R in [int(v) for v in a.rows.split(",")]:
            if route == "same" and R == 1:
                continue
            ins = make_inputs(rng, R, route, a.chain)
            f = lambda x, i, w, _ids: m(x, i, w, fq8=True)
            burn_in(f, R, TOPK)
            us = min(timeit(f, ins) for _ in range(a.reps))
            nb = float(np.mean([per[np.unique(i[3])].sum() for i in ins]))
            roof = nb / (a.gbs * 1e3)
            if a.stages and R == 1:
                x0, i0, w0, _ = ins[0]
                sg = stages(m, x0, i0, w0, R)
                print("   stages us: " + "  ".join(f"{k} {v:.1f}" for k, v in sg.items()) + f"  sum {sum(sg.values()):.1f}", flush=True)
            out[f"{route}_r{R}"] = dict(us=round(us, 1), launches=launches(m, R), MB=round(nb / 1e6, 1),
                                        roof_us=round(roof, 1), x_roof=round(us / roof, 2))
            print(f"{route:8s} R{R}: {us:8.1f} us  launches {launches(m, R):2d}  read {nb / 1e6:7.1f} MB  "
                  f"roof {roof:7.1f} us  ({us / roof:.2f}x roof)", flush=True)
    if a.gate:
        # like-for-like: the impl under test vs a fresh MoeGather with the same construction
        # at the production flags (fq=True, fq8=True). The Gather-vs-Unfused equivalence itself
        # is covered by tests/test_moe.py + the fq=False leg below (exact at these dims).
        ref = M.MoeGather(g, u, d, TOPK, limit=10.0, wpre=True, fq=True)
        rng = np.random.default_rng(11)
        maxrel, biteq = 0.0, True
        for R in [1, 4]:
            for t in range(3):
                ids = np.stack([rng.choice(E, TOPK, replace=False) for _ in range(R)]).astype(np.int32)
                x = mx.array((rng.standard_normal((R, H)) * 0.5).astype(np.float16))
                w = mx.array(rng.random((R, TOPK)).astype(np.float32))
                yo, yref = m(x, mx.array(ids), w, fq8=True), ref(x, mx.array(ids), w, fq8=True)
                mx.eval(yo, yref)
                ao, aref = np.array(yo), np.array(yref)
                maxrel = max(maxrel, float(np.abs(ao - aref).max() / (np.abs(aref).max() + 1e-30)))
                biteq = biteq and bool(mx.array_equal(yo, yref).item())
        print(f"gate: vs MoeGather(prod flags) maxrel {maxrel:.3e}  bit-identical {biteq}", flush=True)
        out["gate"] = dict(maxrel=maxrel, bit_identical=biteq)
        if a.impl == "cur":
            # same-flags Gather(fq=False) vs the per-expert reference (mlx_vlm fq unavailable here)
            r = M.MoeUnfused(g, u, d, TOPK, limit=10.0, wpre=True, fq=False)
            m0 = M.MoeGather(g, u, d, TOPK, limit=10.0, wpre=True, fq=False)
            maxrel0, biteq0 = 0.0, True
            for R in [1, 4]:
                for t in range(2):
                    ids = np.stack([rng.choice(E, TOPK, replace=False) for _ in range(R)]).astype(np.int32)
                    x = mx.array((rng.standard_normal((R, H)) * 0.5).astype(np.float16))
                    w = mx.array(rng.random((R, TOPK)).astype(np.float32))
                    yo, yr = m0(x, mx.array(ids), w), r(x, mx.array(ids), w)
                    mx.eval(yo, yr)
                    ao, ar = np.array(yo), np.array(yr)
                    maxrel0 = max(maxrel0, float(np.abs(ao - ar).max() / (np.abs(ar).max() + 1e-30)))
                    biteq0 = biteq0 and bool(mx.array_equal(yo, yr).item())
            print(f"gate: Gather(nofq) vs MoeUnfused maxrel {maxrel0:.3e}  bit-identical {biteq0}", flush=True)
            out["gate_nofq"] = dict(maxrel=maxrel0, bit_identical=biteq0)
    print("BENCH " + json.dumps(dict(impl=a.impl, res=out)), flush=True)
