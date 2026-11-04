#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""opmoe-2 synthetic bench: routed MoE decode (rows = 1) at the real DS4.1 geometry, old path against the new levers, on the Mac.

One layer: H 5120, I 2304, topk 6, 96 experts (K 2 / 3 in the pack's 243 : 141 mix; 96 instead of 384 experts keeps the three stacks near 1 GiB, bound
checked below at 3.5 GiB; the per-call read is topk experts, so the call does the same work as with 384). Random valid trellises, as tools/moe_fuse_bench.py.
Arms (flag value sets, all other levers as served): stock, moe_finpar (bit-identical combine), moe_tgt 240 / 960 / 1920, moe_ksplit 1 / 2, ksplit with tgt 960, ksplit 2 + finpar. Dependent chains of n and 4n
calls in one eval (slope = us per call), arms interleaved A B C ... C B A over --rounds, median. ms/step = us x 40 layers.
Per-op table (stage launches rot_gu, gemv_gu, mid, gemv_dn, final) for every arm. Quality: rel L2 and max abs error of the layer output against a reference
accumulated in ONE k-split (tgk 1, the longest sequential f32 sum), next to the stock path's own error; the levers only change the order the f32 partials
are added in, so "no worse" = the arm's error is the same order as stock's.
Runs in well under 60 s on an M5 Ultra (first run compiles each kernel variant once).
usage: python tools/moe_k_bench.py [--rounds 5] [--experts 96] [--chain 6]"""
import argparse
import os
import statistics
import sys
import time

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kyojin_mlx import flags as FL, moe as M  # noqa: E402
from moe_fuse_bench import make_stack, expert_bytes  # noqa: E402

H, I, TOPK, LAYERS = 5120, 2304, 6, 40
_OFF = dict(moe_tgt=0, moe_ksplit=0, moe_finpar=0)
ARMS = {"stock": dict(_OFF), "pf2": dict(_OFF, gemv_pf2=1), "finpar": dict(_OFF, moe_finpar=1), "tgt240": dict(_OFF, moe_tgt=240), "tgt960": dict(_OFF, moe_tgt=960),
        "tgt1920": dict(_OFF, moe_tgt=1920), "ks1": dict(_OFF, moe_ksplit=1), "ks2": dict(_OFF, moe_ksplit=2),
        "ks1+tgt960": dict(_OFF, moe_tgt=960, moe_ksplit=1), "ks2+tgt960": dict(_OFF, moe_tgt=960, moe_ksplit=2),
        "all(ks2,fin)": dict(_OFF, moe_ksplit=2, moe_finpar=1)}
LIMIT_GIB = 3.5


def timed(f, ins, n):
    def run(c):
        outs = [f(*ins[i % len(ins)]) for i in range(c)]
        t0 = time.perf_counter()
        mx.eval(*outs)
        return time.perf_counter() - t0
    a, b = [], []
    for r in range(4):
        da, db = run(n), run(4 * n)
        if r:
            a.append(da)
            b.append(db)
    return (min(b) - min(a)) / (3 * n) * 1e6


def stage_us(m, x, ids, wts):
    ids_f, wts_f = ids.reshape(-1).astype(mx.int32), wts.reshape(-1).astype(mx.float32)
    xh = m.rot_gu(x, ids_f, False)
    part = m.gemv_gu(xh, ids_f, 1)
    xh2 = m.mid(part, ids_f, wts_f, 1)
    part2 = m.gemv_dn(xh2, ids_f, 1)
    mx.eval(xh, part, xh2, part2)
    fs = {"rot_gu": lambda: m.rot_gu(x, ids_f, False), "gemv_gu": lambda: m.gemv_gu(xh, ids_f, 1), "mid": lambda: m.mid(part, ids_f, wts_f, 1),
          "gemv_dn": lambda: m.gemv_dn(xh2, ids_f, 1), "final": lambda: m.final(part2, ids_f, wts_f, 1)}
    return {k: min(timed(lambda *_: f(), [(0,)], 8) for _ in range(2)) for k, f in fs.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--experts", type=int, default=96)
    ap.add_argument("--chain", type=int, default=6)
    ap.add_argument("--gbs", type=float, default=834.0)
    a = ap.parse_args()
    mx.set_cache_limit(256 << 20)
    rng = np.random.default_rng(7)
    ks = [3 if r < 141 / 384 else 2 for r in rng.random(a.experts)]
    t0 = time.time()
    g, u, d = make_stack(H, I, ks, 1), make_stack(H, I, ks, 2), make_stack(I, H, ks, 3)
    mx.eval(g.words, u.words, d.words)
    peak = mx.get_peak_memory() / 2 ** 30
    print(f"stacks {time.time() - t0:.1f} s, peak {peak:.2f} GiB (bound {LIMIT_GIB}), {a.experts} experts K2x{ks.count(2)} K3x{ks.count(3)}", flush=True)
    assert peak < LIMIT_GIB, "bench memory bound"
    per = expert_bytes((g, u, d))
    kw = dict(limit=10.0, wpre=True, fq=False)
    FL.set_many(_OFF)
    ref = M.MoeGather(g, u, d, TOPK, tgk_gu=1, tgk_dn=1, **kw)
    m = M.MoeGather(g, u, d, TOPK, **kw)
    ins = []
    for _ in range(a.chain):
        ids = rng.choice(a.experts, TOPK, replace=False)[None].astype(np.int32)
        ins.append((mx.array((rng.standard_normal((1, H)) * 0.5).astype(np.float16)), mx.array(ids),
                    mx.array(rng.uniform(0.1, 1.0, (1, TOPK)).astype(np.float32)), ids))
    mx.eval(*[v for i in ins for v in i[:3]])
    nb = float(np.mean([per[i[3][0]].sum() for i in ins]))
    print(f"read per call {nb / 1e6:.1f} MB, roof {nb / (a.gbs * 1e3):.1f} us at {a.gbs:.0f} GB/s", flush=True)
    refs = [np.array(ref(x, i, w)) for x, i, w, _ in ins]
    err = {}
    for name, fl in ARMS.items():
        FL.set_many(fl)
        outs = [np.array(m(x, i, w)) for x, i, w, _ in ins]
        for _ in range(8):
            mx.eval(m(*ins[0][:3]))
        err[name] = (max(float(np.linalg.norm(o - r) / (np.linalg.norm(r) + 1e-30)) for o, r in zip(outs, refs)),
                     max(float(np.abs(o - r).max()) for o, r in zip(outs, refs)))
    us = {k: [] for k in ARMS}
    order = list(ARMS)
    for r in range(a.rounds):
        for name in (order if r % 2 == 0 else order[::-1]):
            FL.set_many(ARMS[name])
            us[name].append(timed(lambda x, i, w, _i: m(x, i, w), ins, a.chain))
    st = {}
    for name in ARMS:
        FL.set_many(ARMS[name])
        st[name] = stage_us(m, *ins[0][:3])
    FL.set_many(_OFF)
    base = statistics.median(us["stock"])
    print(f"\n{'arm':12s} {'us/call':>9s} {'ms/step':>8s} {'vs stock':>9s}   " + " ".join(f"{k:>8s}" for k in st['stock']) + "     relL2   maxabs")
    for name in ARMS:
        med = statistics.median(us[name])
        print(f"{name:12s} {med:9.1f} {med * LAYERS / 1e3:8.2f} {100 * (med / base - 1):+8.1f}%   " + " ".join(f"{v:8.1f}" for v in st[name].values())
              + f"   {err[name][0]:.2e} {err[name][1]:.2e}")
    print("per-op columns are us per launch in a chain of 8; the error columns compare with the single-k-split f32 reference (stock row = today's error).")


if __name__ == "__main__":
    main()
