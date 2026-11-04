#!/usr/bin/env python3
"""Root-cause table of the weak EXL3 decode ops (macds105): small-output / small-K 1-row linears and same-input groups, synthetic weights at the
real shapes (DS4.1 layer 10, Qwen3.5 4B / 9B), every variant in ONE process, interleaved per rep (same thermal state).
Each measure: one eval (graph built before the clock starts) of a chain of --chain calls, min / median over --reps of the time per call, bytes = trellis + suh + svh.
  pool  : every call of a chain reads its own weight copy (pool of distinct linears, > the SLC) -- what one decode step sees
  same  : the per-op table's method (tools/ds41_raw_slice.py): one linear called --chain times with fresh inputs
  dep   : dependent chain A, A^T, A, ... (A^T = the transposed shape, own weights): call i+1 reads call i's output; time per linear of the pair
  slope : pool chains of --chain and 4x --chain calls: per-call cost without the fixed eval / sync cost of one chain (the per-op table's chain of 8
          carries that fixed cost on every call)
  dslope: dependent chains of --chain and 4x --chain linears: the per-linear cost of a decode step (kernel + its ramp / tail + the barrier)
  dslope3 : dslope with the three-launch linear (rot, GEMV, finalize: KYOJIN_LINEAR_FUSED=0 path)
  tgk   : the k-split (threadgroups along k) swept over --tgk (0 = the engine's pick_tgk)
usage: weak105.py [--shapes ds,q4,q9] [--tgk 0,1,2,4,8,16,32] [--modes pool,same] [--chain 32] [--pool 24] [--reps 9] [--json OUT]"""
import argparse, json, os, statistics, sys, time

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kyojin_mlx import kernels as KN, exl3_linear as EL  # noqa: E402

SHAPES = {   # name: (kin, nout, K)
    "ds": [("ds.wq_a", 5120, 1280, 5), ("ds.wkv", 5120, 512, 5), ("ds.sh_w1", 5120, 2304, 5), ("ds.sh_w2", 2304, 5120, 5),
           ("ds.wq_b", 1280, 32768, 5), ("ds.wo_b", 8192, 5120, 5)],
    "q4": [("q4.k_proj", 2560, 1024, 4), ("q4.in_z", 2560, 4096, 4), ("q4.o_proj", 4096, 2560, 4), ("q4.gate", 2560, 9216, 4)],
    "ex": [(f"ex.w1k{K}", 5120, 2304, K) for K in (2, 3, 4, 5)] + [(f"ex.w2k{K}", 2304, 5120, K) for K in (2, 3, 4)],   # DS4.1 routed expert shapes, uniform K
    "q9": [("q9.k_proj", 4096, 1024, 4), ("q9.o_proj", 4096, 4096, 4), ("q9.gate", 4096, 12288, 3)],
}


def synth(kin, nout, K, seed, tgk=None):
    rng = np.random.default_rng(seed)
    tr = mx.array(rng.integers(-32768, 32768, (kin // 16, nout // 16, 16 * K), dtype=np.int16))
    f16 = lambda n: mx.array((rng.choice([-1.0, 1.0], n) * rng.uniform(0.5, 1.5, n)).astype(np.float16))
    return EL.Exl3Linear(tr, f16(kin), f16(nout), cb="mul1", tune={} if tgk is None else {"tgk": tgk})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapes", default="ds,q4,q9")
    ap.add_argument("--only", default="", help="comma list of shape names")
    ap.add_argument("--tgk", default="0,1,2,4,8,16,32")
    ap.add_argument("--modes", default="pool,same")
    ap.add_argument("--chain", type=int, default=32)
    ap.add_argument("--pool", type=int, default=24)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--bw", type=float, default=131.9)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    mx.set_cache_limit(256 << 20)
    rng = np.random.default_rng(1)
    out = []
    w = synth(4096, 4096, 4, 1)                                # ~2 s of GEMV work first: the GPU clock ramps before the first timed shape
    wx = mx.array(rng.standard_normal((1, 4096)).astype(np.float16))
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 2.0:
        mx.eval([w(wx) for _ in range(64)])
    del w
    for fam in a.shapes.split(","):
        for name, kin, nout, K in SHAPES[fam]:
            if a.only and name not in a.only.split(","):
                continue
            pick = KN.pick_tgk(kin // 16, nout // 128)
            tgks = []
            for t in [int(v) for v in a.tgk.split(",")]:
                t = t or pick
                if t not in tgks and (kin // 16) // (4 * t) >= 1:
                    tgks.append(t)
            mb = (kin * nout * K / 8 + 2 * kin + 2 * nout) / 1e6
            roof_us = mb / a.bw * 1e3
            pools = {t: [synth(kin, nout, K, 100 * t + i, t) for i in range(a.pool)] for t in tgks}
            modes = a.modes.split(",")
            tpool = [synth(nout, kin, K, 7000 + i) for i in range(a.pool)] if any(m.startswith(("dep", "dslope")) for m in modes) else []
            mx.eval(*[l._words for p in pools.values() for l in p], *[l._words for l in tpool])
            xs = [mx.array(rng.standard_normal((1, kin)).astype(np.float16)) for _ in range(a.chain)]
            mx.eval(*xs)
            var = [(m, t) for t in tgks for m in modes]
            cnt = {"slope": 4 * a.chain, "dslope": 4 * a.chain, "dslope3": 4 * a.chain}

            def once(m, t):
                p = pools[t]
                n = cnt.get(m, a.chain)
                if m in ("pool", "slope"):
                    return [p[i % len(p)](xs[i % len(xs)]) for i in range(n)]
                if m.startswith("dep") or m.startswith("dslope"):
                    EL.LINEAR_FUSED = not m.endswith("3")
                    x = xs[0]
                    for i in range(n // 2):
                        x = tpool[i % len(tpool)](p[i % len(p)](x))
                    EL.LINEAR_FUSED = True
                    return x
                return [p[0](xs[i]) for i in range(n)]
            for v in var:
                mx.eval(once(*v))
            T = {v: [] for v in var}
            for _ in range(a.reps):
                for v in var:
                    o = once(*v); t0 = time.perf_counter(); mx.eval(o); T[v].append((time.perf_counter() - t0) * 1e6)
            for (m, t) in var:
                n = cnt.get(m, a.chain)
                base = {"slope": "pool", "dslope": "dep", "dslope3": "dep3"}.get(m)
                if base and (base, t) in T:                      # (t(4n) - t(n)) / 3n, min and median
                    b = (min(T[(m, t)]) - min(T[(base, t)])) / (n - a.chain)
                    md = (statistics.median(T[(m, t)]) - statistics.median(T[(base, t)])) / (n - a.chain)
                else:
                    b, md = min(T[(m, t)]) / n, statistics.median(T[(m, t)]) / n
                r = dict(op=name, kin=kin, nout=nout, K=K, mode=m, tgk=t, pick=t == pick, tgs=(nout // 128) * t, us=round(b, 2), med=round(md, 2),
                         mb=round(mb, 3), pct=round(100 * roof_us / b, 1))
                out.append(r)
                print(f"{name:12s} {kin:5d}x{nout:5d} K{K} {m:4s} tgk {t:2d}{'*' if t == pick else ' '} tgs {r['tgs']:4d}  {b:7.1f} us (med {md:7.1f})"
                      f"  {mb:6.2f} MB  {r['pct']:5.1f} % roof", flush=True)
            del pools
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
