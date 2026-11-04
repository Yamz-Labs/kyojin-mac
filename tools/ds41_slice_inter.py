#!/usr/bin/env python3
"""Interleaved decode-step A/B on the 2-layer EXL3 DS4.1 slice (one process, variants toggled per step).

Variants toggle live module flags (installs are once): stock, s5, mhc, attn, index, all.
Each variant owns a cache warmed identically; steps interleave round-robin so machine drift hits all arms equally.
Reports median/min ms + bit-identity vs stock (or maxrel where approximate).
usage: ds41_slice_inter.py [--steps 25]
"""
import os, statistics, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.expanduser("~/kyojin-mlx/vlm_pkg"))

import mlx.core as mx
import numpy as np


def main():
    from ds41_slice_census import build as _b, stub_routed
    from kyojin_mlx import archs
    from kyojin_mlx import fuse as FU
    import kyojin_mlx.fuse_attn as FA
    import kyojin_mlx.fuse_index as FI
    from kyojin_mlx import fused_group as FG
    lm = _b("pair", exl3=True, dtype=mx.float16)
    FU.install(archs.DS41)
    stub_routed(lm)

    def fwd():
        cache = lm.make_cache()
        mx.eval(lm(mx.array([[1, 2, 3, 4, 5, 6, 7, 8]]), cache=cache))
        mx.eval(lm(mx.array([[9]]), cache=cache))
    info = FG.install(lm, fwd)
    print("S5:", {k: v for k, v in info.items() if k != "names"}, flush=True)

    def setv(name):
        FU.ENABLED = FU.MHC = name in ("mhc", "all")
        FA.ENABLED = name in ("attn", "all")
        FI.ENABLED = name in ("index", "all")
        FG.ENABLED = name in ("s5", "all")

    rng = np.random.default_rng(7)
    ids = [int(x) for x in rng.integers(3, 1024, 8)]
    names = ["stock", "s5", "mhc", "attn", "index", "all"]
    caches = {}
    for n in names:
        setv(n)
        c = lm.make_cache()
        mx.eval(lm(mx.array([ids]), cache=c))
        for t in range(15):
            mx.eval(lm(mx.array([[t % 997 + 3]]), cache=c))
        caches[n] = c
    steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 25
    res = {n: [] for n in names}
    lg0 = {}
    for s in range(steps):
        for n in names:
            setv(n)
            c = caches[n]
            t0 = time.perf_counter()
            lg = lm(mx.array([[(s % 997) + 3]]), cache=c).logits
            mx.eval(lg)
            res[n].append(time.perf_counter() - t0)
            if s == steps - 1:
                lg0[n] = np.array(lg[0, -1]).astype(np.float32)
    setv("stock")
    for n in names:
        v = res[n]
        print(f"{n:7s} median {1e3 * statistics.median(v):6.3f} min {1e3 * min(v):6.3f} ms over {steps}", flush=True)
    a = lg0["stock"]
    for n in names[1:]:
        b = lg0[n]
        if np.array_equal(a, b):
            print(f"gate {n}: BIT-IDENTICAL")
        else:
            d = float(np.max(np.abs(a - b))) / (float(np.max(np.abs(a))) + 1e-30)
            print(f"gate {n}: maxrel {d:.3e}")


if __name__ == "__main__":
    main()
