#!/usr/bin/env python3
"""pfroof step 1 table: per M bucket (uniform, 384 experts, real DS4.1 dims),
production cfg = NX.pick_cfg(counts): combined (before) vs split (after),
paired ratios, medians. Sustained clocks (burn-in), interleaved."""
import os, sys, time
import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from bench_moe import synth_stack  # noqa: E402
from kyojin_mlx import metal_nax as NX  # noqa: E402

mx.set_cache_limit(256 << 20)
POOL, H, I, K = 48, 5120, 2304, 3
g0 = synth_stack(H, I, [K] * POOL, "mul1", 1)
u0 = synth_stack(H, I, [K] * POOL, "mul1", 2)


def virtual(st, V):
    pool = st.experts
    phys = np.arange(V) % pool
    off = np.array(st.offset)[:pool][phys]
    off = np.concatenate([off, off[-1:] + 1]).astype("<u4")
    return type(st)(st.words, mx.array(off), mx.array(np.array(st.k)[phys]),
                    mx.array(np.array(st.suh)[phys]), mx.array(np.array(st.svh)[phys]),
                    st.kt, st.nt, st.cb)


rng = np.random.default_rng(0)
print("M E cfg-before=same pick_cfg | combined TF med/min | split TF med/min | split/comb med_ratio", flush=True)
for M in (16, 32, 64, 128, 256, 512):
    E = 192 if M == 512 else 384
    g, u = virtual(g0, E), virtual(u0, E)
    eids = mx.array(np.arange(E, dtype=np.int32))
    counts = np.full(E, M)
    C = NX.pick_cfg(counts)
    n = E * M
    A = mx.array(NX.permute_k(rng.standard_normal((2 * n, H)).astype(np.float16)))
    cl = (np.arange(E + 1) * M).astype(np.uint32)
    ro2 = mx.array(np.concatenate([cl[:-1], n + cl]))
    ro1 = mx.array(cl)
    mx.eval(A)
    fl = 4.0 * E * M * H * I  # gate+up flops
    fns = {"c": lambda: NX.expert_gemm((g, u), eids, ro2, A, 2 * n, M, C, True),
           "s": lambda: mx.concatenate([NX.expert_gemm((g,), eids, ro1, A[:n], n, M, C, True),
                                        NX.expert_gemm((u,), eids, ro1, A[n:], n, M, C, True)], axis=0)}
    for k, f in fns.items():
        mx.eval(f())
    y0 = np.array(fns["c"]())
    y1 = np.array(fns["s"]())
    bi = np.array_equal(y0, y1)
    for k, f in fns.items():
        for _ in range(4):
            mx.eval(f())
    ratios, tc, ts = [], [], []
    for _ in range(7):
        got = {}
        for k in (("c", "s") if rng.random() < 0.5 else ("s", "c")):
            mx.synchronize(); t0 = time.perf_counter()
            y = fns[k](); mx.eval(y)
            got[k] = time.perf_counter() - t0
        ratios.append(got["s"] / got["c"]); tc.append(got["c"]); ts.append(got["s"])
    print(f"M={M:4d} E={E} cfg={C} biteq={bi} | "
          f"comb {fl / np.median(tc) / 1e12:5.1f}/{fl / min(tc) / 1e12:5.1f}TF | "
          f"split {fl / np.median(ts) / 1e12:5.1f}/{fl / min(ts) / 1e12:5.1f}TF | "
          f"ratio {np.median(ratios):.3f} [{min(ratios):.3f},{max(ratios):.3f}]", flush=True)
print(f"peak {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)
