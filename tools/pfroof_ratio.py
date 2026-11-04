#!/usr/bin/env python3
"""pfroof: paired-ratio combined-vs-split (uniform and skewed routing, one
process, pairwise ratios robust to contention drift)."""
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
C = (32, 2, 32, 64)
E = 384


def bench(counts, label, pairs=11):
    Mmax = int(counts.max())
    g, u = virtual(g0, E), virtual(u0, E)
    eids = mx.array(np.arange(E, dtype=np.int32))
    n = int(counts.sum())
    A = mx.array(NX.permute_k(rng.standard_normal((2 * n, H)).astype(np.float16)))
    cl = np.concatenate([[0], np.cumsum(counts)]).astype(np.uint32)
    ro2 = mx.array(np.concatenate([cl[:-1], n + cl]))
    ro1 = mx.array(cl)
    mx.eval(A)
    fl = 2.0 * n * H * I  # per projection; gu = 2x
    ratios, tc, ts = [], [], []
    for _ in range(pairs):
        fns = {"c": lambda: NX.expert_gemm((g, u), eids, ro2, A, 2 * n, Mmax, C, True),
               "s": lambda: mx.concatenate([NX.expert_gemm((g,), eids, ro1, A[:n], n, Mmax, C, True),
                                            NX.expert_gemm((u,), eids, ro1, A[n:], n, Mmax, C, True)], axis=0)}
        for k, f in fns.items():
            mx.eval(f())
        order = ["c", "s"] if rng.random() < 0.5 else ["s", "c"]
        got = {}
        for k in order:
            mx.synchronize(); t0 = time.perf_counter()
            y = fns[k](); mx.eval(y)
            got[k] = time.perf_counter() - t0
        ratios.append(got["s"] / got["c"]); tc.append(got["c"]); ts.append(got["s"])
    print(f"{label}: pairs={pairs} split/comb med_ratio {np.median(ratios):.3f} "
          f"p10 {sorted(ratios)[len(ratios) // 10]:.3f} p90 {sorted(ratios)[-1 - len(ratios) // 10]:.3f} | "
          f"comb med {np.median(tc) * 1e3:.1f}ms {2 * fl / np.median(tc) / 1e12:.1f}TF | "
          f"split med {np.median(ts) * 1e3:.1f}ms {2 * fl / np.median(ts) / 1e12:.1f}TF", flush=True)


M = 128
bench(np.full(E, M), f"uniform M={M}")
hot = rng.choice(E, 96, replace=False)
p = np.full(E, 0.25 / (E - 96)); p[hot] = 0.75 / 96
ids = np.stack([rng.choice(E, 6, replace=False, p=p) for _ in range(8192)])
cnt = np.bincount(ids.reshape(-1), minlength=E)
print(f"skewed counts: min {cnt[cnt > 0].min()} max {cnt.max()} mean {cnt.mean():.0f}", flush=True)
bench(cnt, "skewed-8K")
print(f"peak {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)
