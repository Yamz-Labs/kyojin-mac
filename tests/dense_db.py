"""Per-shape bench of the dense EXL3 GEMV (synthetic trellises, real DS4.1 shapes and bitrates).

Method (decroof-compatible): marginal cost per call = slope of the chain time, (t(16) - t(4)) / 12, medians over reps; the calls inside one chain are
independent, so the slope carries the launch gap + kernel time and not the mx.eval overhead. Bytes = trellis bytes (only the weight stream matters at
these sizes); roof = bytes / 1000 GB/s (decroof's sustained stream read).
"""
from __future__ import annotations

import os
import statistics
import time

import numpy as np
import mlx.core as mx

import kyojin_mlx  # noqa: F401  (import check: must be this worktree)
from kyojin_mlx import exl3_linear as EL, kernels as KN

BW = float(os.environ.get("DB_BW", "1000"))          # GB/s roof (decroof: 1036 sustained min; table uses 1000)
# (name, kin, nout, K): DS4.1 dense tensors (hidden 5120, q_lora 1280, 64 heads x 512, shared expert 2304)
SHAPES = [
    ("wq_a", 5120, 1280, 5),
    ("wkv", 5120, 512, 5),
    ("sh.gate", 5120, 2304, 5),
    ("sh.up", 5120, 2304, 5),
    ("sh.down", 2304, 5120, 5),
    ("idx.wq_b", 1280, 4096, 5),
    ("wq_b", 1280, 32768, 5),
    ("wo_b", 8192, 5120, 5),
]
BY = {n: (kin, nout, k) for n, kin, nout, k in SHAPES}


def synth(kin: int, nout: int, K: int, cb: str = "mul1", seed: int = 1) -> EL.Exl3Linear:
    rng = np.random.default_rng(seed)
    kt, nt = kin // 16, nout // 16
    tr = mx.array(rng.integers(-32768, 32768, (kt, nt, 16 * K), dtype=np.int16))
    f16 = lambda n: mx.array((rng.choice([-1.0, 1.0], n) * rng.uniform(0.5, 1.5, n)).astype(np.float16))
    lin = EL.Exl3Linear(tr, f16(kin), f16(nout), cb=cb)
    mx.eval(lin.parameters())
    return lin


def nbytes(lin: EL.Exl3Linear) -> int:
    return lin._words.size * 4


def tg_of(lin: EL.Exl3Linear, rows: int = 1) -> tuple[int, int]:
    g = lin._g[rows] if rows in lin._g else lin._gemv(rows)
    return g.blocks * g.tgk, g.tgk


def slope(fn, xs, n_lo: int = 4, n_hi: int = 16, reps: int = 9) -> float:
    """Marginal ms per call: median over reps of (t(n_hi) - t(n_lo)) / (n_hi - n_lo)."""
    fn(xs[0])
    mx.eval(*[fn(xs[0]) for _ in range(2)])
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(*[fn(xs[i % len(xs)]) for i in range(n_lo)])
        t1 = time.perf_counter()
        mx.eval(*[fn(xs[i % len(xs)]) for i in range(n_hi)])
        t2 = time.perf_counter()
        out.append((t2 - t1 - (t1 - t0)) / (n_hi - n_lo) * 1e3)
    return statistics.median(out)


def median_ms(fn, xs, chain: int = 16, reps: int = 9) -> float:
    fn(xs[0])
    mx.eval(*[fn(xs[0]) for _ in range(2)])
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(*[fn(xs[i % len(xs)]) for i in range(chain)])
        out.append((time.perf_counter() - t0) / chain * 1e3)
    return statistics.median(out)
