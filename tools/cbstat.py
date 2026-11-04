#!/usr/bin/env python3
"""Commits and residency events per decode step. Swizzles -[MTLCommandBuffer commit] (tools/cbcount) and the MTLResidencySet calls (tools/cbcount/rscount.mm).

usage:  tools/cbstat.py PACK [--steps 16] [--len 135]      real decode steps of a DS4.1 pack slice (R.step, the serving path), one line per step
        tools/cbstat.py --synth [--layers 8]               bounded synthetic step (tools/cbill_bench.py shapes), no pack
Set the levers in the environment: KYOJIN_DENSE_VIEWS=1, KYOJIN_MOE_VIEWS=0, KYOJIN_WIRED_FIT=1 (applied by --fit), MLX_MAX_OPS_PER_BUFFER, KYOJIN_BOUND_DEC_EVERY, ...
columns: commits (command buffers MLX committed during the step), gpu_ms (sum of their GPU time), rs_add / rs_n (residency set addAllocation(s) calls /
allocations added), rs_commit (residency set commits), rs_req (requestResidency), act_mb / cache_mb (change in MLX active / cache memory over the step),
wall_ms. The libraries build with clang++ on the Mac; without them the Metal counters read 'n/a' and the memory columns still print."""
from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))


def _lib(name: str, src: str):
    """ctypes handle of tools/cbcount/lib<name>.dylib (built on first use), or None."""
    path = os.path.join(HERE, "cbcount", f"lib{name}.dylib")
    if not os.path.isfile(path):
        cmd = ["clang++", "-O2", "-std=c++17", "-fobjc-arc", "-dynamiclib", "-framework", "Metal", "-framework", "Foundation", os.path.join(HERE, "cbcount", src), "-o", path]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except (OSError, subprocess.CalledProcessError):
            return None
    try:
        return ctypes.CDLL(path)
    except OSError:
        return None


class Counters:
    """snapshot() before and after a region; diff() between two snapshots."""

    def __init__(self):
        import mlx.core as mx
        self.mx = mx
        self.cb = _lib("cbcount", "cbcount.mm")
        self.rs = _lib("rscount", "rscount.mm")
        self.has_cb = self.has_rs = False
        if self.cb is not None:
            self.cb.cb_install()
            self.cb.cb_commits.restype = ctypes.c_long
            self.cb.cb_gpu_us.restype = ctypes.c_double
            self.has_cb = True
        if self.rs is not None:
            self.has_rs = self.rs.rs_install() == 0
            self.rs.rs_counts.argtypes = [ctypes.POINTER(ctypes.c_long)]
        self.overhead = 0                                     # commits of one snapshot() itself (mx.synchronize), subtracted by delta()
        if self.has_cb:
            a = self.snapshot()
            self.overhead = self.snapshot()["commits"] - a["commits"]

    def snapshot(self) -> dict:
        mx = self.mx
        mx.synchronize()
        s = {"t": time.perf_counter(), "act": mx.get_active_memory(), "cache": mx.get_cache_memory()}
        if self.has_cb:
            s["commits"], s["gpu_us"] = self.cb.cb_commits(), self.cb.cb_gpu_us()
        if self.has_rs:
            a = (ctypes.c_long * 6)()
            self.rs.rs_counts(a)
            s["rs"] = list(a)
        return s

    def delta(self, a: dict, b: dict) -> dict:
        d = self.diff(a, b)
        if "commits" in d:
            d["commits"] -= self.overhead
        return d

    @staticmethod
    def diff(a: dict, b: dict) -> dict:
        d = {"wall_ms": (b["t"] - a["t"]) * 1e3, "act_mb": (b["act"] - a["act"]) / 2 ** 20, "cache_mb": (b["cache"] - a["cache"]) / 2 ** 20}
        if "commits" in a:
            d["commits"], d["gpu_ms"] = b["commits"] - a["commits"], (b["gpu_us"] - a["gpu_us"]) / 1e3
        if "rs" in a:
            r = [y - x for x, y in zip(a["rs"], b["rs"])]
            d.update(rs_add=r[0], rs_n=r[1], rs_commit=r[4], rs_req=r[5])
        return d


COLS = ("commits", "gpu_ms", "rs_add", "rs_n", "rs_commit", "rs_req", "act_mb", "cache_mb", "wall_ms")


def row(d: dict) -> str:
    return "  ".join(f"{c}={d[c]:.2f}" if c in d and isinstance(d[c], float) else f"{c}={d.get(c, 'n/a')}" for c in COLS)


def run_pack(ctr: Counters, pack: str, steps: int, plen: int) -> list:
    import numpy as np
    import mlx.core as mx
    sys.path.insert(0, os.path.expanduser("~/kyojin-mlx/vlm_pkg"))
    from kyojin_mlx import memfit, patch as P, rows as R, spec as S
    model, spec, cfg = P.build_model(pack)
    P.patch_model(model, pack, spec, cfg); R.enable(model, spec); S.install(model, spec)
    memfit.apply()                                           # WIRED_FIT / CACHE_FIT, as build_app does
    rng = np.random.default_rng(1)
    cache0, lg, _, _ = S.prefill(model, rng.integers(16, cfg["text_config"]["vocab_size"] - 16, plen).tolist())
    tok = mx.array([int(mx.argmax(lg[0]).item())])
    c = R.fork(cache0)
    out = []
    for i in range(steps + 2):                               # two warm steps
        a = ctr.snapshot()
        lg = R.step(model, c, tok); tok = mx.argmax(lg, -1).reshape(1).astype(mx.int32); mx.eval(tok, lg)
        d = ctr.delta(a, ctr.snapshot())
        if i >= 2:
            out.append(d)
            print(f"step {i - 2:3d}  {row(d)}", flush=True)
    return out


def run_synth(ctr: Counters, layers: int) -> list:
    import cbill_bench as B
    return B.run_counted(ctr, layers)


def wire() -> None:
    """The served path wires memory before the load (serve.backends.wire_memory); the counters must run in that state, not in the unwired one."""
    try:
        from kyojin_mlx.serve.backends import wire_memory
        wire_memory()
    except Exception as e:                               # a missing optional dependency of the serve package
        print(f"cbstat: wire_memory not applied ({type(e).__name__}: {e}); the baseline is the UNWIRED state", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack", nargs="?"); ap.add_argument("--steps", type=int, default=16); ap.add_argument("--len", type=int, default=135)
    ap.add_argument("--synth", action="store_true"); ap.add_argument("--layers", type=int, default=8)
    a = ap.parse_args()
    if not a.pack and not a.synth:
        ap.error("PACK or --synth")
    wire()
    ctr = Counters()
    print(f"counters: commits {'on' if ctr.has_cb else 'n/a (no libcbcount)'}, residency {'on' if ctr.has_rs else 'n/a (no librscount or no MTLResidencySet)'}", flush=True)
    res = run_synth(ctr, a.layers) if a.synth else run_pack(ctr, a.pack, a.steps, a.len)
    keys = [k for k in COLS if res and k in res[0]]
    print("mean  " + "  ".join(f"{k}={sum(r[k] for r in res) / len(res):.2f}" for k in keys))


if __name__ == "__main__":
    main()
