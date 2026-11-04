#!/usr/bin/env python3
"""Command-buffer billing bench: a chain of dense decode layers at the real DS4.1 dense geometry (hidden 5120, q_lora 1280, wq_b 32768 out, wo_b 8192 in,
8 x wo_a groups, shared expert 2304, K = 5 words), every layer with its OWN weight buffers (a bound array is billed once per command buffer, so reusing one
weight for every layer would hide the effect). Old path = KYOJIN_DENSE_VIEWS=0, new path = 1; the chain output is compared bit for bit.

  python tools/cbill_bench.py [--layers 12] [--steps 40] [--reps 5] [--sweep] [--cap-gib 3.5]

Default: one process, arms interleaved (old new old new ...), prints per arm: ms per step (min / median), ms per layer, ms per weighted launch, commits per
step when the Metal counter library builds (tools/cbstat.py). --sweep re-runs the chain in worker processes for the import-time knobs (MLX_MAX_OPS_PER_BUFFER,
MLX_MAX_MB_PER_BUFFER), each with views off / on, so the commit limit and the billing are separated.
Size: ~97 MB of weights per layer; --layers x 90 MB must stay under --cap-gib (default 3.5, so at most 37 layers; the default 12 is ~1.1 GB). The head
(496 MB) is added with --head. Time: ~10 s per worker. A 40-layer step is an extrapolation (layers x 40 / L); the printout labels it as such.
Nothing here was run in the authoring sandbox (no Metal)."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

SH = {"wq_a": (5120, 1280), "wkv": (5120, 512), "wq_b": (1280, 32768), "wo_b": (8192, 5120), "wo_a": (4096, 1024),
      "sh_gate": (5120, 2304), "sh_up": (5120, 2304), "sh_down": (2304, 5120), "head": (5120, 129280)}
K = 5
LAUNCHES_PER_LAYER = 6          # wq_a|wkv group, wq_b, wo_b, wo_a grouped, gate|up group, sh_down: the weighted launches of a layer


def layer_bytes() -> int:
    one = sum(kin * nout * K // 8 for n, (kin, nout) in SH.items() if n not in ("head", "wo_a"))
    return one + 8 * SH["wo_a"][0] * SH["wo_a"][1] * K // 8


class Chain:
    def __init__(self, layers: int, head: bool, cap_gib: float, submit_every: int = 2):
        import numpy as np
        import mlx.core as mx
        from kyojin_mlx.exl3_linear import Exl3Linear
        from kyojin_mlx.fused_group import Exl3Group
        from kyojin_mlx.grouped import GroupedGemv
        need = layers * layer_bytes() + (SH["head"][0] * SH["head"][1] * 6 // 8 if head else 0)
        if need > cap_gib * 2 ** 30:
            raise SystemExit(f"cbill_bench: {need / 2 ** 30:.2f} GiB of weights over the {cap_gib} GiB cap (--layers / --cap-gib)")
        self.mx, self.layers, self.bytes, self.submit_every = mx, layers, need, submit_every
        rng = np.random.default_rng(0)
        src = {}

        def arrays(name, kk=K):                     # host arrays made once per shape, uploaded once per layer: distinct device buffers
            if name not in src:
                kin, nout = SH[name]
                tr = rng.integers(-32768, 32768, (kin // 16, nout // 16, 16 * kk)).astype(np.int16)
                suh = (rng.choice([-1.0, 1.0], kin) * rng.uniform(0.5, 1.5, kin)).astype(np.float16)
                svh = (rng.choice([-1.0, 1.0], nout) * rng.uniform(0.5, 1.5, nout)).astype(np.float16)
                src[name] = (tr, suh, svh)
            tr, suh, svh = src[name]
            t, u, v = mx.array(tr), mx.array(suh), mx.array(svh)
            mx.eval(t, u, v)
            return t, u, v

        def lin(name):
            return Exl3Linear(*arrays(name), cb="mul1", out_dtype=mx.float16)

        self.L = []
        for _ in range(layers):
            t, u, v = arrays("wo_a")
            kt, nt, w = t.shape
            tt = mx.contiguous(mx.broadcast_to(t.reshape(1, kt, nt, w), (8, kt, nt, w)))
            uu = mx.contiguous(mx.broadcast_to(u.reshape(1, -1), (8, u.size)))
            vv = mx.contiguous(mx.broadcast_to(v.reshape(1, -1), (8, v.size)))
            mx.eval(tt, uu, vv)
            self.L.append(dict(qa=Exl3Group([lin("wq_a"), lin("wkv")]), qb=lin("wq_b"), ob=lin("wo_b"), oa=GroupedGemv(tt, uu, vv),
                               gu=Exl3Group([lin("sh_gate"), lin("sh_up")]), dn=lin("sh_down")))
        self.head = Exl3Linear(*arrays("head", 6), cb="mul1", out_dtype=mx.float16) if head else None
        self.x0 = mx.array((rng.standard_normal((1, 5120)) * .3).astype(np.float16))

    def step(self):
        mx = self.mx
        x = self.x0
        for i, ly in enumerate(self.L):
            qa = ly["qa"].run(x)[0]                                           # [1, 1280]
            q = ly["qb"](qa)                                                  # [1, 32768]
            x = x + ly["ob"](q[:, :8192])
            oa = ly["oa"](q.reshape(1, 8, 1, 4096))                           # [1, 8, 1, 1024]
            x = x + mx.sum(oa).astype(mx.float16) * mx.array(1e-3, dtype=mx.float16)
            g, u = ly["gu"].run(x)[:2]
            x = x + ly["dn"](g * u)
            if self.submit_every and (i + 1) % self.submit_every == 0 and i + 1 < len(self.L):
                mx.async_eval(x)                                              # the served decode step submits every KYOJIN_BOUND_DEC_EVERY (2) layers
        return self.head(x) if self.head is not None else x


def run_arm(chain: Chain, views: bool, steps: int, ctr=None) -> dict:
    mx = chain.mx
    from kyojin_mlx import cbill
    cbill.DENSE_VIEWS = views
    mx.eval(chain.step()); mx.synchronize()                                  # warm (builds the views, the kernels)
    t0 = ctr.snapshot() if ctr else None
    ts = []
    for _ in range(steps):
        a = time.perf_counter()
        y = chain.step(); mx.eval(y); mx.synchronize()
        ts.append((time.perf_counter() - a) * 1e3)
    out = {"ms": ts, "y": y}
    if ctr and ctr.has_cb:
        d = ctr.delta(t0, ctr.snapshot())
        out["commits"] = d["commits"] / steps
    return out


def compare(layers: int, steps: int, reps: int, head: bool, cap_gib: float, submit_every: int = 2) -> None:
    import numpy as np
    from cbstat import Counters, wire
    wire()
    ctr = Counters()
    chain = Chain(layers, head, cap_gib, submit_every)
    res = {False: [], True: []}
    ys = {}
    for _ in range(reps):
        for views in (False, True):
            r = run_arm(chain, views, steps, ctr)
            res[views].append(r)
            ys[views] = np.array(r["y"])
    same = np.array_equal(ys[False], ys[True])
    print(f"layers {layers} ({chain.bytes / 2 ** 30:.2f} GiB of weights, own buffers per layer), steps {steps} x reps {reps}, arms interleaved; bit-identical output: {same}")
    print(f"{'arm':14s}{'ms/step min':>12s}{'median':>9s}{'ms/layer':>10s}{'us/launch':>11s}{'commits/step':>14s}")
    for views, name in ((False, "old (views 0)"), (True, "new (views 1)")):
        allms = [m for r in res[views] for m in r["ms"]]
        mn, md = min(allms), statistics.median(allms)
        cm = [r["commits"] for r in res[views] if "commits" in r]
        print(f"{name:14s}{mn:12.3f}{md:9.3f}{md / layers:10.3f}{md / (layers * LAUNCHES_PER_LAYER) * 1e3:11.1f}{(statistics.mean(cm) if cm else float('nan')):14.1f}")
    o, n = statistics.median([m for r in res[False] for m in r["ms"]]), statistics.median([m for r in res[True] for m in r["ms"]])
    print(f"median delta {n - o:+.3f} ms per {layers} layers; 40-layer extrapolation (an estimate, not a measurement): {(n - o) * 40 / layers:+.2f} ms")


def worker(layers: int, steps: int, reps: int, head: bool, cap_gib: float, submit_every: int = 2) -> None:
    from cbstat import Counters, wire
    wire()
    ctr = Counters()
    chain = Chain(layers, head, cap_gib, submit_every)
    out = {}
    for views in (False, True):
        ms = []
        cm = []
        for _ in range(reps):
            r = run_arm(chain, views, steps, ctr)
            ms += r["ms"]
            cm += [r["commits"]] if "commits" in r else []
        out[int(views)] = {"med": statistics.median(ms), "min": min(ms), "commits": statistics.mean(cm) if cm else None}
    print(json.dumps(out), flush=True)


SWEEP = {"default": {}, "ops200": {"MLX_MAX_OPS_PER_BUFFER": "200"}, "mb4k": {"MLX_MAX_MB_PER_BUFFER": "4096"},
         "one buffer": {"MLX_MAX_OPS_PER_BUFFER": "100000", "MLX_MAX_MB_PER_BUFFER": "1000000"}}


def sweep(a) -> None:
    print(f"{'MLX limits':12s}{'views 0 ms':>12s}{'views 1 ms':>12s}{'commits 0':>11s}{'commits 1':>11s}")
    for name, env in SWEEP.items():
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", "--layers", str(a.layers), "--steps", str(a.steps), "--reps", str(a.reps),
                            "--cap-gib", str(a.cap_gib), "--submit-every", str(a.submit_every)] + (["--head"] if a.head else []), env={**os.environ, **env}, capture_output=True, text=True)
        if r.returncode:
            print(name, "failed:", r.stderr[-300:]); continue
        d = json.loads(r.stdout.strip().splitlines()[-1])
        f = lambda v: "n/a" if v is None else f"{v:.1f}"
        print(f"{name:12s}{d['0']['med']:12.3f}{d['1']['med']:12.3f}{f(d['0']['commits']):>11s}{f(d['1']['commits']):>11s}")


def run_counted(ctr, layers: int) -> list:
    """cbstat --synth: per-step counter rows of the new-path chain (views on if KYOJIN_DENSE_VIEWS=1)."""
    from kyojin_mlx import cbill
    import mlx.core as mx
    chain = Chain(layers, False, 3.5)
    mx.eval(chain.step()); mx.synchronize()
    out = []
    for i in range(16):
        a = ctr.snapshot()
        mx.eval(chain.step())
        d = ctr.delta(a, ctr.snapshot())
        out.append(d)
        print(f"step {i:3d}  {__import__('cbstat').row(d)}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", type=int, default=12); ap.add_argument("--steps", type=int, default=40); ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--head", action="store_true"); ap.add_argument("--cap-gib", type=float, default=3.5)
    ap.add_argument("--submit-every", type=int, default=2, help="async_eval every n layers (the served KYOJIN_BOUND_DEC_EVERY; 0 = one eval per step)")
    ap.add_argument("--sweep", action="store_true"); ap.add_argument("--worker", action="store_true")
    a = ap.parse_args()
    if a.worker:
        worker(a.layers, a.steps, a.reps, a.head, a.cap_gib, a.submit_every)
    elif a.sweep:
        sweep(a)
    else:
        compare(a.layers, a.steps, a.reps, a.head, a.cap_gib, a.submit_every)


if __name__ == "__main__":
    main()
