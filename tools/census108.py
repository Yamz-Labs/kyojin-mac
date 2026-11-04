#!/usr/bin/env python3
"""Per-op census of one DS4.1 decode step (R.step of the serving path, rows.enable + spec.install, warm) on a slice: dispatches by op type, the serial
chain (longest dependency path counted in dispatches, views free), the widest level, the step time, and the cost of one dependent launch on this GPU
(slope of a chain of tiny dependent ops in one eval vs the same number of independent ones) -> GPU idle estimate = chain x (dependent - independent).
usage: census108.py PACK [--len 135] [--reps 15] [--top 25] [--first N]"""
import argparse, collections, os, re, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "100000"); os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "1000000")

import numpy as np
import mlx.core as mx
from kyojin_mlx import patch as P, rows as R, spec as S
from count_launches import VIEWS

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("pack"); ap.add_argument("--len", type=int, default=135); ap.add_argument("--reps", type=int, default=15)
ap.add_argument("--top", type=int, default=25); ap.add_argument("--first", type=int, default=0)
a = ap.parse_args()
FREE = VIEWS | {"Full", "Arange"}                    # no dispatch (views) or a constant folded at build


def dag(out):
    path = os.path.join(tempfile.gettempdir(), f"census108_{os.getpid()}.dot")
    mx.export_to_dot(path, out)
    txt = open(path).read(); os.unlink(path)
    lab = dict(re.findall(r'\{ (\d+) \[label ?= ?"([^"]*)", shape=rectangle\]', txt))
    ins, outs = collections.defaultdict(list), {}
    for s, d in re.findall(r'^"?([^"\s]+)"? -> "?([^"\s]+)"?$', txt, re.M):
        if d in lab:
            ins[d].append(s)                          # array s -> op d
        elif s in lab:
            outs[d] = s                               # op s -> array d
    depth, memo = {}, {}
    sys.setrecursionlimit(100000)

    def dep(op):                                      # dispatches on the longest path ending at op (op included)
        if op in memo:
            return memo[op]
        best = 0
        for arr in ins[op]:
            src = outs.get(arr)
            if src is not None:
                best = max(best, dep(src))
        memo[op] = best + (0 if lab[op] in FREE else 1)
        return memo[op]
    for op in lab:
        depth[op] = dep(op)
    disp = [o for o in lab if lab[o] not in FREE]
    lv = collections.Counter(depth[o] for o in disp)
    path, op = [], max(depth, key=depth.get) if depth else None    # the longest path, sink to source
    while op is not None:
        if lab[op] not in FREE:
            path.append(lab[op][:40])
        nxt = [outs[x] for x in ins[op] if x in outs]
        op = max(nxt, key=lambda o: memo[o]) if nxt else None
    return dict(dispatches=len(disp), chain=max(depth.values()) if depth else 0, width=max(lv.values()) if lv else 0,
                by_type=collections.Counter(lab[o] for o in disp), path=path[::-1])


model, spec, cfg = P.build_model(a.pack)
P.patch_model(model, a.pack, spec, cfg)
R.enable(model, spec)
S.install(model, spec)
if a.first:
    model.language_model.layers = model.language_model.layers[:a.first]
tc = cfg["text_config"]
rng = np.random.default_rng(1)
cache, lg, _, _ = S.prefill(model, rng.integers(16, tc["vocab_size"] - 16, a.len).tolist())
tok = mx.array([int(mx.argmax(lg[0]).item())])
step = lambda: R.step(model, R.fork(cache), tok)
for _ in range(3):
    mx.eval(step())
c = dag(step())
ts = []
for _ in range(a.reps):
    o = step(); t0 = time.perf_counter(); mx.eval(o); ts.append((time.perf_counter() - t0) * 1e3)
nl = len(model.language_model.layers)
print(f"{os.path.basename(a.pack)} ({nl} layers) decode step: {c['dispatches']} dispatches ({c['dispatches'] / nl:.1f}/layer), serial chain {c['chain']} "
      f"({c['chain'] / nl:.1f}/layer, {c['chain'] / max(c['dispatches'], 1) * 100:.0f}% of the dispatches), widest level {c['width']}; step eval {min(ts):.2f} ms "
      f"(med {np.median(ts):.2f})", flush=True)
print("serial chain: " + " > ".join(c["path"]), flush=True)
print("by type: " + ", ".join(f"{k} {v}" for k, v in c["by_type"].most_common(a.top)), flush=True)

x0 = mx.ones((1, 64), mx.float16); mx.eval(x0)
def tiny(n, dep_):
    def run():
        if dep_:
            y = x0
            for _ in range(n):
                y = y * 1.0001
            outs = [y]
        else:
            outs = [x0 * (1 + i * 1e-4) for i in range(n)]
        t0 = time.perf_counter(); mx.eval(*outs); return time.perf_counter() - t0
    return run
T = {}
for d in (True, False):
    for n in (64, 256):
        r = tiny(n, d); r()
        T[(d, n)] = min(r() for _ in range(7))
per = {d: (T[(d, 256)] - T[(d, 64)]) / 192 * 1e6 for d in (True, False)}
gap = per[True] - per[False]
print(f"tiny op slope: dependent {per[True]:.2f} us, independent {per[False]:.2f} us -> dependent-launch gap {gap:.2f} us; "
      f"chain x gap = {c['chain'] * gap / 1e3:.2f} ms = {c['chain'] * gap / 1e3 / min(ts) * 100:.0f}% of the step (GPU idle estimate); "
      f"all dispatches x dependent = {c['dispatches'] * per[True] / 1e3:.2f} ms", flush=True)
