#!/usr/bin/env python3
"""S5 on a real patched DS4.1 / GLM slice pack (e.g. models/ds2: 2 real layers, 32 experts, 4096 vocab): groups found, launches per decode step,
bit-exactness of the logits (S5 off vs on, separate caches fed the same tokens), CPU time per step with the GPU idle at submit.
usage: s5_slice.py PACK_DIR [--rows 1,4] [--steps 8] [--cpu-steps 300] [--json OUT.json]
Run it under `/usr/bin/time -l` for the peak memory."""
import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import numpy as np
import mlx.core as mx

from count_launches import graph_census
from kyojin_mlx import fused_group as FG
from kyojin_mlx import patch as P

ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("--rows", default="1,4"); ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--cpu-steps", type=int, default=300); ap.add_argument("--json", default=None)
a = ap.parse_args()
model, spec, cfg = P.build_model(a.pack)
rep = P.patch_model(model, a.pack, spec, cfg)
lm = model.language_model
vocab = cfg["text_config"]["vocab_size"]
print("patched:", rep, flush=True)
rng = np.random.default_rng(0)
seq = rng.integers(3, vocab, 4 + 4 * a.steps + a.cpu_steps + 40).tolist()


def prefill(n=5):
    cache = lm.make_cache()
    mx.eval(lm(mx.array([seq[:n]]), cache=cache).logits)
    return cache


info = FG.install(model)
print(f"S5 install: {info['groups']} groups, {info['members']} members, {info['launches_saved']} launches saved")
for g in info["names"]:
    print("   group:", ", ".join(n.replace("language_model.", "") for n in g))
res = dict(install=info, rep=rep)

# ---- launches per step
for rows in [int(r) for r in a.rows.split(",")]:
    cen = {}
    for v, en in (("off", False), ("on", True)):
        FG.ENABLED = en
        cache = prefill()
        cen[v] = graph_census(lm(mx.array([seq[5:5 + rows]]), cache=cache).logits, rows)
    FG.ENABLED = True
    off, on = cen["off"], cen["on"]
    new = {k: on["copylike"].get(k, 0) - off["copylike"].get(k, 0) for k in set(on["copylike"]) | set(off["copylike"])}
    print(f"rows {rows}: dispatches/step off {off['dispatches']} on {on['dispatches']} ({on['dispatches'] - off['dispatches']:+d}); nodes {off['nodes']} -> {on['nodes']}; "
          f"CustomKernel {off['by_type'].get('CustomKernel', 0)} -> {on['by_type'].get('CustomKernel', 0)}; new copy-like nodes: {({k: v for k, v in new.items() if v > 0}) or 'none'}")
    res[f"census_rows{rows}"] = cen
    res[f"new_copy_nodes_rows{rows}"] = {k: v for k, v in new.items() if v > 0}

# ---- exactness: separate caches, same tokens, rows 1 then 2
cs = {v: prefill() for v in ("off", "on")}
same, maxd = True, 0.0
for i in range(a.steps):
    n = 1 + (i % 2)
    tok = mx.array([seq[5 + 2 * i: 5 + 2 * i + n]])
    outs = {}
    for v, en in (("off", False), ("on", True)):
        FG.ENABLED = en
        outs[v] = lm(tok, cache=cs[v]).logits
        mx.eval(outs[v])
    FG.ENABLED = True
    eq = bool(mx.array_equal(outs["off"], outs["on"]).item())
    same &= eq
    if not eq:
        maxd = max(maxd, float(mx.abs(outs["off"].astype(mx.float32) - outs["on"].astype(mx.float32)).max().item()))
print(f"logits S5 off vs on over {a.steps} decode steps (1 and 2 rows): {'BIT-IDENTICAL' if same else 'DIFFER, max |d| ' + str(maxd)}")
res["logits_bit_identical"] = same

# ---- CPU per step, GPU idle at submit
if a.cpu_steps:
    cs = {v: prefill() for v in ("off", "on")}
    rec = {False: ([], []), True: ([], [])}
    for i in range(a.cpu_steps):
        en = bool(i % 2)
        FG.ENABLED = en
        tok = mx.array([[seq[5 + i % 200]]])
        b0 = time.perf_counter(); out = lm(tok, cache=cs["on" if en else "off"]).logits; b1 = time.perf_counter()
        c0 = time.thread_time(); mx.async_eval(out); c1 = time.thread_time()
        mx.eval(out)
        if i >= 8:
            rec[en][0].append((b1 - b0) * 1e3); rec[en][1].append((c1 - c0) * 1e3)
    FG.ENABLED = True
    iso = {("on" if k else "off"): dict(build_ms=statistics.median(v[0]), encode_cpu_ms=statistics.median(v[1])) for k, v in rec.items()}
    for v in ("off", "on"):
        iso[v]["cpu_ms"] = iso[v]["build_ms"] + iso[v]["encode_cpu_ms"]
        print(f"CPU per step, S5 {v:3s}: build {iso[v]['build_ms']:.3f} ms + async_eval encode {iso[v]['encode_cpu_ms']:.3f} ms = {iso[v]['cpu_ms']:.3f} ms")
    print(f"CPU per step on vs off: {100 * (iso['on']['cpu_ms'] / iso['off']['cpu_ms'] - 1):+.1f} % ({iso['on']['cpu_ms'] - iso['off']['cpu_ms']:+.3f} ms)")
    res["isolated_cpu"] = iso
print(f"mlx peak {mx.get_peak_memory() / 2 ** 30:.2f} GiB")
if a.json:
    json.dump(res, open(a.json, "w"), indent=1)
sys.exit(0 if same else 1)
