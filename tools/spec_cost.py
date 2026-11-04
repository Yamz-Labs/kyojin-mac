"""Cost of one speculative round on a pack, in units of a solo decode step: chained verify of 1..5 rows, and the draft call when the pack holds a draft head.
usage: spec_cost.py PACK [--len 135] [--reps 9] [--rows 5] [--json F] [--set key=json ...]
Medians of --reps interleaved runs (each synced). t(1 row chained) is the plain step of the speculative loop; t(n)/t(1) is what a round pays for n - 1 drafts.
On a slice the vocab is cut, so the head (a fixed share of a real step) is not in these numbers."""
import argparse, json, statistics, time

import mlx.core as mx
import numpy as np

from kyojin_mlx import dspark as D, patch as P, rows as R, spec as S

ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("--len", type=int, default=135); ap.add_argument("--reps", type=int, default=9)
ap.add_argument("--chunk", type=int, default=0); ap.add_argument("--rows", type=int, default=5); ap.add_argument("--json"); ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()
over = {k: json.loads(v) for k, v in (s.split("=", 1) for s in a.set)}
model, spec, cfg = P.build_model(a.pack, over)
P.patch_model(model, a.pack, spec, cfg)
R.enable(model, spec)
tc = {**cfg["text_config"], **over}
cfg = {**cfg, "text_config": tc}
has_draft = spec.mtp is not None and int(tc.get(spec.mtp.keys["stages"], 0)) > 0
drafter = D.build_drafter(model, a.pack, spec, cfg, conf_threshold=0.0) if has_draft else None
S.install(model, spec, tap_layers=set(drafter.tap_ids) if drafter else ())
rng = np.random.default_rng(1)
prompt = rng.integers(16, tc["vocab_size"] - 16, a.len).tolist()
cache, lg, taps, first = S.prefill(model, prompt, drafter.tap_ids if drafter else (), chunk=a.chunk or None)
tok = [int(mx.argmax(lg[0]).item())] + rng.integers(16, tc["vocab_size"] - 16, a.rows).tolist()
if drafter:
    drafter.append([taps[l] for l in drafter.tap_ids], first)
    print(f"memory after load: peak {mx.get_peak_memory() / 2**30:.2f} GiB", flush=True)
times = {f"verify{n}": [] for n in range(1, a.rows + 1)}
times["solo"], times["draft"] = [], []
for rep in range(a.reps + 2):
    for n in range(1, a.rows + 1):
        t = time.perf_counter(); lgs, _, _ = S.verify(model, cache, tok[:n]); mx.eval(lgs); dt = time.perf_counter() - t
        if rep >= 2: times[f"verify{n}"].append(dt)
    c = R.fork(cache)
    t = time.perf_counter(); lg1 = R.step(model, c, mx.array([tok[0]])); mx.eval(lg1); dt = time.perf_counter() - t
    if rep >= 2: times["solo"].append(dt)
    if drafter:
        t = time.perf_counter(); drafter.draft(tok[0], a.len, 3); dt = time.perf_counter() - t
        if rep >= 2: times["draft"].append(dt)
med = {k: 1e3 * statistics.median(v) for k, v in times.items() if v}
base = med["verify1"]
for k, v in med.items():
    print(f"{k:9s} {v:8.2f} ms   x{v / base:.2f} of verify1")
print(f"peak memory {mx.get_peak_memory() / 2**30:.2f} GiB")
if a.json:
    json.dump(dict(pack=a.pack, over=over, ms=med, peak_gib=mx.get_peak_memory() / 2**30), open(a.json, "w"), indent=1)
