"""CPU cost of a speculative round, in solo steps: python graph build + async_eval encode of the chained verify of 1..5 rows and of the plain step,
and the wall time of the draft call (it synchronises inside, so that one includes the GPU wait: an upper bound of its CPU cost).
usage: spec_cpu.py PACK [--len 135] [--reps 25] [--rows 5] [--json F]   (a pack with draft tensors, e.g. ds2m)
On a machine whose GPU is faster than its CPU (the Ultra) a step costs its CPU time, so these are the ratios of the round cost there.
Variants run interleaved, medians and minima."""
import argparse, json, os, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("M10_LIGHT"):
    import light                                              # custom-kernel grids divided by F: the GPU is never the limit (outputs are garbage)

import mlx.core as mx
import numpy as np

from kyojin_mlx import dspark as D, patch as P, rows as R, spec as S

ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("--len", type=int, default=135); ap.add_argument("--reps", type=int, default=25)
ap.add_argument("--rows", type=int, default=5); ap.add_argument("--json")
a = ap.parse_args()
model, spec, cfg = P.build_model(a.pack)
P.patch_model(model, a.pack, spec, cfg)
R.enable(model, spec)
tc = cfg["text_config"]
has_draft = spec.mtp is not None and int(tc.get(spec.mtp.keys["stages"], 0)) > 0
drafter = D.build_drafter(model, a.pack, spec, cfg, conf_threshold=0.0) if has_draft else None
S.install(model, spec, tap_layers=set(drafter.tap_ids) if drafter else ())
rng = np.random.default_rng(1)
prompt = rng.integers(16, tc["vocab_size"] - 16, a.len).tolist()
cache, lg, taps, first = S.prefill(model, prompt, drafter.tap_ids if drafter else ())
tok = [int(mx.argmax(lg[0]).item())] + rng.integers(16, tc["vocab_size"] - 16, a.rows).tolist()
if drafter:
    drafter.append([taps[l] for l in drafter.tap_ids], first)
names = [f"verify{n}" for n in range(1, a.rows + 1)] + ["solo"]
build, enc, draft, dparts = {k: [] for k in names}, {k: [] for k in names}, [], []
for rep in range(a.reps + 2):
    for k in names:
        if k == "solo":
            c = R.fork(cache)
            t0 = time.perf_counter(); out = R.step(model, c, mx.array([tok[0]])); t1 = time.perf_counter(); mx.async_eval(out); t2 = time.perf_counter(); mx.eval(out)
        else:
            n = int(k[6:])
            t0 = time.perf_counter(); out = S.verify(model, cache, tok[:n])[0]; t1 = time.perf_counter(); mx.async_eval(out); t2 = time.perf_counter(); mx.eval(out)
        if rep >= 2:
            build[k].append(t1 - t0); enc[k].append(t2 - t1)
    if drafter:
        mark = {}
        real_eval = mx.eval

        def hook(*xs):                                        # the draft call ends with one mx.eval: split it into build (before), encode (async_eval) and GPU wait
            mark["b"] = time.perf_counter()
            mx.async_eval(*xs); mark["e"] = time.perf_counter()
            real_eval(*xs); mark["w"] = time.perf_counter()
        mx.eval = hook
        t = time.perf_counter(); drafter.draft(tok[0], a.len, 3); dt = time.perf_counter() - t
        mx.eval = real_eval
        if rep >= 2:
            draft.append(dt); dparts.append((mark["b"] - t, mark["e"] - mark["b"], mark["w"] - mark["e"]))
cpu = {k: [b + e for b, e in zip(build[k], enc[k])] for k in names}
res = {}
print(f"{'':9s} {'build':>7s} {'encode':>7s} {'cpu med':>8s} {'cpu min':>8s} {'x solo (min)':>13s}")
for k in names:
    res[k] = dict(build=1e3 * statistics.median(build[k]), enc=1e3 * statistics.median(enc[k]), med=1e3 * statistics.median(cpu[k]), min=1e3 * min(cpu[k]))
    print(f"{k:9s} {res[k]['build']:7.2f} {res[k]['enc']:7.2f} {res[k]['med']:8.2f} {res[k]['min']:8.2f} {res[k]['min'] / (1e3 * min(cpu['solo'])):13.2f}")
if draft:
    res["draft_wall"] = dict(min=1e3 * min(draft), med=1e3 * statistics.median(draft))
    print(f"draft call wall (includes GPU wait): min {res['draft_wall']['min']:.2f} ms, med {res['draft_wall']['med']:.2f} ms = x{res['draft_wall']['min'] / (1e3 * min(cpu['solo'])):.2f} of the solo CPU")
    bm, em, wm = (1e3 * statistics.median(x) for x in zip(*dparts))
    bn, en = 1e3 * min(b for b, _, _ in dparts), 1e3 * min(e for _, e, _ in dparts)
    cmin = 1e3 * min(b + e for b, e, _ in dparts)
    res["draft_cpu"] = dict(build_med=bm, enc_med=em, wait_med=wm, build_min=bn, enc_min=en, cpu_min=cmin)
    print(f"draft call split: build {bm:.2f} ms + encode {em:.2f} ms (medians; CPU min {cmin:.2f} ms = x{cmin / (1e3 * min(cpu['solo'])):.2f} of the solo CPU) + GPU wait {wm:.2f} ms")
if a.json:
    json.dump(res, open(a.json, "w"), indent=1)
