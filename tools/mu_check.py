"""Multi-user decode exactness on a patched mlx-vlm model slice: row i of a batched step vs the solo decode of user i, bit for bit.
usage: mu_check.py PACK [--users 4] [--steps 24] [--lens 37,64,90,131] [--set key=int ...] [--stock] [--seed 0]
Users prefill solo (stock ops) with prompts of different lengths; the solo reference and the batched run start from forks of the same
prefill cache. The batched run is teacher forced with the solo greedy tokens (a mismatch cannot snowball); it is run for N = 1..users
(the first N users) so a row is also checked against the number of rows. --stock: the stock model (before `enable`) decodes user 0 greedily,
to compare the canonical solo path with the unmodified ops."""
import argparse, json, sys, time

import numpy as np
import mlx.core as mx

from kyojin_mlx import patch as P, rows as R

ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("--users", type=int, default=4); ap.add_argument("--steps", type=int, default=24)
ap.add_argument("--lens", default="37,64,90,131"); ap.add_argument("--set", nargs="*", default=[]); ap.add_argument("--stock", action="store_true")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--union", action="store_true"); ap.add_argument("--json")
ap.add_argument("--core", default=None, help="user = per-user DS4.1 attention cores (A/B), default = the spec's adapter")
a = ap.parse_args()
over = {k: ([int(t) for t in v.split(":")] if ":" in v else int(v)) for k, v in (s.split("=", 1) for s in a.set)}   # key=7 or key=4:4 (list)
model, spec, cfg = P.build_model(a.pack, over)
P.patch_model(model, a.pack, spec, cfg)
vocab = cfg["text_config"]["vocab_size"]
rng = np.random.default_rng(a.seed)
lens = [int(v) for v in a.lens.split(",")][:a.users]
prompts = [rng.integers(16, vocab - 16, n).tolist() for n in lens]
users = [R.prefill(model, ids) for ids in prompts]
first = [int(mx.argmax(lg, axis=-1).item()) for _, lg in users]


def solo(cache, tok, n, teacher=None):
    toks, lgs = [], []
    for s in range(n):
        lg = R.step(model, cache, mx.array([tok])); mx.eval(lg)
        lgs.append(np.array(lg[0]))
        tok = int(mx.argmax(lg[0]).item()) if teacher is None else teacher[s]
        toks.append(tok)
    return toks, lgs


out = {"pack": a.pack, "lens": lens, "overrides": over}
if a.stock:
    stock = [solo(R.fork(c), f, a.steps) for (c, _), f in zip(users, first)]
rep = R.enable(model, spec, union=a.union, core=a.core)
print("enable:", rep, flush=True)
batched_solo = spec.rows.adapter == "dsv4_core" and a.core is None


def fresh_solo(c):
    """The solo decode of a user: a batch of one for the batched DS4.1 core (the canonical solo), the user's own cache otherwise."""
    return R.batch([R.fork(c)]) if batched_solo else R.fork(c)


ref = [solo(fresh_solo(c), f, a.steps) for (c, _), f in zip(users, first)]
if a.stock:
    agree = [sum(int(x == y) for x, y in zip(st[0], rf[0])) for st, rf in zip(stock, ref)]
    out["canonical_vs_stock_greedy"] = [agree, a.steps]
    print(f"canonical solo vs stock solo, free-running greedy tokens equal per user: {agree} / {a.steps}", flush=True)
    tf = [solo(fresh_solo(c), f, a.steps, teacher=st[0]) for (c, _), f, st in zip(users, first, stock)]
    top1 = [sum(int(np.argmax(l1) == np.argmax(l2)) for l1, l2 in zip(t[1], st[1])) for t, st in zip(tf, stock)]
    dmax = max(float(np.abs(l1 - l2).max()) for t, st in zip(tf, stock) for l1, l2 in zip(t[1], st[1]))
    scale = max(float(np.abs(l2).max()) for st in stock for l2 in st[1])
    out["canonical_vs_stock_teacher_forced"] = dict(top1_agree=top1, steps=a.steps, max_abs_logit_diff=dmax, logit_scale=scale)
    print(f"teacher forced on the stock tokens: top-1 agrees {top1} / {a.steps}; max |logit diff| {dmax:.3g} (max |logit| {scale:.3g})", flush=True)
out["exact"] = {}
for N in range(1, len(users) + 1):
    bc = R.batch([R.fork(c) for c, _ in users[:N]])
    cur = mx.array(first[:N])
    bad_steps, maxd, tok_ok = [], 0.0, True
    for s in range(a.steps):
        lg = R.step(model, bc, cur); mx.eval(lg)
        lg = np.array(lg)
        for i in range(N):
            if not np.array_equal(lg[i].view(np.uint32), ref[i][1][s].view(np.uint32)):
                bad_steps.append((s, i)); maxd = max(maxd, float(np.abs(lg[i] - ref[i][1][s]).max()))
            tok_ok &= int(lg[i].argmax()) == ref[i][0][s]
        cur = mx.array([ref[i][0][s] for i in range(N)])
    out["exact"][N] = dict(bitwise_all_steps=not bad_steps, mismatches=len(bad_steps), first_bad=bad_steps[:3], max_abs_diff=maxd, greedy_equal=tok_ok)
    print(f"N={N}: logits bitwise equal to solo on all {a.steps} steps x {N} rows: {not bad_steps}"
          f"{'' if not bad_steps else f' ({len(bad_steps)} bad (step,row), first {bad_steps[:3]}, max|d| {maxd:.3g})'}; greedy tokens equal: {tok_ok}", flush=True)
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
