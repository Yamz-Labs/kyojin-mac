"""Chained verify exactness on a patched mlx-vlm DS4.1 slice: speculative rounds with a scripted drafter vs plain solo decode, bit for bit.
usage: spec_check.py PACK [--steps 64] [--len 135] [--depth 3] [--set key=json ...] [--neg nolink|norollback|stock_matmul|unpinned_moe] [--seed 0] [--json F]

The drafter is an oracle: the true greedy continuation (from a plain solo run on the same enabled model) with one wrong token injected at
draft index k. k cycles over -1 (all drafts right), 0 .. D - 1, so a round accepts 0 .. D drafts; D cycles 1 .. --depth. For every round:
  * logits of the rows whose prefix is the true one (rows 0 .. a) equal the solo logits of that step, bitwise
  * logits of the rows after the wrong token equal a solo run over the wrong prefix (rows are exact whatever they are fed)
  * the accepted count is the injected index, the emitted tokens are the plain greedy tokens
  * the rollback cache (chain.rows[a]) equals the solo cache after the same tokens, array by array, bitwise
The run then goes on from the rollback cache, so a wrong state would show up in every later round.
--neg nolink: rows do not take over each other's layer state (must fail). norollback: the cache after the last row is kept whatever
was accepted (must fail whenever a < D). stock_matmul / unpinned_moe: one row-invariance measure of rows.py is switched off (must fail on the logits).

KYOJIN_ROWS_NAX=0 (with KYOJIN_ATTEND_MMA=0 KYOJIN_ATTEND_SPLIT=0: the multi-query MMA attention and the split P@V sum in a row-count dependent order, their rounding class is gated by lever_check): everything above is bitwise. KYOJIN_ROWS_NAX=1 (default; verify rows above 4 run the NAX kernel, fp16 rounding class, see tolgate.py):
the same rounds are driven by the true acceptance (the scripted one), and "bitwise" becomes: per row max|d| / max|logit| <= REL_LOGIT, cache arrays
within REL_STATE, greedy top-1 of every row (true prefix and wrong prefix) equal to the solo run in >= 99.5 % of the rows, every flip at a near-tie
margin of the solo logits. The negative controls must still fail."""
import argparse, json, os, sys

import numpy as np
import mlx.core as mx

from kyojin_mlx import patch as P, rows as R, spec as S
import tolgate as T

ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("--steps", type=int, default=64); ap.add_argument("--len", type=int, default=135)
ap.add_argument("--depth", type=int, default=3); ap.add_argument("--set", nargs="*", default=[]); ap.add_argument("--neg", default="")
ap.add_argument("--parallel-gather", action="store_true", help="Engram tables through the parallel pread gather (what day 1 runs)")
ap.add_argument("--tol", action="store_true", help="force the tolerance mode (default: on iff KYOJIN_ROWS_NAX is on)")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--json"); ap.add_argument("--quiet", action="store_true")
a = ap.parse_args()
over = {k: json.loads(v) for k, v in (s.split("=", 1) for s in a.set)}
model, spec, cfg = P.build_model(a.pack, over)
P.patch_model(model, a.pack, spec, cfg)
if a.parallel_gather:
    from kyojin_mlx import engram_rows as ER, engram_mlx as EM
    es = ER.EngramSpec.from_config(cfg)
    assert es is not None, "--parallel-gather: the pack has no Engram layer"
    EM.install_parallel_gather(model, ER.EngramTables(a.pack, es))
vocab = cfg["text_config"]["vocab_size"]
rng = np.random.default_rng(a.seed)
prompt = rng.integers(16, vocab - 16, a.len).tolist()
if a.neg == "stock_matmul":
    import kyojin_mlx.rowinv as RI
    R.rowmm = RI.rowmm = lambda x, w: x @ w.T
rep = R.enable(model, spec)
tol = a.tol or T.tolerant()
print("mode:", "tolerance (NAX rows ON)" if tol else "bitwise (KYOJIN_ROWS_NAX=0)", flush=True)
if a.neg == "unpinned_moe":
    for _, m in model.language_model.named_modules():
        mo = getattr(m, "moe", None)
        if isinstance(mo, P.M.MoeGather):
            mo.pin_rows = None; mo._p.clear()
print("enable:", rep, flush=True)
print("spec.install:", S.install(model, spec, tap_layers=(0, 1)), flush=True)
cache0, lg0 = R.prefill(model, prompt)
first = int(mx.argmax(lg0, axis=-1).item())


def bits(x):
    return np.array(x).view(np.uint32)


# ---- reference: plain solo greedy, a fork of the cache after every step
n_ref = a.steps + 2 * a.depth + 4
g, ref_lg, ref_cache = [first], [], []
cur = R.fork(cache0)
for s in range(n_ref):
    lg = R.step(model, cur, mx.array([g[-1]])); mx.eval(lg)
    ref_lg.append(np.array(lg[0])); ref_cache.append(R.fork(cur))
    g.append(int(mx.argmax(lg[0]).item()))
print(f"reference: {n_ref} solo steps from a {a.len}-token prompt", flush=True)


def same(p, q):
    if p is None or q is None:
        return p is None and q is None
    return p.shape == q.shape and p.dtype == q.dtype and bool(mx.array_equal(p, q, equal_nan=True).item())


def close(p, q):
    """tolerance variant of same(): floats within REL_STATE of the largest value of the array, everything else bitwise"""
    if p is None or q is None or not mx.issubdtype(p.dtype, mx.floating):
        return same(p, q)
    if p.shape != q.shape or p.dtype != q.dtype:
        return False
    r = T.maxrel(np.array(p.astype(mx.float32)), np.array(q.astype(mx.float32)))
    stat["state"] = max(stat["state"], r)
    return r <= T.REL_STATE


stat = dict(state=0.0, logit=0.0, rows=0, flips=0, tie_flips=0)


def rowcheck(got, ref, label, problems):
    """tolerance check of one logit row against its solo row; the greedy top-1 must agree, a flip only at a near tie"""
    r = T.maxrel(got, ref)
    if os.environ.get("TOL_DEBUG"):
        d = np.asarray(got, np.float64) - np.asarray(ref, np.float64)
        print(f"   {label}: max|ref| {np.abs(ref).max():.3g} rms(ref) {np.sqrt((np.asarray(ref, np.float64) ** 2).mean()):.3g} max|d| {np.abs(d).max():.3g} rms(d) {np.sqrt((d ** 2).mean()):.3g} margin {T.margin(ref):.3g}")
    stat["logit"] = max(stat["logit"], r)
    stat["rows"] += 1
    if r > T.REL_LOGIT:
        problems.append(f"{label} logits rel {r:.3g} > {T.REL_LOGIT}")
    ag, ar = int(np.argmax(got)), int(np.argmax(ref))
    if ag != ar:
        stat["flips"] += 1
        if T.near_tie(ref, ag, float(np.abs(np.asarray(got, np.float64) - ref).max())):
            stat["tie_flips"] += 1
        else:
            problems.append(f"{label} top-1 flip {ar} -> {ag} not at a near tie (gap {ref.max() - ref[ag]:.3g})")


def state_diff(x, y):
    """Names of the cache slots that differ (bitwise, up to the sign of zero) between two DeepseekV41Cache objects."""
    bad = []
    if x.offset != y.offset:
        bad.append(f"offset {x.offset} != {y.offset}")
    for name in ("window", "compress", "keys", "kv_state", "score_state"):
        for l, (p, q) in enumerate(zip(getattr(x, name), getattr(y, name))):
            if not (close if tol else same)(p, q):
                bad.append(f"{name}[{l}]")
    for name in ("engram",) + tuple(x.HANDOFF):
        if not (close if tol else same)(getattr(x, name), getattr(y, name)):
            bad.append(name)
    return bad


cache, pos = R.fork(cache0), 0                    # pos: index in g of the pending token (consumed = pos tokens after the prompt)
emitted = [first]
rounds, tot_acc, tot_draft, bad = 0, 0, 0, []
k_cycle = [-1, 0, 1, 2, -1, 1, 0, 2]
while pos < a.steps:
    D = 1 + rounds % a.depth
    k = k_cycle[rounds % len(k_cycle)]
    k = k if k < D else -1
    drafts = [g[pos + 1 + i] for i in range(D)]
    if k >= 0:
        drafts[k] = (drafts[k] + 1 + rounds) % vocab
        if drafts[k] == g[pos + 1 + k]:
            drafts[k] = (drafts[k] + 1) % vocab
    toks = [g[pos]] + drafts
    lg, taps, chain = S.verify(model, cache, toks, link=a.neg != "nolink")
    mx.eval(lg)
    lg = np.array(lg)
    am = lg.argmax(axis=-1)
    acc = 0
    while acc < D and int(am[acc]) == drafts[acc]:
        acc += 1
    exp_acc = D if k < 0 else k
    problems = []
    if acc != exp_acc and not tol:
        problems.append(f"accepted {acc} expected {exp_acc}")
    if tol:
        acc = exp_acc                                                  # the scripted acceptance drives the round; a top-1 flip is counted per row below
    for i in range(acc + 1):                                           # rows on the true prefix vs the plain run
        if tol:
            rowcheck(lg[i], ref_lg[pos + i], f"row {i}", problems)
        elif not np.array_equal(bits(lg[i]), bits(ref_lg[pos + i])):
            problems.append(f"row {i} logits != solo (max|d| {np.abs(lg[i] - ref_lg[pos + i]).max():.3g})")
    if acc < D:                                                        # rows after the wrong token vs a solo run over the wrong prefix
        wc = R.fork(ref_cache[pos + acc])                              # cache after toks[0 .. acc], all true
        for i in range(acc + 1, D + 1):
            wl = R.step(model, wc, mx.array([toks[i]])); mx.eval(wl)
            if tol:
                rowcheck(lg[i], np.array(wl[0]), f"wrong-prefix row {i}", problems)
            elif not np.array_equal(bits(lg[i]), bits(np.array(wl[0]))):
                problems.append(f"wrong-prefix row {i} logits != solo over the same tokens")
    new_cache = S.commit(chain, len(toks) - 1 if a.neg == "norollback" else acc)
    sd = state_diff(new_cache[0], ref_cache[pos + acc][0])
    if sd:
        problems.append("cache after rollback differs: " + ", ".join(sd[:6]) + (" ..." if len(sd) > 6 else ""))
    out_toks = [int(am[i]) for i in range(acc + 1)]
    if tol:
        out_toks = g[pos + 1:pos + acc + 2]                            # the plain greedy tokens (the flips are counted per row)
    if out_toks != g[pos + 1:pos + acc + 2]:
        problems.append(f"emitted {out_toks} != plain {g[pos + 1:pos + acc + 2]}")
    if problems:
        bad.append((rounds, D, k, problems))
        if not a.quiet:
            print(f"round {rounds} D={D} k={k}: " + "; ".join(problems), flush=True)
    emitted += out_toks
    tot_acc += acc
    tot_draft += D
    cache, pos, rounds = new_cache, pos + acc + 1, rounds + 1
    if a.neg == "norollback" and acc < D:
        # keep going from the plain state so one failure does not hide the rest: the check above already counted it
        cache = R.fork(ref_cache[pos - 1])
ok = not bad and emitted == g[:len(emitted)]
if tol:
    agree = 1 - stat["flips"] / max(stat["rows"], 1)
    ok = ok and (agree >= T.MIN_AGREE or stat["rows"] < T.MIN_ROWS)             # one flip in fewer rows than that is already below 99.5 %: judged on the pool (tolpool.py)
    print(f"tolerance: logit rel max {stat['logit']:.3g} (<= {T.REL_LOGIT}), state rel max {stat['state']:.3g} (<= {T.REL_STATE}), top-1 agreement vs solo "
          f"{agree:.4f} over {stat['rows']} rows ({stat['flips']} flips, {stat['tie_flips']} at a near tie; >= {T.MIN_AGREE}, enforced per run from {T.MIN_ROWS} rows, else by tolpool.py)", flush=True)
print(f"{rounds} rounds, {tot_acc}/{tot_draft} drafts accepted, {len(emitted)} tokens, greedy equal to plain decode: {emitted == g[:len(emitted)]}; "
      f"rounds with a problem: {len(bad)}  =>  {'EXACT' if ok else 'MISMATCH'}", flush=True)
try:
    from kyojin_mlx import fuse_layer as FL
    print("fuse_layer.STATS:", FL.STATS, flush=True)                     # the compiled chain must have run (call_chain > 0) without a fallback
except Exception:
    pass
if a.json:
    json.dump(dict(pack=a.pack, over=over, neg=a.neg, rounds=rounds, accepted=tot_acc, drafted=tot_draft, tokens=len(emitted), exact=ok,
                   rows=stat["rows"], flips=stat["flips"], tie_flips=stat["tie_flips"], logit_rel=stat["logit"], state_rel=stat["state"] if stat["state"] != float("inf") else -1,
                   bad=[(r, d, k, p[:3]) for r, d, k, p in bad[:10]]), open(a.json, "w"), indent=1)
sys.exit(0 if ok else 1)
