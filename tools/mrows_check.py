"""Batched-core verify exactness on a patched mlx-vlm DS4.1 slice: sessions with several rows per pass vs the same sessions decoded one plain
step at a time, bit for bit.
usage: mrows_check.py PACK [--steps 40] [--lens 37,64,90,131] [--depth 3] [--set key=json ...] [--neg NAME] [--seed 0] [--json F] [--quiet]

The reference of a session is the canonical solo decode (a batch of one, plain steps, teacher forced with its own greedy tokens). The test
run puts all sessions in ONE batched cache and, round after round, feeds session s counts[s] <= depth + 1 consecutive tokens in ONE pass
(`rows.verify`: rows = sum of the counts), accepting accepted[s] <= counts[s] - 1 of them; the rows after the accepted ones carry wrong tokens.
Every round checks
  * the logits of the rows on the true prefix (rows 0 .. accepted[s]) equal the reference step of that position, bitwise
  * the state of every session after the commit equals the reference state after the same number of tokens (arrays, offsets, Engram history)
Some rounds are plain batched steps (one token per session) and some rebatch the sessions (extract / merge in another order), so the stacked
state is also checked across the two kinds of pass and across churn. Counts mix 1 (no draft) with 2 .. depth + 1; accepted covers 0 .. counts - 1.
--neg: fullring (a row sees the ring entries of the later rows), norollback (every session commits its last row), nolens (every row uses the
pool length of its session's last row): each must fail."""
import argparse, json, sys

import numpy as np
import mlx.core as mx

from kyojin_mlx import patch as P, rows as R, dsv4_core as DC

ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("--steps", type=int, default=40); ap.add_argument("--lens", default="37,64,90,131")
ap.add_argument("--depth", type=int, default=3); ap.add_argument("--set", nargs="*", default=[]); ap.add_argument("--neg", default="")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--json"); ap.add_argument("--quiet", action="store_true")
ap.add_argument("--parallel-gather", action="store_true")
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
lens = [int(v) for v in a.lens.split(",")]
S = len(lens)
prompts = [rng.integers(16, vocab - 16, n).tolist() for n in lens]
rep = R.enable(model, spec)
print("enable:", rep, flush=True)
assert rep["adapter"] == "dsv4_core", rep

if a.neg == "fullring":

    def overlay(self, m):
        o = self._ov.get(("full", m))
        if o is None:
            rows, slots, src = [], [], []
            for j in range(self.n):
                for i in range(self.first[self.sess[j]], self.first[self.sess[j]] + self.counts[self.sess[j]]):
                    rows.append(j), slots.append(self.pos[i] % m), src.append(i)
            o = self._ov[("full", m)] = DC.SimpleNamespace(rows=mx.array(rows, dtype=mx.int32), slots=mx.array(slots, dtype=mx.int32),
                                                           src=mx.array(src, dtype=mx.int32))
        return o
    DC.Plan.overlay = overlay
elif a.neg == "nolens":
    orig_emit = DC.Plan.emit

    def emit(self, r):
        e = orig_emit(self, r)
        if self.plain or getattr(e, "_neg", False):
            return e
        import copy
        e = copy.copy(e)
        last = [max(e.lens[self.first[s]:self.first[s] + self.counts[s]]) for s in self.sess]
        e.lens, e.lens_arr, e.tmax, e._neg = last, mx.array(last, dtype=mx.int32), max(last), True
        self._emit[r] = e
        return e
    DC.Plan.emit = emit


def bits(x):
    return np.array(x).view(np.uint32)


def same(p, q):
    if p is None or q is None:
        return p is None and q is None
    return p.shape == q.shape and p.dtype == q.dtype and bool(mx.array_equal(p, q, equal_nan=True).item())


def state_diff(x, y):
    bad = []
    if x.offset != y.offset:
        bad.append(f"offset {x.offset} != {y.offset}")
    for name in ("window", "compress", "keys", "kv_state", "score_state"):
        for l, (p, q) in enumerate(zip(getattr(x, name), getattr(y, name))):
            if not same(p, q):
                bad.append(f"{name}[{l}]")
    for name in ("engram",) + tuple(x.HANDOFF):
        if not same(getattr(x, name), getattr(y, name)):
            bad.append(name)
    return bad


users = [R.prefill(model, ids) for ids in prompts]
first = [int(mx.argmax(lg, axis=-1).item()) for _, lg in users]
n_ref = a.steps + 3 * (a.depth + 1) + 4
g, ref_lg, ref_cache = [], [], []
for s, (c0, _) in enumerate(users):
    gs, ls, cs = [first[s]], [], [R.fork(c0)[0]]
    bc = R.batch([R.fork(c0)])
    for k in range(n_ref):
        lg = R.step(model, bc, mx.array([gs[-1]], dtype=mx.int32)); mx.eval(lg)
        ls.append(np.array(lg[0]))
        cs.append(bc[0].extract(0))
        gs.append(int(mx.argmax(lg[0]).item()))
    g.append(gs), ref_lg.append(ls), ref_cache.append(cs)
print(f"reference: {S} sessions x {n_ref} plain steps (batch of one), prompts {lens}", flush=True)

order = list(range(S))                                         # order[i] = session held by row i of the batched cache
bc = R.batch([R.fork(users[s][0]) for s in order])
pos = [0] * S                                                  # tokens consumed after the prompt, per session
rounds, bad, n_rows, n_acc = 0, [], 0, 0
while min(pos) < a.steps:
    kind = "plain" if rounds % 5 == 4 else "spec"
    if rounds % 7 == 6:                                        # churn: back to per-session caches, other order
        caches = [bc[0].extract(i) for i in range(S)]
        perm = list(reversed(range(S))) if (rounds // 7) % 2 == 0 else list(range(S))
        order = [order[i] for i in perm]
        bc = R.batch([[caches[i]] for i in perm])
    problems = []
    if kind == "plain":
        toks = [g[s][pos[s]] for s in order]
        lg = R.step(model, bc, mx.array(toks, dtype=mx.int32)); mx.eval(lg)
        lg = np.array(lg)
        for i, s in enumerate(order):
            if not np.array_equal(bits(lg[i]), bits(ref_lg[s][pos[s]])):
                problems.append(f"plain: session {s} logits != reference")
            pos[s] += 1
        acc_list = [0] * S
        counts = [1] * S
    else:
        counts = [1 + (rounds * 5 + 3 * i + i * i) % (a.depth + 1) for i in range(S)]
        acc_list = [int(rng.integers(0, c)) for c in counts]
        tok_lists = []
        for i, s in enumerate(order):
            t = [g[s][pos[s] + k] for k in range(counts[i])]
            for k in range(acc_list[i] + 1, counts[i]):
                t[k] = int((t[k] + 1 + rounds + k) % vocab)
            tok_lists.append(t)
        ps = R.verify(model, bc, tok_lists)
        mx.eval(ps.logits)
        lg = np.array(ps.logits)
        for i, s in enumerate(order):
            for k in range(acc_list[i] + 1):
                if not np.array_equal(bits(lg[ps.starts[i] + k]), bits(ref_lg[s][pos[s] + k])):
                    problems.append(f"session {s} row {k} (count {counts[i]}, accepted {acc_list[i]}) logits != reference "
                                    f"(max|d| {np.abs(lg[ps.starts[i] + k] - ref_lg[s][pos[s] + k]).max():.3g})")
        ps.commit([c - 1 for c in counts] if a.neg == "norollback" else acc_list)
        for i, s in enumerate(order):
            pos[s] += acc_list[i] + 1
        n_rows += sum(counts)
        n_acc += sum(acc_list)
    for i, s in enumerate(order):
        sd = state_diff(bc[0].extract(i), ref_cache[s][pos[s]])
        if sd:
            problems.append(f"session {s} state after the pass differs: " + ", ".join(sd[:5]))
    if problems:
        bad.append((rounds, kind, counts, acc_list, problems))
        if not a.quiet:
            print(f"round {rounds} {kind} counts={counts} accepted={acc_list}: " + "; ".join(problems[:4]), flush=True)
        if a.neg == "norollback":
            bc = R.batch([[R.fork(ref_cache[s][pos[s]])] for s in order])
    rounds += 1
ok = not bad
print(f"{rounds} rounds, {n_rows} rows verified, {n_acc} drafts accepted, rounds with a problem: {len(bad)}  =>  {'EXACT' if ok else 'MISMATCH'}", flush=True)
if a.json:
    json.dump(dict(pack=a.pack, over=over, neg=a.neg, rounds=rounds, rows=n_rows, exact=ok, bad=[(r, k, c, ac, p[:3]) for r, k, c, ac, p in bad[:10]]),
              open(a.json, "w"), indent=1)
sys.exit(0 if ok else 1)
