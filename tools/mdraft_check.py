"""Batched draft == per-session draft, bit for bit: the proposals, the confidence values and the draft rings of S sessions drafted and fed in
ONE pass equal those of each session alone (a pass of S = 1), through rounds of mixed session subsets, feed sizes 1..4 and a full ring window.
usage: mdraft_check.py PACK [--rounds 24] [--sessions 4] [--neg ring|mixes|moe|markov] [--json]
Negative controls: ring = a session drafts from the ring of another one (must report DIFFERENT); the other three break one row-invariance mechanism
(they report DIFFERENT only where the stock kernel does pick another reduction order for these row counts: not on the slice); mixes = stock mHC mixes and router (matmul), moe = k-split of the routed experts follows the rows, markov = stock matmul.
KYOJIN_ROWS_NAX=0: bitwise as described. KYOJIN_ROWS_NAX=1 (default; S sessions x block rows > 4 run the NAX kernel, fp16 rounding class, tolerance in tolgate.py):
  * rings of the batched and the per-session drafter within REL_STATE (trunk kv: the rounding class of spec_check),
  * session isolation, bitwise even with NAX on: the block of a session (tokens, confidences, logits) is the same whatever the OTHER rows of the pass hold
    (the other sessions are swapped for other rings and pending tokens, same row count),
  * batched vs per-session proposals, step by step until the first flip (the rest of that block is fed by the flipped token): a flip only where the solo gap is
    within 2 x the logit deviation measured on that step. The deviations themselves are printed, not bounded: the slice draft head is noise (flat, ill-conditioned logits):
    half an fp16 ulp of noise on the taps (--perturb 0.000488, NAX off) gives logit rel 0.38 and the same ~21 % flips as the NAX rows, so no logit tolerance can tell
    a rounding from a fault there. Its flips are harmless (proposals never decide the output): the agreement is reported, enforced only with --min-agree (0.995 on the real head).
Also prints, for information, how often the batched head proposes what the stock drafter (B = 1, block as a sequence) proposes."""
import argparse, json, sys

import mlx.core as mx
import numpy as np

import tolgate as T

from kyojin_mlx.serve.backends import load_backend

ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("--rounds", type=int, default=24); ap.add_argument("--sessions", type=int, default=4)
ap.add_argument("--neg", choices=("ring", "mixes", "moe", "markov")); ap.add_argument("--seed", type=int, default=3); ap.add_argument("--json", action="store_true")
ap.add_argument("--tol", action="store_true", help="force the tolerance mode (default: on iff KYOJIN_ROWS_NAX is on)")
ap.add_argument("--perturb", type=float, default=0.0, help="control: the solo rings are fed with taps scaled by (1 + N(0, X)): the sensitivity of the head to input rounding (X = 2^-11 is half an fp16 ulp)")
ap.add_argument("--min-agree", type=float, default=0.0, help="tolerance mode: required proposal agreement (default 0: the slice draft head is noise, its proposals never reach the output; "
                "use 0.995 on the real DS4.1 head)")
a = ap.parse_args()
mx.set_cache_limit(256 * 2 ** 20)
be, cfg = load_backend(a.pack, mtp=True)
dr = be.drafter
tol = a.tol or T.tolerant()
dr.keep_margin = tol
print("mode:", "tolerance (NAX rows ON)" if tol else "bitwise (KYOJIN_ROWS_NAX=0)", flush=True)
H = cfg["text_config"]["hidden_size"]
V = cfg["text_config"]["vocab_size"]
rng = np.random.default_rng(a.seed)
S_MAX = a.sessions
if a.neg == "markov":                                              # the markov product through the stock matmul: not row invariant
    w = dr.lm.markov_head.weight
    dr.lm.markov_head = type("Stock", (), {"__call__": lambda self, x: x @ w.T, "weight": w})()
elif a.neg == "mixes":                                             # stock mHC mixes and routers
    for blk in dr.lm.layers:
        blk.__class__ = blk.__class__.__mro__[1]
        blk.ffn.gate.__class__ = blk.ffn.gate.__class__.__mro__[1]
elif a.neg == "moe":
    for blk in dr.lm.layers:
        blk.ffn.switch_mlp.moe.pin_rows = None
        blk.ffn.switch_mlp.moe._p.clear()


def make_ring(n):
    d = be.new_draft()
    c = be.make_cache()
    ids = rng.integers(16, V - 16, n).tolist()
    for i in range(0, n, 32):
        be.prefill_chunk(c, ids[i:i + 32], d)
    mx.eval(*be._ring(d))
    return d


def clone(d):
    n = be.new_draft()
    n.rows = list(d.rows)
    return n


lens = [41, 100, 126, 5, 77, 130, 12, 64][:S_MAX]
batched = [make_ring(n) for n in lens]
solo = [clone(d) for d in batched]
pos = [n for n in lens]                                            # position of the pending token of each session
stat = dict(rows=0, flips=0, tie_flips=0, conf=0.0, ring=0.0, logit=0.0)


def block_cmp(tb, cb, lb, tr, cr, lr):
    """batched block (tokens, confidences, logits [L, V]) vs the solo one, step by step until the first flip. The logit / confidence deviations are
    recorded, not bounded (the slice head is ill-conditioned, see the header); a flip only where the solo gap to the flipped token is within 2 x the
    logit deviation measured on that step."""
    pr = []
    for i in range(len(tr)):
        stat["rows"] += 1
        r = T.maxrel(lb[i], lr[i])
        stat["logit"] = max(stat["logit"], r)
        cl = [float(np.log(np.clip(c, 1e-6, 1 - 1e-6) / (1 - np.clip(c, 1e-6, 1 - 1e-6)))) for c in (cb[i], cr[i])]
        rc = abs(cl[0] - cl[1]) / max(abs(cl[1]), 1.0)
        stat["conf"] = max(stat["conf"], rc)
        if tb[i] != tr[i]:
            stat["flips"] += 1
            if T.near_tie(lr[i], tb[i], float(np.abs(lb[i] - lr[i]).max())):
                stat["tie_flips"] += 1
            else:
                pr.append(f"step {i} proposal flip {tr[i]} -> {tb[i]} not at a near tie")
            break                                                      # the following steps are fed by the flipped token
    return not pr, pr


def ring_close(x, y):
    if x is None or y is None:
        return x is None and y is None
    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    if not mx.issubdtype(x.dtype, mx.floating):
        return bool(mx.array_equal(x, y).item())
    r = T.maxrel(np.array(x.astype(mx.float32)), np.array(y.astype(mx.float32)))
    stat["ring"] = max(stat["ring"], r)
    return r <= T.REL_STATE


ok, rep = True, dict(rounds=0, draft_equal=0, conf_equal=0, ring_equal=0, checks=0, iso_checks=0, iso_equal=0, stock_agree=0, stock_total=0)
ntaps = len(dr.tap_ids)
for r in range(a.rounds):
    k = int(rng.integers(1, S_MAX + 1))
    sel = sorted(rng.choice(S_MAX, k, replace=False).tolist())
    pend = [int(rng.integers(16, V - 16)) for _ in sel]
    depths = [int(rng.integers(1, 4)) for _ in sel]
    jobs = [(batched[sel[(j + 1) % k] if a.neg == "ring" else i], pend[j], pos[i], depths[j], None) for j, i in enumerate(sel)]
    got = be.draft_many(jobs)
    raw = (list(dr.last_raw[0]), list(dr.last_raw[1])) if hasattr(dr, "last_raw") else None
    lg_b = np.array(dr.last_lg) if tol else None
    if tol and k >= 2:                                                 # isolation: the other rows of the pass do not matter, bitwise (the session is fed its own ring: a misrouted ring shows here)
        p = int(rng.integers(0, k))
        others = [(batched[(sel[q] + 1) % S_MAX], int(rng.integers(16, V - 16)), pos[(sel[q] + 1) % S_MAX], depths[q], None) if q != p else (batched[sel[p]], pend[p], pos[sel[p]], depths[p], None) for q in range(k)]
        got2 = be.draft_many(others)
        rr2 = dr.last_raw
        rep["iso_checks"] += 1
        iso = got2[p] == got[p] and rr2[1][p] == raw[1][p] and rr2[0][p] == raw[0][p] and bool(np.array_equal(np.array(dr.last_lg)[p], lg_b[p]))
        rep["iso_equal"] += iso
        ok &= iso
        if not iso:
            print(f"round {r}: block of row {p} depends on the other rows of the pass", flush=True)
    for j, i in enumerate(sel):
        ref = be.draft_many([(solo[i], pend[j], pos[i], depths[j], None)])[0]
        rep["checks"] += 1
        rr = dr.last_raw
        if tol:
            ce, pr = block_cmp(raw[0][j], raw[1][j], lg_b[j], rr[0][0], rr[1][0], np.array(dr.last_lg)[0])
            eq = ce                                                    # the proposals themselves are judged step by step in block_cmp
            if pr:
                print(f"round {r} session {i}: " + "; ".join(pr), flush=True)
        else:
            eq = ref == got[j]
            ce = raw is not None and rr[1][0] == raw[1][j] and rr[0][0] == raw[0][j]
        rep["draft_equal"] += eq
        rep["conf_equal"] += bool(ce)
        ok &= eq and bool(ce)
        if not a.neg:                                              # stock drafter on the same ring (information)
            dr.ctx = solo[i]
            st = dr.draft(pend[j], pos[i], 3)
            rep["stock_total"] += 1
            rep["stock_agree"] += st[:len(ref)] == ref[:len(st)] and len(st) == len(ref)
    # feed: taps of n_i rows for each selected session; batched in one call, solo one call each
    ns = [int(rng.integers(1, 5)) for _ in sel]
    taps = [[mx.array(rng.standard_normal((n, H)).astype(np.float16)) for _ in range(ntaps)] for n in ns]
    dr.append_many([(batched[i], taps[j], pos[i]) for j, i in enumerate(sel)])
    for j, i in enumerate(sel):
        dr.append_many([(solo[i], [(t * (1 + a.perturb * mx.array(rng.standard_normal(t.shape).astype(np.float16)))).astype(mx.float16) for t in taps[j]] if a.perturb else taps[j], pos[i])])
        pos[i] += ns[j]
    mx.eval(*[x for d in batched + solo for x in be._ring(d)])
    for i in range(S_MAX):
        same = all(ring_close(x, y) if tol else ((x is None and y is None) or (x is not None and y is not None and bool(mx.array_equal(x, y).item()))) for x, y in zip(batched[i].rows, solo[i].rows))
        rep["ring_equal"] += same
        ok &= same
    rep["rounds"] += 1
rep["ring_checks"] = rep["rounds"] * S_MAX
if tol:
    agree = 1 - stat["flips"] / max(stat["rows"], 1)
    rep.update(rows=stat["rows"], flips=stat["flips"], tie_flips=stat["tie_flips"], conf_abs=stat["conf"], ring_rel=stat["ring"])
    ok = ok and (agree >= a.min_agree)
    print(f"tolerance: proposal logits rel max {stat['logit']:.3g}, confidence logit rel max {stat['conf']:.3g} (not bounded), ring rel max {stat['ring']:.3g} (<= {T.REL_STATE}), isolation {rep['iso_equal']}/{rep['iso_checks']} bitwise, "
          f"proposal agreement {agree:.4f} over {stat['rows']} steps ({stat['flips']} flips, {stat['tie_flips']} at a near tie; min {a.min_agree})", flush=True)
print(json.dumps(rep) if a.json else f"batched == per-session: drafts {rep['draft_equal']}/{rep['checks']}, confidence+tokens {rep['conf_equal']}/{rep['checks']}, "
      f"rings {rep['ring_equal']}/{rep['ring_checks']}; stock drafter agrees on {rep['stock_agree']}/{rep['stock_total']}")
print("EXACT" if ok else "DIFFERENT", flush=True)
sys.exit(0 if ok else 1)
