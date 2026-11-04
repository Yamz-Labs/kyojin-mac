#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Speculation-yield levers on synthetic data of the real geometry (run on the Mac; nothing here was timed by the author, there is no GPU in the
sandbox it was written in).

  lookup   host cost + yield of the prompt-lookup index at --tokens committed tokens: `match_best` (today) vs `match_vote` (spec_lookup_vote, 16
           candidates) vs the same with the short-suffix floor of 4 (spec_lookup_adapt's reach). ms per call, and on a synthetic stream made of
           repeated spans with edits (the last 10 % of positions are probed against an index of the first 90 %): attempts per 100 positions, accepted drafts per attempt, tokens per attempt (accepted + 1). The stream is
           synthetic: the yield numbers compare the arms with each other, they are not served acceptance rates. Pure numpy, runs anywhere.
  tail     the draft head's bigram-biased chain (dspark Drafter._rows_tail, the stub of tests/test_draft_levers.py) at the pack's vocab / rank:
           full-vocab chain (today) vs KYOJIN_DRAFT_CAND=K, ms per call at 1 and 2 sessions x block 5, proposals agreement with the full chain.
           Needs Metal for a meaningful time (runs on CPU too, slowly).
  draft    the pieces of a draft pass that need no pack (trunk head over S*L rows, block SDPA, confidence head) at the real shapes, so the 7 ms a
           round can be split: stages (served model) = spec_t_draft per round minus these.
Memory: the tail arm holds a [vocab, rank] fp16 head (66 MB) and a [vocab, hidden-free] logits block; the lookup arm 5 arrays of --tokens ints. Both are
counted first and the run stops above --cap-gib (3.5). Time budget --budget-s (60): arms stop early and say so.

  python tools/spec_yield_bench.py [--tokens 1048576] [--attempts 2000] [--vocab 129280] [--rank 256] [--k 64] [--reps 20] [--arms lookup,tail,draft]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kyojin_mlx import pld as PLD  # noqa: E402

T0 = time.perf_counter()


def left(budget):
    return budget - (time.perf_counter() - T0)


def synth_stream(n, vocab, seed=0):
    """Tokens made of spans copied from earlier positions with random edits, plus fresh noise: the shape of code / recitation / copy-heavy text."""
    rng = np.random.default_rng(seed)
    out = np.empty(n, dtype=np.int32)
    out[:4096] = rng.integers(0, vocab, 4096)
    i = 4096
    while i < n:
        if rng.random() < 0.7:                                    # copy a span, with a few substitutions
            ln = int(rng.integers(8, 96))
            src = int(rng.integers(0, i - ln))
            ln = min(ln, n - i)
            seg = out[src:src + ln].copy()
            for j in rng.integers(0, ln, int(rng.integers(0, 3))):
                seg[j] = rng.integers(0, vocab)
            out[i:i + ln] = seg
            i += ln
        else:
            ln = min(int(rng.integers(4, 40)), n - i)
            out[i:i + ln] = rng.integers(0, vocab, ln)
            i += ln
    return out


def arm_lookup(a):
    n = int(a.tokens)
    bytes_ = n * (4 + 4 + 4) + 2 * n * 4
    assert bytes_ < a.cap_gib * 2**30, "lookup arm over the memory cap"
    stream = synth_stream(n, a.vocab)
    split = int(n * 0.9)
    ix = PLD.Index(n)
    t = time.perf_counter()
    ix.extend(stream[:split])
    ix.toks[split:] = stream[split:]                          # the tokens after the indexed range are readable (draft / truth), not hashed: occurrences in [split, p) are never found, equally for every arm
    print(f"lookup  index build {split} tokens: {(time.perf_counter() - t) * 1e3:8.1f} ms")
    rng = np.random.default_rng(1)
    pos = rng.integers(split, n - 40, int(a.attempts))
    arms = [("best (today, floor 8)", lambda t1, md: ix.match_best(t1, md), PLD.MIN_SUFFIX),
            ("vote16 (floor 8)", lambda t1, md: ix.match_vote(t1, md, 16), PLD.MIN_SUFFIX),
            ("best, floor 4 (adapt reach)", lambda t1, md: ix.match_best(t1, md), PLD.NGRAM),
            ("vote16, floor 4", lambda t1, md: ix.match_vote(t1, md, 16), PLD.NGRAM)]
    print(f"        {'arm':30s} {'us/call':>9s} {'attempts/100':>13s} {'acc/attempt':>12s} {'tok/attempt':>12s}")
    for name, fn, floor in arms:
        if left(a.budget_s) < 5:
            print("        budget reached, arm skipped:", name)
            continue
        n_att = n_acc = 0
        t = time.perf_counter()
        for p in pos:
            ix.n = int(p)                                         # the index "as of" position p: the tail is stream[p-3:p], the round's first token stream[p]
            m = fn(int(stream[p]), 7)
            if m is None or m.suffix < floor:
                continue
            n_att += 1
            truth = stream[p + 1:p + 1 + len(m.draft)]
            k = 0
            while k < len(m.draft) and int(truth[k]) == m.draft[k]:
                k += 1
            n_acc += k
        dt = (time.perf_counter() - t) * 1e6 / len(pos)
        print(f"        {name:30s} {dt:9.1f} {100.0 * n_att / len(pos):13.1f} {n_acc / max(1, n_att):12.2f} {1 + n_acc / max(1, n_att):12.2f}")
    ix.n = split


def arm_tail(a):
    import mlx.core as mx
    import mlx.nn as nn
    from types import SimpleNamespace
    from kyojin_mlx import dspark as DS
    V, R, L = int(a.vocab), int(a.rank), 5
    if not mx.metal.is_available():                                  # the row-invariant product is a Metal kernel: a plain matmul stands in (CPU smoke run only)
        from kyojin_mlx import rowinv
        rowinv.rowmm = lambda x, w: mx.matmul(x, w.T)
    assert V * R * 2 + 2 * L * V * 4 * 3 < a.cap_gib * 2**30, "tail arm over the memory cap"
    mx.random.seed(0)
    lm = SimpleNamespace(markov_embed=nn.Embedding(V, R), markov_head=nn.Linear(R, V, bias=False), confidence=nn.Linear(5120 + R, 1, bias=False))
    d = SimpleNamespace(lm=lm, block=L, _conf_w=None)
    d._tail_conf = lambda *args: DS.Drafter._tail_conf(d, *args)
    mx.eval(lm.markov_embed.parameters(), lm.markov_head.parameters(), lm.confidence.parameters())
    print(f"tail    vocab {V} rank {R} block {L}  device {mx.default_device()}")
    for S in (1, 2):
        x = mx.random.normal((S * L, 1, 5120)).astype(mx.float16)
        logits = mx.random.normal((S, L, V)) * 3.0
        pend = mx.array(list(range(3, 3 + S)), dtype=mx.int32)
        mx.eval(x, logits, pend)
        res = {}
        for name, cand in (("full vocab (today)", 0), (f"draft_cand K={a.k}", int(a.k))):
            if left(a.budget_s) < 3:
                print("        budget reached, arm skipped:", name)
                continue
            DS.CAND = cand
            for _ in range(3):
                out = DS.Drafter._rows_tail(d, x, logits, pend, S)
                mx.eval(out[0], out[1])
            ts = []
            for _ in range(int(a.reps)):
                t = time.perf_counter()
                out = DS.Drafter._rows_tail(d, x, logits, pend, S)
                mx.eval(out[0], out[1])
                ts.append((time.perf_counter() - t) * 1e3)
            res[name] = out[0].tolist()
            print(f"        S={S}  {name:22s} median {sorted(ts)[len(ts) // 2]:7.3f} ms   min {min(ts):7.3f} ms")
        if len(res) == 2:
            a0, a1 = res.values()
            same = sum(int(u == v) for ru, rv in zip(a0, a1) for u, v in zip(ru, rv))
            print(f"        S={S}  proposals equal to the full chain: {same}/{S * L}")
    DS.CAND = 0


def arm_draft(a):
    """The pieces of a draft pass that need no pack: trunk head on S*L rows, the SDPA of the block over its ring, the confidence head, the markov chain.
    The three MoE stages need the served model: read spec_t_draft (/v1/stats) per round for the whole pass and subtract these."""
    import mlx.core as mx
    V, H, L, W, D, NH = int(a.vocab), 5120, 5, 128, 512, 64
    assert V * H * 2 + 4 * 2**20 < a.cap_gib * 2**30, "draft arm over the memory cap"
    mx.random.seed(0)
    head = (mx.random.normal((V, H)) * 0.02).astype(mx.float16)
    mx.eval(head)

    def med(fn, n):
        for _ in range(3):
            mx.eval(fn())
        ts = []
        for _ in range(int(a.reps)):
            t = time.perf_counter()
            mx.eval(fn())
            ts.append((time.perf_counter() - t) * 1e3)
        return sorted(ts)[len(ts) // 2]

    for S in (1, 2):
        R = S * L
        x = mx.random.normal((R, H)).astype(mx.float16)
        q = mx.random.normal((1, NH, L, D)).astype(mx.float16)
        kv = mx.random.normal((1, 1, W + L, D)).astype(mx.float16)
        sink = mx.zeros((NH,), dtype=mx.float16)
        cw = mx.random.normal((1, H + int(a.rank))).astype(mx.float32)
        cin = mx.random.normal((R, H + int(a.rank))).astype(mx.float32)
        mx.eval(x, q, kv, sink, cw, cin)
        print(f"draft   S={S} rows={R}")
        print(f"        trunk head  x[{R},{H}] @ head[{V},{H}]^T -> f32 logits: {med(lambda: (x @ head.T).astype(mx.float32), 0):7.3f} ms")
        print(f"        block SDPA  {NH} heads x {L} q over {W + L} kv, {S} session(s) in a loop: "
              f"{med(lambda: [mx.fast.scaled_dot_product_attention(q, kv, kv, scale=D ** -0.5, mask=None, sinks=sink) for _ in range(S)], 0):7.3f} ms")
        print(f"        confidence  [{R},{H + int(a.rank)}] @ w: {med(lambda: mx.sigmoid(cin @ cw.T), 0):7.3f} ms")
    print("        (markov chain: arm `tail`; stages: not here, they need the served model)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=1 << 20)
    ap.add_argument("--attempts", type=int, default=2000)
    ap.add_argument("--vocab", type=int, default=129280)
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--arms", default="lookup,tail,draft")
    ap.add_argument("--cap-gib", type=float, default=3.5)
    ap.add_argument("--budget-s", type=float, default=60.0)
    a = ap.parse_args()
    for arm in a.arms.split(","):
        {"lookup": arm_lookup, "tail": arm_tail, "draft": arm_draft}[arm.strip()](a)
    print(f"total {time.perf_counter() - T0:.1f} s")


if __name__ == "__main__":
    main()
