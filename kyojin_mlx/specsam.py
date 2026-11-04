# SPDX-License-Identifier: Apache-2.0
"""Speculative rejection sampling for a sampled session (KYOJIN_SPEC_RS, served path).

A greedy draft head proposes the target's argmax, so "accept when the target sample equals the draft" can only accept with p(top1): on flat prose
that is 60-65 %. This module draws the draft from a proposal distribution q (the draft head's own block logits, tempered like the target and
top-k filtered) and accepts it with min(1, p(x) / q(x)); a rejected row emits a draw from the normalised residual max(0, p - q). The emitted
sequence then follows the target p exactly (temperature / top-k / top-p / min-p applied), while the acceptance rate is 1 - TV(p, q) instead of
p(argmax): on a peaked target the two agree, on a flat one q can accept far more drafts.

q does not have to be the drafter's autoregressive distribution: the block logits of every position are a fixed function of the prefix the drafter
was fed, and each position is drawn independently from its own q_i. The standard accept / residual argument only needs x_i ~ q_i and q_i known
(it conditions on the emitted prefix, not on how q_i was formed), so a chain that keeps its greedy-conditioned logits is a valid proposal.

p is the engine's sampling distribution (serve.sampling: the same filters the plain path applies), q is the same filters on the draft logits with
the same temperature. Greedy sessions (temperature 0) never take this path: their drafts stay the argmax and the engine compares samples, as before,
so their tokens are unchanged. `decide` is what tests/test_specsam.py checks against the exact distributions on synthetic logits."""
from __future__ import annotations

import mlx.core as mx

NEG = -float("inf")
TINY = 1e-30


def _norm(lp: mx.array) -> mx.array:
    return lp - mx.logsumexp(lp, axis=-1, keepdims=True)


def logq(logits: mx.array, temp: float, top_k: int = 0) -> mx.array:
    """Draft logits [..., V] -> log q: temperature, top-k truncation over the last axis, normalised (rows outside the kept set are -inf)."""
    lp = _norm(logits.astype(mx.float32) * (1.0 / max(float(temp), 1e-6)))
    if top_k and 0 < top_k < lp.shape[-1]:
        idx = mx.argpartition(-lp, kth=top_k - 1, axis=-1)[..., top_k:]
        lp = _norm(mx.put_along_axis(lp, idx, mx.array(NEG, lp.dtype), axis=-1))
    return lp


def logp(row: mx.array, sampling) -> mx.array:
    """One target logits row [V] -> log p, the plain sampling distribution (the filters of kyojin_mlx.serve.sampling, then the temperature)."""
    from .serve.sampling import _filtered
    return _norm(_filtered(row, sampling) * (1.0 / max(float(sampling.temperature), 1e-6)))


def draw(lq: mx.array, keys: list) -> mx.array:
    """[n, V] log q -> one token per row, drawn with the row's own key (a list of n keys)."""
    return mx.stack([mx.random.categorical(lq[j], key=keys[j]).astype(mx.int32) for j in range(lq.shape[0])])


def decide(plogits: mx.array, lq: mx.array, drafts: mx.array, sampling, keys: list):
    """Rows of one session: plogits [n, V] the target's logits (row n - 1 is the bonus row after the last draft), lq [n - 1, V] the log q the
    drafts were drawn from (`logq` of the draft head's block logits: the very array `draw` took them from), drafts [n - 1], keys n fresh keys (one
    per row; each is split into the uniform of the acceptance test and the residual's draw).

    Returns (out [n] int32, acc [n - 1] bool): acc[j] is the acceptance of draft j; out[j] is the token row j emits (the draft when accepted, a
    draw from the residual max(0, p - q) when rejected; out[n - 1] is a plain draw from p, the bonus token). The engine keeps the leading run of
    acc and emits drafts[:a] + [out[a]]."""
    n = int(plogits.shape[0])
    lp = mx.stack([logp(plogits[j], sampling) for j in range(n)])                 # [n, V]
    x = drafts.astype(mx.int32)
    xl = x[:, None]
    ratio = mx.exp(mx.minimum(mx.take_along_axis(lp[:n - 1], xl, axis=-1) - mx.take_along_axis(lq, xl, axis=-1), 0.0))[:, 0]
    resid = mx.log(mx.maximum(mx.exp(lp[:n - 1]) - mx.exp(lq), 0.0) + TINY)       # max(0, p - q), renormalised by the draw
    acc, rej = [], []
    for j in range(n - 1):
        ku, kr = mx.random.split(keys[j], 2)
        acc.append(mx.random.uniform(shape=(), key=ku) < ratio[j])                # min(1, p(x) / q(x))
        rej.append(mx.random.categorical(resid[j], key=kr).astype(mx.int32))
    out = mx.concatenate([mx.where(mx.stack(acc), x, mx.stack(rej)),
                          mx.random.categorical(lp[n - 1], key=keys[n - 1]).astype(mx.int32)[None]])
    return out, mx.stack(acc)


def break_even(slope_ms: float, plain_ms: float) -> float:
    """Smallest acceptance p that pays for one more verified row: the row costs `slope_ms`, a token is worth `plain_ms` of a plain step."""
    if not plain_ms or slope_ms <= 0:
        return 0.0
    return min(0.95, max(0.02, float(slope_ms) / float(plain_ms)))


def gate_confidence(cal, p_star: float, steps: int = 100) -> float | None:
    """Smallest raw confidence whose calibrated acceptance reaches `p_star` (None before the calibration saw a round): the confidence gate's
    fixed 50 % assumes the head is calibrated; this asks the engine's measured P(accept | conf) instead."""
    if cal is None or not cal.ready or p_star <= 0.0:
        return None
    for i in range(steps + 1):
        c = i / steps
        if cal.prob(c) >= p_star:
            return c
    return 1.0


