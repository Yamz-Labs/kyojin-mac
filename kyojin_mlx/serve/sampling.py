# SPDX-License-Identifier: Apache-2.0
"""Per-session sampling on a batch of logits. Greedy rows are one argmax for the whole batch; sampled rows get their own
temperature / top-k / top-p / min-p and their own random key (a session's stream does not depend on the other sessions)."""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class Sampling:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    seed: int | None = None

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0


def _filtered(logits: mx.array, p: Sampling) -> mx.array:
    """logits [V] fp32 -> log-probabilities with top-k / top-p / min-p applied (-inf outside the kept set)."""
    from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p
    lp = (logits - mx.logsumexp(logits, axis=-1, keepdims=True))[None]
    if p.top_k and p.top_k > 0:
        lp = apply_top_k(lp, int(p.top_k))
    if p.top_p and 0.0 < p.top_p < 1.0:
        lp = apply_top_p(lp, float(p.top_p))
    if p.min_p and p.min_p > 0.0:
        lp = apply_min_p(lp, float(p.min_p), 1)
    return lp[0]


def sample_rows(logits: mx.array, params: list[Sampling], keys: list) -> mx.array:
    """logits [N, V] fp32, one Sampling per row, keys[i] = a fresh random key for row i (or None for greedy rows).
    Returns [N] int32 (lazy)."""
    greedy = mx.argmax(logits, axis=-1).astype(mx.int32)
    rows = [i for i, p in enumerate(params) if not p.greedy]
    if not rows:
        return greedy
    out = [greedy[i] for i in range(len(params))]
    for i in rows:
        p = params[i]
        lp = _filtered(logits[i], p)
        out[i] = mx.random.categorical(lp * (1.0 / p.temperature), key=keys[i]).astype(mx.int32)
    return mx.stack(out)
