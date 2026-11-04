# SPDX-License-Identifier: Apache-2.0
"""Vectorized prefill pooling for the DS4.1 compressor (mlx-vlm `deepseek_v41` Compressor).

The stock `Compressor.__call__` pools a prompt in one vectorized step only when it starts at position 0. Every later prefill chunk (`start_pos != 0`)
runs a Python loop over its tokens: per token two slot rewrites (concatenates of the whole [batch, ratio, head_dim] state) and, at each group end, one
softmax pool, about eight tiny GPU launches. At 4096 tokens a ratio-2 compressor spends ~200 ms on an M5 in that loop (a 2-layer slice, tools/ced_bench.py
style measurement) against ~20 ms for the first chunk; the loop is launch bound on the CPU, so it does not shrink on a bigger GPU. A DS4.1 prefill has
three or four ratio-2 compressors (the KV source layers of the encoder half), i.e. ~0.6 s of every 4096-token chunk that a 2000 tok/s prefill cannot afford.

This module replaces the loop for chunks of >= MIN_ROWS tokens with the same arithmetic on all groups at once:
  * the open group (start_pos % ratio != 0) is the old state slots [0, a) plus the first ratio - a tokens;
  * the other complete groups are taken straight from the projections, `ratio` tokens each;
  * every group goes through the stock ops in the stock form ([groups, ratio, head_dim], softmax over axis 1, weighted sum over axis 1), so the
    pooled values are bit for bit the loop's (tests/test_compressor_pf.py compares them, and the final state, with the stock loop);
  * the final state is the last `ratio` tokens in their slots (the loop overwrites every slot of the state when the chunk has >= ratio tokens).
Chunks shorter than MIN_ROWS (decode, speculative verify, replay) and ratio 1, start_pos 0 and batched states of another batch size take the stock code.
Flag ttft_floor lowers that floor to `ratio` rows, so the short prompt chunk of a tool round (2..63 rows) pools too; below `ratio` rows nothing changes.
"""
from __future__ import annotations

import os

import mlx.core as mx

MIN_ROWS = 64
FLAG = "_kyojin_pf_pool"
# flag ttft_floor: a short prompt chunk pools too, from `ratio` rows (a chunk that fills the per-token state: the last `ratio` tokens are written into every
# slot). A tool round appends a few dozen tokens to a prompt, so the chunk that pays the fixed cost of a round is 2..63 rows, and that is exactly the band
# MIN_ROWS sends to the stock loop: 8 launches per token per ratio-2 layer against one small graph. Below `ratio` rows the loop stays (the pooled form
# cannot write a whole state from a chunk that closes no group).
TTFT_FLOOR = os.environ.get("KYOJIN_TTFT_FLOOR", "0") != "0"
# Path check (the served-side one, not a code read): `pool` counts the calls that took the vectorized pooling, `stock` the ones that fell back
# (ratio 1, start_pos 0, a chunk under MIN_ROWS, a foreign batch). The serve report carries this dict live, so a served prefill shows whether
# the flag is armed on the path it actually runs (a `pool` of 0 with a long prompt = the stock per-token loop is still in the way).
HITS = {"pool": 0, "stock": 0}


def _pool_call(comp, orig, x, start_pos, cache):
    ratio, n, batch = comp.compress_ratio, x.shape[1], x.shape[0]
    i = comp.layer_idx
    st = cache.kv_state[i]
    floor = MIN_ROWS if not TTFT_FLOOR else max(ratio, 2)
    if ratio == 1 or start_pos == 0 or n < floor or (st is not None and st.shape[0] != batch):
        HITS["stock"] += 1
        return orig(comp, x, start_pos, cache)
    HITS["pool"] += 1
    dtype = x.dtype
    xf = x.astype(mx.float32)
    kv, score = comp.wkv(xf), comp.wgate(xf)
    comp._grow_state(batch, cache)
    state_kv, state_score = cache.kv_state[i], cache.score_state[i]
    d = comp.head_dim
    a = start_pos % ratio
    h = (ratio - a) % ratio                                   # tokens that close the open group
    g = (n - h) // ratio                                      # complete groups after it
    groups_kv, groups_score = [], []
    if a:
        groups_kv.append(mx.concatenate([state_kv[:batch, :a], kv[:, :h]], axis=1)[:, None])
        groups_score.append(mx.concatenate([state_score[:batch, :a], score[:, :h]], axis=1)[:, None])
    groups_kv.append(kv[:, h:h + g * ratio].reshape(batch, g, ratio, d))
    groups_score.append(score[:, h:h + g * ratio].reshape(batch, g, ratio, d))
    gk = groups_kv[0] if len(groups_kv) == 1 else mx.concatenate(groups_kv, axis=1)
    gs = groups_score[0] if len(groups_score) == 1 else mx.concatenate(groups_score, axis=1)
    total = gk.shape[1]
    gk, gs = gk.reshape(batch * total, ratio, d), gs.reshape(batch * total, ratio, d)
    pooled = (gk * mx.softmax(gs, axis=1)).sum(axis=1).reshape(batch, total, d)
    e = (start_pos + n) % ratio                               # next slot to write: the last `ratio` tokens sit at slots e, e + 1, ... (mod ratio)
    last_kv, last_score = kv[:, n - ratio:], score[:, n - ratio:]
    if e:
        last_kv = mx.concatenate([last_kv[:, ratio - e:], last_kv[:, :ratio - e]], axis=1)
        last_score = mx.concatenate([last_score[:, ratio - e:], last_score[:, :ratio - e]], axis=1)
    cache.kv_state[i], cache.score_state[i] = last_kv, last_score
    return comp.norm(pooled.astype(dtype))


def report() -> dict:
    """Copy of the path counters (pool = vectorized pooling calls, stock = fallbacks), for the serve report / a gate print."""
    return dict(HITS)


def install(model) -> int:
    """Use the vectorized pooling in every ratio > 1 compressor of `model` (patched mlx-vlm model or its language model). Idempotent. Returns how many
    compressors it armed."""
    from .ced import _override
    lm = getattr(model, "language_model", model)
    n = 0
    for _, mod in lm.named_modules():
        if type(mod).__name__ == "Compressor" and getattr(mod, "compress_ratio", 1) > 1 and not getattr(type(mod), FLAG, False):
            _override(mod, "__call__", _pool_call)
            type(mod)._kyojin_pf_pool = True                  # on the per-instance subclass _override made: a second install skips it
            n += 1
    return n
