# SPDX-License-Identifier: Apache-2.0
"""Compiled decode regions for a hybrid (gated-delta + attention) foreign pack, one row (KYOJIN_FOREIGN_REGIONS, default on).

The Python graph build of a one-row decode step (every layer's ~60 Python-level ops) is paid once per region (mx.compile trace) instead of once per token:
the host is the bottleneck of a small-active MoE, not the GPU. Everything between two attention cores is ONE compiled region: the attention layer's
output seam and MLP (add + norm, MLP / MoE), the gated-delta layers (their conv / recurrent states go in and out as arrays) and the input norm of the
next attention layer. The attention core (cache append, rope at the cache offset, sdpa) stays eager: its shapes and Python offset change every token.
The region bodies are the same calls as decrows._body, so the launches and the arithmetic are the eager ones.

Falls back to the eager body for more than one row, a state that is not there yet (first token after a prefill that left none), a captured pass (qspec.CAP)
or a gated-delta cache that is not a plain ArraysCache."""
from __future__ import annotations

import os

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import ArraysCache

from . import decfuse as DF
from .depchain import chain as _dep_chain      # test-only slope probe: K dependent kernels per layer (KYOJIN_TEST_DEP_CHAIN), identity when 0

ENABLED = os.environ.get("KYOJIN_FOREIGN_REGIONS", "1") not in ("0", "")
STATS = {"fast": 0, "fallback": 0, "regions_traced": 0}


class _State:
    """The two arrays of a gated-delta cache entry, as a functional stand-in inside a traced region (no cache object leaks a tracer)."""
    lengths = None

    def __init__(self, conv, ssm):
        self.s = [conv, ssm]

    def __getitem__(self, i):
        return self.s[i]

    def __setitem__(self, i, v):
        self.s[i] = v

    def advance(self, n):
        pass


class _Plan:
    def __init__(self, tm):
        self.tm = tm
        self.layers = L = list(tm.layers)
        self.attn = [i for i, l in enumerate(L) if not l.is_linear]
        from . import gdn_pf
        for l in L:
            if l.is_linear:
                gdn_pf.prep_ba(l.linear_attn)
        bounds = [-1] + self.attn + [len(L)]
        self.regions = [(list(range(bounds[r] + 1, bounds[r + 1])), self._make(r, bounds[r] + 1, bounds[r + 1])) for r in range(len(bounds) - 1)]

    def _make(self, r, lo, hi):
        tm, L = self.tm, self.layers
        first, last = r == 0, hi == len(L)
        prev_attn = None if first else L[lo - 1]
        next_attn = None if last else L[hi]

        def fn(*args):
            n = 1 if first else 2
            if first:
                x, pend = tm.embed_tokens(args[0]), None
            else:
                x, h = DF.add_rms(args[0], args[1], prev_attn.post_attention_layernorm)
                pend = DF.mlp(prev_attn.mlp, h, defer=True)
            st, new = args[n:], []
            for j, i in enumerate(range(lo, hi)):
                layer = L[i]
                s = _State(st[2 * j], st[2 * j + 1])
                x, h = DF.norm_of(x, pend, layer.input_layernorm)
                out = _dep_chain(layer.linear_attn(h, None, s))
                x, h = DF.add_rms(x, out, layer.post_attention_layernorm)
                pend = DF.mlp(layer.mlp, h, defer=True)
                new += s.s
            if last:
                return (DF.norm_of(x, pend, tm.norm)[1], *new)
            x, h = DF.norm_of(x, pend, next_attn.input_layernorm)
            return (x, h, *new)
        STATS["regions_traced"] += 1
        return mx.compile(fn)


def forward(tm, inputs, cache):
    """The one-row decode step in compiled regions, or None when this call is not eligible (the caller runs the eager body)."""
    from . import qspec
    if not ENABLED or qspec.CAP is not None or inputs.shape != (1, 1):
        return None
    plan = getattr(tm, "_fr_plan", None)
    if plan is None:
        plan = tm._fr_plan = _Plan(tm)
    L = plan.layers
    if any(type(cache[i]) is not ArraysCache or cache[i][0] is None or cache[i][1] is None or cache[i].lengths is not None
           for i, l in enumerate(L) if l.is_linear):
        STATS["fallback"] += 1
        return None
    STATS["fast"] += 1
    r_out = None
    for r, (gl, fn) in enumerate(plan.regions):
        st = [s for i in gl for s in (cache[i][0], cache[i][1])]
        res = fn(inputs, *st) if r == 0 else fn(x, r_out, *st)
        last = r == len(plan.regions) - 1
        ns = 1 if last else 2
        head, new = res[:ns], res[ns:]
        for j, i in enumerate(gl):
            cache[i][0], cache[i][1] = new[2 * j], new[2 * j + 1]
            cache[i].advance(1)
        if last:
            return head[0]
        x, h = head
        ai = plan.attn[r]
        r_out = _dep_chain(L[ai].self_attn(h, create_attention_mask(h, cache[ai]), cache[ai]))
