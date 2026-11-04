# SPDX-License-Identifier: Apache-2.0
"""Compiled decode regions for the Qwen3.5 hybrid stack (gated-delta layers + full-attention layers), one token, batch 1.

The python graph build of a decode token (1.2k nodes on the 0.8B, ~2.7k on the 4B) is paid once per region (mx.compile trace) instead of once per token.
The attention core (rope at the cache offset, KV append, sdpa) stays eager: its shapes and python offset change every token and the KV append must stay
in place. Everything between two attention cores is ONE compiled region: finish of the previous attention layer (o_proj, residual, MLP), the gated-delta
layers (their conv / recurrent states go in and out as arrays), the pre-attention part of the next attention layer (norm, q/k/v projections, q/k norms).
The region bodies go through decfuse (fused same-input projections and residual add + RMSNorm: bit-equal; fused gated-delta front end: within
rounding, KL vs stock 3-5e-4 on 4B / 9B); KYOJIN_DECFUSE=0 keeps the region layout with the stock op bodies (792 dispatches on 4B vs 864 before the rewrite), logits bit for bit with the stock forward (tools/host_gap.py --check).

    step_regions.enable(model)      # model = mlx-lm Qwen3.5 Model (or its language model); KYOJIN_STEP_REGIONS=0 makes it a no-op
Falls back to the stock forward for anything but one token of one sequence with plain KVCache / ArraysCache entries."""
from __future__ import annotations

import os

import mlx.core as mx
from mlx_lm.models import base as _base
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache

from . import decfuse as DF
from .kvwin import WinKVCache
from . import replay as RP

ENABLED = os.environ.get("KYOJIN_STEP_REGIONS", "1") != "0"
_W = os.environ.get("KYOJIN_STEP_REGIONS_WIDE")
WIDE = None if _W is None else _W == "1"                         # None = tuning.chip()["step_regions_wide"] (chip_defaults.json, default 0). 1: also the served solo step and WinKVCache / TurboQuant caches. Default 0 = only bare KVCache (the served solo
# step (multiuser RowsKV of one cache) and the window / TurboQuant caches (WinKVCache, read through cache.attend) fall back to the uncompiled forward). On: bit-equal only within the decfuse rounding, and a solo session no longer matches the same session in a 2-user batch
STATS = {"regions_traced": 0, "fast": 0, "fallback": 0}


def _gdn(layer, x, pend, st):
    """One gated-delta decoder layer; pend = the previous sub-layer output not yet added to the stream x (decfuse: add + norm in one launch)."""
    x, h = DF.norm_of(x, pend, layer.input_layernorm)
    r, s = DF.gdn(layer.linear_attn, h, st)
    x, h = DF.add_rms(x, r, layer.post_attention_layernorm)
    return x, DF.mlp(layer.mlp, h), s


def _pre(layer, x, pend):
    a = layer.self_attn
    B, L, _ = x.shape
    x, h = DF.norm_of(x, pend, layer.input_layernorm)
    qo, k, v = DF.qkv(a, h)
    q, gate = mx.split(qo.reshape(B, L, a.num_attention_heads, -1), 2, axis=-1)
    gate = gate.reshape(B, L, -1)
    q = a.q_norm(q).transpose(0, 2, 1, 3)
    k = a.k_norm(k.reshape(B, L, a.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
    v = v.reshape(B, L, a.num_key_value_heads, -1).transpose(0, 2, 1, 3)
    return x, q, k, v, gate


def _post(layer, x, out, gate):
    a = layer.self_attn
    x, h = DF.add_rms(x, a.o_proj(out * mx.sigmoid(gate)), layer.post_attention_layernorm)
    return x, DF.mlp(layer.mlp, h)


class _Plan:
    def __init__(self, tm):
        self.tm = tm
        L = list(tm.pipeline_layers)
        self.layers = L
        self.attn = [i for i, l in enumerate(L) if not l.is_linear]
        self.gdn = [i for i, l in enumerate(L) if l.is_linear]
        a = self.attn
        bounds = [-1] + a + [len(L)]
        # region r covers the gated-delta layers strictly between bounds[r] and bounds[r+1]
        self.regions = []
        for r in range(len(bounds) - 1):
            lo, hi = bounds[r], bounds[r + 1]
            self.regions.append(self._make(r, [i for i in range(lo + 1, hi)], bool(r == 0), bool(r == len(bounds) - 2),
                                           a[r - 1] if r > 0 else None, a[r] if r < len(a) else None))

    def _make(self, r, gl, first, last, prev_attn, next_attn):
        tm, L = self.tm, self.layers
        # args: first: (ids,) ; else: (x, out, gate) ; then states flat (conv, ssm) per gated layer
        def fn(*args):
            n = 1 if first else 3
            if first:
                x, pend = tm.embed_tokens(args[0]), None
            else:
                x, pend = _post(L[prev_attn], args[0], args[1], args[2])
            st = args[n:]
            new = []
            for j, i in enumerate(gl):
                x, pend, s = _gdn(L[i], x, pend, (st[2 * j], st[2 * j + 1]))
                new += list(s)
            if last:
                return (DF.norm_of(x, pend, tm.norm)[1], *new)
            x, q, k, v, gate = _pre(L[next_attn], x, pend)
            return (x, q, k, v, gate, *new)
        return gl, RP.region(mx.compile(fn), f"qwen_step{r}")     # KYOJIN_REPLAY=1: recorded once, replayed as an indirect command buffer


def _chip_wide() -> bool:
    global WIDE
    from .tuning import chip
    WIDE = bool(chip().get("step_regions_wide", 0))
    return WIDE


def _solo(c):
    """The one cache behind an attention-layer cache of the fast path, else None: a KVCache, or with WIDE a WinKVCache (TurboQuant included) and
    the single-user multiuser RowsKV around either (served solo decode)."""
    if not (WIDE if WIDE is not None else _chip_wide()):
        return c if isinstance(c, KVCache) else None
    rows = getattr(c, "caches", None)
    if rows is not None and type(c).__name__ == "RowsKV":
        if len(rows) != 1:
            return None
        c = rows[0]
    return c if isinstance(c, (KVCache, WinKVCache)) else None


def _forward(self, inputs, cache=None, input_embeddings=None):
    plan = self._kr_plan
    att = [_solo(cache[i]) for i in plan.attn] if plan is not None and cache is not None and len(cache) == len(plan.layers) else None
    ok = (att is not None and input_embeddings is None and inputs.shape == (1, 1)
          and all(c is not None for c in att)
          and all(type(cache[i]) is ArraysCache and cache[i][0] is not None and cache[i][1] is not None for i in plan.gdn))
    if not ok:
        STATS["fallback"] += 1
        return self._kr_orig(inputs, cache, input_embeddings=input_embeddings)
    STATS["fast"] += 1
    L = plan.layers
    out = None
    x = gate = None
    for r, (gl, fn) in enumerate(plan.regions):
        st = [s for i in gl for s in (cache[i][0], cache[i][1])]
        args = (inputs, *st) if r == 0 else (x, out, gate, *st)
        res = fn(*args)
        ns = 1 if plan.regions[r] is plan.regions[-1] else 5
        head, new = res[:ns], res[ns:]
        for j, i in enumerate(gl):
            cache[i][0], cache[i][1] = new[2 * j], new[2 * j + 1]
            cache[i].advance(1)
        if ns == 1:
            return head[0]
        x, q, k, v, gate = head
        ai = plan.attn[r]
        c, a = att[r], L[ai].self_attn
        q = a.rope(q, offset=c.offset)
        k = a.rope(k, offset=c.offset)
        k, v = c.update_and_fetch(k, v)
        mask = create_attention_mask(x, c)
        o = _base.scaled_dot_product_attention(q, k, v, cache=c, scale=a.scale, mask=mask)      # looked up per call: sdpa_dec.install patches it
        out = o.transpose(0, 2, 1, 3).reshape(1, 1, -1)


def enable(model):
    """Install the segmented decode forward on the text model of `model`. Returns True when installed."""
    if not ENABLED:
        return False
    lm = getattr(model, "language_model", model)
    tm = getattr(lm, "model", None)
    if tm is None or not hasattr(tm, "pipeline_layers") or not hasattr(tm.layers[0], "is_linear") or getattr(tm, "_kr_plan", None) is not None:
        return False
    if getattr(tm, "pipeline_size", 1) != 1:
        return False
    cls = type(tm)
    tm._kr_orig = lambda *a, **k: cls.__call__(tm, *a, **k)
    DF.prepare(tm)
    tm._kr_plan = _Plan(tm)
    tm.__class__ = type("Kr" + cls.__name__, (cls,), {"__call__": _forward})
    return True
