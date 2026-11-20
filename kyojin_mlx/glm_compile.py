# SPDX-License-Identifier: Apache-2.0
"""Compiled decode layers of GLM-5.3 (flag KYOJIN_GLM_COMPILE, default off).

The decode step builds ~70 launches per layer in Python (mHC glue, norms, projections, gated-delta step / indexer, router, experts). With mx.compile the Python graph
build is paid once per (layer, rows) trace and later tokens replay it; MLX also fuses the elementwise chains between the custom kernels.

Region 1: a LINEAR-ATTENTION layer (34 of 45) is a pure function of (x, conv state, recurrent state): its cache entry is a fixed-shape ArraysCache, there is no
position-dependent Python state. The compiled function builds a fresh vendor ArraysCache around the traced states, runs the layer body (stock, or the G3b / G2c / G3
custom-kernel versions: they are plain primitives inside the trace), and returns (x, conv', ssm'); the wrapper stores them back and advances the cache.

Region 2: a DSA layer (11 of 45) is cut around the attention core (indexer, latent-cache update, host offsets, sparse attention: growing shapes, python state):
    A  mHC collapse + input norm, qkv_a / q_b projections, the two latent norms   -> (branch input, q latent, q, new latent, mHC post / comb or fused pc)
    .  core, eager (Glm5NextAttention.__call__ after the projections: cache update, indexer, _attend)
    B  o_proj, mHC expand, FFN mHC collapse + norm, FFN (router, experts), mHC expand
Both regions are pure functions of arrays (the weights are constants of the trace). The cross-layer G3b slot (the fused expand's partial sums for the next layer's mHC)
crosses the region boundaries as an explicit input / output, so no tracer is ever left in glm_hc.SLOT.

Falls back to the stock body (counted in STATS) for: more than one token per row, a mask, a speculative / lengths cache, a state that is not an array yet (first token
after a prefill that left none), or the first call of a (layer, rows) variant (lazily built kernel parameters evaluate there; a
trace may not eval).
"""
from __future__ import annotations

import os
import types

import mlx.core as mx

from . import fired as FD

ENABLED = os.environ.get("KYOJIN_GLM_COMPILE", "0") not in ("0", "")
DSA = os.environ.get("KYOJIN_GLM_COMPILE_DSA", "1") not in ("0", "")        # the DSA layers' regions A / B (the linear layers are compiled whenever ENABLED)
STATS = {"fast": 0, "fallback": 0, "traced": 0, "dsa_fast": 0, "dsa_fallback": 0, "dsa_traced": 0}


def _plain(cache) -> bool:
    c = getattr(cache, "cache", None)
    return (c is not None and len(c) >= 2 and isinstance(c[0], mx.array) and isinstance(c[1], mx.array)
            and not getattr(cache, "history_capacity", 0) and getattr(cache, "lengths", None) is None)


# ------------------------------------------------------------------------------------------------------------------------------ G3b slot
def _hc():
    from . import glm_hc
    return glm_hc


def _g3b_on(layer, x) -> bool:
    """The layer runs the G3b fused mHC glue for this x (same test as glm_hc._layer_call)."""
    if not getattr(layer.__class__, "_g3b_hc", False):
        return False
    H = _hc()
    return H.ok_shape(x, layer.attn_hc) and H.ok_shape(x, layer.ffn_hc)


def _slot_take(layer, x):
    """The partial sums the previous layer's fused expand left for this x (consumed), or None."""
    s = _hc().SLOT
    if s.x is x and s.fn is layer.attn_hc.fn:
        part = s.part
        s.x = s.part = s.fn = None
        return part
    return None


def _slot_set(x, part, fn):
    s = _hc().SLOT
    s.x, s.part, s.fn = x, part, fn


def _slot_put(layer, y, part):
    nxt = layer.__dict__.get("_g3b_next")
    if part is not None and nxt is not None:
        _slot_set(y, part, nxt.fn)


# ------------------------------------------------------------------------------------------------------------------------------ linear layers
def _linear_run(cls):
    from mlx_vlm.models.cache import ArraysCache

    def run(self, x, cache, prev_topk_indices):
        st = self.__dict__["_gc"]
        g3b = _g3b_on(self, x)
        part = _slot_take(self, x) if g3b else None
        has_part = part is not None
        key = ("L", x.shape[0], x.dtype, cache.cache[0].shape, cache.cache[1].shape, g3b, has_part)
        if key not in st.seen:                                   # first call of this variant runs the stock body (lazy kernel params may evaluate)
            st.seen.add(key)
            STATS["fallback"] += 1
            FD.fb("compile_lin", "first_call_stock")
            if has_part:                                         # give the slot back to the body
                _slot_set(x, part, self.attn_hc.fn)
            return cls.__call__(self, x, None, cache, prev_topk_indices)
        fn = st.fns.get(key)
        if fn is None:
            def body(x, *rest):
                if has_part:
                    p, conv, ssm = rest
                    _slot_set(x, p, self.attn_hc.fn)
                else:
                    conv, ssm = rest
                c = ArraysCache(size=2)
                c.cache = [conv, ssm]
                try:
                    y, _ = cls.__call__(self, x, None, c, prev_topk_indices)
                    s = _hc().SLOT if g3b else None
                    pout = s.part if s is not None and s.x is y else None
                finally:
                    if g3b:
                        _slot_set(None, None, None)
                st.pout[key] = pout is not None
                return (y, c.cache[0], c.cache[1]) + ((pout,) if pout is not None else ())
            fn = st.fns[key] = mx.compile(body)
            STATS["traced"] += 1
        r = fn(x, *((part,) if has_part else ()), cache.cache[0], cache.cache[1])
        cache.cache[0], cache.cache[1] = r[1], r[2]
        if g3b and st.pout.get(key):
            _slot_put(self, r[0], r[3])
        if hasattr(cache, "advance"):
            cache.advance(1)
        STATS["fast"] += 1
        FD.fire("compile_lin")
        return r[0], prev_topk_indices
    return run


# ------------------------------------------------------------------------------------------------------------------------------ DSA layers
def _lang():
    from . import archs as _AR
    return _AR.lang(_AR.GLM5)


def _core(attn, xo, q_resid, q, new_latent, cache, prev_topk):
    """Glm5NextAttention.__call__ between the projections and o_proj (eager): latent cache update, indexer, attention. -> (out [B, 1, H * v_dim], topk)."""
    batch = xo.shape[0]
    length = xo.shape[1]
    kv_cache = cache[0]
    cache_offset = kv_cache.offset
    if attn.indexer is None:
        index_cache = pool_cache = None
        projected_cache = cache[1]
    else:
        index_cache, pool_cache, projected_cache = cache[1], cache[2], cache[3]
    latent, _ = kv_cache.update_and_fetch(new_latent, mx.zeros((batch, 1, length, 0), dtype=new_latent.dtype))
    if attn.indexer is None:
        if prev_topk is None:
            raise ValueError("Shared indexer layer has no previous top-k indices.")
        topk = prev_topk
    else:
        topk = attn.indexer(xo, q_resid, None, index_cache, pool_cache, cache_offset)
    return attn._attend(q, latent, new_latent, topk, kv_cache, projected_cache, False)


def _dsa_regions(self, g3b, has_part, has_next, st, key):
    """The two compiled functions of DSA layer `self`.
    A(x, [part]) -> (xo, q_resid, q, new_latent, post, comb | pc);  B(out, x, post, comb | pc) -> (x', [part'])."""
    L = _lang()
    attn, a, f = self.self_attn, self.attn_hc, self.ffn_hc
    from mlx_vlm.models.deepseek_v4 import hyper_connection as HCV
    H = _hc() if g3b else None

    def A(x, *rest):
        if g3b:
            part = rest[0] if has_part else H.partials(x, a.fn)
            xo, pc = H.pre(x, part, a, self.input_layernorm.weight, self.input_layernorm.eps)
            pcs = (pc,)
        else:
            collapsed, post, comb = a(x)
            xo, pcs = self.input_layernorm(collapsed), (post, comb)
        b, l, _ = xo.shape
        q_a, kv_a = mx.split(L.linear(attn.qkv_a_proj, xo), (attn.q_lora_rank,), axis=-1)
        q_resid = attn.q_a_layernorm(q_a)
        q = L.linear(attn.q_b_proj, q_resid).reshape(b, l, attn.num_heads, attn.q_head_dim).transpose(0, 2, 1, 3)
        new_latent = attn.kv_a_layernorm(kv_a)[:, None]
        return (xo, q_resid, q, new_latent) + pcs

    def B(out, x, *pcs):
        o = L.linear(attn.o_proj, out)
        if g3b:
            x1, part = H.expand(o, x, pcs[0], f.fn)
            xo, pc = H.pre(x1, part, f, self.post_attention_layernorm.weight, self.post_attention_layernorm.eps)
            o2 = self.mlp(xo)
            x2, part2 = H.expand(o2, x1, pc, self.__dict__["_g3b_next"].fn if has_next else None)
            st.pout[key] = part2 is not None
            return (x2, part2) if part2 is not None else (x2,)
        post, comb = pcs
        x1 = HCV.hc_expand(o, x, post, comb, use_kernel=not a.training)
        return (f.apply_branch(x1, self.post_attention_layernorm, self.mlp),)

    return mx.compile(A), mx.compile(B)


def _dsa_run(cls, attn_call):
    def run(self, x, cache, prev_topk_indices):
        st = self.__dict__["_gc"]
        g3b = _g3b_on(self, x)
        part = _slot_take(self, x) if g3b else None
        has_part = part is not None
        key = ("D", x.shape[0], x.dtype, g3b, has_part)
        if key not in st.seen:
            st.seen.add(key)
            STATS["dsa_fallback"] += 1
            FD.fb("compile_dsa", "first_call_stock")
            if has_part:
                _slot_set(x, part, self.attn_hc.fn)
            return cls.__call__(self, x, None, cache, prev_topk_indices)
        fns = st.fns.get(key)
        if fns is None:
            fns = st.fns[key] = _dsa_regions(self, g3b, has_part, self.__dict__.get("_g3b_next") is not None, st, key)
            STATS["dsa_traced"] += 1
        fa, fb = fns
        r = fa(x, *((part,) if has_part else ()))
        xo, q_resid, q, new_latent = r[:4]
        out, topk = _core(self.self_attn, xo, q_resid, q, new_latent, cache, prev_topk_indices)
        rb = fb(out, x, *r[4:])
        if g3b and st.pout.get(key):
            _slot_put(self, rb[0], rb[1])
        STATS["dsa_fast"] += 1
        FD.fire("compile_dsa")
        return rb[0], topk
    return run


def _why(self, st, x, mask, cache, attn_call) -> str:
    """Flag audit: why the compiled body was not taken (only called under fired.ON; the guards of `__call__`, same order)."""
    if not ENABLED:
        return "flag_off"
    if st is None:
        return "not_armed"
    if x.ndim != 4 or x.shape[1] != 1:
        return f"rows>1 (L={x.shape[1] if x.ndim == 4 else '?'})"
    if mask is not None:
        return "mask"
    if cache is None:
        return "no_cache"
    if self.is_linear:
        c = getattr(cache, "cache", None)
        if c is None or len(c) < 2 or not isinstance(c[0], mx.array) or not isinstance(c[1], mx.array):
            return "state_not_array"
        if getattr(cache, "history_capacity", 0):
            return "speculative_cache"
        if getattr(cache, "lengths", None) is not None:
            return "batch_cache_lengths"
        return "?"
    if not DSA:
        return "dsa_off"
    if type(self.self_attn).__call__ is not attn_call:
        return "attn_call_overridden"
    return "?"


def _make_cls(cls, attn_call):
    lin = _linear_run(cls)
    dsa = _dsa_run(cls, attn_call)

    def __call__(self, x, mask=None, cache=None, prev_topk_indices=None):
        st = self.__dict__.get("_gc")
        ok = ENABLED and st is not None and x.ndim == 4 and x.shape[1] == 1 and mask is None and cache is not None
        if ok and self.is_linear and _plain(cache):
            return lin(self, x, cache, prev_topk_indices)
        if ok and not self.is_linear and DSA and type(self.self_attn).__call__ is attn_call:
            return dsa(self, x, cache, prev_topk_indices)
        STATS["fallback" if self.is_linear else "dsa_fallback"] += 1
        if FD.AUDIT:
            FD.fb("compile_lin" if self.is_linear else "compile_dsa", _why(self, st, x, mask, cache, attn_call))
        return cls.__call__(self, x, mask, cache, prev_topk_indices)
    return __call__


def install(lm) -> int:
    """Arm every decoder layer of the language model `lm` (idempotent). The behaviour is switched at call time by ENABLED."""
    n = 0
    attn_call = _lang().Glm5NextAttention.__call__
    for layer in lm.model.layers:
        if "_gc" in layer.__dict__:
            continue
        cls = type(layer)
        layer._gc = types.SimpleNamespace(fns={}, seen=set(), pout={})       # not a dict: nn.Module keeps dict values as items
        layer.__class__ = type("Gc" + cls.__name__, (cls,), {"__call__": _make_cls(cls, attn_call)})
        n += 1
    return n
