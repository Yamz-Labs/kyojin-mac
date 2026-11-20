# SPDX-License-Identifier: Apache-2.0
"""ablate_struct: structural classes of the KYOJIN_ABLATE probe for any model that runs on the generic decoder skeleton (foreign_skeleton: Hy4, GLM,
every kit model), found by STRUCTURE (the attributes a decoder layer has, the methods a module defines), never by model name.

Same contract as ablate.py: the flag names op classes; each listed class is replaced on the decode path (<= MAX_ROWS rows: the 1-row plain step and the
verify rows; prompt chunks stay real, so caches are valid) by the cheapest stand-in that keeps every shape and dtype the rest of the step reads. Outputs are
garbage: a timing probe. Run the same step with and without a class in ONE process, arms interleaved after >= 4 warm-up reps, and subtract the whole-step
times (hy4_resident.py decode request, arms {"flags": {"ablate": "attn"}}); no per-op stamps are taken, so nothing is serialised.

  attn        layer.self_attn call -> zeros (cache untouched)
  attn_proj   attention `project` (q_a / q_b / kv_a projections, norms, rope) -> zeros of the same shapes   (modules that define `project`)
  attn_core   attention `attend` (scores, softmax, value mix)                                                (modules that define `attend`)
  attn_out    attention `_finish` (output gate + o_proj)                                                     (modules that define `_finish`)
  idx         kpool / indexer: `rows`, `query` of the layer's `kpool` + the selection `keep_*` calls -> zeros / all-visible masks
  mlp         layer.mlp call (MoE or dense) -> zeros
  router      GenericMoE.route (router GEMV + top-k + weight norm) -> fixed ids / uniform weights (constants)
  moe_routed  the routed-expert stack (`switch_mlp`) -> zeros [T, K, W]
  moe_shared  GenericMoE.shared_out -> zeros
  hc          hyper-connection stream mixes + sinkhorn + collapse + expand -> stream mean + add (the sub-layer still runs)
  norm        the layer norms (input / post-attention) -> identity
  engram      layer.engram write -> identity (the host table lookup of Model.engram_source still runs)
  head        lm_head -> zeros [B, L, V]

Wrappers are installed once, from foreign_skeleton.Model.__call__ the first time a class is on (`prepare`), on the classes found on the model's layers; they
read `ablate.ACTIVE` per call (flags.set bumps flags.GEN, compiled regions retrace). The first call of every shape signature runs real and records the output
shapes; later ablated calls return constants of those shapes."""
from __future__ import annotations

import mlx.core as mx

from . import ablate as AB

MAX_ROWS = 16
DECODE = [False]                      # set by prepare() per model call: True when this call is a decode / verify step
_META: dict = {}                      # (class, method, input signature) -> output meta tree
_PATCHED: set = set()
_DONE: set = set()                    # id() of models already prepared
_cache: dict = {}


def _sig(a, k):
    def one(v):
        if isinstance(v, mx.array):
            return (tuple(v.shape), str(v.dtype))
        if isinstance(v, (list, tuple)):
            return tuple(one(i) for i in v)
        if isinstance(v, (int, bool, str, type(None))):
            return v
        return type(v).__name__
    return (tuple(one(v) for v in a), tuple(sorted((n, one(v)) for n, v in k.items())))


def _meta(o):
    if isinstance(o, mx.array):
        return ("a", tuple(o.shape), o.dtype)
    if isinstance(o, tuple):
        return ("t", tuple(_meta(i) for i in o))
    if isinstance(o, list):
        return ("l", [_meta(i) for i in o])
    return ("v", o)


def _const(shape, dtype, fill):
    key = (shape, dtype, fill)
    c = _cache.get(key)
    if c is None:
        import numpy as np
        npd = {mx.float16: np.float16, mx.float32: np.float32, mx.int32: np.int32, mx.bool_: np.bool_, mx.uint32: np.uint32}.get(dtype)
        if npd is None:                                   # bf16: a lazy fill (elementwise consumers only)
            c = mx.zeros(shape, dtype=dtype) if not fill else mx.ones(shape, dtype=dtype)
        else:
            c = mx.array((np.ones if fill else np.zeros)(shape, dtype=npd))
        if len(_cache) > 512:
            _cache.clear()
        _cache[key] = c
    return c


def _build(m):
    t = m[0]
    if t == "a":
        return _const(m[1], m[2], m[2] == mx.bool_)
    if t == "t":
        return tuple(_build(i) for i in m[1])
    if t == "l":
        return [_build(i) for i in m[1]]
    return m[1]


def _wrap(cls, name, klass, standin=None, only=None):
    """cls.name -> wrapper; `standin(self, meta, a, k)` builds the ablated result (default: constants of the recorded shapes). `only`: ids of the
    instances that are ablated (a class shared with unrelated modules, e.g. nn.Linear)."""
    if (cls, name) in _PATCHED or not hasattr(cls, name):
        return
    _PATCHED.add((cls, name))
    orig = getattr(cls, name)

    def w(self, *a, **k):
        if only is not None and id(self) not in only:
            return orig(self, *a, **k)
        key = (cls, name, _sig(a, k))
        m = _META.get(key)
        if klass in AB.ACTIVE and DECODE[0] and m is not None:
            AB.hit(klass)
            return standin(self, m, a, k) if standin else _build(m)
        out = orig(self, *a, **k)
        if m is None:
            _META[key] = _meta(out)
        return out
    w.__wrapped__ = orig
    setattr(cls, name, w)


def _wrap_plain(cls, name, klass, standin):
    """Stand-in that needs no recorded shapes (identity-like): `standin(self, a, k)`."""
    if (cls, name) in _PATCHED or not hasattr(cls, name):
        return
    _PATCHED.add((cls, name))
    orig = getattr(cls, name)

    def w(self, *a, **k):
        if klass in AB.ACTIVE and DECODE[0]:
            AB.hit(klass)
            return standin(self, a, k)
        return orig(self, *a, **k)
    w.__wrapped__ = orig
    setattr(cls, name, w)


def _route_standin(self, m, a, k):
    """route(x2 [T, H]) -> (inds [T, K] int, weights [T, K]): row r takes experts r*K .. r*K+K-1 (mod E), weight 1 / K (host constants)."""
    import numpy as np
    _, (mi, mw) = m
    (T, K), idt, wshape, wdt = mi[1], mi[2], mw[1], mw[2]
    E = max(1, int(getattr(getattr(self, "spec_", None), "n_expert", K)))
    ids = _cache.get(("rid", T, K, E, idt))
    if ids is None:
        ids = _cache[("rid", T, K, E, idt)] = mx.array(((np.arange(T)[:, None] * K + np.arange(K)[None, :]) % E).astype(np.int32)).astype(idt)
    return ids, _const(wshape, wdt, True) * (1.0 / K)


def _hc_standin(self, a, k):
    """HcStream.__call__(h [B, L, hc, D], sub) -> h + sub(mean over streams)."""
    h, sub = a[0], a[1]
    y = sub(h.mean(axis=2).astype(h.dtype))
    return h + y[:, :, None, :].astype(h.dtype)


def _hc_fused_standin(self, a, k):
    """HcStream.fused(h, sub, pre_norm, post_norm)."""
    h, sub, pre, post = a[0], a[1], a[2], a[3]
    x = h.mean(axis=2).astype(h.dtype)
    if pre is not None:
        x = pre(x)
    y = sub(x)
    if post is not None:
        y = post(y)
    return h + y[:, :, None, :].astype(h.dtype)


def prepare(model, inputs) -> None:
    """Called at the top of every skeleton Model call while a class is on: decode flag + one-time wrapper installation on the classes of `model`."""
    DECODE[0] = int(inputs.shape[0]) * int(inputs.shape[1]) <= MAX_ROWS
    if id(model) in _DONE:
        return
    _DONE.add(id(model))
    layers = list(getattr(model, "layers", []) or [])
    for layer in layers:
        a = getattr(layer, "self_attn", None)
        if a is not None:
            ca = type(a)
            _wrap(ca, "__call__", "attn")
            _wrap(ca, "project", "attn_proj")
            _wrap(ca, "attend", "attn_core")
            _wrap(ca, "_finish", "attn_out")
            kp = getattr(a, "kpool", None)
            if kp is not None:
                _wrap(type(kp), "rows", "idx")
                _wrap(type(kp), "query", "idx")
            if getattr(a, "kp", None):
                from . import foreign_skeleton_kpool as KP
                for n in ("keep_fused_fixed", "keep_fused", "keep_cached"):
                    _wrap(KP.KpoolSelect, n, "idx")
        f = getattr(layer, "mlp", None)
        if f is not None:
            _wrap(type(f), "__call__", "mlp")
            blk = f.block() if hasattr(f, "block") else f
            if hasattr(blk, "route") and hasattr(blk, "switch_mlp"):
                _wrap(type(blk), "route", "router", _route_standin)
                _wrap(type(blk), "shared_out", "moe_shared")
                _wrap(type(blk.switch_mlp), "__call__", "moe_routed")
        for nm in ("input_layernorm", "post_attention_layernorm"):
            n = getattr(layer, nm, None)
            if n is not None:
                _wrap_plain(type(n), "__call__", "norm", lambda self, a, k: a[0])
        e = layer["engram"] if "engram" in layer else None
        if e is not None:
            _wrap_plain(type(e), "__call__", "engram", lambda self, a, k: a[0])
        for nm in ("hc_attn",):
            hp = getattr(layer, nm, None)
            if hp is not None and hasattr(hp, "stream"):
                hs = hp.stream(model.args)
                _wrap_plain(type(hs), "__call__", "hc", _hc_standin)
                _wrap_plain(type(hs), "fused", "hc", _hc_fused_standin)
    lm = getattr(model, "lm_head", None)
    if lm is not None:
        _wrap(type(lm), "__call__", "head", only={id(lm)})
