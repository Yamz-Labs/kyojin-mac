# SPDX-License-Identifier: Apache-2.0
"""Multi-user batched decode for the patched mlx-vlm models (every ArchSpec with a `rows` table): N users, one token each, one forward per step.

    model, spec, cfg = build_model(pack); patch_model(model, pack, spec, cfg)
    enable(model, spec)                                   # row-invariant decode ops, pinned MoE k-split, attention adapter
    users = [prefill(model, ids) for ids in prompts]     # solo prefill (S > 1: stock ops), (cache, last logits)
    bc = batch([c for c, _ in users])                     # one cache list for the batched step
    logits = step(model, bc, tokens)                      # [N, vocab]; row i equals the solo decode of user i

Exactness contract: row i of a batched step equals the solo step of user i (`step` with one user, same enabled model) bit for bit.
That needs every op of the decode step to give a row the same bits whatever the number of rows:
  EXL3 linears        the multi-row GEMV (<= 4 rows per launch, up to 16 rows as launches of 4) decodes each weight once per launch, per row arithmetic unchanged
  routed experts      MoeGather with the k-split pinned to the 1-row choice (pin_rows = 1), pairs of all rows in the same 5 launches
  dense matmuls       rowdot (rowinv.py): summation order a function of (K, N) only; the sites are listed in ArchSpec.rows.sites
  attention           "batch_cache": the model's own batch caches; "dsv4_core": projections and the whole core for all users at once
                      (dsv4_core.py: stacked ragged state, fixed-order kernels); "dsv4_hoist": projections at once, per-user cores
Ops that are row independent by construction (norms, elementwise, per-row top-k, fake quant blocks) need nothing."""
from __future__ import annotations

from . import archs as _A
import importlib
import os
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from . import headlast as HL, patch as P
from .archs import ArchSpec
from .exl3_linear import Exl3Linear
from .rowinv import RowLinear, rowdot, rowmm
from . import fired as FD, fuse as F, ablate as AZ, taps as TAPS

# pf_route (KYOJIN_PF_ROUTE, default 0): the prompt-path router as the gate matmul + one selection launch (fuse.router_select), the
# shape the decode path already runs; same ids, same weights. Read per call so POST /admin/flags switches it.
PF_ROUTE = os.environ.get("KYOJIN_PF_ROUTE", "0") not in ("0", "")
# router_w16 (KYOJIN_ROUTER_W16, default 0): the decode router reads its fp32 gate through an exact bf16 / fp16 copy (lowp.exact_lowp), half the bytes, same logits.
ROUTER_W16 = os.environ.get("KYOJIN_ROUTER_W16", "0") not in ("0", "")


# --------------------------------------------------------------------------------------------------------------- site overrides
def _lang(mod):
    """The mlx-vlm language module of a model class. A subclass defined elsewhere (the Engram pipeline classes of engram_mlx) is looked through to its base."""
    import sys
    for c in type(mod).__mro__:
        m = sys.modules.get(c.__module__)
        if m is not None and hasattr(m, "fake_quant_fp8_ue8m0"):
            return m
    return sys.modules[type(mod).__module__]


def mix_vector(self, x, hc_fn):
    """The [B, 1, hc * hidden] x [24, hc * hidden] mix of the block's mHC (after the rms norm of the flattened streams) through rowdot: [B, 1, 24] fp32."""
    x32 = F.take_f32(x) if F.ENABLED and F.MHC else None
    z = mx.fast.rms_norm((x32 if x32 is not None else x.astype(mx.float32)).flatten(-2), None, self.norm_eps)
    return rowmm(z, hc_fn)


def _hc_block_mixes(self, orig, x, hc_fn, hc_scale, hc_base):
    """V4.1 block mHC coefficients (compiled in the stock model): the [B, 1, hc * hidden] x [24, hc * hidden] mix through rowdot."""
    if x.shape[1] != 1:
        return orig(self, x, hc_fn, hc_scale, hc_base)
    if AZ.ACTIVE and AZ.on("mhc"):                       # ablate: constants of the mean / add stand-ins, no mix GEMV, no sinkhorn
        AZ.hit("mhc")
        return AZ.mhc_coeffs(x.shape[0], x.shape[2])
    from mlx_vlm.models.deepseek_v4.hyper_connection import hc_split_sinkhorn
    return hc_split_sinkhorn(mix_vector(self, x, hc_fn), hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)


def _engram_rows(self, orig, x, hash_ids, token_mask=None):
    """V4.1 Engram layer: the key / value projection `wkv` (K = n_hash_cols * head_dim) through rowdot. The gate and the residual write are
    reductions over the last axis of one row (compiled elementwise code), the same for any number of rows."""
    if x.ndim != 4 or x.shape[1] != 1 or "weight" not in self.wkv or "bias" in self.wkv:
        return orig(self, x, hash_ids, token_mask)
    m = _lang(self)
    w = self.wkv.weight
    if w.dtype not in (mx.float16, mx.float32):                                    # bf16 table (dequantized fp8): fp32 is lossless, kept once
        if getattr(self, "_w32", None) is None:
            self._w32 = SimpleNamespace(w=w.astype(mx.float32))
        w = self._w32.w
    n_cols = hash_ids.shape[-1]
    emb = self.embed(hash_ids).astype(x.dtype).reshape(*hash_ids.shape[:-1], n_cols * self.head_dim)
    kv = rowmm(m.fake_quant_fp8_ue8m0(emb), w)
    key, value = mx.split(kv, [self.hc_mult * self.dim], axis=-1)
    key = key.reshape(*key.shape[:-1], self.hc_mult, self.dim)
    score = m._engram_gate_score(x, key, self.q_weight, self.k_weight, mx.array(self.dim ** -0.5, dtype=mx.float32), self.eps, self.clamp_value)
    return m._engram_apply_gate(x, value, mx.sigmoid(score), token_mask)


def _hc_branch_mix(self, orig, z):
    """GLM hyper-connection mix."""
    if z.shape[1] != 1:
        return orig(self, z)
    return rowmm(z, self.fn)


def _router_sqrtsoftplus(self, orig, x, image_mask=None):
    """V4.1 router (sqrt(softplus) scores, correction bias for selection only). Flag router_tap: a decode call also hands its applied weights to the probe."""
    out = _router_sqrtsoftplus_impl(self, orig, x, image_mask)
    if TAPS.ROUTER and x.ndim == 3 and x.shape[1] == 1:
        TAPS.router_tap(self, out[1])
    return out


def _router_sqrtsoftplus_impl(self, orig, x, image_mask=None):
    if PF_ROUTE and x.ndim == 3 and x.shape[1] != 1 and image_mask is None and F.ENABLED and F.ROUTER \
            and self.weight.dtype == mx.float32 and self.bias.dtype == mx.float32 and self.weight.shape[0] % 32 == 0 \
            and 0 < self.top_k < self.weight.shape[0]:
        # Prompt path: the gate GEMV (unchanged, the same fp32 matmul the stock gate runs) then one selection launch instead of the
        # sqrt / softplus / add / neg / argpartition / take / sum / div chain (~10 launches per layer). The selection is
        # `fuse.router_select`: same sqrt(softplus(score)) in fp32, same (key desc, lower index first) order as MLX argpartition,
        # same normalisation, so ids and weights are the stock values bit for bit.
        FD.hit("pf_route")
        lg = (x.astype(mx.float32) @ self.weight.T).reshape(-1, self.weight.shape[0])
        ids, wts = F.router_select(lg, self.bias, self.top_k, bool(self.norm_topk_prob and self.top_k > 1), self.routed_scaling_factor)
        return ids.reshape(*x.shape[:-1], self.top_k), wts.reshape(*x.shape[:-1], self.top_k)
    if x.ndim != 3 or x.shape[1] != 1:
        if PF_ROUTE and x.ndim == 3:
            FD.hit("pf_route_miss")                                   # PF_ROUTE on, a gate (image_mask / dtype / shape / fuse off) kept the stock chain
        return orig(self, x, image_mask)
    if AZ.ACTIVE and AZ.on("router"):                                         # ablate: fixed ids / weights, no gate GEMV, no top-k
        AZ.hit("router")
        return AZ.fixed_route(x.shape[0], self.top_k, self.weight.shape[0], self.routed_scaling_factor)
    if F.ENABLED and F.ROUTER and (image_mask is None or image_mask.size == x.shape[0]) and self.weight.dtype == mx.float32 and self.bias.dtype == mx.float32 and x.dtype in (mx.float16, mx.bfloat16, mx.float32):
        # 2 launches instead of 14: rowdot reads x in its own dtype (the fp32 cast of the stock sequence is exact), one kernel selects and normalises
        from . import fuse_router as FR, launchcut as LC
        args = (self.bias, self.top_k, bool(self.norm_topk_prob and self.top_k > 1), self.routed_scaling_factor, None if image_mask is None else self.bias_vl, image_mask)
        w = self.weight
        if ROUTER_W16:
            from . import lowp
            n16 = lowp.exact_lowp(w)                                                             # same values, half the bytes; fp32 stays when not exact
            FD.hit("router_w16" if n16 is not None else "router_w16_fp32")
            w = w if n16 is None else n16
        if LC.on("router") and FR.route_ok(x[:, 0], w, self.bias, image_mask):                  # launchcut: the gate GEMV and the selection in one launch
            LC.STATS["router"] += 1
            ids, wts = FR.route(x[:, 0], w, *args)
        else:
            ids, wts = F.router_select(rowdot(x[:, 0], w, mx.float32), *args)
        return ids[:, None], wts[:, None]
    scores = mx.sqrt(nn.softplus(rowmm(x.astype(mx.float32), self.weight)))
    bias = self.bias if image_mask is None else mx.where(image_mask[..., None], self.bias_vl, self.bias)
    inds = mx.stop_gradient(mx.argpartition(-(scores + bias), kth=self.top_k - 1, axis=-1)[..., :self.top_k].astype(mx.int32))
    weights = mx.take_along_axis(scores, inds, axis=-1)
    if self.norm_topk_prob and self.top_k > 1:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    return inds, weights * self.routed_scaling_factor


def _router_sigmoid_group(self, orig, x):
    """GLM router (sigmoid scores, grouped top-k)."""
    if x.ndim != 3 or x.shape[1] != 1:
        return orig(self, x)
    return _lang(self)._expert_select(rowmm(x.astype(mx.float32), self.weight), self.e_score_correction_bias, self.top_k, self.n_group,
                                      self.topk_group, self.routed_scaling_factor, self.norm_topk_prob)


def _index_gate_keys(self, orig, x):
    """GLM indexer key projection and pooling gate (the gate is a plain fp32 matmul on the activation)."""
    if x.shape[1] != 1:
        return orig(self, x)
    return self.k_norm(self.wk(x)), rowmm(x.astype(mx.float32), self.index_kpool_compress_gate)


def _mla_attend_rows(self, orig, q, latent, new_latent, topk, kv_cache, projected_cache, last_only=False):
    """GLM MLA decode over the top-k latents for N users: fast SDPA does not give a (batch, head) the same bits for every batch size, so
    each user gets its own SDPA call (shape of the solo call; masked positions contribute exact zeros whatever the gathered values are)."""
    batch, _, length, _ = q.shape
    if length != 1 or batch == 1:
        return orig(self, q, latent, new_latent, topk, kv_cache, projected_cache, last_only)
    sdpa = _lang(self).scaled_dot_product_attention
    sel = topk[:, 0:1]
    valid = (sel >= 0) & (sel < latent.shape[2])
    selected = mx.take_along_axis(latent, mx.clip(sel, 0, max(latent.shape[2] - 1, 0))[:, None, 0, :, None], axis=2)
    query = self.embed_q(q)
    out = mx.concatenate([sdpa(query[i:i + 1], selected[i:i + 1], selected[i:i + 1], cache=kv_cache, scale=self.scale, mask=valid[i:i + 1, None])
                          for i in range(batch)], axis=0)
    out = self.unembed_out(out)
    return out.transpose(0, 2, 1, 3).reshape(batch, 1, -1), topk


SITES = {"engram_rows": _engram_rows, "mla_attend_rows": _mla_attend_rows, "hc_block_mixes": _hc_block_mixes, "hc_branch_mix": _hc_branch_mix, "router_sqrtsoftplus": _router_sqrtsoftplus,
         "router_sigmoid_group": _router_sigmoid_group, "index_gate_keys": _index_gate_keys}


def _override(mod: nn.Module, method: str, fn) -> None:
    """Replace one method of one module instance (the class of the instance becomes a subclass)."""
    cls = type(mod)
    orig = getattr(cls, method)

    def wrapper(self, *a, **k):
        return fn(self, orig, *a, **k)

    wrapper._kyojin_site = fn
    mod.__class__ = type(cls.__name__, (cls,), {method: wrapper, "__module__": cls.__module__})


# --------------------------------------------------------------------------------------------------------------- hoisted attention
class Hoist(nn.Module):
    """A linear of the attention. While `st.pre` is set, a call returns the row `st.row` of an output computed for all users at once."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.st = SimpleNamespace(pre=None, row=0)

    def __call__(self, x):
        st = self.st
        if st.pre is None:
            return self.inner(x)
        return st.pre if st.row is None else st.pre[st.row:st.row + 1]                 # row None: `pre` already is the row (a compiled region emits one array per row)


def out_features(m: nn.Module) -> int:
    """Output width of a patched or a stock linear: the EXL3 linears carry `out_features`, an unpatched nn.Linear / MultiLinear only its weight
    ([out, in], or [groups, out, in] for the grouped form). A tiny stock fixture (tests/mini_ds41.py) is not patched, so the chained verify has to
    read the shape there; on the served pack the modules are patched and this never looks at the weight."""
    n = getattr(m, "out_features", None)
    if n is not None:
        return int(n)
    w = getattr(m, "weight", None)
    if w is None:
        raise AttributeError(f"{type(m).__name__} has neither out_features nor weight")
    return int(w.shape[1] if w.ndim == 3 else w.shape[0])


class Sink(nn.Module):
    """A linear at the end of the attention. While `st.grab` is a list, a call stores its input and returns zeros of the output shape:
    the rest of the single-user code builds lazy arrays that are never evaluated."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.st = SimpleNamespace(grab=None)

    def __call__(self, x):
        g = self.st.grab
        if g is None:
            return self.inner(x)
        g.append(x)
        if x.ndim == 4:                                                          # grouped [B, G, S, in] -> [B, G, S, out]
            return mx.zeros((*x.shape[:3], out_features(self.inner)), dtype=mx.float16)
        return mx.zeros((*x.shape[:-1], out_features(self.inner)), dtype=mx.float16)


def _walk(root, path: str):
    """(parent, leaf name) of a dotted path under `root`, or None when an intermediate module is absent."""
    parts = path.split(".")
    m = root
    for p in parts[:-1]:
        m = getattr(m, p, None)
        if m is None:
            return None
    return (m, parts[-1]) if getattr(m, parts[-1], None) is not None else None


def _install_dsv4_hoist(model, spec: ArchSpec, dispatch: bool = True) -> int:
    """Wrap the attention linears of the hoist / sink tables (pass-through unless an adapter arms them). dispatch=True also routes the
    decode step of a batched cache to the per-user cores (adapter "dsv4_hoist"); dispatch=False only adds the wrappers, for adapters
    (the chained verify of kyojin_mlx.spec) that arm them themselves while the batched core stays in charge of the batched cache."""
    AttnCls = _A.cls("attn")
    n = 0
    for layer in model.language_model.layers:
        attn = getattr(layer, spec.rows.attn_attr)
        if not isinstance(attn, AttnCls) or getattr(attn, "rows", None) is not None:
            continue
        hoist, sink = {}, {}
        for path in spec.rows.hoist:
            w = _walk(attn, path)
            if w:
                parent, leaf = w
                h = Hoist(getattr(parent, leaf))
                setattr(parent, leaf, h)
                hoist[path] = h
        for path in spec.rows.sink:
            parent, leaf = _walk(attn, path)
            s = Sink(getattr(parent, leaf))
            setattr(parent, leaf, s)
            sink[path] = s
        attn.rows = SimpleNamespace(hoist=hoist, sink=sink)
        n += 1
        if not dispatch:
            continue
        cls = type(attn)

        def __call__(self, x, cache, _orig=cls.__call__):
            BatchCacheCls = _A.cls("batch_cache")
            if isinstance(cache, BatchCacheCls) and x.shape[1] == 1:
                return dsv4_attention_rows(self, x, cache)
            return _orig(self, x, cache)

        attn.__class__ = type(cls.__name__, (cls,), {"__call__": __call__, "__module__": cls.__module__})
    return n


def dsv4_attention_rows(attn, x: mx.array, cache) -> mx.array:
    """One decode step of the V4.1 attention for N users (x [N, 1, H]). The projections run once for all users, the attention core of a
    user is the model's own `_attention` over that user's cache with the projection outputs replayed; wo_a / wo_b run once for all users."""
    fq = _lang(attn).fake_quant_fp8_ue8m0
    h, s = attn.rows.hoist, attn.rows.sink
    xq = fq(x)
    qa = h["wq_a"].inner(xq)
    qrq = fq(attn.q_norm(qa))
    pre = {"wq_a": qa, "wq_b": h["wq_b"].inner(qrq), "wkv": h["wkv"].inner(xq)}
    comp, idx = attn.compressor, attn.indexer
    if comp is not None:
        if comp.compress_ratio == 1:
            pre["compressor.wkv"] = h["compressor.wkv"].inner(x)
        else:
            xf = x.astype(mx.float32)
            pre["compressor.wkv"], pre["compressor.wgate"] = h["compressor.wkv"].inner(xf), h["compressor.wgate"].inner(xf)
    if idx is not None:
        pre["indexer.wq_b"], pre["indexer.weights_proj"] = h["indexer.wq_b"].inner(qrq), h["indexer.weights_proj"].inner(x)
    grabbed = []
    try:
        for i, row in enumerate(cache.rows):
            for name, mod in h.items():
                mod.st.pre, mod.st.row = pre.get(name), i
            s["wo_a"].st.grab = s["wo_b"].st.grab = g = []
            attn._attention(row, x[i:i + 1])
            grabbed.append(g[0])
    finally:
        for mod in h.values():
            mod.st.pre = None
        for mod in s.values():
            mod.st.grab = None
    o = mx.concatenate(grabbed, axis=0)                                           # [N, G, 1, in]
    o = s["wo_a"].inner(o).transpose(0, 2, 1, 3).flatten(-2)
    return s["wo_b"].inner(fq(o))


def _install_dsv4_core(model, spec: ArchSpec) -> int:
    BatchCacheCls, AttnCls = _A.cls("batch_cache"), _A.cls("attn")
    from . import dsv4_core
    n = 0
    for layer in model.language_model.layers:
        attn = getattr(layer, spec.rows.attn_attr)
        if not isinstance(attn, AttnCls):
            continue
        cls = type(attn)

        def __call__(self, x, cache, _orig=cls.__call__):
            if isinstance(cache, BatchCacheCls) and x.shape[1] == 1:
                return dsv4_core.core(self, x, cache)
            return _orig(self, x, cache)

        attn.__class__ = type(cls.__name__, (cls,), {"__call__": __call__, "__module__": cls.__module__})
        n += 1
    return n


ADAPTERS = {"dsv4_core": _install_dsv4_core, "dsv4_hoist": _install_dsv4_hoist, "batch_cache": lambda model, spec: 0}
ADAPTER_CORE = {"user": "dsv4_hoist"}          # enable(core="user"): the per-user attention cores (before dsv4_core), kept for A/B runs


# --------------------------------------------------------------------------------------------------------------- enable
def make_row_invariant(model: nn.Module) -> int:
    """Replace every non-EXL3 nn.Linear of the model by a RowLinear. Returns the number replaced."""
    mods = [(n, RowLinear(m)) for n, m in model.named_modules() if type(m) is nn.Linear and not isinstance(m, Exl3Linear)]
    model.update_modules(tree_unflatten(mods))
    return len(mods)


def enable(model, spec: ArchSpec, union: bool = False, grouped: bool | str | None = None, nax: bool | None = None, core: str | None = None) -> dict:
    """Make the patched model's decode step row-invariant and batchable (idempotent per model). Returns counts.
    grouped=True (or KYOJIN_ROWS_GROUPED=1): routed experts for 2+ rows read each distinct expert once for all the rows routed to it (bitwise equal to the
    per-row path); grouped="nax" (KYOJIN_ROWS_GROUPED=nax): groups of KYOJIN_GROUP_HOTK (4) or more rows also go through the NAX GEMM (rounding class).
    nax=True (or KYOJIN_ROWS_NAX=1): dense EXL3 linears with more than 4 rows run the fused NAX kernel (cost flat in rows) instead of GEMV chunks
    of 4 rows; a row then matches its solo bits only to fp16 rounding (1e-3 relative), nax=False keeps bit equality.
    core="user": per-user attention cores (the spec's `hoist` / `sink` tables) instead of the batched core, for A/B runs."""
    assert spec.rows is not None, f"{spec.name}: no multi-user table"
    if nax is not None:
        from . import exl3_linear as _EL
        _EL.ROWS_NAX = bool(nax)
    lm = model.language_model
    if getattr(lm, "_kyojin_rows", None) is not None:
        return lm._kyojin_rows
    rep = dict(row_linears=make_row_invariant(lm), sites=0, moe=0)
    for mod in [m for _, m in lm.named_modules()]:
        for cname, method, impl in spec.rows.sites:
            if type(mod).__name__ == cname:
                _override(mod, method, SITES[impl])
                rep["sites"] += 1
        moe = getattr(mod, "moe", None)
        if isinstance(moe, P.M.MoeGather):
            moe.pin_rows, moe.union = 1, union                                  # k-split of the 1-row call: a row does not depend on N
            msh = getattr(mod, "msh", None)
            if msh is not None:                                                 # port_sumoe: the routed + shared launches pin the same split (its union stays on)
                msh.pin_rows = 1
                msh._p.clear()
            if grouped is not None:
                moe.grouped = bool(grouped)
                moe.hotk = int(os.environ.get("KYOJIN_GROUP_HOTK", "4")) if grouped == "nax" else 0
                if moe.hotk and not P.M.NX.nax_available():
                    moe.hotk = 0
            moe._p.clear()
            rep["moe"] += 1
    from . import exl3_linear as EL
    EL.ROW_SPLIT_MAX = 4 * EL.KN.MAX_ROWS                                      # up to 16 users: EXL3 GEMV launches of <= 4 rows, no dense fallback
    P.MOE_MAX_ROWS = P.M.MAX_PAIRS                                              # up to MAX_PAIRS // topk rows in one MoE call
    rep["fused"] = F.install(spec)
    rep["adapter"] = ADAPTER_CORE[core] if core else spec.rows.adapter
    rep["attention"] = ADAPTERS[rep["adapter"]](model, spec)
    from . import fused_group as FG
    if FG.ENABLED and getattr(lm, "_kyojin_s5", None) is None:
        lm._kyojin_s5 = FG.install(model)                                       # S5: same-input linears share one rot + one finalize (bit-identical)
        rep["s5_groups"] = lm._kyojin_s5["groups"]
    lm._kyojin_rows = rep
    return rep


def enable_draft(lm, spec: ArchSpec) -> dict:
    """The same row invariance for the draft head (kyojin_mlx.dspark): its plain linears, mHC mixes, routers and routed experts give a row the
    bits it has alone, so one draft pass for several sessions equals a pass per session. Idempotent."""
    if getattr(lm, "_kyojin_rows", None) is not None:
        return lm._kyojin_rows
    rep = dict(row_linears=make_row_invariant(lm), sites=0, moe=0)
    for mod in [m for _, m in lm.named_modules()]:
        for cname, method, impl in spec.rows.sites:
            if type(mod).__name__ == cname and impl in ("hc_block_mixes", "router_sqrtsoftplus", "hc_branch_mix", "router_sigmoid_group"):
                _override(mod, method, SITES[impl])
                rep["sites"] += 1
        moe = getattr(mod, "moe", None)
        if isinstance(moe, P.M.MoeGather):
            moe.pin_rows = 1
            moe._p.clear()
            msh = getattr(mod, "msh", None)
            if msh is not None:
                msh.pin_rows = 1
                msh._p.clear()
            rep["moe"] += 1
    lm._kyojin_rows = rep
    return rep


# --------------------------------------------------------------------------------------------------------------- caches and steps
def fork(obj):
    """Independent copy of a cache (any nesting of lists / tuples / dicts / cache objects); MLX arrays are values, so the copies share
    buffers until one of them writes (copy on write)."""
    import copy
    if isinstance(obj, mx.array):
        return obj[...]
    if hasattr(obj, "raw_arrays"):                                              # packed fp4 slots: copy the codes, never unpack
        return obj.copy()
    if isinstance(obj, (list, tuple)):
        return type(obj)(fork(v) for v in obj)
    if isinstance(obj, dict):
        return {k: fork(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__") and not isinstance(obj, (type, SimpleNamespace)):
        new = copy.copy(obj)
        for k, v in vars(obj).items():
            setattr(new, k, fork(v))
        return new
    return obj


PREFIX_COMPACT = os.environ.get("KYOJIN_PREFIX_COMPACT", "0") not in ("0", "")        # flags.py prefix_compact


def compact(obj, out: list):
    """`obj` (a fork: its lists and cache objects are its own) with every array in a buffer of exactly its size. New arrays are appended to `out` (lazy, the caller evaluates).
    The capacity buffers of the vendor cache (`_compress_buffers`, `_key_buffers`) are dropped: the next append rebuilds them, the published arrays are the same bits."""
    from .fp4pool import own
    if isinstance(obj, mx.array):
        a = own(obj)
        out.append(a)
        return a
    if hasattr(obj, "compacted"):
        new = obj.compacted()
        out.extend(a for e in list.__iter__(new) if not hasattr(e, "segs") or hasattr(e, "codes") for a in ((e.codes, e.scales) if hasattr(e, "codes") else (e,)) if a is not None)
        return new
    if isinstance(obj, (list, tuple)):
        return type(obj)(compact(v, out) for v in obj)
    if isinstance(obj, dict):
        return {k: compact(v, out) for k, v in obj.items()}
    if hasattr(obj, "__dict__") and not isinstance(obj, (type, SimpleNamespace)):
        for k, v in list(vars(obj).items()):
            setattr(obj, k, [None] * len(v) if k in ("_compress_buffers", "_key_buffers") and isinstance(v, list) else compact(v, out))
    return obj


def snapshot(obj, compacted: bool | None = None):
    """Fork for a long-lived prefix snapshot: the stored KV only. The derived dense buffers (depthflat mirrors, cross-layer handoffs) are rebuilt
    or republished by the next forward, so keeping them would only pin a dense copy per snapshot once the session writes (copy on write).
    compacted (default: flag prefix_compact): also copy every array into a buffer of exactly its size (see `compact`)."""
    snap = fork(obj)
    from . import fp4pool as FP
    if FP.SEG > 0:                                                              # KYOJIN_KVSEG: share the full pool / key segments, copy only the open tail (fp4pool doc)
        for live, c in zip(obj if isinstance(obj, (list, tuple)) else [obj], snap if isinstance(snap, (list, tuple)) else [snap]):
            for name in ("compress", "keys"):
                slots = getattr(live, name, None)
                if hasattr(slots, "segment_snapshot"):
                    setattr(c, name, slots.segment_snapshot())
    for c in snap if isinstance(snap, (list, tuple)) else [snap]:
        if hasattr(c, "reset_handoff"):
            c.reset_handoff()
        for slots in (getattr(c, "compress", None), getattr(c, "keys", None)):
            if hasattr(slots, "drop_mirrors"):
                slots.drop_mirrors()
    if PREFIX_COMPACT if compacted is None else compacted:
        out: list = []
        snap = compact(snap, out)
        mx.eval(*out)
    return snap


def restore(obj):
    """A cache restored from a stored prefix snapshot (`fork`), with room for its next appends: a snapshot is a view of its own buffers, so the first append of a
    hit copies the whole capacity-sized pool to write in place, and an exact snapshot (prefix_compact, a disk checkpoint) grows it instead. Each packed pool /
    key layer takes a private code buffer of `room` capacity first (fp4pool.restore, flag pf_grow: one copy, at the restore, then the appends write in place).
    Off = `fork`, the hit as it was."""
    out = fork(obj)
    for c in (out if isinstance(out, (list, tuple)) else [out]):
        for name in ("compress", "keys"):
            slots = getattr(c, name, None)
            if hasattr(slots, "restore"):
                slots.restore()
    return out


def prefill(model, ids: list[int], cache: list | None = None) -> tuple[list, mx.array]:
    """Solo prefill of one user. Returns (cache, logits of the last position [1, vocab])."""
    lm = model.language_model
    cache = cache if cache is not None else lm.make_cache()
    with HL.last_row():                                                         # only the last row is read
        lg = lm(mx.array(ids)[None], cache=cache).logits[:, -1].astype(mx.float32)
    mx.eval(lg)
    return cache, lg


def batch(users: list[list]) -> list:
    """Per-user caches (each a list over layers / cache groups) -> one cache list for a batched step (the model's own merge)."""
    return [type(col[0]).merge(list(col)) for col in zip(*users)]


def step(model, cache: list, tokens: mx.array) -> mx.array:
    """One decode step. tokens [N] int, cache from `batch` (or a solo cache with N = 1). Returns the fp32 logits [N, vocab]; evaluating them
    also evaluates the state the step wrote (stacked attention state of the batched DS4.1 core)."""
    lg = model.language_model(tokens[:, None], cache=cache).logits[:, -1].astype(mx.float32)
    dep = [a for c in cache if hasattr(c, "stacked_arrays") for a in c.stacked_arrays()]
    return mx.depends(lg, dep) if dep else lg


class Pass:
    """One verify pass over the sessions of a batched cache (`verify`). The model ran counts[s] consecutive positions of session s as rows of
    one forward; nothing is committed yet. logits [rows, V] fp32 (row i of session s = the logits after its token i), taps {layer: [rows, 1, H]}.
    `commit(accepted)` keeps, for each session, its first accepted[s] + 1 rows (its pending token and accepted[s] drafts)."""

    def __init__(self, entry, logits, taps, plan, hist):
        self.entry, self.logits, self.taps, self.plan, self.hist = entry, logits, taps, plan, hist
        self.starts = plan.first
        self.eh = None                                                  # the Engram hash ids of the rows (verify sets them)

    def commit(self, accepted: list) -> None:
        assert len(accepted) == self.plan.S and all(0 <= a < c for a, c in zip(accepted, self.plan.counts)), (accepted, self.plan.counts)
        self.entry.commit(list(accepted), self.plan, self.hist)


def _ids(toks):
    """Token ids of one session as an int32 array: a host list, or an array that may still be on the GPU (a lazy draft)."""
    return toks.astype(mx.int32) if isinstance(toks, mx.array) else mx.array(toks, dtype=mx.int32)


def _engram_pass(lm, entry, token_lists, read_ahead=True):
    """Engram hashes [rows, 1, layers, cols] of the rows of a pass and the new token history of every session (the session's own state is
    left as it was until the commit). (None, None) when the model has no Engram layer."""
    lm._ensure_engram_hash()
    if lm.engram_hash is None:
        return None, None
    parts, hist = [], []
    ahead = getattr(lm.engram_hash, "_kyojin_ahead", None)                # the rows ahead (engram_mlx): one job set for the joined rows, not per session
    if ahead is not None:
        ahead.late = None                                                 # an earlier pass that died before its first Engram layer
    if ahead is not None:
        ahead.mute = True
    try:
        for row, toks in zip(entry.rows, token_lists):
            old = row.engram
            h = lm.engram_hash(_ids(toks)[None], row.offset, row, token_mask=None)      # [1, n_s, layers, cols]
            parts.append(h[0][:, None])
            hist.append(row.engram)
            row.engram = old
    finally:
        if ahead is not None:
            ahead.mute = False
    eh = mx.concatenate(parts, axis=0)
    if ahead is not None:
        if read_ahead:
            from . import engram_mlx as EM
            if EM.PRESTART and any(isinstance(t, mx.array) for t in token_lists):
                ahead.defer(eh)                                           # verify_prestart: the read waits for the first Engram layer, the pre-Engram layers are built under the draft
            else:
                ahead.start(eh)                                           # the embeds see [rows, 1, cols] per layer
        else:
            ahead.jobs = {}                                               # no stale job of an earlier pass for the stage's embeds
    return eh, hist


def verify(model, cache: list, token_lists: list, stage=None) -> Pass:
    """One forward over the sessions of the batched cache `cache` (from `batch`), session s feeding `token_lists[s]` (its pending token and its
    drafts: consecutive positions): rows = sum of the lengths, per-session causal state, one pass through the batched core for every row.
    Row i of session s equals the plain decode step that consumes token i of that session (same code: plain decode is the case of one row per
    session). Returns a Pass; `Pass.commit(accepted)` keeps the first accepted[s] + 1 rows of each session and drops the others.
    stage: an engram_mlx.EngramStage: the forward is built inside `stage.deferred()` (the Engram rows are registered, not read: the caller fills
    them from `Pass.eh` before the pass is committed to the GPU, see VlmBackend.spec_verify_lazy)."""
    from . import dsv4_core as DC
    lm = model.language_model
    st = DC.stack_rows(cache[0])
    assert len(token_lists) == len(st.rows), (len(token_lists), len(st.rows))
    st.begin([len(t) for t in token_lists])
    try:
        plan = DC.plan_of(st)
        eh, hist = _engram_pass(lm, st, token_lists, read_ahead=stage is None)   # a stage fills its own row slots
        kw = {} if eh is None else {"engram_hashes": eh}
        st.taps = {}
        if any(isinstance(t, mx.array) for t in token_lists):        # lazy drafts: the ids stay on the GPU, no host read
            ids = mx.concatenate([_ids(t) for t in token_lists])
        else:
            ids = mx.array([t for toks in token_lists for t in toks], dtype=mx.int32)
        def eager():
            return lm(ids[:, None], cache=cache, **kw).logits[:, -1].astype(mx.float32)
        from . import verify_replay as VR
        import contextlib
        from . import exl3_linear as EL
        nax = EL.ROWS_NAX
        EL.ROWS_NAX = nax and os.environ.get("KYOJIN_VERIFY_NAX", "0") == "1"
        if EL.ROWS_NAX and ids.shape[0] > 4:
            FD.hit("verify_nax")   # verify rows keep their solo (GEMV) bits: exact vs plain at any depth
        try:
            with stage.deferred() if stage is not None else contextlib.nullcontext():
                lg = VR.forward(lm, st, plan, ids, kw, eager) if VR.ENABLED else eager()
        finally:
            EL.ROWS_NAX = nax
    except BaseException:
        st._counts = None
        st.pend = {}
        raise
    finally:
        taps = st.__dict__.pop("taps", {})
        ahead = getattr(getattr(lm, "engram_hash", None), "_kyojin_ahead", None)
        if ahead is not None:
            ahead.late = None                                            # verify_prestart: never leaves hashes behind for a later forward
    ps = Pass(st, mx.depends(lg, st.stacked_arrays()), taps, plan, hist)
    ps.eh = eh
    return ps
