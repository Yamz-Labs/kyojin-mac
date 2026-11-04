# SPDX-License-Identifier: Apache-2.0
"""CED prefill (DeepSeek-V4.1 report sections 2.2 and 3.2.2, arXiv 2609.19969): the vendor's own prefill for models whose layer layout
splits into an encoder half and a decoder half that reads ONE shared global KV. The server turns it on wherever the layout allows
(`auto_enable`), `--no-ced` turns it off; the library itself changes nothing until `enable()` is called.

Layout (`layout()` checks it on the model config, no model name involved): an even layer split with the midpoint layer `mid = n // 2`;
the midpoint is the KV, index and candidate source of the whole second half, every layer from `mid` on has compress ratio 1 (one
compressed entry per token, no cross-token state), no KV source comes after `mid`, no Engram layer sits at or after `mid`.

What the decoder half needs from a chunk of `n` prompt tokens: the midpoint's compressed KV and index keys of ALL tokens (they project the
hidden states that the encoder half produced, which are exact), and the per-layer sliding-window KV, queries and MoE of the LAST `tail`
tokens only. So a chunk longer than `tail` runs
  layers < mid          unchanged, all n rows
  layer  mid            compressor and index-key projection over n rows; queries, window, indexer scores, attention, MoE over the tail rows
  layers > mid          tail rows only, window attention restarted at the tail (no history before the tail start)
and the model returns the logits of the tail rows (the last row is the next-token distribution). The approximation: the first tail rows of
the decoder half see a window shorter than `sliding_window`; the global KV, the encoder half, the caches of the encoder half and the
midpoint's compressed KV and index keys are bit for bit the exact ones. `tail >= sliding_window`, so after a CED chunk the window ring of
every decoder layer holds the last `sliding_window` positions and a following chunk (or decode) finds its history.

The result depends on the chunk grid (each chunk cuts its own tail); a cache built with CED must not be mixed with exact chunks of another
grid (namespace snapshots by `signature()`). `exact(model)` turns it off for a block. Decode, speculative
verify and batched rows have at most a few rows (<= tail) and never take this path.

Install order: `enable()` before `spec.install` (the drafter taps wrap the block calls; they must see the full input at the midpoint). The
drafter taps of layers >= mid then cover the tail rows only: keep `tail >= keep` of `spec.install_taps` (default 128).

Tail size: `tail = sliding_window` (default, DeepSeek) is approximate; `tail = receptive_rows(layout)` (`enable(tail="exact")`) makes the last-row logits
and the window rings of the decoder layers equal the exact prefill up to the rounding of other matmul shapes, and still skips the decoder half
of every row before it (a gain once the chunk is several times larger than that). A per-call `ced_tail=` keyword of the language model overrides it.

Gate: tools/ced_gate.py (KL of the last-token logits, top-1, greedy answer match, needle). The vendor reports a negligible loss: a large
drop in the gate means a bug here, not a price to pay."""
from __future__ import annotations

from . import archs as _A
import os
from dataclasses import dataclass, field

import mlx.core as mx

from . import ablit as AB
from . import fired as FD

STATE_ATTR = "_kyojin_ced"
# ced_skip_dec (KYOJIN_CED_SKIP_DEC, default 0, read per call by the engine through `backend.ced_can_skip`): 0 = off, 1 = a chunk skips its decoder half only where no
# prefix snapshot / disk checkpoint is taken at its end, 2 = also where one would be (the engine then takes none at that boundary), 3 = also where one would be and the
# engine takes it, marked stale (decoder rings not the full prefill's): a restore uses it only before a chunk of more than `tail` rows, else an earlier whole one. See `skip_decoder`.
SKIP_DEC = int(os.environ.get("KYOJIN_CED_SKIP_DEC", "0") or 0)


def _override(mod, method: str, fn) -> None:
    """Replace one method of one module instance (the class of the instance becomes a subclass)."""
    cls = type(mod)
    orig = getattr(cls, method)

    def wrapper(self, *a, **k):
        return fn(self, orig, *a, **k)

    mod.__class__ = type(cls.__name__, (cls,), {method: wrapper, "__module__": cls.__module__})


def _g(cfg, *names, default=None):
    for n in names:
        if hasattr(cfg, n):
            return getattr(cfg, n)
    return default


# --------------------------------------------------------------------------------------------------------------- layout
@dataclass(frozen=True)
class Layout:
    mid: int
    n_layers: int
    window: int


def explain(cfg, mid: int | None = None) -> str | None:
    """None when the layer layout supports CED, else the first reason it does not. `mid`: explicit first decoder layer (tests and slices that
    are too shallow for an even split); default is the even split `n // 2`."""
    n = int(_g(cfg, "num_hidden_layers"))
    window = int(_g(cfg, "sliding_window", default=0))
    ratios = list(_g(cfg, "compress_ratios"))[:n]
    kv_src = set(_g(cfg, "kv_source_layer_ids", default=()))
    idx_src = set(_g(cfg, "index_source_layer_ids", default=()))
    engram = list(_g(cfg, "engram_layer_ids", default=()) or ())
    cand = int(_g(cfg, "candidate_source_layer_id", default=-1))
    if mid is None:
        if n % 2:
            return f"{n} layers: no even split"
        mid = n // 2
    if not 0 <= mid < n:
        return f"mid {mid} outside 0..{n - 1}"
    if window <= 0:
        return "no sliding window"
    if mid not in kv_src or mid not in idx_src:
        return f"layer {mid} is not a KV and index source"
    if len(ratios) < n or any(r != 1 for r in ratios[mid:n]):
        return f"layers {mid}..{n - 1} are not all compress ratio 1"
    if any(i > mid for i in kv_src):
        return "a KV source after the midpoint"
    if any(i >= mid for i in engram):
        return "an Engram layer at or after the midpoint"
    if 0 <= cand < mid:
        return f"candidate source {cand} is before the midpoint"
    return None


def receptive_rows(lay: Layout) -> int:
    """Rows a decoder-half tail needs for its last `window` rows (the ones decode and the next chunk read) to be what the exact prefill computes:
    row p at decoder layer j depends on the rows of the window chain, (j + 1) * (window - 1) positions back; with D decoder layers the last
    `window` rows need D * (window - 1) + window rows in all. The global KV is exact whatever the tail, so nothing else leaks."""
    return (lay.n_layers - lay.mid) * (lay.window - 1) + lay.window


def supported(cfg, mid: int | None = None) -> bool:
    return explain(cfg, mid) is None


def layout(cfg, mid: int | None = None) -> Layout:
    why = explain(cfg, mid)
    if why is not None:
        raise ValueError(f"CED prefill is not supported by this layer layout: {why}")
    n = int(_g(cfg, "num_hidden_layers"))
    return Layout(mid=n // 2 if mid is None else mid, n_layers=n, window=int(_g(cfg, "sliding_window")))


# --------------------------------------------------------------------------------------------------------------- state
@dataclass
class Ced:
    lay: Layout
    tail: int
    on: bool = True
    force: bool = False                         # tests: also take the CED path when the chunk is exactly `tail` rows (nothing is cut)
    skip: bool = False                          # set by `skip_decoder` for one chunk: the decoder half of that chunk is not run
    stats: dict = field(default_factory=lambda: dict(chunks=0, rows=0, rows_cut=0, skip_cache=0))

    def signature(self) -> str:
        """Namespace for snapshots / disk checkpoints: approximate caches never mix with exact ones."""
        return f"ced1-m{self.lay.mid}-t{self.tail}" if self.on else "exact"


@dataclass
class _Ctx:
    n: int          # rows of the chunk
    s0: int         # position of its first row
    tail: int
    skip: bool = False      # ced_skip_dec: the midpoint publishes its KV and index keys, the decoder half is not run

    @property
    def s_t(self) -> int:
        return self.s0 + self.n - self.tail


def state(model) -> Ced | None:
    return getattr(getattr(model, "language_model", model), STATE_ATTR, None)


def signature(model) -> str:
    st = state(model)
    return "exact" if st is None else st.signature()


class exact:
    """`with ced.exact(model):` runs the block with the exact prefill (CED switched off, then back to its previous setting)."""

    def __init__(self, model):
        self.st = state(model)

    def __enter__(self):
        if self.st is not None:
            self.prev, self.st.on = self.st.on, False
        return self

    def __exit__(self, *exc):
        if self.st is not None:
            self.st.on = self.prev


def _is_ds_cache(entry) -> bool:
    """The DS4.1 cache or a subclass (fp4pool); batched caches never take the CED path."""
    return entry is not None and _A.is_model_cache(entry)


def takes_ced(st: Ced, n: int, tail: int, ds: bool) -> bool:
    """Whether a prompt call of `n` rows takes the CED path: the one predicate of `_lm_call`."""
    return bool(st.on and ds and n >= tail and (n > tail or st.force))


def next_chunk_is_ced(model, cache, n: int) -> bool:
    """Whether a prompt chunk of `n` rows on `cache` would run the CED path (the engine asks this about the NEXT chunk before it lets the current one skip its decoder half)."""
    st = state(model)
    return st is not None and takes_ced(st, int(n), st.tail, _is_ds_cache(cache[0] if cache else None))


def rewrites_rows(model, n: int) -> bool:
    """`rewrites_decoder` without a cache in hand (the prefix lookup asks it before the cache exists; a DS4.1 model only runs DS4.1 caches)."""
    st = state(model)
    return st is not None and st.on and int(n) > st.tail


def rewrites_decoder(model, cache, n: int) -> bool:
    """Whether a chunk of `n` rows on `cache` is a CED chunk that restarts its decoder half at its own tail (n > tail): it rewrites the window ring of every decoder layer
    whole and never reads the old one. (n == tail under `force` is a CED chunk that does read the ring: not enough.)"""
    st = state(model)
    return st is not None and int(n) > st.tail and next_chunk_is_ced(model, cache, n)


class skip_decoder:
    """`with ced.skip_decoder(model):` the next CED chunk skips its decoder half (ced_skip_dec). The encoder half and the midpoint's compressed KV and
    index keys of ALL its rows are computed as ever; the midpoint's queries, window, indexer scores, attention and MoE and the layers after it are not, and
    the call returns the logits of one meaningless row. What is lost is the window ring of the decoder layers (and the drafter taps of those layers):
    both are rewritten from scratch, bit for bit, by the next chunk iff that chunk is itself a CED chunk of more than `tail` rows (its decoder half
    restarts at its tail and never reads the ring). The caller owns that guarantee (`next_chunk_is_ced`), must not store a snapshot of the cache at
    this boundary unless it marks it stale (mode 3: see PrefixCache.lookup, kvdisk "stale"), and must not read the taps. A chunk that does not take the CED path ignores the hint and runs whole."""

    def __init__(self, model):
        self.st = state(model)

    def __enter__(self):
        if self.st is not None:
            self.st.skip = True
        return self

    def __exit__(self, *exc):
        if self.st is not None:
            self.st.skip = False


def set_on(model, on: bool) -> None:
    st = state(model)
    assert st is not None, "CED was not enabled on this model"
    st.on = bool(on)


# --------------------------------------------------------------------------------------------------------------- DS4.1 (mlx-vlm deepseek_v41)
def _L():
    L = _A.lang()
    return L


def _window_nohist(attn, x, start_pos, cache):
    """`DeepseekV41Attention._window_part` for a tail whose preceding positions were skipped: the ring holds nothing valid before `start_pos`,
    so the window is the rows of this call only (a zero key under a valid mask would still draw softmax weight)."""
    L = _L()
    batch, seqlen = x.shape[0], x.shape[1]
    win = attn.window_size
    kv = attn.kv_norm(attn.wkv(L.fake_quant_fp8_ue8m0(x)))
    kv = attn.rope(kv[:, None], start_pos).reshape(batch, seqlen, attn.head_dim)
    kv = L.fake_quant_fp8_ue8m0(kv)
    buffer = cache.window[attn.layer_idx]
    if buffer is None:
        buffer = mx.zeros((batch, win, attn.head_dim), dtype=kv.dtype)
    elif buffer.shape[0] < batch:
        buffer = mx.pad(buffer, [(0, batch - buffer.shape[0]), (0, 0), (0, 0)])
    end_pos = start_pos + seqlen
    retained = min(seqlen, win)
    buffer[:batch, mx.arange(end_pos - retained, end_pos) % win] = kv[:, -retained:]
    cache.window[attn.layer_idx] = buffer
    positions = mx.arange(start_pos, end_pos)
    valid = (positions[None, :] <= positions[:, None]) & (positions[None, :] > positions[:, None] - win)
    return kv, valid[None, None]


def _window_part(attn, orig, x, start_pos, cache):
    if getattr(cache, "_ced_nohist", False):
        return _window_nohist(attn, x, start_pos, cache)
    return orig(attn, x, start_pos, cache)


def _mid_kv(attn, x_full, cache, s0: int):
    """The part of the midpoint that the whole chunk feeds: the compressed KV of ALL rows into the pool, the index keys of the owner. Returns the latent."""
    L = _L()
    ratio = attn.compress_ratio
    latent = attn.compressor(x_full, s0, cache)
    n_latent = latent.shape[1]
    positions = (s0 // ratio + mx.arange(n_latent)) * ratio
    pool_kv = L._apply_rope_at_positions(latent, positions, attn.config.qk_rope_head_dim, attn._latent_theta, attn._latent_yarn)
    cache.append_compressed(attn.layer_idx, L.fake_quant_fp4_e4m3(pool_kv))
    cache.compress_kv = cache.compress[attn.layer_idx]
    return latent


def _attention_mid(attn, x_full, x_tail, cache, s0: int, s_t: int):
    """`DeepseekV41Attention._attention` of the midpoint (mode "full", ratio 1) with two row sets: the compressed KV and the index keys come
    from ALL rows of the chunk (`x_full`, positions s0..), the queries, the window and the indexer scores from the tail (`x_tail`, from s_t)."""
    L = _L()
    batch, n, t = x_full.shape[0], x_full.shape[1], x_tail.shape[1]
    ratio = attn.compress_ratio
    qr = attn.q_norm(attn.wq_a(L.fake_quant_fp8_ue8m0(x_tail)))
    q = attn.wq_b(L.fake_quant_fp8_ue8m0(qr)).reshape(batch, t, attn.n_heads, attn.head_dim)
    q = attn.rope(q.transpose(0, 2, 1, 3), s_t)
    window_kv, window_mask = attn._window_part(x_tail, s_t, cache)
    latent = _mid_kv(attn, x_full, cache, s0)
    pool = cache.compress_kv[:batch, :(s_t + t) // ratio]
    ix = attn.indexer
    if ix.owns_k:
        ix._publish_keys(latent, s0, cache)
    idxs = ix(x_tail, qr, None, s_t, 0, cache)
    cache.topk_idxs = idxs
    out = attn._sparse_attention(q, window_kv, pool, idxs, window_mask)
    out = attn.rope(out, s_t, inverse=True)
    out = out.reshape(batch, attn.o_groups, -1, t, attn.head_dim)
    out = out.transpose(0, 1, 3, 2, 4).flatten(-2)
    out = attn.wo_a(out)
    out = out.transpose(0, 2, 1, 3).flatten(-2)
    return attn.wo_b(L.fake_quant_fp8_ue8m0(out))


def _block_mid(blk, h, pre_mix, image_mask, cache, ctx: _Ctx):
    """`DeepseekV41Block.__call__` of the midpoint: attention input from all rows, everything else on the tail rows."""
    L = _L()
    t = ctx.tail
    h_t = h[:, -t:]
    attn_pre, attn_post, attn_comb = blk.hc_mixes(h_t, blk.hc_attn_fn, blk.hc_attn_scale, blk.hc_attn_base)
    x_full = blk.attn_norm(blk.hc_pre(h, pre_mix))
    cache._ced_nohist = t < ctx.n
    try:
        a = _attention_mid(blk.attn, x_full, x_full[:, -t:], cache, ctx.s0, ctx.s_t)
    finally:
        cache._ced_nohist = False
    x = L.hc_expand(AB.attn_out(blk, a), h_t, attn_post, attn_comb)
    residual = x
    ffn_pre, ffn_post, ffn_comb = blk.hc_mixes(x, blk.hc_ffn_fn, blk.hc_ffn_scale, blk.hc_ffn_base)
    x = blk.hc_pre(x, attn_pre)
    x = blk.ffn_norm(x)
    x = AB.ffn_out(blk, blk.ffn(x, None if image_mask is None else image_mask[:, -t:]))
    return L.hc_expand(x, residual, ffn_post, ffn_comb), ffn_pre


def _block_mid_skip(blk, h, pre_mix, cache, ctx: _Ctx):
    """ced_skip_dec: the midpoint of a chunk whose decoder half is not run. Everything the later chunks read from it is written (the compressed KV and the index
    keys of all rows, from the same ops as `_attention_mid`); the tail rows' queries, window, indexer scores, attention and FFN are not computed. Returns one row, so
    the layers after it and the head have nothing to do."""
    x_full = blk.attn_norm(blk.hc_pre(h, pre_mix))
    attn = blk.attn
    latent = _mid_kv(attn, x_full, cache, ctx.s0)
    if attn.indexer.owns_k:
        attn.indexer._publish_keys(latent, ctx.s0, cache)
    return h[:, -1:], pre_mix[:, -1:]


def _block_call(blk, orig, h, pre_mix, image_mask, cache, *, mid: int):
    ctx = getattr(cache, "_ced", None)
    if ctx is None:
        return orig(blk, h, pre_mix, image_mask, cache)
    if ctx.skip:
        return _block_mid_skip(blk, h, pre_mix, cache, ctx) if blk.layer_idx == mid else (h, pre_mix)
    if blk.layer_idx == mid:
        return _block_mid(blk, h, pre_mix, image_mask, cache, ctx)
    off = cache.offset
    cache.offset, cache._ced_nohist = ctx.s_t, ctx.tail < ctx.n
    try:
        return orig(blk, h, pre_mix, None if image_mask is None else image_mask[:, -ctx.tail:], cache)
    finally:
        cache.offset, cache._ced_nohist = off, False


def _lm_call(lm, orig, input_ids=None, inputs_embeds=None, cache=None, image_mask=None, engram_hashes=None, inputs=None, n_to_process=None,
             ced_tail=None, **kw):
    st: Ced = getattr(lm, STATE_ATTR)
    tail = st.tail if ced_tail is None else int(ced_tail)
    entry = cache[0] if cache else None
    ids = input_ids if input_ids is not None else inputs
    src = inputs_embeds if inputs_embeds is not None else ids
    n = 0 if src is None else src.shape[1]
    full = dict(input_ids=input_ids, inputs_embeds=inputs_embeds, cache=cache, image_mask=image_mask, engram_hashes=engram_hashes, inputs=inputs,
                n_to_process=n_to_process, **kw)
    ds = _is_ds_cache(entry)
    st.stats["skip_cache"] += st.on and n > tail and not ds                     # a prompt chunk CED could cut that the cache type keeps whole
    if not takes_ced(st, n, tail, ds):
        return orig(lm, **full)
    if tail < st.lay.window:
        raise ValueError(f"ced_tail {tail} < sliding window {st.lay.window}")
    entry._ced = _Ctx(n=n, s0=int(entry.offset), tail=tail, skip=st.skip)
    st.stats["chunks"] += 1
    st.stats["rows"] += n
    st.stats["rows_cut"] += n - tail
    if st.skip:
        st.stats["skip_dec"] = st.stats.get("skip_dec", 0) + 1                  # (keys appear on first use: the stats of a run without the lever are unchanged)
        st.stats["skip_dec_rows"] = st.stats.get("skip_dec_rows", 0) + tail
        FD.hit("ced_skip_dec")
    try:
        return orig(lm, **full)
    finally:
        entry._ced = None


def _install_dsv4(model, spec, st: Ced) -> int:
    lm = getattr(model, "language_model", model)
    _override(lm, "__call__", _lm_call)
    mid = st.lay.mid
    for i in range(mid, st.lay.n_layers):
        blk = lm.layers[i]
        _override(blk.attn, "_window_part", _window_part)

        def call(self, orig, h, pre_mix, image_mask, cache, _mid=mid):
            return _block_call(self, orig, h, pre_mix, image_mask, cache, mid=_mid)

        _override(blk, "__call__", call)
    from . import bound_layers as BL
    BL.rewrap(lm)                                                  # count the starts of the decoder classes in a bound, when bound_layers was installed first
    return st.lay.n_layers - mid


ADAPTERS = {"dsv4": _install_dsv4}


def enable(model, spec, tail: int | str | None = None, mid: int | None = None) -> dict:
    """Arm CED prefill on `model` (a patched mlx-vlm model or its language model). Idempotent: a second call only changes `tail` and switches
    the path on. tail: rows of the decoder half per chunk (None = sliding_window, "exact" = receptive_rows). Raises ValueError when the layer
    layout is not supported. Returns {mid, tail, layers}."""
    assert spec.ced is not None, f"{spec.name}: no CED description in its ArchSpec (ArchSpec.ced)"
    lm = getattr(model, "language_model", model)
    cfg = lm.config
    mid, tail = spec_defaults(spec, mid, tail)
    lay = layout(cfg, mid)
    tail = lay.window if tail is None else receptive_rows(lay) if tail == "exact" else int(tail)
    if tail < lay.window:
        raise ValueError(f"tail {tail} < sliding window {lay.window}: the window ring would miss positions")
    st = getattr(lm, STATE_ATTR, None)
    if st is not None:
        st.tail, st.on = tail, True
        return dict(mid=st.lay.mid, tail=tail, layers=0)
    st = Ced(lay=lay, tail=tail)
    n = ADAPTERS[spec.ced.adapter](model, spec, st)
    setattr(lm, STATE_ATTR, st)
    return dict(mid=lay.mid, tail=tail, layers=n)


def spec_defaults(spec, mid, tail):
    """Explicit arguments win; else the ArchSpec's CedSpec (`mid` / `tail`); None stays None = the even split n // 2 and the sliding window (what DS4.1 runs: 20 and 128)."""
    cs = getattr(spec, "ced", None)
    if cs is None:
        return mid, tail
    return (cs.mid if mid is None else mid), (cs.tail if tail is None else tail)


def auto_enable(model, spec, disabled: bool = False, tail: int | str | None = None, log=print, mid: int | None = None) -> dict | None:
    """Server default: arm CED when the pack's layout supports it, unless `disabled` (--no-ced). Returns enable()'s dict, or None (disabled, no
    CED description for the architecture, or layout not supported; the reason goes to `log`). mid: explicit first decoder layer (slices)."""
    if disabled:
        log("ced: off (--no-ced)")
        return None
    if getattr(spec, "ced", None) is None:
        return None
    lm = getattr(model, "language_model", model)
    mid, tail = spec_defaults(spec, mid, tail)
    why = explain(lm.config, mid)
    if why:
        log(f"ced: not used, layout not supported: {why}")
        return None
    r = enable(model, spec, tail=tail, mid=mid)
    log(f"ced: on, midpoint layer {r['mid']}, tail {r['tail']} rows per chunk, {r['layers']} decoder layers (--no-ced to turn off)")
    return r
