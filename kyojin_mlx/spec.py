# SPDX-License-Identifier: Apache-2.0
"""Speculative decode on the patched mlx-vlm models: draft D tokens, verify D + 1 positions of ONE sequence in one batched pass, keep the
longest prefix the target agrees with. Greedy output is token identical to plain decode.

    model, spec, cfg = build_model(pack); patch_model(model, pack, spec, cfg); rows.enable(model, spec); spec_mod.install(model, spec)
    logits, taps, chain = verify(model, cache, [t0, d1, .., dD])   # row i = the solo decode step of token i (bit for bit)
    new_cache = chain.rows[a]                                       # rollback to "t0..ta consumed": no replay, no copy

Why the rows are exact. The batched rows path (rows.py) already gives a row the bits of its solo step, whatever the number of rows: EXL3
GEMVs of <= 4 rows, row-invariant dense products, pinned MoE k-split. Users have private caches there. Here the rows are consecutive
positions of one sequence, so row j needs the state row j - 1 left behind, layer by layer. The pass is therefore "chained":
  * row j owns a fork of the cache (`R_j`, offset base + j). Copy on write: the forks share buffers until one writes.
  * layer l, row j: the layer-l slots (window ring, compressed pool, index keys, compressor state) are re-pointed to the slots row j - 1
    left after its own layer-l core, then the model's own `_attention` runs on `R_j`. Same code as a solo step.
  * the cross-layer handoff (index_k, candidates, compress_kv, topk_idxs) lives on `R_j`: every layer reads what a source layer of the same
    row wrote earlier in the same pass, as in a solo step.
  * rollback after acceptance of `a` drafts = `chain.rows[a]`, the cache of a solo run over t0..ta. Nothing to undo, nothing to replay.
Projections, mHC mixes, router, wo_a / wo_b, MoE and the head run once for all rows (weights read once): that is the speed-up.

Arch hook: `ArchSpec.rows.adapter` selects the chain adapter (CHAIN_ADAPTERS). Only the DS4.1 style hoisted attention has one."""
from __future__ import annotations

from . import archs as _A
import os
from types import SimpleNamespace

import mlx.core as mx

from . import headlast as HL, rows as R, wcache as WC
from .moe import MoeGather
from .archs import ArchSpec

_SLOTS = ("window", "compress", "keys", "kv_state", "score_state", "_compress_buffers", "_key_buffers")


def _cow(a):
    return None if a is None else a[...]


def copy_layer(dst, src, l: int) -> None:
    """Point the layer-l slots of cache `dst` at (copy on write views of) the layer-l slots of `src`."""
    for name in _SLOTS:
        d, s = getattr(dst, name), getattr(src, name)
        if hasattr(d, "raw_arrays"):                                            # packed fp4 slots: move the codes, no unpack / repack
            d.set_raw(l, d.cow(s.raw(l)))
        else:
            d[l] = _cow(s[l])


class ChainCache:
    """The caches R_0 .. R_n of a chained verify pass (see module doc). Quacks like BatchDeepseekV41Cache for the model loop.
    `starts`: indices of the rows that begin a segment (one sequence = one segment; a server pass holds one segment per session). Only the
    rows inside a segment see each other's state: `cont[i]` is True when row i takes over row i - 1."""

    def __init__(self, rows: list, link: bool = True, base=None, starts=(0,)):
        self.rows = list(rows)
        self.link = link                                    # False = negative control: rows do not see each other's state
        self.base = base                                    # per row: the cache its pass segment started from (a segment's first row, and every row without link, takes its layer slots from it)
        self.starts = list(starts)
        self.cont = [link and i not in self.starts for i in range(len(self.rows))]
        self.taps: dict = {}
        self.left_padding = [0] * len(rows)
        self._lengths = [None] * len(rows)

    @property
    def offset(self):
        return mx.array([r.offset for r in self.rows])

    def advance(self, width: int) -> None:
        for r in self.rows:
            r.offset += width

    def map(self, function, *arrays):
        raise NotImplementedError("ChainCache.map: the engram hashes of a chain are computed by spec.engram_hashes")

    def take_layer(self, j: int, l: int) -> None:
        """Re-point the layer-l slots of row j just before its layer-l core runs: row j takes (copy on write views of) the slots row j - 1 left after its
        own layer-l core, the first row of a segment (and every row of an unlinked chain) the slots of its base cache."""
        copy_layer(self.rows[j], self.rows[j - 1] if self.cont[j] else self.base[j], l)


def shell(base):
    """A cache that starts as `base` without a view of every layer slot: the slot lists are new lists holding base's arrays, the other state (offset, engram
    history, handoff) is forked. A chain re-points each layer slot (take_layer) just before that layer's core runs for the row, so the n x layers x slots
    views of an up front fork are not built; a slot of a layer that never runs for the row stays base's array, which nothing writes."""
    import copy
    new = copy.copy(base)
    for k, v in vars(base).items():
        setattr(new, k, (v.copy() if hasattr(v, "raw_arrays") else list(v)) if k in _SLOTS else R.fork(v))
    return new


def make_chain(base, n: int, link: bool = True) -> ChainCache:
    """`base`: the model's cache list (one DeepseekV41Cache entry) or the entry itself."""
    return make_chains([base], [n], link)


def make_chains(bases: list, ns: list, link: bool = True) -> ChainCache:
    """One pass over several sequences: segment k = ns[k] consecutive positions after the cache `bases[k]` (not modified). Rows are laid
    out segment after segment."""
    rows, starts, rbase = [], [], []
    for base, n in zip(bases, ns):
        base = base[0] if isinstance(base, list) else base
        starts.append(len(rows))
        for j in range(n):
            r = shell(base)
            r.offset = base.offset + j
            rows.append(r)
            rbase.append(base)
    return ChainCache(rows, link, rbase, starts)


# --------------------------------------------------------------------------------------------------------------- DS4.1 chain adapter
def dsv4_attention_chain(attn, x: mx.array, chain: ChainCache) -> mx.array:
    """The decode step of the V4.1 attention for n chained positions (x [n, 1, H]). Same structure as rows.dsv4_attention_rows
    (projections once for all rows, the core of a row is the model's own `_attention` over that row's cache with the projection outputs
    replayed, wo_a / wo_b once), plus the chain link: before the core of row j its layer slots come from row j - 1."""
    fq = R._lang(attn).fake_quant_fp8_ue8m0
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
        for i, row in enumerate(chain.rows):
            chain.take_layer(i, attn.layer_idx)
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
    o = mx.concatenate(grabbed, axis=0)                                           # [n, G, 1, in]
    o = s["wo_a"].inner(o).transpose(0, 2, 1, 3).flatten(-2)
    return s["wo_b"].inner(fq(o))


def _install_dsv4_chain(model, spec: ArchSpec) -> int:
    AttnCls = _A.cls("attn")
    n = 0
    for layer in model.language_model.layers:
        attn = getattr(layer, spec.rows.attn_attr)
        if not isinstance(attn, AttnCls) or getattr(attn, "_kyojin_chain", False):
            continue
        cls = type(attn)

        def __call__(self, x, cache, _prev=cls.__call__):
            if isinstance(cache, ChainCache):
                return dsv4_attention_chain(self, x, cache)
            return _prev(self, x, cache)

        attn.__class__ = type(cls.__name__, (cls,), {"__call__": __call__, "__module__": cls.__module__})
        attn._kyojin_chain = True
        n += 1
    return n


def _install_dsv4_chain_over_core(model, spec: ArchSpec) -> int:
    """The batched core (adapter "dsv4_core") serves BatchDeepseekV41Cache steps; the chained verify runs the model's own `_attention` per
    row, with the projections hoisted: add the pass-through hoist / sink wrappers (no dispatch change), then the chain hook."""
    R._install_dsv4_hoist(model, spec, dispatch=False)
    return _install_dsv4_chain(model, spec)


CHAIN_ADAPTERS = {"dsv4_hoist": _install_dsv4_chain, "dsv4_core": _install_dsv4_chain_over_core}


# --------------------------------------------------------------------------------------------------------------- taps
def install_taps(model, layer_ids, keep: int = 128) -> int:
    """Before layer l runs (after its Engram), store the mean of the hyper-connection streams of its INPUT in `cache.taps[l]` when the
    cache entry has a `taps` dict: the hidden state the drafter reads (a DSpark head taps the attention input of its target layers).
    Shape [rows, positions, H]; a call with more than `keep` positions (prefill) keeps the last `keep` of them."""
    n = 0
    for l in sorted(set(layer_ids)):
        blk = model.language_model.layers[l]
        if getattr(blk, "_kyojin_tap", False):
            continue
        cls = type(blk)

        def __call__(self, h, pre_mix, image_mask, cache, _prev=cls.__call__, _keep=keep):
            tp = getattr(cache, "taps", None)
            if tp is not None:
                tp[self.layer_idx] = h[:, -getattr(cache, "taps_keep", _keep):].mean(axis=2)          # taps_keep: flag pf_cut_merge keeps more rows for one pass
            return _prev(self, h, pre_mix, image_mask, cache)

        blk.__class__ = type(cls.__name__, (cls,), {"__call__": __call__, "__module__": cls.__module__})
        blk._kyojin_tap = True
        blk._kyojin_tap_keep = keep
        n += 1
    if n:
        from . import bound_layers as BL
        BL.rewrap(getattr(model, "language_model", model))         # the tap classes are layer calls too: one bound per layer, not one per wrapper
    return n


GROUPED = os.environ.get("KYOJIN_MTP_GROUPED", "1") != "0"             # chained verify rows (2..5): one trellis decode per distinct expert across the rows


def _group_experts(model) -> int:
    """Route the multi-row MoE calls (the chained verify rows, batched decode) through the grouped expert path (bitwise equal to the per-row path);
    a MoE whose `grouped` was set by rows.enable(grouped=...) or KYOJIN_ROWS_GROUPED keeps it."""
    n = 0
    for _, m in model.language_model.named_modules():
        moe = getattr(m, "moe", None)
        if GROUPED and isinstance(moe, MoeGather) and not moe.grouped:
            moe.grouped = True
            moe._p.clear()
            n += 1
    return n


def install(model, spec: ArchSpec, tap_layers=(), keep: int = 128) -> dict:
    """Make the enabled (rows.enable) model able to verify chained positions. Idempotent."""
    assert spec.rows is not None, f"{spec.name}: no multi-user table"
    lm = model.language_model
    assert getattr(lm, "_kyojin_rows", None) is not None, "call rows.enable(model, spec) first"
    adapter = lm._kyojin_rows.get("adapter", spec.rows.adapter)
    assert adapter in CHAIN_ADAPTERS, f"{spec.name}: no chained verify for adapter {adapter}"
    rep = dict(chain_attention=CHAIN_ADAPTERS[adapter](model, spec), taps=install_taps(model, tap_layers, keep))
    rep["moe_grouped"] = _group_experts(model)
    return rep


def prefill(model, ids: list, tap_layers=(), chunk: int | None = None, keep: int = 128):
    """Solo prefill that also returns the taps of the last `keep` positions: (cache, logits [1, V], {layer: [positions, H]}, first position).
    chunk: prompt chunk size (None = one pass, bit-identical to rows.prefill); the taps of all chunks are joined before the tail is kept."""
    lm = model.language_model
    cache = lm.make_cache()
    step = chunk or max(len(ids), 1)
    parts: dict = {}
    with WC.prefill_scope(model, step, -(-len(ids) // step)):                   # dense weights decoded once for the whole prompt
        for s in range(0, len(ids), step):
            cache[0].taps = {}
            with HL.last_row():                                                 # only the last row is read
                lg = lm(mx.array(ids[s:s + step])[None], cache=cache).logits[:, -1].astype(mx.float32)
            mx.eval(lg)
            for l, t in cache[0].taps.items():
                parts.setdefault(l, []).append(t[0])
    del cache[0].taps
    taps = {l: (v[0] if len(v) == 1 else mx.concatenate(v, axis=0))[-keep:] for l, v in parts.items()}
    n = taps[next(iter(taps))].shape[0] if taps else 0
    return cache, lg, taps, len(ids) - n


# --------------------------------------------------------------------------------------------------------------- verify
def engram_hashes(model, chain: ChainCache, tokens: list):
    """Engram n-gram hashes of the positions of a chain: row j hashes over the history row j - 1 left (fed one token at a time, each row
    forked from the previous one after its own update). None when the model has no Engram layer."""
    lm = model.language_model
    lm._ensure_engram_hash()
    if lm.engram_hash is None:
        return None
    out = []
    for j, row in enumerate(chain.rows):
        if chain.cont[j]:
            row.engram = chain.rows[j - 1].engram                       # arrays are values: the history row j - 1 just wrote
        out.append(lm.engram_hash(mx.array(tokens[j:j + 1])[None], row.offset, row, token_mask=None))
    return mx.concatenate(out, axis=0)


def commit(chain: ChainCache, accepted: int) -> list:
    """The model cache list after `accepted` drafts were kept: the cache of row `accepted` (a solo run over the pending token + drafts)."""
    return [chain.rows[accepted]]


def _run_chain(model, chain: ChainCache, tokens: list):
    lm = model.language_model
    kw = {}
    eh = engram_hashes(model, chain, tokens) if getattr(lm, "layout", None) is not None else None
    if eh is not None:
        kw["engram_hashes"] = eh
    out = lm(mx.array(tokens, dtype=mx.int32)[:, None], cache=[chain], **kw)
    return out.logits[:, -1].astype(mx.float32), chain.taps, chain


def verify(model, base, tokens: list, link: bool = True):
    """One batched pass over `tokens` (t0 = the pending token, then the drafts) from cache `base` (not modified).
    Returns (fp32 logits [n, vocab], taps {layer: [n, 1, H]} or {}, chain); row i of the logits is the logits after tokens[i]."""
    return _run_chain(model, make_chain(base, len(tokens), link), tokens)


def verify_many(model, bases: list, token_lists: list, link: bool = True):
    """verify() for several sequences in ONE pass (a server step: every active session contributes its pending token and its drafts).
    Returns (logits [sum n_k, vocab], taps {layer: [sum n_k, 1, H]}, chain); segment k owns the rows chain.starts[k] .. + n_k - 1, and its
    cache after `a` accepted drafts is `commit_seg(chain, k, a)`. Every row equals the solo step of its position bit for bit, whatever the
    other segments are (the rows of different segments never see each other)."""
    chain = make_chains(bases, [len(t) for t in token_lists], link)
    return _run_chain(model, chain, [t for toks in token_lists for t in toks])


def commit_seg(chain: ChainCache, k: int, accepted: int) -> list:
    """The model cache list of segment k after `accepted` drafts were kept."""
    return [chain.rows[chain.starts[k] + accepted]]


# --------------------------------------------------------------------------------------------------------------- generate
def plain_greedy(model, prompt: list, n: int, chunk: int | None = None) -> list:
    """The reference: n greedy tokens by solo decode steps of the (rows-enabled) model (same prefill chunking as the speculative run)."""
    cache, lg, _, _ = prefill(model, prompt, chunk=chunk)
    tok = int(mx.argmax(lg[0]).item())
    out = [tok]
    while len(out) < n:
        lg = R.step(model, cache, mx.array([tok]))
        tok = int(mx.argmax(lg[0]).item())
        out.append(tok)
    return out


LAZY_APPEND = os.environ.get("KYOJIN_DSPARK_LAZY", "1") != "0"


def chain_greedy(model, prompt: list, n: int, chunk: int | None = None) -> list:
    """n greedy tokens, one position per chained verify pass: the same ops as the speculative decode with D = 0 (the model's own `_attention`
    per row). With the batched DS4.1 core (adapter "dsv4_core") `plain_greedy` rounds differently from this path (logit diffs ~0.1 on the
    slice, a different token where two logits are that close), so the speculative decode is exact against this one."""
    cache, lg, _, _ = prefill(model, prompt, chunk=chunk)
    tok = int(mx.argmax(lg[0]).item())
    out = [tok]
    while len(out) < n:
        lgs, _, chain = verify(model, cache, [tok])
        cache = commit(chain, 0)
        tok = int(mx.argmax(lgs[0]).item())
        out.append(tok)
    return out


def reference_greedy(model, prompt: list, n: int, chunk: int | None = None) -> list:
    """The solo decode the speculative decode must equal token for token: the chain path when plain steps run on the batched core, else the
    plain steps (their attention is the stock one then, the chain's too)."""
    batched_core = (getattr(model.language_model, "_kyojin_rows", None) or {}).get("adapter") == "dsv4_core"
    return (chain_greedy if batched_core else plain_greedy)(model, prompt, n, chunk)


def stream(model, drafter, prompt: list, depth: int = 3, stats: dict | None = None, chunk: int | None = None):
    """Greedy speculative decode as a generator: yields the list of tokens each round emits (the first yield is the first token, from the prefill).
    A round drafts up to `depth` tokens, verifies them with the pending token in one chained pass, keeps the longest prefix the trunk agrees with
    and the trunk's own token after it (the accept rule of the CUDA / ROCm engine, greedy). The tokens are the ones plain greedy decode would emit.
    All state of the session (cache, pending token, draft ring) lives in the generator, so several sessions can share one model and one drafter.
    stats (optional dict): rounds, drafted, accepted, skipped, p_end, t_draft, t_verify, t_append (seconds, wall clock incl. sync)."""
    import time
    st = stats if stats is not None else {}
    for k in ("rounds", "drafted", "accepted", "skipped"):
        st[k] = 0
    st["t_draft"] = st["t_verify"] = st["t_append"] = 0.0
    ctx = drafter.new_ctx()
    drafter.ctx = ctx
    cache, lg, taps, first = prefill(model, prompt, drafter.tap_ids, chunk)
    drafter.append([taps[l] for l in drafter.tap_ids], first)
    pending = int(mx.argmax(lg[0]).item())
    st["p_end"] = cache[0].offset
    yield [pending]
    while True:
        drafter.ctx = ctx
        p0 = cache[0].offset
        t0 = time.perf_counter()
        drafts = drafter.draft(pending, p0, depth) if depth > 0 else []
        t1 = time.perf_counter()
        toks = [pending] + drafts
        lgs, tp, chain = verify(model, cache, toks)
        am = mx.argmax(lgs, axis=-1).tolist()
        t2 = time.perf_counter()
        a = 0
        while a < len(drafts) and am[a] == drafts[a]:
            a += 1
        cache = commit(chain, a)
        drafter.append([tp[l][:a + 1, 0] for l in drafter.tap_ids], p0)
        if not LAZY_APPEND or depth <= 0:
            mx.eval(ctx.rows)                                    # else the rows are computed inside the next draft's eval: one host sync fewer per round (no draft = nothing consumes them)
        t3 = time.perf_counter()
        pending = am[a]
        st["rounds"] += 1
        st["drafted"] += len(drafts)
        st["accepted"] += a
        st["skipped"] += int(not drafts)
        st["p_end"] = p0 + a + 1
        st["t_draft"] += t1 - t0
        st["t_verify"] += t2 - t1
        st["t_append"] += t3 - t2
        yield drafts[:a] + [am[a]]


def generate(model, drafter, prompt: list, n: int, depth: int = 3, stats: dict | None = None, chunk: int | None = None) -> list:
    """n greedy tokens by speculative decode (see `stream`); token identical to reference_greedy."""
    out = []
    for new in stream(model, drafter, prompt, depth, stats, chunk):
        out += new
        if len(out) >= n:
            return out[:n]
