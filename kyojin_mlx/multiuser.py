# SPDX-License-Identifier: Apache-2.0
"""Multi-user batched decode on top of mlx-lm models (Qwen3.5 / Qwen3-Next family): N users, one forward per step.

Dense and attention projections see one [N, d] activation (the EXL3 multi-row GEMV decodes every weight once for all rows).
Row-independent ops (norms, GDN recurrence, conv) run batched. The only per-user part is the attention core of the full
attention layers: every user owns a private `KVCache` (own length, no left padding), so rope / cache write / SDPA run per
row exactly as in a solo run. Results are bit-identical to solo decode by construction (see tests/test_multiuser.py).

    users = [prefill(model, ids_i) for i in range(N)]          # solo prefill, or fork a shared prefix: fork_cache()
    bc = batch_caches(users)
    tokens = decode_batched(model, bc, first_tokens, n)         # list of N lists

The recurrent (GDN) states of the users are stacked once on the batch axis and stay stacked for the whole decode."""
from __future__ import annotations

import functools
import inspect
import os
import sys
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from mlx_lm.models import qwen3_next
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

from . import kvwin
from .kvwin import WinKVCache, make_prompt_cache                       # make_prompt_cache: KYOJIN_KVWIN=k:v:g:sink:window swaps in the quantized + fp16 window cache (default off)

from . import wcache as WC
from . import mu_attn as MA


class RowsKV:
    """Full-attention cache of a batch of users: one private KVCache per user. Decode only (one token per user)."""

    def __init__(self, caches: list[KVCache]):
        self.caches = list(caches)

    @property
    def offset(self) -> mx.array:                       # per-row rope offsets (mx.fast.rope takes a vector; bit-identical to int)
        return mx.array([c.offset for c in self.caches])

    def make_mask(self, N: int, **kw):
        assert N == 1, "RowsKV is decode only (one token per user)"
        return None

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        ks, vs = [], []
        for i, c in enumerate(self.caches):
            k, v = c.update_and_fetch(keys[i:i + 1], values[i:i + 1])
            ks.append(k)
            vs.append(v)
        return ks, vs

    def unbatch(self) -> list[KVCache]:
        return self.caches


_ROWDOT = r"""
    // y[r, o] = sum_k x[r, k] * w[o, k]: one simdgroup per output o, all MR rows at once (the weight row is read once).
    // The summation order of a row does not depend on MR: lane l takes k = 8 l + 256 i + j (j = 0..7 in order, fp32 fma),
    // then one simd_sum per row. Row r of an MR-row launch is bit-identical to the same row alone.
    const uint o = thread_position_in_grid.x / 32;
    const uint lane = thread_index_in_simdgroup;
    const uint K = params[0], N = params[1];
    if (o >= N) return;
    float acc[MR];
    for (uint r = 0; r < MR; ++r) acc[r] = 0.0f;
    const device half* wp = w + ulong(o) * K;
    for (uint kk = lane * 8; kk < K; kk += 256) {
        float wv[8];
        for (uint j = 0; j < 8; ++j) wv[j] = (kk + j < K) ? float(wp[kk + j]) : 0.0f;
        for (uint r = 0; r < MR; ++r) {
            const device half* xp = x + ulong(r) * K + kk;
            for (uint j = 0; j < 8; ++j) acc[r] = fma(wv[j], (kk + j < K) ? float(xp[j]) : 0.0f, acc[r]);
        }
    }
    for (uint r = 0; r < MR; ++r) {
        const float t = simd_sum(acc[r]);
        if (lane == 0) y[ulong(r) * N + o] = half(t);
    }
"""
_rowdot_kern = None
ROWDOT_MAX = 8


def rowdot(x: mx.array, w: mx.array) -> mx.array:
    """x [B, K] fp16, w [N, K] fp16 -> [B, N] fp16, row-invariant (see _ROWDOT). B > ROWDOT_MAX is split into launches."""
    global _rowdot_kern
    if _rowdot_kern is None:
        _rowdot_kern = mx.fast.metal_kernel(name="kyojin_rowdot", input_names=["x", "w", "params"], output_names=["y"],
                                            source=_ROWDOT, header="#pragma METAL fp contract(off)\n")
    B, K = x.shape
    N = w.shape[0]
    if B > ROWDOT_MAX:
        return mx.concatenate([rowdot(x[i:i + ROWDOT_MAX], w) for i in range(0, B, ROWDOT_MAX)], axis=0)
    return _rowdot_kern(inputs=[x, w, mx.array([K, N], dtype=mx.uint32)], template=[("MR", B)], grid=(N * 32, 1, 1),
                        threadgroup=(64, 1, 1), output_shapes=[(B, N)], output_dtypes=[mx.float16])[0]


class RowLinear(nn.Module):
    """Plain fp16 linear (router, tiny projections) whose decode output does not depend on the number of users.
    mx.matmul of [N, k] x [k, n] picks other kernels for other N (about 2 % of rows differ in the last bit, even 1-row 3-D vs
    2-D inputs differ), so decode steps (one token per user) use `rowdot`; prompts (S > 1) use the stock matmul."""

    def __init__(self, lin: nn.Linear):
        super().__init__()
        self.weight = lin.weight
        if "bias" in lin:
            self.bias = lin.bias

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim == 3 and x.shape[1] == 1 and x.dtype == mx.float16 and self.weight.dtype == mx.float16:
            y = rowdot(x[:, 0], self.weight)[:, None]
        else:
            y = x @ self.weight.T
        return y + self.bias if "bias" in self else y


def make_row_invariant(model) -> int:
    """Replace every non-EXL3 nn.Linear of the model by a RowLinear. Returns the number replaced."""
    from .exl3_linear import Exl3Linear
    mods = [(n, RowLinear(m)) for n, m in model.named_modules() if type(m) is nn.Linear and not isinstance(m, Exl3Linear)]
    model.update_modules(tree_unflatten(mods))
    return len(mods)


_orig_sdpa = None
_mu_sdpa = None


def configure(mode: str | None = None) -> str:
    """Pick the EXL3 linear path of the batched decode rows (env KYOJIN_MU_MODE, default "default"):
    exact   rows are bitwise equal to the solo run: <= 4 rows the fused GEMV, 5..16 rows ONE 8-row GEMV launch (ROW_SPLIT_MAX=16, no NAX);
    mma     speed: 4 rows on the packed-fp16 GEMV, 5..16 rows on the simdgroup_matrix GEMV (not bitwise; KL-gated);
    default the stock thresholds (<= 4 rows GEMV, 5..8 the simdgroup_matrix GEMV, above: fused NAX split along k)."""
    from . import exl3_linear as EL
    mode = mode or os.environ.get("KYOJIN_MU_MODE", "default")
    if mode == "exact":
        EL.ROW_SPLIT_MAX, EL.ROWS_NAX, EL.MMA_ROWS = 4 * EL.KN.MAX_ROWS, False, 0
    elif mode == "mma":
        EL.ROW_SPLIT_MAX, EL.MMA_ROWS = 0, EL.MMA_ROWS if EL.MMA_ROWS > 0 else 4
        if "KYOJIN_HALF_ROWS" not in os.environ:
            EL.KN.S.HALF_ROWS = 4                                                    # 4 rows: packed-fp16 FMAs in the fused GEMV (-6 % at 4 users, KL 1e-5; 2..3 rows gain nothing)
    elif mode != "default":
        raise ValueError(mode)
    return mode


FOLD = os.environ.get("KYOJIN_GQA_FOLD", "1") == "1"       # verify rows (2..FOLD_MAX query rows): the G rows of a kv head x T rows go to the SDPA as ONE long query block
FOLD_MAX = 8
_fold_mask: dict = {}


def fold_sdpa(q, k, v, scale):
    """Causal-tail SDPA of T query rows per head over N keys (the last T keys are the rows themselves). MLX's vector kernel (q_len <= 8) reads the whole
    K / V once per query row and per query head; folding the G = Hq / Hk heads of a kv head and the T rows into the query axis ([B, Hk, G * T, D],
    row g * T + t) makes it one blocked-attention pass per kv head that reads K / V once, with an explicit additive mask for the causal tail."""
    B, Hq, T, D = q.shape
    Hk, N = k.shape[1], k.shape[2]
    G = Hq // Hk
    pad = max(0, 9 - G * T)                  # MLX takes its vector kernel (one pass over K / V per row) up to 8 query rows: pad to the blocked kernel
    key = (N, T, G, q.dtype, pad)
    m = _fold_mask.get(key)
    if m is None:
        pos = mx.arange(N)[None, :]
        lim = (N - T + mx.arange(T))[:, None]
        m = mx.tile(mx.where(pos <= lim, 0.0, -mx.inf).astype(q.dtype), (G, 1))
        if pad:
            m = mx.concatenate([m, mx.zeros((pad, N), dtype=m.dtype)], axis=0)
        m = m[None, None]
        if len(_fold_mask) > 8:
            _fold_mask.clear()
        _fold_mask[key] = m
    qf = q.reshape(B, Hk, G * T, D)
    if pad:
        qf = mx.concatenate([qf, mx.zeros((B, Hk, pad, D), dtype=q.dtype)], axis=2)
    o = mx.fast.scaled_dot_product_attention(qf, k, v, scale=scale, mask=m)
    return o[:, :, :G * T].reshape(B, Hq, T, D)


POOL_MIN = int(os.environ.get("KYOJIN_MU_POOL_MIN", "256"))   # shortest shared prefix worth a PoolKV: below it the tail copies + prefix check cost more
                                                                # than reading the prefix once saves (8 users sharing only the ~20-token chat template: N8 -7..-26 %)
_NO_POOL = False          # set by install() for models whose attention has learned sinks (gpt-oss): the pooled kernel has no sink term, RowsKV does


def install(model=None) -> int:
    """Teach the mlx-lm Qwen attention to run its SDPA per user when its cache is a RowsKV, and make the model's dense linears
    row-invariant (idempotent). Returns the number of linears replaced."""
    global _orig_sdpa, _mu_sdpa, _NO_POOL
    if model is not None and any(hasattr(m, "sinks") for _, m in model.named_modules()) and os.environ.get("KYOJIN_MU_POOL_SINKS", "1") == "0":
        _NO_POOL = True                                       # KYOJIN_MU_POOL_SINKS=0: sink models stay on RowsKV (the pooled merge adds the sink term by default)
    from . import sdpa_dec as SD
    n = make_row_invariant(model) if model is not None else 0
    SD.install()
    from . import tq_step
    tq_step.install(model)
    if _orig_sdpa is not None:
        _route_module(model)
        return n
    _orig_sdpa = qwen3_next.scaled_dot_product_attention

    def _row_sdpa(q, k, v, scale, sinks=None):
        out = SD.route(q, k, v, None, scale, None, sinks)
        return out if out is not None else mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=None, sinks=sinks)

    def sdpa(queries, keys, values, cache, scale, mask, sinks=None):
        if isinstance(cache, MA.PoolKV):
            return cache.attend(queries, scale, sinks)
        if isinstance(cache, RowsKV):
            if sinks is not None and any(isinstance(kk, kvwin.KVView) for kk in keys):
                raise ValueError("attention sinks on a quantized KV window (KYOJIN_KVWIN / TurboQuant) are not supported: serve with --kv-mode fp16")
            outs = [keys[i].cache.attend(queries[i:i + 1], scale) if isinstance(keys[i], kvwin.KVView)
                    else _row_sdpa(queries[i:i + 1], keys[i], values[i], scale, sinks) for i in range(len(keys))]
            return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
        out = SD.route(queries, keys, values, cache, scale, mask, sinks)       # split-K decode / verify attention (sdpa_dec.py)
        if out is not None:
            return out
        if FOLD and sinks is None and 2 <= queries.shape[2] <= FOLD_MAX and isinstance(mask, str) and not hasattr(cache, "bits") and queries.shape[1] > keys.shape[1]:
            return fold_sdpa(queries, keys, values, scale)
        return _orig_sdpa(queries, keys, values, cache=cache, scale=scale, mask=mask, sinks=sinks)

    qwen3_next.scaled_dot_product_attention = _mu_sdpa = sdpa
    _route_module(model)
    return n


def _route_module(model) -> None:
    """The per-user attention also serves the module of a text-only class (Qwen3, Qwen3-MoE read as stored)."""
    mod = sys.modules.get(type(model).__module__) if model is not None else None
    if mod is not None and hasattr(mod, "scaled_dot_product_attention"):
        mod.scaled_dot_product_attention = _mu_sdpa
    if model is not None and os.environ.get("KYOJIN_VISIONGEN", "0") != "0" and "language_model" in model:     # flag visiongen: an mlx-vlm wrapper (llava ...) keeps its attention in the text module's own sdpa
        for _, m in model.language_model.named_modules():
            sub = sys.modules.get(type(m).__module__)
            if sub is not None and hasattr(sub, "scaled_dot_product_attention") and sub.scaled_dot_product_attention is not _mu_sdpa:
                sub.scaled_dot_product_attention = _mu_sdpa


# ---------------------------------------------------------------------------------------------- caches
def _dup(x):
    """Structure-preserving copy of a cache state: arrays are values (shared until written), lists / tuples are rebuilt."""
    if isinstance(x, (list, tuple)):
        return type(x)(_dup(v) for v in x) if not hasattr(x, "_fields") else type(x)(*(_dup(v) for v in x))
    return x[...] if isinstance(x, mx.array) else x


def _fork_any(c):
    """KYOJIN_GENERIC_CACHE: any mlx-lm cache object (its own state / meta_state round trip), CacheList included."""
    if hasattr(c, "caches") and hasattr(type(c), "from_state"):
        return type(c)(*_fork_any_list(c.caches))
    if hasattr(type(c), "from_state"):
        return type(c).from_state(_dup(c.state), c.meta_state)
    raise TypeError(f"fork_cache: unsupported cache {type(c).__name__} (no from_state)")


def _fork_any_list(cs) -> list:
    return [_fork_any(x) if type(x).__name__ not in ("KVCache", "ArraysCache") else fork_cache([x])[0] for x in cs]


def fork_cache(cache: list) -> list:
    """Independent copy of a prompt cache (shared-prefix fork): same state, no recompute. MLX arrays are values, so the copies
    share buffers until one of them writes (copy on write). Each forked KVCache records its fork point (`mu_prefix`: (source tag, length) pairs,
    nested forks included): batch_caches reads them to store and read the shared prefix once (PoolKV). Assumes a fork never trims below its fork
    point (batch_caches also checks the last prefix key of every user)."""
    out = []
    generic = os.environ.get("KYOJIN_GENERIC_CACHE", "0") != "0"
    for c in cache:
        if generic and type(c) is not KVCache and not isinstance(c, (WinKVCache, ArraysCache)):
            n = _fork_any(c)                                          # a subclass of KVCache keeps its class (plain KVCache() would drop its own update path)
        elif isinstance(c, KVCache):
            n = KVCache()
            n.keys = c.keys[...] if c.keys is not None else None
            n.values = c.values[...] if c.values is not None else None
            n.offset = c.offset
            if c.offset:
                n.mu_prefix = getattr(c, "mu_prefix", ()) + ((c.__dict__.setdefault("mu_tag", object()), c.offset),)
        elif isinstance(c, WinKVCache):
            n = c.fork()
            if c.offset:                                              # same fork record as a KVCache: the pool keys below the fork point are shared (tq.attend_shared)
                n.mu_prefix = getattr(c, "mu_prefix", ()) + ((c.__dict__.setdefault("mu_tag", object()), c.offset),)
        elif isinstance(c, RotatingKVCache):                          # sliding-window layers (gemma3 ...): the ring, its write position and its counters; arrays are values
            n = RotatingKVCache(c.max_size, c.keep)
            n.keys = c.keys[...] if c.keys is not None else None
            n.values = c.values[...] if c.values is not None else None
            n.offset, n._idx = c.offset, c._idx
        elif isinstance(c, ArraysCache):
            n = ArraysCache(len(c.cache))
            n.cache = [x[...] if x is not None else None for x in c.cache]
        elif os.environ.get("KYOJIN_GENERIC_CACHE", "0") != "0":
            n = _fork_any(c)
        else:
            raise TypeError(f"fork_cache: unsupported cache {type(c).__name__}")
        out.append(n)
    return out


def shared_prefix(col: list) -> int:
    """Length of the prefix the caches of one layer share by construction (fork_cache records), 0 if none."""
    chains = [dict(getattr(c, "mu_prefix", ())) for c in col]
    return max([min(ch[t] for ch in chains) for t in chains[0] if all(t in ch for ch in chains)], default=0)


def _same_prefix(col: list, P: int) -> bool:
    k = mx.stack([c.keys[0, :, P - 1] for c in col])
    return bool(mx.all(k == k[:1]).item())


def _mla_call(self, x, mask=None, cache=None):
    """MLA attention (deepseek_v3 class: DeepSeek V2 / V3, Kimi, Moonlight, GLM-4.7-Flash lite) under a RowsKV: one user at a time through the stock
    call on its own KVCache (the latent cache is multiplied as one array, q_pe @ k_pe.T); the MoE / MLP and the other layers stay batched."""
    if isinstance(cache, RowsKV):
        outs = [self._mla_stock(x[i:i + 1], None, c) for i, c in enumerate(cache.caches)]
        return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
    return self._mla_stock(x, mask, cache)


def install_mla(model) -> int:
    """Swap the call of every MLA attention module (kv_a_proj_with_mqa + embed_q). Returns the number of modules."""
    n = 0
    for _, mod in model.named_modules():
        if hasattr(mod, "kv_a_proj_with_mqa") and hasattr(mod, "embed_q"):
            cls = mod.__class__
            if not getattr(cls, "_mla_rows", False):
                mod.__class__ = type("Rows" + cls.__name__, (cls,), {"__call__": _mla_call, "_mla_stock": cls.__call__, "_mla_rows": True})
            n += 1
    return n


def batch_caches(users: list[list], prefix_len: int | None = None) -> list:
    """Per-user prompt caches (each a list over layers) -> one cache list over layers for a batched decode step. Users forked (fork_cache) from one
    prompt cache share its prefix (prefix_len: None = from the fork records, 0 = off): with >= 2 users their full-attention layers become one
    PoolKV (prefix stored and read once per step for all users, mu_attn.py) where its kernel runs, else one private KVCache per user (RowsKV)."""
    out = []
    generic = os.environ.get("KYOJIN_GENERIC_CACHE", "0") != "0"
    for col in zip(*users):
        col = list(col)
        P = (shared_prefix(col) if prefix_len is None else prefix_len) if len(col) > 1 and type(col[0]) is KVCache else 0
        if isinstance(col[0], ArraysCache):
            out.append(ArraysCache.merge(col))
        elif generic and type(col[0]) is not KVCache and not isinstance(col[0], (KVCache, WinKVCache)) and hasattr(type(col[0]), "merge"):
            out.append(type(col[0]).merge(col))                       # KYOJIN_GENERIC_CACHE: the cache's own merge (mlx-vlm ArraysCache, CacheList, rotating)
        elif P >= max(POOL_MIN, 1) and not _NO_POOL and MA.supported(col, P) and _same_prefix(col, P):
            out.append(MA.PoolKV(col, P))
        else:
            out.append(RowsKV(col))
    return out


def unbatch_caches(bc: list, n: int) -> list[list]:
    """Inverse of batch_caches (e.g. to move a user out of the batch): list over users of per-layer caches."""
    users = [[] for _ in range(n)]
    for c in bc:
        if isinstance(c, (RowsKV, MA.PoolKV)):
            for u, k in zip(users, c.unbatch()):
                u.append(k)
        else:
            for i in range(n):
                u = c.extract(i)
                users[i].append(u)
    return users


# ---------------------------------------------------------------------------------------------- forward
def logits_last(model, tokens: mx.array, cache) -> mx.array:
    """Logits of the last position of each row (the head sees one row per user, never the whole prompt)."""
    lm = model.language_model if "language_model" in model else model      # a text-only mlx-lm class (Qwen3 ...) has the head at the top
    t = _trunk(lm)
    h = (t(tokens, cache=cache) if _kw_cache(type(t)) else t(tokens, cache))[:, -1:]
    return head(lm, h)[:, -1]


HEAD_ROUTE = os.environ.get("KYOJIN_MU_HEAD", "1") == "1"   # 0: tied heads go straight to embed_tokens.as_linear (pre-macmu100: skipped the q4mma few-row head)


@functools.lru_cache(maxsize=None)
def _kw_cache(cls) -> bool:
    """A trunk whose __call__ takes `inputs_embeds` before `cache` (the mlx-vlm text classes) must get the cache by name: positionally it lands in inputs_embeds."""
    return "inputs_embeds" in inspect.signature(cls.__call__).parameters


def _trunk(lm):
    """The layer stack of a text class: `.model` (Llama / Qwen / DeepSeek ...), `.backbone` (the Mamba family)."""
    return lm.model if "model" in lm else lm.backbone


def head(lm, h: mx.array) -> mx.array:
    """Output head of the language model: an installed head (q4mma.Q4Head on tied packs, set in lm.__dict__) wins over the tied embedding."""
    hd = lm.__dict__.get("lm_head") if HEAD_ROUTE else None
    if hd is not None:
        return hd(h)
    tie = getattr(getattr(lm, "args", None) or lm.config, "tie_word_embeddings", None)           # mlx-vlm text classes keep `config`, mlx-lm classes `args`
    if tie is None:                                               # gemma3_text keeps no such arg: its sanitize drops lm_head when the file has none
        tie = "lm_head" not in lm
    if tie:
        t = _trunk(lm)
        return (t.embed_tokens if "embed_tokens" in t else t.embeddings).as_linear(h)
    return lm.lm_head(h)


def prefill(model, ids: list[int], cache: list | None = None, chunk: int = 2048) -> tuple[list, mx.array]:
    """Solo prefill (optionally on top of a forked prefix cache). Returns (cache, last-position logits [1, V])."""
    cache = cache if cache is not None else make_prompt_cache(model)
    lg = None
    with WC.prefill_scope(model, chunk, -(-len(ids) // chunk)):                 # dense weights decoded once for the whole prompt
        for s in range(0, len(ids), chunk):
            lg = logits_last(model, mx.array(ids[s:s + chunk])[None], cache)
            mx.eval(lg)
    return cache, lg


def _run(step, first: mx.array, n: int, async_: bool) -> list[mx.array]:
    """Greedy loop: n tokens (first included). async_ = mlx-lm style, step k+1 is queued while the GPU runs step k."""
    toks, cur = [first], first
    steps = n - 1
    if steps <= 0:
        return toks
    if async_:
        nxt = step(cur)
        mx.async_eval(nxt)
        for i in range(steps):
            cur = nxt
            if i + 1 < steps:
                nxt = step(cur)
                mx.async_eval(nxt)
            mx.eval(cur)
            toks.append(cur)
    else:
        for _ in range(steps):
            cur = step(cur)
            mx.eval(cur)
            toks.append(cur)
    return toks


def decode_batched(model, bc: list, first: mx.array, n: int, async_: bool = True) -> list[list[int]]:
    """Greedy decode of N users for n tokens. first: [N] int tokens (argmax of each user's prefill logits).
    Returns the N lists of n generated tokens (first token included)."""
    toks = _run(lambda y: mx.argmax(logits_last(model, y[:, None], bc), axis=-1), first, n, async_)
    return [[int(t[i].item()) for t in toks] for i in range(first.shape[0])]


def decode_solo(model, cache: list, first: mx.array, n: int, async_: bool = False) -> list[int]:
    """Stock mlx-lm single-user greedy decode (the reference every batched row must reproduce). first: [1]."""
    toks = _run(lambda y: mx.argmax(logits_last(model, y[:, None], cache), axis=-1), first, n, async_)
    return [int(t[0].item()) for t in toks]
