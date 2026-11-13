# SPDX-License-Identifier: Apache-2.0
"""KVWIN: quantized KV cache + fp16 overlay (attention sinks + most recent window) for the mlx-lm full-attention layers.

Write-through design (same as the CUDA/HIP plan): the quantized pool (mx.quantize affine, `kbits`/`vbits`, `group_size`) holds EVERY
token, so fork / trim / save keep working on the pool alone. Next to it, a small fp16 overlay keeps the first `sink` tokens and the
last `window` tokens (the window includes the query's own key). Attention is decode-faithful: row p of a chunk sees the keys in
[p - window + 1, p] and the sinks at fp16 and everything else through the quantized pool, whatever the chunk size (the pool is
never overwritten, the fp16 values come from the overlay + the keys of the current chunk).

    KYOJIN_KVWIN=kbits:vbits:group:sink:window     e.g. 4:4:64:4:128   (unset / "0" / "off": plain fp16 KVCache, default)
    KYOJIN_KVWIN_F32=1                             fp32 scores/softmax/P@V (default 0 = fp16 like mlx-lm; KL unchanged within noise on Qwen3.5-2B)
    cache = kvwin.make_prompt_cache(model)         # drop-in for mlx_lm make_prompt_cache; honours the switch
    sink = window = 0 is exactly mlx-lm's QuantizedKVCache (bit-identical, tests/test_kvwin.py).

The attention core uses mx.quantized_matmul on the pool (no dequantized copy of the context) and two small fp16 matmuls for the
overlay; it is reached through mlx-lm's quantized-SDPA hook (the cache has `bits`/`group_size`, update_and_fetch returns views)."""
from __future__ import annotations

import os

import mlx.core as mx
from mlx_lm.models import base as _base
from mlx_lm.models.cache import KVCache, _BaseCache, make_prompt_cache as _mlx_make_prompt_cache

ENV = "KYOJIN_KVWIN"


class KVView:
    """What update_and_fetch returns: a handle on the cache (the attention core reads the pool and the overlay from it)."""

    def __init__(self, cache: "WinKVCache"):
        self.cache = cache


class WinKVCache(_BaseCache):
    step = 256

    def __init__(self, kbits: int = 4, vbits: int = 4, group_size: int = 64, sink: int = 4, window: int = 128, margin: int = 16, f32: bool = False):
        self.f32 = f32                                      # scores / softmax / P@V in fp32 (mx.quantized_matmul takes fp32 queries); default False = mlx-lm's fp16 path (no KL gain measured)
        self.kbits, self.vbits, self.group_size = kbits, vbits, group_size
        self.bits = kbits                                   # read by mlx-lm's scaled_dot_product_attention to choose the quantized hook
        self.sink, self.window, self.margin = sink, window, margin
        self.kq = self.vq = None                            # (packed, scales, biases) each [B, H, cap, .]
        self.offset = 0
        self.sk = self.sv = None                            # fp16 sink rows [B, H, <=sink, D]
        self.rk = self.rv = None                            # fp16 ring: last <= window + margin tokens [B, H, n, D]
        self._prev = None                                   # (rk, rv, prev offset) before the last update, for the chunk attention
        self._chunk = None

    # ------------------------------------------------------------------ write
    def _alloc(self, B, H, kd, vd, dtype, n):
        steps = (self.step + n - 1) // self.step * self.step

        def new(dim, bits):
            return (mx.zeros((B, H, steps, dim * bits // 32), dtype=mx.uint32),   # 5 / 6 bit: no integer number of values per word
                    mx.zeros((B, H, steps, dim // self.group_size), dtype=dtype),
                    mx.zeros((B, H, steps, dim // self.group_size), dtype=dtype))

        if self.kq is None:
            self.kq, self.vq = new(kd, self.kbits), new(vd, self.vbits)
        else:
            if self.offset % self.step != 0:
                self.kq = tuple(x[..., :self.offset, :] for x in self.kq)
                self.vq = tuple(x[..., :self.offset, :] for x in self.vq)
            self.kq = tuple(mx.concatenate([a, b], axis=-2) for a, b in zip(self.kq, new(kd, self.kbits)))
            self.vq = tuple(mx.concatenate([a, b], axis=-2) for a, b in zip(self.vq, new(vd, self.vbits)))

    def _cap(self):
        return self.kq[0].shape[-2]

    def _write_pool(self, prev, keys, values):
        for dst, src in ((self.kq, mx.quantize(keys, group_size=self.group_size, bits=self.kbits)),
                         (self.vq, mx.quantize(values, group_size=self.group_size, bits=self.vbits))):
            for i in range(3):
                dst[i][..., prev:self.offset, :] = src[i]

    def _pool_scores(self, q, L, rep, ex):
        """q (scaled) against the pool keys [0, L): [.., T, L] scores (a subclass with another pool format overrides this and _pool_pv)."""
        return mx.quantized_matmul(q, *ex(self.kq), transpose=True, group_size=self.group_size, bits=self.kbits)

    def _pool_pv(self, p, L, rep, ex):
        return mx.quantized_matmul(p, *ex(self.vq), transpose=False, group_size=self.group_size, bits=self.vbits)

    def update_and_fetch(self, keys, values):
        B, H, n, kd = keys.shape
        prev = self.offset
        if self.kq is None or prev + n > self._cap():
            self._alloc(B, H, kd, values.shape[-1], keys.dtype, n)
        self.offset += n
        self._write_pool(prev, keys, values)
        self._prev = (self.rk, self.rv, prev)
        self._chunk = (keys, values)
        if prev < self.sink:                                # first rows of the sequence
            m = min(self.sink - prev, n)
            self.sk = keys[..., :m, :] if self.sk is None else mx.concatenate([self.sk, keys[..., :m, :]], axis=-2)
            self.sv = values[..., :m, :] if self.sv is None else mx.concatenate([self.sv, values[..., :m, :]], axis=-2)
        cap = self.window + self.margin
        if self.window > 0:
            rk = keys if self.rk is None else mx.concatenate([self.rk, keys], axis=-2)
            rv = values if self.rv is None else mx.concatenate([self.rv, values], axis=-2)
            self.rk, self.rv = rk[..., -cap:, :], rv[..., -cap:, :]
        return KVView(self), KVView(self)

    # ------------------------------------------------------------------ attention
    def attend(self, queries, scale: float):
        B, Hq, T, D = queries.shape
        L = self.offset
        Hk = self.kq[0].shape[1]
        rep = Hq // Hk
        gs = self.group_size
        cdt = mx.float32 if self.f32 else queries.dtype
        q = (queries.astype(cdt) * scale)
        q = q.reshape(B, Hk, rep, T, D) if rep > 1 else q
        ex = (lambda t: tuple(mx.expand_dims(x[..., :L, :], -3) for x in t)) if rep > 1 else (lambda t: tuple(x[..., :L, :] for x in t))
        s = self._pool_scores(q, L, rep, ex)
        S = min(self.sink, L)
        win = self.window
        pos = mx.arange(L - T, L)[:, None]                  # absolute position of each query row
        if win > 0:
            rk, rv, prev = self._prev
            ck, cv = self._chunk
            fk = ck if rk is None else mx.concatenate([rk, ck], axis=-2)
            fv = cv if rv is None else mx.concatenate([rv, cv], axis=-2)
            c0 = max(S, L - fk.shape[-2])                   # first position of the fp16 tail (positions below S are sinks)
            fk, fv = fk[..., c0 - (L - fk.shape[-2]):, :], fv[..., c0 - (L - fv.shape[-2]):, :]
        else:
            c0 = L
        parts_s = []
        if S > 0:
            parts_s.append(q @ _rep(self.sk[..., :S, :].astype(cdt), rep).swapaxes(-1, -2))
        if c0 > S:
            parts_s.append(s[..., S:c0])
        if c0 < L:
            col = mx.arange(c0, L)[None, :]
            inwin = col > pos - win                         # row p: keys [p - win + 1, p] at fp16
            sf = q @ _rep(fk.astype(cdt), rep).swapaxes(-1, -2)
            parts_s.append(mx.where(inwin, sf, s[..., c0:L]))
        if parts_s:
            s = mx.concatenate(parts_s, axis=-1) if len(parts_s) > 1 or parts_s[0].shape[-1] != L else parts_s[0]
        causal = mx.arange(L)[None, :] <= pos
        s = mx.where(causal, s, mx.finfo(s.dtype).min)
        p = mx.softmax(s, axis=-1, precise=True)
        if S == 0 and c0 == L:
            out = self._pool_pv(p, L, rep, ex)
        else:
            pq = [mx.zeros_like(p[..., :S])] if S > 0 else []
            out = None
            if c0 > S:
                pq.append(p[..., S:c0])
            if c0 < L:
                pt = p[..., c0:]
                pq.append(mx.where(inwin, mx.zeros_like(pt), pt))
                out = (mx.where(inwin, pt, mx.zeros_like(pt)) @ _rep(fv.astype(cdt), rep)).astype(mx.float32)
            pq = mx.concatenate(pq, axis=-1) if len(pq) > 1 else pq[0]
            oq = self._pool_pv(pq, L, rep, ex).astype(mx.float32)
            out = oq if out is None else oq + out
            if S > 0:
                out = out + (p[..., :S] @ _rep(self.sv[..., :S, :].astype(cdt), rep)).astype(mx.float32)
        out = out.astype(queries.dtype)
        return out.reshape(B, Hq, T, D) if rep > 1 else out

    # ------------------------------------------------------------------ cache protocol
    def make_mask(self, *args, **kwargs):
        return None                                         # the causal mask is built inside attend (per row, chunk-size independent)

    def is_trimmable(self):
        return True

    def trim(self, n):
        """Drop the last n tokens (speculative rollback). The ring keeps `margin` spare rows, so a rollback up to `margin` tokens
        keeps the window exact; a longer one refills the missing window rows from the quantized pool (lossy, still correct)."""
        n = min(self.offset, n)
        if n == 0:
            return 0
        self.offset -= n
        if self.sk is not None and self.sk.shape[-2] > self.offset:
            self.sk, self.sv = self.sk[..., :self.offset, :], self.sv[..., :self.offset, :]
        if self.rk is not None:
            keep = max(self.rk.shape[-2] - n, 0)
            self.rk, self.rv = self.rk[..., :keep, :], self.rv[..., :keep, :]
            need = min(self.window, self.offset)                # the ring may hold window..window+margin rows: refill only below the window
            if keep < need:                                 # rollback deeper than the margin: dequantize the missing rows
                lo = self.offset - need
                hi = self.offset - keep
                dk, dv = self._deq_rows(lo, hi)
                self.rk = dk if keep == 0 else mx.concatenate([dk, self.rk], axis=-2)
                self.rv = dv if keep == 0 else mx.concatenate([dv, self.rv], axis=-2)
        self._prev = self._chunk = None
        return n

    def _deq_rows(self, lo, hi):
        return (mx.dequantize(*(x[..., lo:hi, :] for x in self.kq), group_size=self.group_size, bits=self.kbits),
                mx.dequantize(*(x[..., lo:hi, :] for x in self.vq), group_size=self.group_size, bits=self.vbits))

    def fork(self) -> "WinKVCache":
        """Independent copy (shared-prefix fork): MLX arrays are values, buffers are shared until one side writes."""
        c = WinKVCache(self.kbits, self.vbits, self.group_size, self.sink, self.window, self.margin, self.f32)
        cp = lambda t: None if t is None else (tuple(x[...] for x in t) if isinstance(t, tuple) else t[...])
        c.kq, c.vq, c.sk, c.sv, c.rk, c.rv = (cp(x) for x in (self.kq, self.vq, self.sk, self.sv, self.rk, self.rv))
        c.offset = self.offset
        return c

    def empty(self):
        return self.kq is None

    @property
    def nbytes(self):
        t = [x for x in (self.kq, self.vq) if x is not None]
        n = sum(a.nbytes for tt in t for a in tt)
        return n + sum(a.nbytes for a in (self.sk, self.sv, self.rk, self.rv) if a is not None)

    NPOOL = 6                                                                       # leading entries of SLOTS / state that grow with the context
    SLOTS = ("kq0", "kq1", "kq2", "vq0", "vq1", "vq2", "sk", "sv", "rk", "rv")      # order of `state`; the six pool arrays grow along axis 2, the overlay is fixed-size

    @property
    def state(self):
        """[kq (packed, scales, biases), vq (same), sink k, sink v, ring k, ring v] cut to `offset` rows (None = absent); [] when empty. The
        overlay is part of the state: a restored cache attends bit for bit like the one that was saved."""
        if self.kq is None:
            return []
        return [*(x[..., :self.offset, :] for x in (*self.kq, *self.vq)), self.sk, self.sv, self.rk, self.rv]

    @state.setter
    def state(self, v):
        if not v:
            self.kq = self.vq = self.sk = self.sv = self.rk = self.rv = None
            self.offset = 0
            return
        self.kq, self.vq = tuple(v[0:3]), tuple(v[3:6])
        self.sk, self.sv, self.rk, self.rv = v[6:10]
        self.offset = self.kq[0].shape[-2]
        self._prev = self._chunk = None


def _rep(x, rep):
    return mx.expand_dims(x, -3) if rep > 1 else x


# ---------------------------------------------------------------------------------------------- switch / hook
_installed = False
_orig_qsdpa = None


def install() -> None:
    """Route mlx-lm's quantized-SDPA hook to WinKVCache.attend when it is handed a KVView (idempotent)."""
    global _installed, _orig_qsdpa
    if _installed:
        return
    _orig_qsdpa = _base.quantized_scaled_dot_product_attention

    def qsdpa(queries, q_keys, q_values, scale, mask, group_size=64, bits=8):
        if isinstance(q_keys, KVView):
            return q_keys.cache.attend(queries, scale)
        return _orig_qsdpa(queries, q_keys, q_values, scale=scale, mask=mask, group_size=group_size, bits=bits)

    _base.quantized_scaled_dot_product_attention = qsdpa
    _installed = True


def config():
    """Parse KYOJIN_KVWIN -> dict(kbits, vbits, group_size, sink, window) or None (off)."""
    v = os.environ.get(ENV, "").strip().lower()
    if v in ("", "0", "off", "false"):
        return None
    f = [int(x) for x in v.split(":")]
    if len(f) != 5:
        raise ValueError(f"{ENV}={v!r}: expected kbits:vbits:group:sink:window")
    return dict(kbits=f[0], vbits=f[1], group_size=f[2], sink=f[3], window=f[4], f32=os.environ.get(ENV + "_F32", "0") == "1")


def make_prompt_cache(model, cfg: dict | None = None, plain: bool = False):
    """mlx-lm make_prompt_cache with the full-attention KVCache layers swapped for WinKVCache when the switch (or cfg) is on. plain: the stock caches
    (a short session under KYOJIN_TQ_FROM)."""
    cache = _mlx_make_prompt_cache(model)
    if plain:
        return cache
    mla = _mla_layers(model, len(cache))                     # an MLA layer caches (latent, k_pe), widths 512 / 64: the K = V head-dim pools do not apply, it keeps its KVCache
    from . import tq                                         # KYOJIN_TQ=kbits:vbits:sink:window: TurboQuant pool (+ the same fp16 overlay)
    tcfg = tq.config() if cfg is None else None
    if tcfg is not None:
        install()
        skip = mla | _tq_unsupported_layers(model, len(cache), tcfg["kbits"], tcfg["vbits"])
        return [tq.TQKVCache(**tcfg) if type(c) is KVCache and i not in skip else c for i, c in enumerate(cache)]
    cfg = cfg if cfg is not None else config()
    if cfg is None:
        return cache
    install()
    return [WinKVCache(**cfg) if type(c) is KVCache and i not in mla else c for i, c in enumerate(cache)]


def _tq_unsupported_layers(model, n: int, kbits: int, vbits: int) -> set:
    """Indices of the cache entries whose attention head_dim has no TurboQuant codebook / rotation (Gemma 4 global layers: 512 next to 256 on the sliding ones): those layers
    keep their plain KVCache. Cache entry i belongs to layer i (models that share KV between layers have fewer entries than layers, the shared ones come last)."""
    from . import tq
    inner = getattr(model, "language_model", model)
    layers = getattr(getattr(inner, "model", inner), "layers", None)
    if layers is None or len(layers) < n:
        return set()
    out = set()
    for i in range(n):
        for name in ("self_attn", "attn", "attention"):
            d = getattr(getattr(layers[i], name, None), "head_dim", None)
            if isinstance(d, int):
                if not (tq.supported(d, kbits) and tq.supported(d, vbits)):
                    out.add(i)
                break
    return out


def _mla_layers(model, n: int) -> set:
    """Indices of the cache entries whose layer runs MLA attention (kv_a_proj_with_mqa), when the trunk has one cache per layer."""
    inner = getattr(model, "language_model", model)
    layers = getattr(getattr(inner, "model", inner), "layers", None)
    if layers is None or len(layers) != n:
        return set()
    return {i for i, l in enumerate(layers) if hasattr(getattr(l, "self_attn", None), "kv_a_proj_with_mqa")}


def bytes_per_token(D: int, kbits: int, vbits: int, group_size: int, sink: int, window: int, ctx: int, kvh: int = 1) -> float:
    """Pool + amortized overlay bytes per token per layer for kvh heads of head dim D (scales and biases are fp16)."""
    pool = sum(bits * D / 8 + 4 * D / group_size for bits in (kbits, vbits))
    ov = 2 * D * 2 * min(sink + window, ctx) / ctx
    return kvh * (pool + ov)
