# SPDX-License-Identifier: Apache-2.0
"""TurboQuant cache for MLA latent rows (GLM-5.x sparse-attention layers: one 512-wide latent per token, no rope part).

The pool holds each 256-wide half of a latent row as (rotation + Lloyd-Max codebook, `bits` per value, one fp16 norm): 6.06 bits per value instead of 16.
The first `sink` rows and the last `window` rows stay fp16 (the engine's tq6w overlay). The attention reads the dense latent that `update_and_fetch`
returns: pool rows are dequantized on the fly (a transient of one layer), sink and window rows are the exact fp16 inputs. The model sees the same API as
`KVCache`: `offset`, `trim`, `extract` / `merge` (to the dense batch cache), `state` (packed, for prefix / disk checkpoints), `nbytes`.

Rows whose second slot is non-empty (a rope part) are not supported: GLM-5.x has `qk_rope_head_dim = 0`."""
from __future__ import annotations

import mlx.core as mx

from mlx_vlm.models.cache import BatchKVCache, KVCache

from . import tq

HALF = 256


class LatentTQCache(KVCache):
    step = 256

    def __init__(self, bits: int = 6, sink: int = 4, window: int = 256):
        super().__init__()
        self.qbits, self.sink_n, self.window = bits, sink, window
        self.words = self.scales = None                      # [B, H, cap, halves, NW] uint32 / [B, H, cap, halves] fp16
        self.sk = self.rk = None                             # fp16 sink rows / last-window rows [B, H, n, D]
        self.dim = 0

    # ------------------------------------------------------------ write
    def _pack(self, x):
        B, H, n, D = x.shape
        y = tq.rotate(x.reshape(-1, HALF), HALF)
        w, s = tq.tq_write(y, self.qbits)
        h = D // HALF
        return w.reshape(B, H, n, h, -1), s.reshape(B, H, n, h)

    def _unpack(self, w, s, dtype):
        B, H, n, h, nw = w.shape
        x = tq.unrotate(tq.tq_dequant(w.reshape(-1, nw), s.reshape(-1), self.qbits, HALF, mx.float32), HALF)
        return x.reshape(B, H, n, h * HALF).astype(dtype)

    def update_and_fetch(self, keys, values):
        if values.shape[-1] != 0:
            raise ValueError("LatentTQCache: a rope part (non-empty second slot) is not supported")
        B, H, n, D = keys.shape
        assert D % HALF == 0 and tq.supported(HALF, self.qbits)
        prev = self.offset
        w, s = self._pack(keys)
        if self.words is None or prev + n > self.words.shape[2]:
            steps = (self.step + n - 1) // self.step * self.step
            nw = w.shape[-1]
            nwords = mx.zeros((B, H, steps, D // HALF, nw), dtype=mx.uint32)
            nscales = mx.zeros((B, H, steps, D // HALF), dtype=mx.float16)
            if self.words is None:
                self.words, self.scales = nwords, nscales
            else:
                self.words = mx.concatenate([self.words[:, :, :prev], nwords], axis=2)
                self.scales = mx.concatenate([self.scales[:, :, :prev], nscales], axis=2)
        self.words[:, :, prev:prev + n] = w
        self.scales[:, :, prev:prev + n] = s
        self.offset = prev + n
        self.dim = D
        if prev < self.sink_n:
            m = min(self.sink_n - prev, n)
            part = keys[:, :, :m]
            self.sk = part if self.sk is None else mx.concatenate([self.sk, part], axis=2)
        if self.window > 0:
            rk = keys if self.rk is None else mx.concatenate([self.rk, keys], axis=2)
            self.rk = rk[:, :, -self.window:]
        return self._dense(keys.dtype), mx.zeros((B, H, self.offset, 0), dtype=keys.dtype)

    def _dense(self, dtype):
        n = self.offset
        s = 0 if self.sk is None else min(self.sk.shape[2], n)
        r = 0 if self.rk is None else self.rk.shape[2]
        t0 = max(s, n - r)                                   # first row served by the ring
        parts = []
        if s:
            parts.append(self.sk[:, :, :s].astype(dtype))
        if t0 > s:
            parts.append(self._unpack(self.words[:, :, s:t0], self.scales[:, :, s:t0], dtype))
        if n > t0:
            parts.append(self.rk[:, :, r - (n - t0):].astype(dtype))
        return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=2)

    def gather(self, idx, dtype):
        """Rows `idx` [B, 1, K, 1] (int, within [0, offset)) of the dense latent, as `_dense` would serve them, but dequantizing only those K rows (sink and
        window rows are the exact fp16 inputs): the decode attention reads the top-k rows, not the whole latent."""
        n = self.offset
        s = 0 if self.sk is None else min(self.sk.shape[2], n)
        r = 0 if self.rk is None else self.rk.shape[2]
        t0 = max(s, n - r)
        idx = idx.astype(mx.int32)
        out = None
        if t0 > s:
            ip = mx.clip(idx, s, t0 - 1)
            w = mx.take_along_axis(self.words, ip[..., None], axis=2)
            sc = mx.take_along_axis(self.scales, ip, axis=2)
            out = self._unpack(w, sc, dtype)
        if s:
            rows = mx.take_along_axis(self.sk, mx.clip(idx, 0, s - 1), axis=2).astype(dtype)
            out = rows if out is None else mx.where(idx < s, rows, out)
        if n > t0:
            rows = mx.take_along_axis(self.rk, mx.clip(idx - (n - r), 0, r - 1), axis=2).astype(dtype)
            out = rows if out is None else mx.where(idx >= t0, rows, out)
        return out

    # ------------------------------------------------------------ cache API
    def size(self):
        return self.offset

    def empty(self):
        return self.words is None

    @property
    def nbytes(self):
        if self.words is None:
            return 0
        return sum(a.nbytes for a in (self.words[:, :, :self.offset], self.scales[:, :, :self.offset], self.sk, self.rk) if a is not None)

    @property
    def state(self):
        if self.words is None:
            return None, None, None, None
        n = self.offset
        return self.words[:, :, :n], self.scales[:, :, :n], self.sk, self.rk

    @state.setter
    def state(self, v):
        self.words, self.scales, self.sk, self.rk = v
        self.offset = 0 if self.words is None else self.words.shape[2]
        self.dim = 0 if self.words is None else self.words.shape[3] * HALF

    @property
    def meta_state(self):
        return ""

    @meta_state.setter
    def meta_state(self, v):
        pass

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        if self.rk is not None:
            self.rk = self.rk[:, :, :max(0, self.rk.shape[2] - n)]
        if self.sk is not None and self.sk.shape[2] > self.offset:
            self.sk = self.sk[:, :, :self.offset]
        return n

    def prefix_cache_reserve(self, min_capacity_tokens):
        return ()

    def prefix_cache_merge(self, rows, prefix_lens):
        return None

    def to_dense(self, dtype=None):
        """The exact dense latent (sink / window rows as written, pool rows dequantized) as a `LatentDense` row; dtype defaults to the dtype the rows were written in
        (fp16 / bf16 inputs are never narrowed to bf16: that would round every sink / window row)."""
        c = LatentDense()
        if self.words is not None:
            dt = dtype or (self.rk.dtype if self.rk is not None else mx.bfloat16)
            c.keys = mx.contiguous(self._dense(dt))
            c.values = mx.zeros((*c.keys.shape[:3], 0), dtype=dt)
            c.offset = self.offset
            c.tq_cfg = (self.qbits, self.sink_n, self.window)
            c.rolled = self.offset - (0 if self.rk is None else self.rk.shape[2])     # rows below this are already the dequantized pool rows
        return c

    def extract(self, idx):
        c = self.to_dense()
        if c.keys is not None:
            c.keys, c.values = c.keys[idx:idx + 1], c.values[idx:idx + 1]
        return c

    @classmethod
    def merge(cls, caches):
        return _merge_rows(caches)


def _dense_rows(caches):
    """Rows of a mixed membership as plain dense caches: a LatentTQCache (a session fresh from prefill) is dequantized, a dense row (one that was
    batched, then extracted) is kept as is."""
    return [c.to_dense() if isinstance(c, LatentTQCache) else c for c in caches]


def _merge_rows(caches):
    """One batch cache from rows that are LatentTQCache and / or dense (`LatentDense`, a stock KVCache). BatchKVCache.merge reads `.keys`, which a
    LatentTQCache does not hold (None -> an all-zero row): every row goes through `to_dense` first, whatever type the first row has."""
    return LatentBatch.merge(_dense_rows(caches))


def _roundtrip(x, bits):
    """Rows `x` [B, H, n, D] as the pool would serve them (rotate + Lloyd-Max write, then dequant): what a window row becomes when it leaves the ring."""
    B, H, n, D = x.shape
    w, s = tq.tq_write(tq.rotate(x.reshape(-1, HALF), HALF), bits)
    y = tq.unrotate(tq.tq_dequant(w, s, bits, HALF, mx.float32), HALF)
    return y.reshape(B, H, n, D).astype(x.dtype)


def _roll(keys, cfg, lo, hi, rolled, pad):
    """Dense rows `keys` [B, H, L, D] of a batch: the columns [lo, hi) left the exact window this step, so each row of the batch that has not rolled them yet
    (`rolled` = per-row first column still exact, `pad` = per-row left padding: an array or a host list) gets the quantized value its solo `LatentTQCache` serves
    for that position (sink rows and padding stay as they are). Without this a merged row keeps rows exact that its solo run has already rounded, and every decode step
    widens the gap (each row of the window that rolls out is one more difference in the attention of a context shorter than index_topk, where the
    indexer keeps every row).
    Which (row, column) pairs roll is known on the host (rolled / pad are host values), so the common case where every pair rolls writes the rounded columns straight
    into the buffer: no mask arithmetic, no select (G3d: those were 8 of the 14 launches of the step's roll)."""
    bits, sink, _ = cfg
    cols = keys[:, :, lo:hi]
    q = _roundtrip(cols, bits)
    ph = pad.tolist() if hasattr(pad, "tolist") else list(pad)
    todo = [[c >= rolled[b] and c - ph[b] >= sink for c in range(lo, hi)] for b in range(len(ph))]
    if all(all(r) for r in todo):
        keys[:, :, lo:hi] = q
        return
    if not any(any(r) for r in todo):
        return
    keys[:, :, lo:hi] = mx.where(mx.array(todo)[:, None, :, None], q, cols)


class LatentDense(KVCache):
    """A latent row that left a batch (or was densified for one): a KVCache whose `merge` accepts LatentTQCache rows next to dense ones. The engine merges with
    `type(first row).merge`, so a batch that has an extracted row first and a fresh LatentTQCache later must not fall to `KVCache.merge`.
    It keeps the TurboQuant overlay of the row it came from (`tq_cfg` = bits, sink, window; `rolled` = rows below it are already quantized) and rounds the
    rows that leave the window as they would have in `LatentTQCache`, so a row decoded dense is the row decoded solo."""
    tq_cfg = None
    rolled = 0

    @classmethod
    def merge(cls, caches):
        return _merge_rows(caches)

    def update_and_fetch(self, keys, values):
        out = super().update_and_fetch(keys, values)
        if self.tq_cfg is None:
            return out
        hi = self.offset - self.tq_cfg[2]
        if hi > self.rolled:
            _roll(self.keys, self.tq_cfg, self.rolled, hi, [self.rolled], [0])
            self.rolled = hi
            return self.keys[..., :self.offset, :], self.values[..., :self.offset, :]
        return out

    def trim(self, n):
        n = super().trim(n)
        self.rolled = min(self.rolled, self.offset)
        return n


class LatentBatch(BatchKVCache):
    """The merged latent cache: stock BatchKVCache whose rows extract back to `LatentDense`. Like `LatentDense` it rounds, per row, the rows that leave the
    TurboQuant window (`_rolled[i]` = first physical column of row i still exact), so row i decodes as it does solo."""
    tq_cfg = None

    @classmethod
    def merge(cls, caches):
        batch = super().merge(caches)
        cfgs = [c.tq_cfg for c in caches if getattr(c, "tq_cfg", None) is not None]
        if cfgs and batch.keys is not None:
            batch.tq_cfg = cfgs[0]
            top = batch.keys.shape[2]
            pads = batch.left_padding.tolist()
            batch._rolled = [p + (c.rolled if getattr(c, "tq_cfg", None) is not None else c.offset) for p, c in zip(pads, caches)]   # a row without overlay never rolls
            batch._rolled = [min(r, top) for r in batch._rolled]
        return batch

    def update_and_fetch(self, keys, values):
        out = super().update_and_fetch(keys, values)
        if self.tq_cfg is None:
            return out
        hi = self._idx - self.tq_cfg[2]
        lo = min(self._rolled)
        if hi > lo:
            _roll(self.keys, self.tq_cfg, lo, hi, self._rolled, self.left_padding)
            self._rolled = [max(r, hi) for r in self._rolled]
            return self.keys[..., :self._idx, :], self.values[..., :self._idx, :]
        return out

    def trim(self, n):
        n = super().trim(n)
        if self.tq_cfg is not None:
            self._rolled = [min(r, self._idx) for r in self._rolled]
        return n

    def filter(self, batch_indices):
        idx = batch_indices.tolist() if hasattr(batch_indices, "tolist") else list(batch_indices)
        idx = [idx] if isinstance(idx, int) else idx
        idx0 = self._idx
        super().filter(batch_indices)
        if self.tq_cfg is not None:
            shift = idx0 - self._idx
            self._rolled = [max(0, self._rolled[i] - shift) for i in idx]

    def extend(self, other):
        lefts = (max(self._idx, other._idx) - self._idx, max(self._idx, other._idx) - other._idx)
        mine = [r + lefts[0] for r in (self._rolled if self.tq_cfg is not None else [self._idx] * self.offset.size)]            # no overlay: never rolls
        theirs = [r + lefts[1] for r in (other._rolled if other.tq_cfg is not None else [other._idx] * other.offset.size)]
        cfg = self.tq_cfg or other.tq_cfg
        super().extend(other)
        self.tq_cfg = cfg
        if cfg is not None:
            self._rolled = mine + theirs

    def extract(self, idx):
        c = super().extract(idx)
        d = LatentDense()
        d.keys, d.values, d.offset = c.keys, c.values, c.offset
        if self.tq_cfg is not None and c.keys is not None:
            d.tq_cfg = self.tq_cfg
            d.rolled = max(0, min(self._rolled[idx] - self.left_padding[idx].item(), d.offset))
        return d


def wrap_latent_layers(cache, cfg: dict):
    """Swap the latent KVCache of every GLM CacheList for a LatentTQCache (in place); returns how many."""
    n = 0
    for c in cache:
        subs = getattr(c, "caches", None)
        if subs and type(subs[0]) is KVCache and subs[0].keys is None:
            c.caches = (LatentTQCache(bits=cfg["kbits"], sink=cfg["sink"], window=cfg["window"]), *subs[1:])
            n += 1
    if n:
        install_attend()
    return n


def latent_ok(lm) -> bool:
    """MLA latent caches without a rope part (GLM-5.x: qk_rope_head_dim 0, kv_lora_rank a multiple of 256)."""
    c = getattr(lm, "config", None) or getattr(getattr(lm, "args", None), "text_config", None)
    return c is not None and int(getattr(c, "qk_rope_head_dim", 1) or 0) == 0 and int(getattr(c, "kv_lora_rank", 0) or 0) % HALF == 0 and int(getattr(c, "kv_lora_rank", 0) or 0) > 0


def install_attend() -> bool:
    """Decode attention of GLM-5.x on a LatentTQCache reads the packed latent through `gather` (only the selected rows are dequantized) instead of the
    dense latent `update_and_fetch` returns (whose lazy full dequant would otherwise be evaluated every step). Prefill / blocks / other caches: stock code."""
    import os
    if os.environ.get("KYOJIN_LATENT_GATHER", "1") == "0":
        return False
    from . import archs as _AR; L = _AR.lang(_AR.GLM5)
    cls = L.Glm5NextAttention
    if getattr(cls, "_kyojin_latent_gather", False):
        return True
    stock = cls._attend

    def _attend(self, q, latent, new_latent, topk, kv_cache, projected_cache, last_only=False):
        from . import glm_dsa as _H
        if q.shape[2] != 1 or self.training or not isinstance(kv_cache, LatentTQCache) or kv_cache.words is None or topk is None:
            k = "attend:stock:" + type(kv_cache).__name__ + (":L>1" if q.shape[2] != 1 else "")
            _H.HITS[k] = _H.HITS.get(k, 0) + 1
            return stock(self, q, latent, new_latent, topk, kv_cache, projected_cache, last_only)
        _H.HITS["attend:tq"] = _H.HITS.get("attend:tq", 0) + 1
        batch = q.shape[0]
        n = kv_cache.offset
        from . import glm_dsa as _G
        out = None
        if _G.gather_usable(kv_cache, q, topk, new_latent):
            if _G.attend_usable(self, q):
                rows, vm = _G.gather_rows(kv_cache, topk[0, 0], new_latent.dtype, pad=32)       # rows + validity, then the attention over them (see glm_dsa)
                out = _G.attend(self.embed_q(q), rows, vm, self.scale)
            else:
                rows, vm = _G.gather_rows(kv_cache, topk[0, 0], new_latent.dtype)
                selected, valid = rows[None, None], vm[None, None]
        else:
            sel = topk[:, 0:1]
            valid = (sel >= 0) & (sel < n)
            safe = mx.clip(sel, 0, max(n - 1, 0))
            selected = kv_cache.gather(safe[:, None, 0, :, None], new_latent.dtype)
        if out is None:
            query = self.embed_q(q)
            out = L.scaled_dot_product_attention(query, selected, selected, cache=kv_cache, scale=self.scale, mask=valid[:, None])
        out = self.unembed_out(out)
        if last_only:
            topk = topk[:, -1:]
        return out.transpose(0, 2, 1, 3).reshape(batch, 1, -1), topk

    cls._attend = _attend
    cls._kyojin_latent_gather = True
    return True
