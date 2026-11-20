# SPDX-License-Identifier: Apache-2.0
"""KYOJIN_HY4_FIXED_KV: a fixed-state KV cache for the skeleton's MLA layers (one latent kv head per layer: row = [latent | k_pe | kpool value | kpool gate]).

Why: step_compile compiles a whole decoder layer only when the layer cache's entire state is a list of fixed-shape arrays (`cache.cache`); a KVCache
(python offset, per-token concatenate / slice copy) keeps its attention eager between two programs. This cache keeps
    cache.cache = [buffer [1, 1, C, D], write offset int32 [1]]
with C a multiple of `chunk` (and of every kpool group size); the attention (foreign_skeleton_mla.MlaAttention._call_fixed) writes the row with a
device-offset slice_update and builds its masks from the offset array, so the layer program has static shapes and replays every step. Growth (a new
chunk) is eager and host side (Model.__call__ -> ensure) and changes the shape key: one retrace per layer per chunk. `offset` / `_n` is the host mirror of
the device offset (advanced by Model.__call__ after the layers), so nothing ever reads the device.

Windowed layers keep every row (a decode step reads only the `window` rows ending at the new row through a device-offset slice, so the cost is
O(window) like a ring, without the ring's wrap-around for a multi-row prefill chunk). Ring storage would save memory only.
"""
from __future__ import annotations

import math
import os

import mlx.core as mx
from mlx_lm.models.cache import _BaseCache

ENABLED = os.environ.get("KYOJIN_HY4_FIXED_KV", "0") == "1"
# KYOJIN_HY4_SWA_RING=1 (read by make_caches): windowed layers keep window + chunk + 16 rows (slot = position % rows) instead of every row: 1M tokens x 30 layers x 2.3 KB = 69 GiB saved
PK_DTYPE = mx.float16 if os.environ.get("KYOJIN_HY4_FIXED_PKCACHE_F16", "0") == "1" else mx.float32     # pooled keys carried in cache[2]: fp16 halves the scorer's read at depth (the scorer widens on load)
RING_ALIGN = 256                                                  # ring rows are a multiple of this (tests use a small value to force wrap-around)
RING_ROWS = int(os.environ.get("KYOJIN_HY4_SWA_RING_ROWS", "4864"))   # initial ring rows (>= window + prefill chunk + 16; grows when a longer chunk arrives)
CHUNK = int(os.environ.get("KYOJIN_HY4_FIXED_KV_CHUNK", "1024"))


class FixedKVCache(_BaseCache):
    fixed_kv = True                                          # step_compile._fixed_cache: whole state in `.cache`, the host offset is only a mirror

    def __init__(self, width: int, align: int = 1, window: int = 0, chunk: int = CHUNK, pk_dx: int = 0, ring: bool = False):
        self.cache = [None, None]
        self.ring = bool(ring and window)
        self.width, self.window, self.align, self.pk_dx = width, window, align, pk_dx      # pk_dx > 0 (flag KYOJIN_HY4_FIXED_PKCACHE, kpool layers): cache[2] = the pooled keys [1, C // align, pk_dx]
        self.step = chunk * align // math.gcd(chunk, align)
        self._n = 0

    @property
    def offset(self) -> int:
        return self._n

    @offset.setter
    def offset(self, v):
        self._n = int(v)

    def ensure(self, need: int, dtype) -> None:
        """Eager, host side: room for `need` rows (rounded up to a chunk); the new buffer keeps the rows written so far."""
        if self.ring:
            return self._ensure_ring(need, dtype)
        buf = self.cache[0]
        slack = self.align - 1 if self.pk_dx else 0
        if buf is not None and need + slack <= buf.shape[2]:
            return
        cap = max(-(-max(need + slack, self.window, 1) // self.step) * self.step, self.step)       # pk_dx: the pooled-key update reads up to align - 1 rows past the write end
        nb = mx.zeros((1, 1, cap, self.width), dtype=dtype if buf is None else buf.dtype)
        if buf is not None and self._n:
            nb = mx.concatenate([buf[:, :, :self._n], nb[:, :, self._n:]], axis=2)
        if self.pk_dx:
            pk = mx.zeros((1, cap // self.align, self.pk_dx), dtype=PK_DTYPE)
            if buf is not None and len(self.cache) > 2 and self.cache[2] is not None:
                j0 = self.cache[2].shape[1]
                pk = mx.concatenate([self.cache[2], pk[:, j0:]], axis=1)
            if len(self.cache) > 2:
                self.cache[2] = pk
            else:
                self.cache.append(pk)
        self.cache[0] = nb
        if self.cache[1] is None:
            self.cache[1] = mx.array([self._n], dtype=mx.int32)

    def _ensure_ring(self, need: int, dtype) -> None:
        """Ring rows R >= window + 16 (speculation rollback) + the rows of this call; a longer call rebuilds the ring (rows still needed re-placed at position % R)."""
        L = max(need - self._n, 1)
        req = -(-(self.window + 16 + L) // RING_ALIGN) * RING_ALIGN
        buf = self.cache[0]
        if buf is not None and req <= buf.shape[2]:
            return
        cap = max(req, RING_ROWS)
        nb = mx.zeros((1, 1, cap, self.width), dtype=dtype if buf is None else buf.dtype)
        if buf is not None and self._n:
            R0 = buf.shape[2]
            pos = mx.arange(max(0, self._n - R0), self._n)
            nb[:, :, pos % cap, :] = buf[:, :, pos % R0, :]
        self.cache[0] = nb
        if self.cache[1] is None:
            self.cache[1] = mx.array([self._n], dtype=mx.int32)

    def update_and_fetch(self, keys, values):
        raise NotImplementedError("FixedKVCache is written by MlaAttention._call_fixed")

    @property
    def state(self):
        return [a for a in self.cache if a is not None]

    @state.setter
    def state(self, v):
        v = list(v)
        self.cache = v + [None] * (2 - len(v))
        if v:
            self._n = int(v[1].item())

    @property
    def meta_state(self):
        return ""

    def is_trimmable(self):
        return True

    def trim(self, n: int) -> int:
        """Speculation rollback: drop the last `n` rows by lowering the write offset (host mirror and device array). Rows past the offset are masked
        (visibility / validity) and the next write overwrites them; the kpool value / gate rows live in the same buffer rows, so nothing else holds state."""
        n = min(int(n), self._n)
        if n <= 0:
            return 0
        if self.ring and n > self.cache[0].shape[2] - self.window:
            raise ValueError("ring KV: a rollback of more than rows - window rows needs rows the ring has overwritten (no prefix reuse on a ring cache)")
        self._n -= n
        if self.cache[1] is not None:
            self.cache[1] = self.cache[1] - n
        return n

    def fork(self):
        """An independent session cache with the same rows (prefix snapshot / restore): the arrays are values and every write here is a functional slice_update that
        returns a new array, so the copy shares the buffer until one side writes (copy on write)."""
        n = object.__new__(type(self))
        n.__dict__.update(self.__dict__)
        n.cache = [None if a is None else a[...] for a in self.cache]
        return n

    @property
    def nbytes(self):
        return sum(a.nbytes for a in self.cache if a is not None)


def ring_write(buf, k, off, L):
    """buf [1, 1, R, D] with the L rows k [1, 1, L, D] written at slots (off + i) % R (a ring; off int32 [1] device array). Functional: buf itself is untouched."""
    R = buf.shape[2]
    out = buf[...]
    out[:, :, (off + mx.arange(L, dtype=mx.int32)) % R, :] = k
    return out


def make_caches(model) -> list:
    """One FixedKVCache per layer of a skeleton model with MLA layers (the row width / kpool group / window read off the layer's attention)."""
    out = []
    for layer in model.layers:
        at = layer.self_attn
        if not hasattr(at, "kv") or not hasattr(at, "kp"):
            raise ValueError("KYOJIN_HY4_FIXED_KV: the layer's attention is not the MLA block")
        kp = at.kp
        width = at.kv + at.r + (2 * kp["dx"] if kp else 0)
        out.append(FixedKVCache(width, kp["kp"] if kp else 1, getattr(at, "window", 0) if getattr(at, "swa", False) else 0, pk_dx=kp["dx"] if kp and os.environ.get("KYOJIN_HY4_FIXED_PKCACHE", "0") == "1" else 0, ring=os.environ.get("KYOJIN_HY4_SWA_RING", "0") == "1" and not kp))      # flag: kpool layers keep their pooled keys in cache[2], updated by the groups a call completes
    return out
