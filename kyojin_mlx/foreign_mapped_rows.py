# SPDX-License-Identifier: Apache-2.0
"""A GGUF table the model class reads by row id, kept in the mapped file (flag KYOJIN_MAPPED_ROW_TABLE, default off).

Qwen3.8-Flash-Next's `per_layer_token_embd` is [rows, 160] IQ4_NL, 28.8 GB. It is not indexed by token id: the class hashes the token and its two predecessors into
16 row ids per token (one per n-gram head) and gathers those rows. The class's own table object takes (B, T, heads) row ids and returns (B, T, heads, row_width).
This module is that object for a table that stays as stored: the uint8 view of the mapped tensor (GgufFile.raw, zero copy) is indexed by the ids, so only the pages
of the rows asked for are read, and those rows alone are decoded (numpy, any GGUF type foreign_gguf.DEQUANT reads). No resident copy of the table ever exists.
Memory touched per step = distinct rows x row bytes (IQ4_NL, 160 wide: 90 bytes a row) plus the decoded fp16 rows (320 bytes a row)."""
from __future__ import annotations

import os
import threading

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from . import foreign_gguf as GF


def row_prefetch_gib():
    """Flag KYOJIN_MAPPED_ROW_PREFETCH (default 0, read at load): 0/empty = off; 1 = pre-touch the whole table; a number > 1 = pre-touch the first N GiB (a table larger than the page cache)."""
    v = os.environ.get("KYOJIN_MAPPED_ROW_PREFETCH", "0")
    if v in ("0", ""):
        return 0.0
    return float("inf") if v == "1" else float(v)


def pretouch(raw: np.ndarray, max_gib: float = float("inf"), chunk_mib: int = 64, page: int = 16384, stop=None) -> dict:
    """Read one byte of every `page` of the mapped table (read only: no value changes, no copy, no write). The first touch of a mapped page costs a CPU page fault even when the
    file page is in the page cache; paid here in a background thread, the decode steps' row gathers (MappedRows.decode_flat) then hit mapped pages. Clean file-backed pages."""
    flat = raw.reshape(-1)
    n = int(min(flat.size, max_gib * (1 << 30))) if max_gib != float("inf") else int(flat.size)
    step, acc, touched = chunk_mib << 20, 0, 0
    for a in range(0, n, step):
        if stop is not None and stop.is_set():
            break
        v = flat[a:min(n, a + step):page]
        acc += int(v.sum(dtype=np.uint64))
        touched += len(v)
    return dict(pages=touched, bytes=n, checksum=acc)


class _Pretouch:
    """The background pre-touch thread of one table (shared, not copied, by a deepcopy of the module)."""

    def __init__(self, raw, gib):
        self.stats, self.stop, self.thr = None, threading.Event(), None
        if gib:
            self.thr = threading.Thread(target=self._run, args=(raw, gib), daemon=True, name="mapped-row-prefetch")
            self.thr.start()

    def _run(self, raw, gib):
        import time
        t = time.perf_counter()
        self.stats = dict(pretouch(raw, gib, stop=self.stop), s=round(time.perf_counter() - t, 2))

    def __deepcopy__(self, memo):
        return self


class MappedRows(nn.Module):
    """ids (any shape, int) -> rows (*ids.shape, width) fp16, decoded from the stored blocks of the mapped tensor."""

    def __init__(self, raw: np.ndarray, tname: str, width: int):
        super().__init__()
        if tname not in GF.DEQUANT:
            raise GF.GgufTypeError(f"mapped row table: GGUF type {tname} has no row decode (foreign_gguf.DEQUANT)")
        self._raw, self._tname, self.width = raw, tname, int(width)         # numpy attributes are not module parameters
        self.num_embeddings, self.row_bytes = int(raw.shape[0]), int(raw.shape[1])
        self.stats = dict(calls=0, rows=0, distinct=0, bytes=0)
        self._capture = None          # RowBatch: list that takes the ids instead of a lookup (zeros come back)
        self._stash = None            # RowBatch: rows already decoded for this call
        self._pf = _Pretouch(raw, row_prefetch_gib())    # flag KYOJIN_MAPPED_ROW_PREFETCH: pre-touch the mapped pages in the background (values never change)

    @property
    def prefetch_stats(self):
        return self._pf.stats

    def decode_flat(self, flat: np.ndarray) -> tuple:
        """(uniq, decoded fp16 [len(uniq), width], inv) of the int64 ids `flat`; the one place rows are read and decoded."""
        if flat.size and (flat.min() < 0 or flat.max() >= self.num_embeddings):
            raise IndexError(f"mapped row table: row id outside [0, {self.num_embeddings})")
        uniq, inv = np.unique(flat, return_inverse=True)
        blocks = np.ascontiguousarray(self._raw[uniq])                          # the only read of the file: these rows' bytes
        dec = GF.dequant(self._tname, blocks, self.width).reshape(len(uniq), self.width).astype(np.float16)
        return uniq, dec, inv, blocks

    def __call__(self, ids: mx.array) -> mx.array:
        if self._capture is not None:
            self._capture.append(ids)
            return mx.zeros((*tuple(ids.shape), self.width), mx.float16)
        if self._stash is not None:
            out, self._stash = self._stash, None
            return out
        flat = np.asarray(ids).reshape(-1).astype(np.int64)
        uniq, dec, inv, blocks = self.decode_flat(flat)
        self.stats["calls"] += 1
        self.stats["rows"] += int(flat.size)
        self.stats["distinct"] += int(len(uniq))
        self.stats["bytes"] += int(blocks.nbytes)
        return mx.array(dec[inv.reshape(-1)]).reshape(*tuple(ids.shape), self.width)


class _Window:
    """Stand-in for the layer cache while ids are probed: serves the n-gram window (slot 3), writes nothing."""

    def __init__(self, cache):
        self._w = cache[3] if cache is not None else None

    def __getitem__(self, i):
        return self._w if i == 3 else None

    def update_window(self, *a, **k):
        pass


class RowBatch:
    """Flag KYOJIN_MAPPED_ROW_BATCH (default off): the n-gram rows of ALL layers of a step are read in ONE host call.

    Without it each of the ~48 PLE layers evaluates its own row ids (np.asarray(ids)) in the middle of the forward: a GPU sync + a host decode + an upload per layer,
    so the GPU idles between layers. The ids depend only on the tokens and the 2-token window, never on a hidden state, so the first layer to run in a step computes the
    ids of every layer (the class's own __call__ with the lookup swapped for a capture: identical ids by construction), evaluates them together (one sync), decodes
    the union of the distinct rows once (same MappedRows.decode_flat) and hands each layer its rows. Steps longer than `max_t` tokens (prefill chunks) keep the per-layer path."""

    def __init__(self, parents, max_t=64):
        self.parents, self.max_t, self.key, self.rows, self.seen = parents, max_t, None, None, set()
        self.enabled = True                                                   # a profile cell flips it to A/B in one process
        self.stats = dict(prefetch=0, layer_calls=0, rows=0, distinct=0, bytes=0)

    def prefetch(self, input_ids, cache):
        self.key, self.rows, self.seen = input_ids, None, set()
        if input_ids.shape[-1] > self.max_t:
            return
        probe = _Window(cache)
        caps = []
        for par in self.parents:
            tab = par.ngram_embedding
            tab._capture = []
            try:
                super(type(par), par).__call__(input_ids, probe)
            finally:
                cap, tab._capture = tab._capture, None
            if not cap:                                                        # the parent did not reach the lookup (e.g. a profiler stub): per-layer path
                self.key = None
                return
            caps.append(cap[0])
        mx.eval(*caps)                                                         # the one sync of the step
        flats = [np.asarray(c).reshape(-1).astype(np.int64) for c in caps]
        tab0 = self.parents[0].ngram_embedding
        uniq, dec, inv, blocks = tab0.decode_flat(np.concatenate(flats))
        up = mx.array(dec[inv])                                               # one upload for every layer
        self.rows, o = {}, 0
        for par, c, f in zip(self.parents, caps, flats):
            self.rows[id(par)] = up[o:o + f.size].reshape(*tuple(c.shape), tab0.width)
            o += f.size
        self.stats["prefetch"] += 1
        self.stats["rows"] += int(inv.size)
        self.stats["distinct"] += int(len(uniq))
        self.stats["bytes"] += int(blocks.nbytes)


def install_batch(model, max_t: int = 64):
    """Swap the class of every n-gram embedding that holds a MappedRows table for one that asks the shared RowBatch. Returns the RowBatch (None: nothing to do)."""
    parents = [m for _p, m in model.named_modules() if isinstance(getattr(m, "ngram_embedding", None), MappedRows) and not hasattr(m, "_row_batch")]
    if not parents:
        return None
    batch = RowBatch(parents, max_t)

    for par in parents:
        base = type(par)

        def __call__(self, input_ids, cache, _base=base):
            b = self._row_batch
            if not b.enabled:
                return _base.__call__(self, input_ids, cache)
            if b.key is not input_ids or id(self) in b.seen:       # a layer seen twice under one array = a new step on a reused array
                b.prefetch(input_ids, cache)
            b.seen.add(id(self))
            b.stats["layer_calls"] += 1
            r = b.rows.get(id(self)) if b.rows else None
            self.ngram_embedding._stash = r
            try:
                return _base.__call__(self, input_ids, cache)
            finally:
                self.ngram_embedding._stash = None
        __call__._rowbatch = True
        par._row_batch = batch
        par.__class__ = type(base.__name__, (base,), {"__call__": __call__})
    return batch


def install_step_hooks(model) -> int:
    """Step-compile host hooks (KYOJIN_STEP_COMPILE_PREP): every module that owns a mapped row table (`ngram_embedding`) declares
    `step_host_prep(args) -> [rows]` (the host part: ids -> rows gather -> upload, run eagerly BEFORE the compiled layer; `args` = the layer's argument dict, it
    reads `input_ids` / `cache`) and `step_host_feed(arrs)` (inside the traced layer the rows come in as an input instead of a lookup). step_compile finds
    them by duck typing. Outside a feed the module behaves as before. Returns the number of modules hooked."""
    n = 0
    for _p, m in model.named_modules():
        tab = getattr(m, "ngram_embedding", None)
        if not isinstance(tab, (MappedRows, MappedNvfp4Rows)) or hasattr(m, "step_host_prep"):
            continue
        base = type(m)
        orig = next((c.__dict__["__call__"] for c in base.__mro__ if "__call__" in c.__dict__ and not getattr(c.__dict__["__call__"], "_rowbatch", False)), None)
        if orig is None:
            continue

        def prep(self, args, _orig=orig):
            ids, cache = args["input_ids"], args.get("cache")
            b = getattr(self, "_row_batch", None)
            if b is not None and b.enabled and ids.shape[-1] <= b.max_t:
                if b.key is not ids or id(self) in b.seen:
                    b.prefetch(ids, cache)
                b.seen.add(id(self))
                b.stats["layer_calls"] += 1
                r = b.rows.get(id(self)) if b.rows else None
                if r is not None:
                    return [r]
            t = self.ngram_embedding
            t._capture = []
            try:
                _orig(self, ids, _Window(cache) if cache is not None else None)
            finally:
                cap, t._capture = t._capture, None
            return [t(cap[0])]                                                # concrete ids: the table's own host lookup

        def feed(self, arrs):
            object.__setattr__(self, "_step_fed", arrs[0])

        def __call__(self, input_ids, cache, _base=base, _orig=orig):
            fed = self.__dict__.get("_step_fed")
            if fed is None:
                return _base.__call__(self, input_ids, cache)
            self.__dict__["_step_fed"] = None
            t = self.ngram_embedding
            t._stash = fed
            try:
                return _orig(self, input_ids, cache)
            finally:
                t._stash = None
        m.__class__ = type(base.__name__, (base,), {"__call__": __call__, "step_host_prep": prep, "step_host_feed": feed})
        n += 1
    return n


class MappedNvfp4Rows(nn.Module):
    """The same row table for the safetensors layout NVIDIA ships (modelopt NVFP4, FP4-Local pack): `shards` = [(nibbles, scales)] views of the mapped file, one pair per row shard.

    nibbles uint8 [rows_i, k/2] (element 2b in the low nibble, 2b+1 in the high one), scales 8-bit E4M3 [rows_i, k/16] (plain, not CSF: one scale per 16 elements ALONG the row),
    one float32 global scale for the whole table (multiplied). Every row is therefore decodable alone: its block scales sit in the same row of the scale tensor.
    The decode is the arithmetic of foreign_pack.dequant for fmt nvfp4 (E2M1 code x E4M3 block scale, then x global), so rows equal the full dequant bit for bit.
    A global row id is split on the cumulative shard sizes (the class's ShardedEmbedding does the same)."""

    def __init__(self, shards, gscale: float, width: int | None = None):
        super().__init__()
        from . import foreign_pack as FP
        if not shards:
            raise ValueError("mapped nvfp4 row table: no shard")
        k = int(shards[0][0].shape[1]) * 2
        for q, s in shards:
            if q.ndim != 2 or s.ndim != 2 or q.shape[0] != s.shape[0] or q.shape[1] * 2 != k or s.shape[1] * 16 != k:
                raise ValueError(f"mapped nvfp4 row table: shard shapes {q.shape} / {s.shape} do not fit k={k} (nibbles [rows, k/2], one scale per 16 elements [rows, k/16])")
        if not np.isfinite(gscale) or gscale == 0.0:
            raise ValueError(f"mapped nvfp4 row table: global scale {gscale} is not usable")
        self._shards, self._g, self.width = list(shards), np.float32(gscale), int(width or k)
        if self.width != k:
            raise ValueError(f"mapped nvfp4 row table: row width {self.width} != {k} decoded elements")
        self._offs = np.concatenate([[0], np.cumsum([int(q.shape[0]) for q, _ in shards])]).astype(np.int64)
        self._capture = None          # step hooks: list that takes the ids instead of a lookup (zeros come back)
        self._stash = None            # step hooks: rows already decoded for this call
        self.num_embeddings, self.k = int(self._offs[-1]), k
        pair = np.empty((256, 2), np.float32)
        pair[:, 0], pair[:, 1] = FP.E2M1[np.arange(256) & 15], FP.E2M1[np.arange(256) >> 4]
        self._pair, self._sc = pair, FP._e4m3(np.arange(256, dtype=np.uint8))
        self.stats = dict(calls=0, rows=0, distinct=0, bytes=0)

    def decode(self, uniq: np.ndarray) -> np.ndarray:
        """float32 [len(uniq), k] of the sorted distinct global row ids `uniq` (only those rows' bytes are read)."""
        sh = np.searchsorted(self._offs, uniq, side="right") - 1
        qs, sc = [], []
        for i in np.unique(sh):
            loc = uniq[sh == i] - self._offs[i]
            qs.append(np.ascontiguousarray(self._shards[i][0][loc]))
            sc.append(np.ascontiguousarray(self._shards[i][1][loc]).view(np.uint8))
        q, s = np.concatenate(qs), np.concatenate(sc)
        self.stats["bytes"] += int(q.nbytes + s.nbytes)
        m = len(uniq)
        w = self._pair[q].reshape(m, self.k // 16, 16) * self._sc[s][..., None]
        return w.reshape(m, self.k) * self._g

    def __call__(self, ids: mx.array) -> mx.array:
        if self._capture is not None:
            self._capture.append(ids)
            return mx.zeros((*tuple(ids.shape), self.width), mx.float16)
        if self._stash is not None:
            out, self._stash = self._stash, None
            return out
        flat = np.asarray(ids).reshape(-1).astype(np.int64)
        if flat.size and (flat.min() < 0 or flat.max() >= self.num_embeddings):
            raise IndexError(f"mapped row table: row id outside [0, {self.num_embeddings})")
        uniq, inv = np.unique(flat, return_inverse=True)
        dec = self.decode(uniq).astype(np.float16)
        self.stats["calls"] += 1
        self.stats["rows"] += int(flat.size)
        self.stats["distinct"] += int(len(uniq))
        return mx.array(dec[inv.reshape(-1)]).reshape(*tuple(ids.shape), self.width)
