# SPDX-License-Identifier: Apache-2.0
"""DS4.1 Engram table lookup on MLX with parallel preads. Replaces mlx-vlm's `OffloadedEngramEmbedding` (np.memmap + one fancy-index gather =
one page fault at a time) by `EngramTables.read_rows` (16 threads of pread, rows deduplicated). Same values: the rows are the mxfp8 bytes of the
shard, dequantised by `mx.dequantize` exactly as mlx-vlm does (tests/test_engram_rows.py compares the two on a pack with an Engram layer).
A 4096-token prefill chunk touches about 100k distinct rows per Engram layer; a decode token 24."""
from __future__ import annotations

import contextlib
import ctypes
import os
import time
import types

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from . import engram_rows as ER
from .engram_rows import EngramTables
from .serve import hostloop as _HL                   # serve_prof section marks (a boolean test when off)


class ParallelRowEmbedding(nn.Module):
    """`engram.embed` stand-in: indices [...] -> bf16 rows [..., 256]; no parameters (the table stays on disk)."""

    def __init__(self, tables: EngramTables, layer_pos: int):
        super().__init__()
        object.__setattr__(self, "_tables", tables)
        object.__setattr__(self, "_pos", layer_pos)
        self.reads = 0

    def __call__(self, indices: mx.array) -> mx.array:
        ahead = getattr(self._tables, "ahead", None)
        job = None if ahead is None else ahead.take(self._pos, indices.shape)
        if job is not None:                                                # prefill: rows read in the background since the forward's entry
            shape, uniq, inv, w, s = job.result()
            idx = types.SimpleNamespace(shape=shape)
        else:
            idx = np.asarray(indices)                                      # small int array: synchronises, like the mlx-vlm class
            uniq, inv = np.unique(idx.reshape(-1), return_inverse=True)
            w, s = self._tables.read_rows(self._pos, uniq)
        self.reads += int(uniq.size)
        rows = mx.dequantize(mx.array(w).view(mx.uint32), mx.array(s), None, group_size=32, bits=8, mode="mxfp8", dtype=mx.bfloat16)
        return rows[mx.array(inv.astype(np.int32))].reshape(*idx.shape, rows.shape[-1])


def install_parallel_gather(model, tables: EngramTables) -> list:
    """Swap the `engram.embed` of every Engram layer of an mlx-vlm DS4.1 model for a `ParallelRowEmbedding` on `tables`; returns the modules."""
    out = []
    for pos, lid in enumerate(tables.spec.layer_ids):
        layer = model.language_model.layers[lid]
        layer.engram.embed = ParallelRowEmbedding(tables, pos)
        out.append(layer.engram.embed)
    install_rows_ahead(model, tables)
    return out


# ----------------------------------------------------------------------------------------------------------------- prefill rows ahead
# The hash ids of a forward are known at its entry, but the embed reads them back at its own layer (np.asarray drains the queue) and the GPU idles
# during the read: ~4 s of a 10.4 s 8192-row DS4.1 chunk on the Ultra (2 layers x ~2 s at 3 % GPU, powermetrics). With the rows ahead, a prompt-sized
# hash call (>= MIN_ROWS positions) reads every Engram layer's rows in the background, in layer order, and each Engram layer submits the graph built
# so far before it waits: the rows of layer 14 are read while the GPU runs layers 1-13. Same rows, same dequantize: same logits.
# Decode and MTP verify too (3-4 rows): the embed's own read used to stop the build at layer 1 and again at 14 with the GPU idle (12.7 of a 48 ms
# DS4.1 verify on the Ultra); with the rows ahead and a submit at each Engram layer, the reads and the build of layers 14-39 overlap the GPU.
PRESTART = os.environ.get("KYOJIN_VERIFY_PRESTART", "0") not in ("0", "")   # flags.py verify_prestart: a lazy-draft verify pass reads its Engram hash ids at the first Engram layer, not at its entry


class EngramAhead:
    MIN_ROWS = 64                                                         # warm only: prompt-sized chunks
    START_ROWS = int(os.environ.get("KYOJIN_ENGRAM_AHEAD_ROWS", "1"))     # 64 = prefill only (the pre-30/10 rule)

    def __init__(self, tables: EngramTables, token_map=None):
        self.tables = tables
        self.jobs: dict = {}
        self.started = 0
        self.token_map = token_map                                        # the model's own map (warm: host hashes of the next chunk)
        self.warmed = 0
        self.warmed_first = 0                                             # flag pf_warm0: prompt heads warmed at admission
        self.rows_total = self.rows_uniq = 0                              # counters (tools/pfcontent.py): positions x columns, distinct rows read
        self.job_seconds = 0.0
        self._warm = None
        self.mute = False                                                 # set by a caller that starts the jobs itself on the joined hashes
        self.waits: dict = {}                                             # prefill: host seconds per Engram layer in its call (row wait + build)
        self.late = None                                                  # verify_prestart: the hash ids of a lazy-draft pass whose host read waits for the first Engram layer
        self.adopt_pending: dict = {}                                     # flag engram_adopt: session id -> _Adopt (the first chunk's row jobs, submitted at admission)
        self.adopt_cur = None                                             # the entry the chunk about to run may adopt (set by adopt_select)
        self.adopt_hit = self.adopt_miss = self.adopt_made = 0
        self.last_adopted = False                                         # the last prefill-sized start() adopted (stats: wait of an adopted vs a cold first chunk)
        self.adopt_debug = os.environ.get("KYOJIN_ENGRAM_ADOPT_DEBUG", "0") not in ("0", "")
        self._rm = None                                                   # RowMap of the admission worker (one thread only)

    def start(self, hashes) -> None:
        if self.mute:
            return
        self.jobs = {}
        if not isinstance(hashes, mx.array) or hashes.ndim != 4 or hashes.shape[1] < self.START_ROWS:
            return
        try:
            h = np.asarray(hashes)                                        # the token embed and the hash kernels only: the GPU queue is short here
        except Exception:                                                 # a compiled trace: no host read, the embeds read as before
            return
        self.started += 1
        small = ER.DEC_QUEUE and h.shape[0] * h.shape[1] < self.MIN_ROWS      # a decode pass: its queue, not behind a prefill chunk's reads and its warm job
        if self.adopt_cur is not None and h.shape[1] >= self.MIN_ROWS:
            got = self._adopt(h)                                          # flag engram_adopt: the jobs submitted at admission, if they are this chunk's own
            if got is not None:
                self.jobs = got
                if self._warm is not None:
                    self.tables._pre.submit(self._warm_job, *self._warm)
                    self._warm = None
                return
        ex = self.tables._pre_small if small else self.tables._pre
        for p in range(h.shape[2]):
            hp = h[:, :, p, :]
            self.jobs[p] = (hp.shape, ex.submit(self._job, p, hp))
        if self._warm is not None:
            self.tables._pre.submit(self._warm_job, *self._warm)
            self._warm = None

    def nested(self, cb):
        """The hook for a decode round inside a prefill chunk (flag mix_slice_ms). The round's own passes reset `jobs`; the chunk's later Engram layers still need theirs,
        or they read their rows again on the engine thread. Flag engram_dec_queue."""
        def run():
            if not ER.DEC_QUEUE:
                return cb()
            saved = self.jobs, self.late
            self.jobs, self.late = {}, None
            try:
                return cb()
            finally:
                self.jobs, self.late = saved
        return run

    def defer(self, hashes) -> None:
        """`start` for a verify pass whose rows are still behind a lazy draft (flag verify_prestart). `start` reads the hash ids back, which waits for the draft to
        finish before the host builds a single layer of the pass, so the GPU idles from the end of the draft until the pre-Engram layers are built and encoded. Here
        the hash kernels are submitted at once (queued right behind the draft), the build goes on, and the first Engram layer submits what is built so far and then
        calls `release` (= `start`): the same ids are read, the same rows fetched, only the order of the host and the GPU changes."""
        self.jobs = {}
        self.late = hashes
        mx.async_eval(hashes)

    def release(self) -> None:
        h, self.late = self.late, None
        if h is not None:
            from . import fired as FD
            FD.hit("verify_prestart")                                     # engage check: the deferred read happened at an Engram layer
            self.start(h)

    def _job(self, p: int, hp: np.ndarray):
        t0 = time.perf_counter()
        uniq, inv = np.unique(hp.reshape(-1), return_inverse=True)
        w, s = self.tables.read_rows(p, uniq)
        self.rows_total += int(hp.size)
        self.rows_uniq += int(uniq.size)
        self.job_seconds += time.perf_counter() - t0
        return hp.shape, uniq, inv, w, s

    def warm(self, ids, a: int, b: int) -> None:
        """Bring the rows of prompt positions [a, b) into the page cache in the background (the engine calls it before the chunk that ends at a:
        the next chunk's reads are then hot, ~0.2 s instead of ~1.4 s cold per layer for 8192 rows). A hint only: the forward reads its own rows.
        The read starts after the current chunk's own reads (same single-worker queue)."""
        if self.token_map is None or b - a < self.MIN_ROWS:
            return
        lo = max(0, a - self.tables.spec.max_ngram_size + 1)                  # the n-grams of position a reach back max_ngram - 1 tokens
        self._warm = (list(ids[lo:b]), a - lo)                                # queued behind this chunk's own reads by `start` (no contention)

    def _warm_job(self, seg, skip):
        from .engram_rows import RowMap
        h = RowMap(self.tables.spec, self.token_map)(seg)[skip:]
        for p in range(h.shape[1]):
            self.tables.read_rows(p, np.unique(h[:, p, :]))
        self.warmed += 1

    def warm_first(self, ids, a: int, b: int) -> bool:
        """Flag pf_warm0: bring the rows of the first chunk of a prompt (positions [a, b)) into the page cache at admission, on a worker of its own.
        `warm` only covers chunks 1+; chunk 0 reads its distinct rows cold (~1.4 s per Engram layer at 8192 rows against ~0.2 s hot). A hint only: the
        forward reads its own rows (same bytes). A separate worker, so a chunk's own row jobs (one worker, layer order) never queue behind it."""
        if self.token_map is None or b - a < self.MIN_ROWS:
            return False
        self.tables._pre_warm.submit(self._warm_job, list(ids[a:b]), 0)
        self.warmed_first += 1
        return True

    # ---- flag engram_adopt: the first chunk's row jobs, submitted when the request is admitted
    def adopt_first(self, sid, ids, a: int, b: int) -> bool:
        """Flag engram_adopt: at admission, hash the prompt positions [a, b) on the host (RowMap, history = the tokens before `a`, as the GPU hash reads it from the cache)
        and submit the REAL row jobs `start` would submit for that chunk, on the chunk's own queue. The chunk adopts them (`_adopt`) when its device hashes are equal.
        The hashing and the submits run on a worker (`_pre_warm`), never on the engine thread. False when nothing was submitted (no token map, a decode-sized chunk)."""
        if self.token_map is None or b - a < self.MIN_ROWS or a < 0:
            return False
        self.adopt_drop(sid)
        while len(self.adopt_pending) >= 4:                               # a queue of admitted prompts: the oldest entry goes (its rows are a page-cache hint then)
            self.adopt_pending.pop(next(iter(self.adopt_pending)))
            self.adopt_miss += 1
        lo = max(0, a - self.tables.spec.max_ngram_size + 1)
        seg, skip = list(ids[lo:b]), a - lo
        ent = types.SimpleNamespace(a=a, b=b, fut=self.tables._pre_warm.submit(self._adopt_job, seg, skip), t=time.perf_counter())
        self.adopt_pending[sid] = ent
        self.adopt_made += 1
        return True

    def _adopt_job(self, seg, skip):
        if self._rm is None:
            self._rm = ER.RowMap(self.tables.spec, self.token_map)
        self._rm.reset()
        h = self._rm(seg)[skip:]                                          # [L, n_layers, n_cols] int64: the device hash's values
        jobs = {}
        for p in range(h.shape[1]):
            hp = np.ascontiguousarray(h[None, :, p, :])                   # [1, L, n_cols], the shape start() gives its jobs
            jobs[p] = (hp.shape, self.tables._pre.submit(self._job, p, hp))
        return h, jobs

    def adopt_drop(self, sid) -> None:
        if self.adopt_pending.pop(sid, None) is not None:
            self.adopt_miss += 1
        if self.adopt_cur is not None and getattr(self.adopt_cur, "sid", None) == sid:
            self.adopt_cur = None

    def adopt_select(self, sid, a: int, b: int) -> None:
        """The engine knows the real first chunk [a, b) (after lookup / restore): the entry of `sid` is armed when it is the same chunk, else dropped (a miss)."""
        ent = self.adopt_pending.pop(sid, None)
        self.adopt_cur = None
        if ent is None:
            return
        if (ent.a, ent.b) != (a, b):
            self.adopt_miss += 1                                          # another cache hit than the peek, or another chunk bounds
            return
        ent.sid = sid
        self.adopt_cur = ent

    def _adopt(self, h: np.ndarray):
        """The jobs of the armed entry as `start` would have submitted them, or None (fallback to today's path) unless its hashes equal the device's bit for bit."""
        ent, self.adopt_cur = self.adopt_cur, None
        self.last_adopted = False
        try:
            if not ent.fut.done():                                        # the worker has not finished the hashes: do not wait for it on the chunk's clock
                self.adopt_miss += 1
                return None
            hh, jobs = ent.fut.result()
            if h.shape[0] != 1 or h.shape[1:] != (hh.shape[0], hh.shape[1], hh.shape[2]) or not np.array_equal(h[0], hh):
                self.adopt_miss += 1
                return None
        except Exception:
            self.adopt_miss += 1
            return None
        if self.adopt_debug:                                              # the adopted rows against a fresh job from the device hashes
            for p in range(h.shape[2]):
                want = self._job(p, h[:, :, p, :])
                got = jobs[p][1].result()
                assert all(np.array_equal(x, y) for x, y in zip(want, got)), f"engram_adopt: layer {p} rows differ"
        self.adopt_hit += 1
        self.last_adopted = True
        return jobs

    def take(self, pos: int, shape):
        job = self.jobs.pop(pos, None)
        return None if job is None or tuple(job[0]) != tuple(shape) else job[1]

    def waiting(self, pos: int) -> bool:
        """True while the rows of Engram layer `pos` are owed by a job (read or not): the layer submits the graph built so far first."""
        return pos in self.jobs


def _ahead_hash(cls, ahead):
    class AheadHash(cls):
        def __call__(self, *a, **k):
            out = super().__call__(*a, **k)
            ahead.start(out)
            return out
    return AheadHash


def _ahead_engram(cls, ahead):
    class AheadEngram(cls):
        def __call__(self, x, hash_ids, token_mask=None):
            if ahead.late is not None:                                    # verify_prestart: the layers built so far queue behind the draft and the hash kernels, then the host reads
                mx.async_eval(x)
                ahead.release()
            if ahead.waiting(self.layer_hash_index):
                mx.async_eval(x)                                          # the GPU runs the layers before this one during the read and the rest of the build
            if x.ndim < 2 or x.shape[1] < ahead.MIN_ROWS:
                return super().__call__(x, hash_ids, token_mask)
            t0 = time.perf_counter()
            out = super().__call__(x, hash_ids, token_mask)
            k = self.layer_hash_index
            ahead.waits[k] = ahead.waits.get(k, 0.0) + time.perf_counter() - t0
            return out
    return AheadEngram


def install_rows_ahead(model, tables: EngramTables):
    """Arm the prefill rows ahead on `tables` (idempotent); KYOJIN_ENGRAM_AHEAD=0 = off. Returns the EngramAhead or None."""
    if os.environ.get("KYOJIN_ENGRAM_AHEAD", "1") == "0" or getattr(tables, "ahead", None) is not None:
        return getattr(tables, "ahead", None)
    lm = model.language_model
    lm._ensure_engram_hash()
    hs = getattr(lm, "engram_hash", None)
    if hs is None:
        return None
    for pos, lid in enumerate(tables.spec.layer_ids):
        assert lm.layers[lid].engram.layer_hash_index == pos, (lid, pos)
    ahead = EngramAhead(tables, np.asarray(hs._token_map) if hasattr(hs, "_token_map") else None)
    hs.__class__ = _ahead_hash(type(hs), ahead)
    object.__setattr__(hs, "_kyojin_ahead", ahead)
    for lid in tables.spec.layer_ids:
        eng = lm.layers[lid].engram
        eng.__class__ = _ahead_engram(type(eng), ahead)
    tables.ahead = ahead
    return ahead


# ----------------------------------------------------------------------------------------------------------------- no host sync in the decode step
# `ParallelRowEmbedding` (above) reads the indices back with np.asarray INSIDE the Python graph build, at the Engram layer (layer 1 of DS4.1): the build
# of the other 42 layers and the whole encode wait for the previous token, so the CPU work of step N + 1 never overlaps the GPU work of step N. Two
# ways out, by where the table lives:
#   DeviceRowEmbedding  the table is resident in unified memory: a pure MLX gather + dequantize on the device, no host read at all. Needs the tables
#                       (2 x 95 GiB for DS4.1) in RAM next to the model: not the 256 GB Ultra (172 GiB pack), a 512 GB machine yes.
#   StagedRowEmbedding  the table stays on disk. The graph is built first with pre-allocated row slots as stand-ins, the 24 rows of each Engram layer
#                       are pread into the slots afterwards (one parallel batch for all layers), and only then is the graph encoded. The hash ids come
#                       from the previous step's own submission (`hashes_ahead`), so waiting for them commits no command buffer.
def _dequant(w, s):
    """mxfp8 rows (uint32 [..., 64] = 256 e4m3 bytes, uint8 [..., 8] e8m0 scales) -> bf16 [..., 256]; the call of the mlx-vlm and pread embeds."""
    return mx.dequantize(w, s, None, group_size=32, bits=8, mode="mxfp8", dtype=mx.bfloat16)


class _Slot:
    """Row buffers the graph reads and the CPU fills in place (an evaluated MLX array is plain unified memory)."""
    __slots__ = ("w", "s", "w_np", "s_np")

    def __init__(self, n: int, head_dim: int):
        self.w = mx.array(np.zeros((n, head_dim // 4), dtype=np.uint32))
        self.s = mx.array(np.zeros((n, head_dim // 32), dtype=np.uint8))
        mx.eval(self.w, self.s)
        self.w_np = np.frombuffer(memoryview(self.w), dtype=np.uint8).reshape(n, head_dim)
        self.s_np = np.frombuffer(memoryview(self.s), dtype=np.uint8).reshape(n, head_dim // 32)


# Commit gate. The staged rows arrive after the step is encoded: with `stage.gate` the command buffers MLX commits for a step are HELD (tools/commitgate,
# a swizzle of -[MTLCommandBuffer commit]) until the host has read the rows into the slots and calls `release`. The step is then encoded, as ONE command
# buffer if need be, while the previous step still runs, and the GPU waits only for the host chain (wake on the ids, row reads, commit), not for the encode.
# Why the wait is on the commit and not in a kernel: a GPU kernel polling a flag the CPU writes does not see the write when the GPU has read that line
# before (measured on the M5, tools/gate_probe.py: a reused flag line never showed the new value, with volatile, atomic, read-modify-write and
# coherent(device) polls alike; only a never-read line did), whereas Metal guarantees that CPU writes made before a commit are visible to that buffer.
_cg_lib = None


def ensure_commitgate(build: bool = True) -> str | None:
    """Path of libcommitgate.dylib (tools/commitgate/commitgate.mm), built with clang++ on first use when it is missing (no machine-local binary in the repo).
    KYOJIN_COMMITGATE overrides the path. None when there is no library and no compiler (the gated decode then stays off)."""
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.environ.get("KYOJIN_COMMITGATE") or os.path.join(here, "..", "tools", "commitgate", "libcommitgate.dylib")
    if os.path.exists(path) or not build or os.environ.get("KYOJIN_COMMITGATE"):
        return path if os.path.exists(path) else None
    src = os.path.join(here, "..", "tools", "commitgate", "commitgate.mm")
    if not os.path.exists(src):
        return None
    try:
        tmp = path + f".{os.getpid()}.tmp"
        subprocess.run(["clang++", "-O2", "-std=c++17", "-fobjc-arc", "-dynamiclib", "-framework", "Metal", "-framework", "Foundation", src, "-o", tmp],
                       check=True, capture_output=True, timeout=180)
        os.replace(tmp, path)                                                # atomic: two servers starting together cannot read a half-written file
        return path
    except Exception:
        return None


def _commit_gate():
    """The libcommitgate.dylib handle (built with tools/commitgate/commitgate.mm; KYOJIN_COMMITGATE overrides the path)."""
    global _cg_lib
    if _cg_lib is None:
        path = ensure_commitgate()
        if path is None:
            raise RuntimeError("libcommitgate.dylib is missing and could not be built (clang++)")
        lib = ctypes.CDLL(path)
        lib.cg_hold.argtypes = [ctypes.c_long]
        lib.cg_release.restype = ctypes.c_long
        lib.cg_seen.restype = ctypes.c_long
        lib.cg_install()
        _cg_lib = lib
    return _cg_lib


class EngramStage:
    """The deferred row fills of one model: `StagedRowEmbedding`s register their slots while the graph is built inside `deferred()`, `fill` reads the
    rows into them. Between the build and `fill` nothing that depends on an Engram row may be evaluated (KYOJIN_ENGRAM_POISON=1 fills the slots
    with e4m3 NaN at registration, so a premature eval shows as NaN logits)."""

    def __init__(self, tables: EngramTables):
        self.tables = tables
        self.embeds: list = []
        self.depth = 0
        self.pending: list = []
        self.pre = None                                                       # input of the first Engram layer of the step being built (see _tap)
        self.gate = False                                                     # hold the step's commits until `release` (pipelined_greedy submits first)
        self.cg_k = None                                                      # command buffers committed per step (measured on the first gated step)
        self.cg_tries = 0
        self.cg_probe: list = []                                              # commits seen per calibration step
        self.cg_expect = 0                                                    # commits that must have been seen before the next step is armed
        self.tolerant = False                                                 # GatedDecode: buffers committed between two steps by other evals are not an error
        self.early = True                                                     # submit the layers before the first Engram layer while the rows are read
        self.rows_read = 0
        self.read_log: list = []                                              # seconds spent in the row reads of each fill (the harness clears it)
        self.poison = bool(os.environ.get("KYOJIN_ENGRAM_POISON"))

    @contextlib.contextmanager
    def deferred(self):
        ahead = getattr(self.tables, "ahead", None)                          # the slots are filled by `fill`: no rows ahead, no submit at the Engram layers
        mute = None if ahead is None else ahead.mute
        if ahead is not None:
            ahead.mute, ahead.jobs = True, {}
        self.depth += 1
        global DEFER_DEPTH
        DEFER_DEPTH += 1
        try:
            yield self
        finally:
            self.depth -= 1
            DEFER_DEPTH -= 1
            if ahead is not None:
                ahead.mute = mute

    def fill(self, host_ids=None) -> int:
        """Read the rows of every registered slot. host_ids: numpy hash ids [B, L, n_layers, n_cols] of the step just built (the evaluated `engram_hashes`);
        None = read each embed's own index array (a host sync when it is not evaluated yet)."""
        reqs = []
        for emb, indices, slot in self.pending:
            idx = np.asarray(indices) if host_ids is None else host_ids[:, :, emb._pos, :]
            assert idx.size * 1 == slot.w_np.shape[0] and tuple(idx.shape) == tuple(indices.shape), (idx.shape, indices.shape)
            reqs.append((emb._pos, idx.reshape(-1), slot.w_np, slot.s_np))
        self.pending = []
        self.pre = None
        t0 = time.perf_counter()
        n = self.tables.read_rows_into(reqs) if reqs else 0
        self.read_log.append(time.perf_counter() - t0)
        self.rows_read += n
        return n

    def resync(self) -> None:
        """Forget the commits of anything else (an mx.synchronize, an eval outside the loop): call when no step of the loop is in flight."""
        self.cg_expect = _commit_gate().cg_seen()

    def hold(self) -> None:
        """Arm the commit gate for the step about to be submitted. The commits of the step before must all have been issued first (they run on MLX's
        stream thread; arming earlier would hold the wrong buffers, and the wait for the ids of that step would never end)."""
        cg = _commit_gate()
        t0 = time.perf_counter()
        while cg.cg_seen() < self.cg_expect:
            if time.perf_counter() - t0 > 10.0:
                raise RuntimeError(f"commit gate: {self.cg_expect - cg.cg_seen()} commits of the previous step never arrived (buffers per step {self.cg_k})")
            time.sleep(0)
        if cg.cg_seen() > self.cg_expect:
            if not self.tolerant:
                raise RuntimeError(f"commit gate: the previous step committed {cg.cg_seen() - self.cg_expect + self.cg_k} buffers, not {self.cg_k}")
            self.cg_expect = cg.cg_seen()                                   # extra buffers of an eval between two steps: already committed, nothing to hold
        cg.cg_hold(self.cg_k)
        self.cg_expect += self.cg_k

    def release(self) -> None:
        """Commit the held buffers of the step submitted last (the rows are in the slots; call after `fill`, and also when the fill failed so that nothing
        stays held)."""
        _commit_gate().cg_release()


class StagedRowEmbedding(nn.Module):
    """`engram.embed` stand-in for a disk-resident table that does not wait for the indices while the graph is built. Outside `EngramStage.deferred()`
    (prefill chunks, verify chains, any caller that does not drive `fill`) it behaves exactly like `ParallelRowEmbedding`."""
    RING = 2                                                                  # slot sets per row count: a step in flight is never overwritten

    def __init__(self, stage: EngramStage, layer_pos: int):
        super().__init__()
        object.__setattr__(self, "_stage", stage)
        object.__setattr__(self, "_pos", layer_pos)
        object.__setattr__(self, "_slots", {})
        object.__setattr__(self, "_next", {})
        object.__setattr__(self, "_sync", ParallelRowEmbedding(stage.tables, layer_pos))
        self.reads = 0

    def _slot(self, n: int) -> _Slot:
        ring = self._slots.get(n)
        if ring is None:
            ring = self._slots[n] = [_Slot(n, self._stage.tables.spec.head_dim) for _ in range(self.RING)]
            self._next[n] = 0
        i = self._next[n]
        self._next[n] = (i + 1) % self.RING
        return ring[i]

    def __call__(self, indices: mx.array) -> mx.array:
        st = self._stage
        if not st.depth:
            out = self._sync(indices)
            self.reads = self._sync.reads
            return out
        n = 1
        for d in indices.shape:
            n *= d
        slot = self._slot(n)
        if st.poison:
            slot.w_np[:] = 0xFF
            slot.s_np[:] = 0xFF
        st.pending.append((self, indices, slot))
        rows = _dequant(slot.w, slot.s)
        return rows.reshape(*indices.shape, rows.shape[-1])


def _tap(engram_cls):
    """Engram subclass that hands the stage its input `x`: everything before the first Engram layer can be submitted to the GPU while the rows are read."""
    class TappedEngram(engram_cls):
        def __call__(self, x, hash_ids, token_mask=None):
            st = self._stage
            if st.depth and st.pre is None:
                st.pre = x
            return super().__call__(x, hash_ids, token_mask)
    return TappedEngram


def install_staged_gather(model, tables: EngramTables) -> EngramStage:
    """Swap every Engram layer's `engram.embed` for a `StagedRowEmbedding`; the returned stage drives the deferred fills (see `pipelined_greedy`)."""
    stage = EngramStage(tables)
    for pos, lid in enumerate(tables.spec.layer_ids):
        emb = StagedRowEmbedding(stage, pos)
        eng = model.language_model.layers[lid].engram
        eng.embed = emb
        if not hasattr(eng, "_stage"):
            object.__setattr__(eng, "_stage", stage)
            eng.__class__ = _tap(type(eng))
        stage.embeds.append(emb)
    install_rows_ahead(model, tables)
    return stage


class DeviceRowEmbedding(nn.Module):
    """`engram.embed` for a table resident in unified memory: indices [...] -> bf16 rows [..., 256] by a device gather of the raw mxfp8 bytes and
    the same `mx.dequantize` as the host paths (so the rows are bit for bit the same). No host read, no sync, nothing to fill. The table is held as
    row chunks so that no single buffer exceeds the device's max buffer length (the real 95 GiB layer is far above it on most machines)."""

    def __init__(self, w_chunks, s_chunks, chunk_rows: int):
        super().__init__()
        object.__setattr__(self, "_w", list(w_chunks))
        object.__setattr__(self, "_s", list(s_chunks))
        object.__setattr__(self, "_cr", int(chunk_rows))

    @classmethod
    def from_tables(cls, tables: EngramTables, layer_pos: int, chunk_rows: int | None = None, block_rows: int = 1 << 22) -> "DeviceRowEmbedding":
        """Load the layer's whole table into device memory (reads it from disk once)."""
        lid = tables.spec.layer_ids[layer_pos]
        hd = tables.spec.head_dim
        rows = tables.loc[lid]["w"][2][0]
        if chunk_rows is None:
            chunk_rows = max(1, min(rows, int(mx.device_info()["max_buffer_length"] * 0.5) // hd))
        w_chunks, s_chunks = [], []
        for c0 in range(0, rows, chunk_rows):
            n = min(chunk_rows, rows - c0)
            parts = []
            for tag, width, dt in (("w", hd, np.uint32), ("s", hd // 32, np.uint8)):
                path, off, _, _, _ = tables.loc[lid][tag]
                src = np.memmap(path, mode="r", dtype=np.uint8, offset=off + c0 * width, shape=(n, width))
                dst = np.empty((n, width // np.dtype(dt).itemsize), dtype=dt)
                raw = dst.view(np.uint8).reshape(n, width)
                for b0 in range(0, n, block_rows):
                    raw[b0:b0 + block_rows] = src[b0:b0 + block_rows]
                parts.append(mx.array(dst))
                del src, dst, raw
            mx.eval(*parts)
            w_chunks.append(parts[0])
            s_chunks.append(parts[1])
        return cls(w_chunks, s_chunks, chunk_rows)

    def __call__(self, indices: mx.array) -> mx.array:
        if len(self._w) == 1:
            return _dequant(self._w[0][indices], self._s[0][indices])
        k = indices // self._cr
        local = indices - k * self._cr
        w = s = None
        for c in range(len(self._w)):
            hit = k == c
            lc = mx.where(hit, local, 0)                                  # rows of other chunks gather row 0 and are masked out
            wc, sc = self._w[c][lc], self._s[c][lc]
            w = wc if w is None else mx.where(hit[..., None], wc, w)
            s = sc if s is None else mx.where(hit[..., None], sc, s)
        return _dequant(w, s)


def install_device_gather(model, tables: EngramTables, chunk_rows: int | None = None) -> list:
    """Swap every Engram layer's `engram.embed` for a `DeviceRowEmbedding` holding the whole table in device memory; returns the modules."""
    out = []
    for pos, lid in enumerate(tables.spec.layer_ids):
        emb = DeviceRowEmbedding.from_tables(tables, pos, chunk_rows)
        model.language_model.layers[lid].engram.embed = emb
        out.append(emb)
    return out


def tables_fit_in_memory(tables: EngramTables, model_bytes: int, wired_bytes: int | None = None, margin: int = 8 << 30) -> bool:
    """True when the Engram tables plus the rest of the model fit under the GPU wired limit (default: `max_recommended_working_set_size`)."""
    wired = wired_bytes if wired_bytes is not None else int(mx.device_info()["max_recommended_working_set_size"])
    need = sum(tables.loc[lid][t][4] for lid in tables.spec.layer_ids for t in ("w", "s"))
    return need + model_bytes + margin <= wired


def install_engram_pipeline(model, tables: EngramTables, model_bytes: int, mode: str = "auto", wired_bytes: int | None = None):
    """Opt-in entry point of the sync-free Engram decode. mode "device" = table resident (needs room for it next to the model: bigger than the 256 GB
    Ultra for DS4.1, whose two tables are 189 GiB beside a 172 GiB pack), "staged" = table on disk, rows read into pre-allocated slots after the graph
    is built, "auto" = device when `tables_fit_in_memory`, else staged. Returns (mode, stage or None); drive the decode with `pipelined_greedy(...,
    stage=stage)`. Nothing is installed by default anywhere: `patch_model` still leaves the stock offloaded embed."""
    if mode == "auto":
        mode = "device" if tables_fit_in_memory(tables, model_bytes, wired_bytes) else "staged"
    if mode == "device":
        install_device_gather(model, tables)
        return mode, None
    assert mode == "staged", mode
    return mode, install_staged_gather(model, tables)


# ----------------------------------------------------------------------------------------------------------------- pipelined greedy decode
def hashes_ahead(model, entry, y: mx.array):
    """Engram hash ids of the positions the tokens `y` [B] will occupy (the cache `entry` is at their offset), built exactly as the model's own forward
    builds them; None when the model has no Engram layer. Updates the history on the cache, so the forward that consumes them must get them as
    `engram_hashes=` and not compute them again."""
    lm = model.language_model
    lm._ensure_engram_hash()
    if lm.engram_hash is None:
        return None
    ids = y[:, None]
    mask = ~(ids == lm.config.image_token_id)
    return entry.map(lambda row, i, m: lm.engram_hash(i, row.offset, row, token_mask=m), ids, mask)


def pipelined_greedy(model, cache: list, token, n: int, stage: EngramStage | None = None, probe=None, trace: list | None = None):
    """Greedy decode with the token feedback on the device, as mlx-lm's generate_step does: step i + 1 is built and submitted before the token of step i
    is read. `token`: the token to feed (int, or mx.array [B]); `cache`: model cache list at that token's offset. Yields the n tokens produced (int, or a
    list for B > 1), the same ones `spec.plain_greedy` produces.
    With `stage` (disk Engram table, `install_staged_gather`) the Engram rows of a step are read AFTER its graph is built and BEFORE it is encoded; the
    hash ids of step i + 1 are part of step i's submission, so that wait commits no command buffer. Without it (resident table, no Engram) there is
    no host read at all between two steps.
    probe(tag), tags "start" / "built" / "filled" / "submitted" per step: instrumentation (timing, command buffer counts). trace: list that receives
    the lazy fp32 logits of every step (for exactness checks; evaluating them early would add commits)."""
    lm = model.language_model
    entry = cache[0]
    note = probe if probe is not None else (lambda tag: None)
    y = token if isinstance(token, mx.array) else mx.array([int(token)])
    ids = hashes_ahead(model, entry, y)
    if ids is not None:
        mx.eval(ids)
    if stage is not None and stage.gate:
        stage.cg_expect = _commit_gate().cg_seen()                                  # nothing of this loop is in flight yet: resync the commit count
    prev = None
    for _ in range(n):
        note("start")
        with stage.deferred() if stage is not None else contextlib.nullcontext():
            lg = lm(y[:, None], cache=cache, engram_hashes=ids).logits[:, -1].astype(mx.float32)
        y_next = mx.argmax(lg, axis=-1)
        ids_next = hashes_ahead(model, entry, y_next)
        if trace is not None:
            trace.append(lg)
        note("built")
        if stage is not None and stage.gate and stage.cg_k is not None:
            # the step is encoded and its commits held while the previous step still runs; the rows are read once that step has finished
            stage.hold()
            note("filled")
            mx.async_eval(y_next) if ids_next is None else mx.async_eval(y_next, ids_next)
            note("submitted")
            try:
                stage.fill(None if ids is None else np.asarray(ids))
            finally:
                stage.release()                                                     # also after a failed fill: nothing may stay held
            note("released")
            if prev is not None:
                yield _py(prev)
            prev, y, ids = y_next, y_next, ids_next
            continue
        if stage is not None:
            if stage.early and not stage.gate and stage.pre is not None:
                mx.async_eval(stage.pre)                                            # the layers before the first Engram layer run while the rows are read
            stage.fill(None if ids is None else np.asarray(ids))                  # waits for step i - 1 only (ids came out of its submission)
        note("filled")
        c0 = _commit_gate().cg_seen() if stage is not None and stage.gate else 0
        mx.async_eval(y_next) if ids_next is None else mx.async_eval(y_next, ids_next)
        if stage is not None and stage.gate:                                        # calibration: commits per step, twice the same value
            mx.eval(y_next)                                                         # waits for the step without a commit of its own (mx.synchronize adds one)
            time.sleep(0.05)                                                        # a trailing buffer (completion signal) can be committed after the results are ready
            k = _commit_gate().cg_seen() - c0
            stage.cg_probe = (stage.cg_probe + [k])[-2:]
            if len(stage.cg_probe) == 2 and stage.cg_probe[0] == stage.cg_probe[1]:
                stage.cg_k, stage.cg_expect = k, _commit_gate().cg_seen()
            else:
                stage.cg_tries += 1
            if stage.cg_k is None and stage.cg_tries >= 6:
                raise RuntimeError(f"commit gate: the number of command buffers per step is not stable ({stage.cg_probe}); run with one buffer per step "
                                   "(MLX_MAX_OPS_PER_BUFFER and MLX_MAX_MB_PER_BUFFER large)")
        note("submitted")
        if prev is not None:
            yield _py(prev)
        prev, y, ids = y_next, y_next, ids_next
    if prev is not None:
        yield _py(prev)


DEFER_DEPTH = 0          # > 0 while a gated decode step is being built (`EngramStage.deferred`): the step must stay one command buffer, nothing inside may sync (dec_filter checks it)


class GatedDecode:
    """The plain decode step of a server (`Engine._launch`) without the Engram host sync: pipelined_greedy's gated step, one launch at a time. Step i + 1 is built
    and encoded while step i runs, its command buffers are held, the Engram rows are read once step i has finished (its hash ids came out of its own submission),
    then the buffers are released: the GPU idles ~1 ms between two steps instead of ~9 (engram_pipe.py, 21.7 vs 30.5 ms/step, bit-identical).
    The hash ids of the NEXT position advance the Engram token history of the cache entry ahead of the step that consumes them: `reset` puts it back
    (call it before anything else touches the batched cache: unbatch, a verify pass, a step outside this class). Needs the staged embeds (`install_staged_gather`),
    libcommitgate.dylib, and one command buffer per step (MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER large, set before mlx is imported)."""

    def __init__(self, model, stage: EngramStage):
        self.model, self.stage = model, stage
        stage.gate = True
        stage.tolerant = True
        self.ahead = None                                                   # (tokens array the ids were made for, ids, history before)
        self.failed = None                                                  # reason, set when the commit count never settles: the owner then drops the gate
        self.ok = True

    @staticmethod
    def _hist(entry):
        return [r.engram for r in entry.rows]

    def reset(self, bc) -> None:
        if self.ahead is not None:
            for r, h in zip(bc[0].rows, self.ahead[2]):
                r.engram = h
            self.ahead = None
        self.stage.pending = []

    def launch(self, bc, toks: mx.array, forward, sample) -> mx.array:
        """forward(toks, engram_hashes) -> fp32 logits (the model step), sample(logits) -> next tokens (lazy). Returns the lazy next tokens, submitted."""
        st, entry = self.stage, bc[0]
        if self.ahead is not None and self.ahead[0] is toks:
            ids = self.ahead[1]
        else:
            self.reset(bc)
            ids = hashes_ahead(self.model, entry, toks)
            mx.eval(ids)
            st.resync()
        with st.deferred():
            lg = forward(toks, ids)
        y = sample(lg)
        hist = self._hist(entry)
        ids_next = hashes_ahead(self.model, entry, y)
        self.ahead = (y, ids_next, hist)
        _HL.mark("gate_build")
        if st.cg_k is not None:
            try:
                st.hold()
            except RuntimeError:                                            # the commit count drifted (something else evaluated between two steps): calibrate again
                st.cg_k, st.cg_probe, st.cg_tries = None, [], 0
        if st.cg_k is not None:
            mx.async_eval(y, ids_next)
            _HL.mark("gate_submit")
            try:
                ids_h = np.asarray(ids)
                _HL.mark("gate_ids_wait")                                   # the GPU idles from here until release(): the inter-step bubble
                st.fill(ids_h)
            finally:
                st.release()
            _HL.mark("gate_fill_release")
            return y
        st.resync()
        c0 = _commit_gate().cg_seen()                                       # calibration: commits per step, twice the same value
        st.fill(np.asarray(ids))
        mx.async_eval(y, ids_next)
        mx.eval(y)
        time.sleep(0.05)
        k = _commit_gate().cg_seen() - c0
        st.cg_probe = (st.cg_probe + [k])[-2:]
        if os.environ.get('KYOJIN_GATE_DEBUG'): print('gate probe k', k, flush=True)
        if len(st.cg_probe) == 2 and st.cg_probe[0] == st.cg_probe[1]:
            st.cg_k, st.cg_expect = k, _commit_gate().cg_seen()
        else:
            st.cg_tries += 1
            if st.cg_tries >= 6:                                                # the step above ran and its tokens are valid: report the failure, do not redo it
                self.failed = f"commit gate: the number of command buffers per step is not stable ({st.cg_probe}); one buffer per step needed"
        return y


def _py(t: mx.array):
    v = np.asarray(t).reshape(-1).tolist()
    return v[0] if len(v) == 1 else v
