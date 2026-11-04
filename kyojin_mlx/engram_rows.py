# SPDX-License-Identifier: Apache-2.0
"""DS4.1 Engram on the Mac: row mapping, shard checks, merged index, row prefetch. numpy + stdlib only (mlx is not needed here).

Engram tables (official shards model-00047 / 00048, F8_E4M3 values [rows, 256] + F8_E8M0 scales [rows, 8], ~95 GiB each) are never loaded: the
model reads the 24 rows (3 n-gram sizes x 8 heads) of every token position per Engram layer. The row ids depend only on the token ids:
    compressed = token_map[token]          (tokens that normalise alike share an id; 99092 ids)
    window     = compressed ids at pos, pos-1, pos-2, pos-3 (pad id before the start)
    hash_n     = (xor of window[k] * mult[layer][k], k <= n)  %  prime[layer][n-1][head]   + offset of that (n, head) bucket range
`RowMap` is a numpy copy of that arithmetic (cross-checked against the box engine and mlx-vlm in tests/test_engram_rows.py and
tools/engram_check.py). `EngramTables` finds the tables in a model directory (merged index or shard headers), validates shapes / dtypes / sizes
against the config and reads rows with parallel preads (`read_rows`, dedicated `prefetch`). The mlx side (a gather module built on `read_rows`,
replacing mlx-vlm's serial memmap gather) is kyojin_mlx/engram_mlx.py."""
from __future__ import annotations

import ctypes
import fcntl
import glob
import json
import os
import time
import struct
import subprocess
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass

import numpy as np

ROW_VALUES = 256                                   # head_dim: one e4m3 byte per value
SCALE_BLOCK = 32                                   # one e8m0 scale per 32 values

# pf_rdadvise (KYOJIN_PF_RDADVISE, default 0): advise the row ranges of a prefill gather before the memory map is touched, so the page cache fills
# in one advisory pass instead of one page fault per cold row (the rival's engram read advises every row). One F_RDADVISE per run of rows within
# ADV_GAP rows of each other; a request that scatters into more than ADV_MAX runs skips the advice (that would cost a syscall per row, which is what
# the map path already pays). The env var is read per call, so the flag is switchable at runtime.
F_RDADVISE = getattr(fcntl, "F_RDADVISE", 44)
ADV_GAP = 16
ADV_MAX = 4096
ADV = {"advised": 0, "runs": 0, "skipped": 0}
STATS = {"calls": 0, "rows": 0, "seconds": 0.0}      # read_rows counters for tools/pfcontent.py


def _rdadvise_on() -> bool:
    return os.environ.get("KYOJIN_PF_RDADVISE", "0") not in ("0", "")


def advise_rows(fd: int, base: int, rows, width: int) -> int:
    """F_RDADVISE the byte range of every row in `rows` (any order) of `width` bytes each, based at file offset `base`. Returns the number of runs
    advised (0 when the flag is off, the list is empty, or it scatters wider than ADV_MAX). Best effort: a refused advice is not an error."""
    if not _rdadvise_on():
        return 0
    r = np.sort(np.asarray(rows, dtype=np.int64).reshape(-1))
    if r.size == 0:
        return 0
    cut = np.flatnonzero(np.diff(r) > ADV_GAP)
    starts = np.concatenate([[0], cut + 1]).tolist()
    ends = np.concatenate([cut + 1, [r.size]]).tolist()
    if len(starts) > ADV_MAX:
        ADV["skipped"] += 1
        return 0
    n = 0
    for a, b in zip(starts, ends):
        try:
            fcntl.fcntl(fd, F_RDADVISE, struct.pack("qi", base + int(r[a]) * width, (int(r[b - 1]) - int(r[a]) + 1) * width))
            n += 1
        except OSError:
            break
    ADV["advised"] += n
    ADV["runs"] += len(starts)
    return n


# ----------------------------------------------------------------------------------------------------------------- safetensors headers
def read_header(path: str) -> tuple[dict, int]:
    """(tensors, data start). tensors: name -> {dtype, shape, data_offsets}; reads only the JSON header."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


# ----------------------------------------------------------------------------------------------------------------- config + hashing
def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def _next_prime(start: int, seen: set) -> int:
    c = start + 1
    while not _is_prime(c) or c in seen:
        c += 1
    return c


@dataclass(frozen=True)
class EngramSpec:
    layer_ids: tuple
    num_embeddings: tuple
    max_ngram_size: int
    vocab_size: int
    n_heads: int
    head_dim: int
    pad_token_id: int
    compressed_vocab_size: int

    @classmethod
    def from_config(cls, cfg: dict) -> "EngramSpec | None":
        """cfg: the pack config.json (or its text_config). None when the model has no Engram layer."""
        t = cfg.get("text_config", cfg)
        if not t.get("engram_layer_ids"):
            return None
        return cls(tuple(t["engram_layer_ids"]), tuple(t["engram_num_embeddings"]), int(t["engram_max_ngram_size"]), int(t["engram_vocab_size"]),
                   int(t["engram_n_heads"]), int(t["engram_head_dim"]), int(t["engram_pad_token_id"]), int(t["engram_compressed_vocab_size"]))

    @property
    def n_cols(self) -> int:
        return (self.max_ngram_size - 1) * self.n_heads

    def primes(self) -> list:
        """[layer][ngram size - 2][head]: one run of primes above vocab_size per (layer, ngram), never reused (as the reference)."""
        out, seen = [], set()
        for _ in self.layer_ids:
            per = []
            for _ in range(self.max_ngram_size - 1):
                cur, sizes = self.vocab_size - 1, []
                for _ in range(self.n_heads):
                    cur = _next_prime(cur, seen)
                    seen.add(cur)
                    sizes.append(cur)
                per.append(sizes)
            out.append(per)
        return out

    def multipliers(self) -> np.ndarray:
        """[layer, max_ngram] odd int64 (numpy PCG64 stream seeded 10007 * layer id), bounded so token * multiplier fits int64."""
        bound = max(1, (np.iinfo(np.int64).max // self.compressed_vocab_size) // 2)
        rows = [np.random.default_rng(10007 * lid).integers(low=0, high=bound, size=(self.max_ngram_size,), dtype=np.int64) * 2 + 1 for lid in self.layer_ids]
        return np.stack(rows)


class RowMap:
    """Token ids -> Engram table rows [positions, n_layers, n_cols] int64 (same values as mlx-vlm NgramHashState and the box NgramHasher).
    `history` carries the compressed ids of the earlier positions between calls (prefill chunks, decode steps)."""

    def __init__(self, spec: EngramSpec, token_map):
        self.spec = spec
        self.token_map = np.asarray(token_map, dtype=np.int64)
        assert int(self.token_map.max()) + 1 == spec.compressed_vocab_size, (int(self.token_map.max()) + 1, spec.compressed_vocab_size)
        self.pad = int(self.token_map[spec.pad_token_id])
        pr = spec.primes()
        self.primes = np.array(pr, dtype=np.int64)                                    # [nl, ng-1, nh]
        flat = [[p for per in layer for p in per] for layer in pr]
        self.offsets = np.array([np.concatenate([[0], np.cumsum(f[:-1])]) for f in flat], dtype=np.int64)    # [nl, n_cols]
        self.mult = spec.multipliers()                                               # [nl, ng]
        self.history = np.zeros(0, dtype=np.int64)

    def reset(self) -> None:
        self.history = np.zeros(0, dtype=np.int64)

    def __call__(self, token_ids, start_pos: int | None = None) -> np.ndarray:
        """Rows for the positions of `token_ids` (1-D, appended to the history; `start_pos` must equal the history length when given)."""
        ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        if start_pos is not None:
            assert start_pos == len(self.history), (start_pos, len(self.history))
        start = len(self.history)
        self.history = np.concatenate([self.history, self.token_map[ids]])
        pos = np.arange(start, len(self.history))
        ng = self.spec.max_ngram_size
        toks = np.stack([np.where(pos - s < 0, self.pad, self.history[np.maximum(pos - s, 0)]) for s in range(ng)], axis=-1)     # [L, ng]
        prod = toks[:, None, :] * self.mult[None]                                     # [L, nl, ng], no overflow by construction
        rolling, hashes = prod[..., 0], []
        for i in range(1, ng):
            rolling = np.bitwise_xor(rolling, prod[..., i])
            hashes.append(rolling[..., None] % self.primes[:, i - 1][None])
        return np.concatenate(hashes, axis=-1) + self.offsets[None]


# ----------------------------------------------------------------------------------------------------------------- value decoding
def _e4m3_lut() -> np.ndarray:
    b = np.arange(256)
    sign = np.where(b & 0x80, -1.0, 1.0)
    e, m = (b >> 3) & 0xF, b & 7
    val = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * 2.0 ** (e.astype(np.float64) - 7))
    val = np.where((e == 15) & (m == 7), np.nan, val)
    return (sign * val).astype(np.float32)


_E4M3 = _e4m3_lut()


def dequant_rows(w: np.ndarray, s: np.ndarray) -> np.ndarray:
    """uint8 values [n, 256] (e4m3) + uint8 scales [n, 8] (e8m0 = 2^(s - 127)) -> float32 [n, 256]."""
    scale = np.exp2(s.astype(np.float32) - 127.0)
    return (_E4M3[w].reshape(len(w), -1, SCALE_BLOCK) * scale[:, :, None]).reshape(len(w), -1)


# ----------------------------------------------------------------------------------------------------------------- tables on disk
def _embed_names(layer_id: int) -> tuple[str, str]:
    return f"layers.{layer_id}.engram.embed.weight", f"layers.{layer_id}.engram.embed.scale"


# ----------------------------------------------------------------------------------------------------------------- native row reads
# A cold prefill-sized read (~196k rows per layer for an 8192-token chunk) is bound by the page-fault path of a memory map: 1.3-2.0 s (values + scales) at 16 threads
# and slower with more (fault contention), ~110k faults/s. Plain preads from 64 native threads reach the NVMe limit (~350k reads/s): 0.40-0.68 s against
# 0.60-1.22 s for the 256-byte value rows on the Ultra. Hot rows (a chunk warmed ahead) stay on the map: preads of cached pages serialize (335 ms
# against 2-65 ms); a mincore sample of the rows picks the path. tools/rowread/rowread.c, built with clang on first use (no machine-local binary in the repo); KYOJIN_ROWREAD=0 or no compiler = the map.
# Narrow rows (the 8-byte scales: 2048 rows a 16 KiB page, a 3 GiB table that stays in the page cache) keep the map: 13-130 ms against ~335 ms of
# preads for 196k rows.
ONE_BATCH = os.environ.get("KYOJIN_ENGRAM_ONE_BATCH", "1") not in ("0", "")   # flags.py engram_one_batch: decode-sized reads of both tags in one batch
DEC_QUEUE = os.environ.get("KYOJIN_ENGRAM_DEC_QUEUE", "0") not in ("0", "")   # flags.py engram_dec_queue
ROWREAD_THREADS = int(os.environ.get("KYOJIN_ROWREAD_THREADS", "64"))
ROWREAD_MIN_WIDTH = 64
ROWREAD_SAMPLE = 512                                                     # rows whose residency picks the path (a warmed chunk is hot: the map)
_rr_fn = None


@dataclass
class _RowRead:
    read: object
    hot: object


def _rowread():
    global _rr_fn
    if _rr_fn is None:
        _rr_fn = False
        if os.environ.get("KYOJIN_ROWREAD", "1") != "0":
            here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "rowread")
            so, src = os.path.join(here, "librowread.dylib"), os.path.join(here, "rowread.c")
            try:
                if os.path.exists(src) and (not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src)):
                    tmp = so + f".{os.getpid()}.tmp"
                    subprocess.run(["clang", "-O2", "-dynamiclib", src, "-o", tmp], check=True, capture_output=True, timeout=120)
                    os.replace(tmp, so)                                  # atomic: two servers starting together cannot load a half-written file
                lib = ctypes.CDLL(so)                                    # CDLL: the GIL is released for the calls
                lib.rr_read.argtypes = [ctypes.c_int, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int]
                lib.rr_read.restype = ctypes.c_int
                lib.rr_hot.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
                lib.rr_hot.restype = ctypes.c_int64
                _rr_fn = _RowRead(lib.rr_read, lib.rr_hot)
            except Exception:
                pass
    return _rr_fn or None


class EngramTables:
    """The Engram embedding tensors of a model directory: located through `model.safetensors.index.json` when it lists them, else by scanning the
    headers of every *.safetensors file. Reads headers only."""

    def __init__(self, directory: str, spec: EngramSpec, threads: int = 16):
        self.dir, self.spec = directory, spec
        idx_path = os.path.join(directory, "model.safetensors.index.json")
        wmap = json.load(open(idx_path))["weight_map"] if os.path.exists(idx_path) else {}
        headers: dict = {}
        self.loc: dict = {}                                    # layer id -> {"w": (path, abs offset, shape, dtype), "s": (...)}
        files_all = sorted(glob.glob(os.path.join(directory, "*.safetensors")))
        for lid in spec.layer_ids:
            self.loc[lid] = {}
            for tag, name in zip(("w", "s"), _embed_names(lid)):
                files = [os.path.join(directory, wmap[name])] if name in wmap else files_all
                for f in files:
                    if f not in headers:
                        headers[f] = read_header(f)
                    h, base = headers[f]
                    if name in h:
                        v = h[name]
                        self.loc[lid][tag] = (os.path.realpath(f), base + v["data_offsets"][0], tuple(v["shape"]), v["dtype"], v["data_offsets"][1] - v["data_offsets"][0])
                        break
        self._fds: dict = {}
        self._mms: dict = {}                                             # (layer id, tag) -> np.memmap of the table (prefill-sized reads)
        self._nthreads = threads
        self._pool = ThreadPoolExecutor(max_workers=threads)             # the parallel preads of read_rows
        self._pre = ThreadPoolExecutor(max_workers=1)                    # prefetch jobs (they call read_rows: a separate pool cannot deadlock it)
        self._pool_small = ThreadPoolExecutor(max_workers=min(threads, 32))   # flag engram_dec_queue: the reads of a decode pass never queue behind a prefill gather
        self._pre_warm = ThreadPoolExecutor(max_workers=1)               # flag pf_warm0: the first chunk's rows, read at admission
        self._pre_small = ThreadPoolExecutor(max_workers=1)              # flag engram_dec_queue: the row jobs of a decode pass, beside the prefill's queue

    # ---- checks
    def validate(self) -> list[str]:
        """Problems found (empty list = the tables match the config): dtypes, shapes, byte sizes, file sizes."""
        sp, bad = self.spec, []
        for pos, lid in enumerate(sp.layer_ids):
            rows = sp.num_embeddings[pos]
            for tag, want_dtype, want_shape in (("w", "F8_E4M3", (rows, sp.head_dim)), ("s", "F8_E8M0", (rows, sp.head_dim // SCALE_BLOCK))):
                name = _embed_names(lid)[0 if tag == "w" else 1]
                if tag not in self.loc[lid]:
                    bad.append(f"{name}: not found in {self.dir}")
                    continue
                path, off, shape, dtype, nbytes = self.loc[lid][tag]
                if dtype != want_dtype:
                    bad.append(f"{name}: dtype {dtype}, expected {want_dtype}")
                if shape != want_shape:
                    bad.append(f"{name}: shape {shape}, expected {want_shape}")
                if nbytes != int(np.prod(shape)):
                    bad.append(f"{name}: {nbytes} bytes for shape {shape}")
                if off + nbytes > os.path.getsize(path):
                    bad.append(f"{name}: data ends at {off + nbytes} beyond the file size {os.path.getsize(path)} ({path})")
        return bad

    # ---- rows
    def _fd(self, path: str) -> int:
        if path not in self._fds:
            self._fds[path] = os.open(path, os.O_RDONLY)
        return self._fds[path]

    def read_rows(self, layer_pos: int, rows) -> tuple[np.ndarray, np.ndarray]:
        """(values uint8 [n, 256], scales uint8 [n, 8]) for the given table rows, in the order asked (parallel preads)."""
        t0 = time.perf_counter()
        out = self._read_rows(layer_pos, rows)
        STATS["calls"] += 1
        STATS["rows"] += out[0].shape[0]
        STATS["seconds"] += time.perf_counter() - t0
        return out

    def _read_rows_one_batch(self, lid: int, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """A decode-sized request (rows < BIG): the value rows and the scale rows of every row in ONE parallel batch (the two tags used to be two
        batches one after the other, each a pool round trip), interleaved over the pool so every NVMe read of the request is in flight together. A
        hot row (page cache, ~1 us) is read on the calling thread: the first slow pread sends the rest to the pool. Same bytes as the per-tag preads."""
        order = np.argsort(rows, kind="stable").tolist()
        bufs, jobs = [], []
        for tag, width in (("w", self.spec.head_dim), ("s", self.spec.head_dim // SCALE_BLOCK)):
            path, off, shape, _, _ = self.loc[lid][tag]
            assert rows.size == 0 or (0 <= rows.min() and rows.max() < shape[0]), f"row out of range for {path}"
            buf = np.empty((rows.size, width), dtype=np.uint8)
            bufs.append(buf)
            mv = memoryview(buf).cast("B")
            fd = self._fd(path)
            jobs.extend((fd, off + int(rows[j]) * width, mv[j * width:(j + 1) * width]) for j in order)
        n_jobs = len(jobs)
        i = 0
        while i < n_jobs:
            fd, pos, view = jobs[i]
            t0 = time.perf_counter()
            if os.preadv(fd, [view], pos) != len(view):
                raise IOError("short read")
            i += 1
            if time.perf_counter() - t0 > self.COLD_S and n_jobs - i >= 8:
                pool = self._pool_small if DEC_QUEUE else self._pool
                nparts = min(self._nthreads, 32, max(1, (n_jobs - i) // 2))
                rest = jobs[i:]

                def work(chunk):
                    for fd, pos, view in chunk:
                        if os.preadv(fd, [view], pos) != len(view):
                            raise IOError("short read")
                list(pool.map(work, [rest[k::nparts] for k in range(nparts)]))
                break
        return bufs[0], bufs[1]

    def _read_rows(self, layer_pos: int, rows) -> tuple[np.ndarray, np.ndarray]:
        lid = self.spec.layer_ids[layer_pos]
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        if ONE_BATCH and rows.size < self.BIG and all(t in self.loc[lid] for t in ("w", "s")):
            return self._read_rows_one_batch(lid, rows)
        out = []
        for tag, width in (("w", self.spec.head_dim), ("s", self.spec.head_dim // SCALE_BLOCK)):
            path, off, shape, _, _ = self.loc[lid][tag]
            assert rows.size == 0 or (0 <= rows.min() and rows.max() < shape[0]), f"row out of range for {path}"
            if rows.size >= self.BIG:
                out.append(self._take_rows(lid, tag, rows, width))
                continue
            fd = self._fd(path)
            buf = np.empty((rows.size, width), dtype=np.uint8)
            order = np.argsort(rows, kind="stable")

            def work(chunk, fd=fd, off=off, width=width, buf=buf):
                for j in chunk:
                    b = os.pread(fd, width, off + int(rows[j]) * width)
                    assert len(b) == width, "short read"
                    buf[j] = np.frombuffer(b, dtype=np.uint8)
            parts = np.array_split(order, min(self._nthreads, max(1, order.size // 4)))
            list((self._pool_small if DEC_QUEUE else self._pool).map(work, parts))
            out.append(buf)
        return out[0], out[1]

    BIG = 2048                                                       # rows from which read_rows gathers from a memory map instead of one pread per row

    def _take_rows(self, lid: int, tag: str, rows: np.ndarray, width: int) -> np.ndarray:
        """read_rows for a prefill-sized request: np.take from a persistent read-only memory map of the table, the sorted rows split in contiguous
        parts over the pool (numpy releases the GIL in the gather; one pread per row costs ~5 us of Python each, 2 s for the ~196k rows of an
        8192-token DS4.1 chunk, against 0.2 s hot / 1.4 s cold here). Same bytes as the preads. The mapped pages are clean file pages (page cache)."""
        k = (lid, tag)
        mm = self._mms.get(k)
        if mm is None:
            path, off, shape, _, _ = self.loc[lid][tag]
            mm = self._mms[k] = np.memmap(path, dtype=np.uint8, mode="r", offset=off, shape=shape)
        rr = _rowread()
        if rr is not None and width >= ROWREAD_MIN_WIDTH:                # wide rows (one page each), mostly cold: native preads, 35-42 % under the map
            r = np.ascontiguousarray(rows, dtype=np.int64)
            if rr.hot(mm.ctypes.data, r.ctypes.data, r.size, width, ROWREAD_SAMPLE) * 2 < min(r.size, ROWREAD_SAMPLE):
                path, off, _, _, _ = self.loc[lid][tag]
                dst = np.empty((rows.size, width), dtype=np.uint8)
                if rr.read(self._fd(path), off, r.ctypes.data, r.size, width, dst.ctypes.data, ROWREAD_THREADS) != 0:
                    raise IOError(f"short read in {path}")
                return dst
        order = None if bool(np.all(rows[1:] >= rows[:-1])) else np.argsort(rows, kind="stable")
        srt = rows if order is None else rows[order]
        path, off, _, _, _ = self.loc[lid][tag]
        advise_rows(self._fd(path), off, srt, width)                      # pf_rdadvise: warm the runs before the map faults them one page at a time
        dst = np.empty((rows.size, width), dtype=np.uint8)
        cut = np.linspace(0, rows.size, self._nthreads + 1).astype(np.int64)
        list(self._pool.map(lambda ab: np.take(mm, srt[ab[0]:ab[1]], axis=0, out=dst[ab[0]:ab[1]]), zip(cut[:-1].tolist(), cut[1:].tolist())))
        if order is None:
            return dst
        buf = np.empty_like(dst)
        buf[order] = dst
        return buf

    COLD_S = 40e-6                                                   # a pread slower than this is a miss of the page cache (see read_rows_into)

    def read_rows_into(self, reqs) -> int:
        """Fill caller-owned buffers in ONE parallel batch of preads. reqs: [(layer_pos, rows, w_out, s_out)] with rows int [n] (any order, repeats
        allowed), w_out / s_out writable C-contiguous uint8 buffers of n * head_dim / n * head_dim // 32 bytes (numpy arrays or a view of an MLX array's
        memory). No copy: each pread lands in its row of the buffer. Returns the number of rows read (all layers)."""
        jobs = []
        for layer_pos, rows, w_out, s_out in reqs:
            lid = self.spec.layer_ids[layer_pos]
            rows = np.asarray(rows, dtype=np.int64).reshape(-1)
            for tag, width, out in (("w", self.spec.head_dim, w_out), ("s", self.spec.head_dim // SCALE_BLOCK, s_out)):
                path, off, shape, _, _ = self.loc[lid][tag]
                assert rows.size == 0 or (0 <= rows.min() and rows.max() < shape[0]), f"row out of range for {path}"
                mv = memoryview(out).cast("B")
                assert len(mv) == rows.size * width, (len(mv), rows.size, width)
                fd = self._fd(path)
                jobs.extend((fd, off + r * width, mv[j * width:(j + 1) * width]) for j, r in enumerate(rows.tolist()))
        if not jobs:
            return 0

        def work(chunk):
            for fd, pos, view in chunk:
                if os.preadv(fd, [view], pos) != len(view):
                    raise IOError("short read")
        # Hot rows (page cache) cost ~1 us a pread: reading them in the calling thread is 5-8x cheaper than waking the pool (0.06 against 0.3-0.5 ms
        # for 24 rows on the Air). Cold rows (NVMe, 80+ us) need the parallel batch: the first slow pread sends the rest to the pool.
        i, n_jobs = 0, len(jobs)
        while i < n_jobs:
            fd, pos, view = jobs[i]
            t0 = time.perf_counter()
            if os.preadv(fd, [view], pos) != len(view):
                raise IOError("short read")
            i += 1
            if time.perf_counter() - t0 > self.COLD_S and n_jobs - i >= 8:
                nparts = min(self._nthreads, max(1, (n_jobs - i) // 4))
                rest = jobs[i:]
                list(self._pool.map(work, [rest[k::nparts] for k in range(nparts)]))
                break
        return n_jobs // 2

    def prefetch(self, hashes: np.ndarray) -> Future:
        """Warm the page cache for hash ids [..., n_layers, n_cols] (RowMap / NgramHashState output) in the background; returns a Future
        (result = rows touched). Useful when the ids are known before the gather (a prefill chunk's ids are: warm chunk N+1 while N computes)."""
        h = np.asarray(hashes, dtype=np.int64)
        per_layer = [np.unique(h[..., p, :].reshape(-1)) for p in range(len(self.spec.layer_ids))]

        def run():
            n = 0
            for p, rows in enumerate(per_layer):
                self.read_rows(p, rows)
                n += rows.size
            return n
        return self._pre.submit(run)

    def close(self) -> None:
        self._pre.shutdown(wait=True)
        self._pre_warm.shutdown(wait=True)
        self._pre_small.shutdown(wait=True)
        self._pool_small.shutdown(wait=True)
        self._pool.shutdown(wait=True)
        for fd in self._fds.values():
            os.close(fd)
        self._fds = {}


# ----------------------------------------------------------------------------------------------------------------- merged index
def merged_index(model_dir: str, spec: EngramSpec | None = None) -> dict:
    """weight_map for a model directory holding converted pack shards and (symlinks to) the official Engram shards. Header-only.
    A file with `.engram.embed.` tensors is an Engram shard: it also carries q_weight / k_weight / wkv of the Engram layers, but the pack has
    its own copy of those, so such a file contributes only its `layers.N.engram.embed.{weight,scale}` (the pack wins on every other name).
    Raises on a name present in two pack files, or (with `spec`) when an Engram table is missing."""
    wmap, sources, engram_files = {}, {}, []
    for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        h, _ = read_header(f)
        if any(".engram.embed." in k for k in h):
            engram_files.append((f, h))
            continue
        for k in h:
            if k in wmap:
                raise ValueError(f"tensor {k} is in {sources[k]} and in {f}")
            wmap[k], sources[k] = os.path.basename(f), f
    for f, h in engram_files:
        for k in h:
            if ".engram.embed." in k:
                if k in wmap:
                    raise ValueError(f"tensor {k} is in {sources[k]} and in {f}")
                wmap[k], sources[k] = os.path.basename(f), f
    if spec is not None:
        for lid in spec.layer_ids:
            for name in _embed_names(lid):
                if name not in wmap:
                    raise ValueError(f"{name} is in no shard of {model_dir}")
    return wmap


def link_engram_shards(model_dir: str, engram_dir: str) -> list[str]:
    """Symlink the Engram shards (the *.safetensors of `engram_dir` with embed tensors) into `model_dir`; returns the link names."""
    names = []
    for f in sorted(glob.glob(os.path.join(engram_dir, "*.safetensors"))):
        h, _ = read_header(f)
        if not any(".engram.embed." in k for k in h):
            continue
        link = os.path.join(model_dir, os.path.basename(f))
        if os.path.lexists(link):
            if os.path.realpath(link) == os.path.realpath(f):
                names.append(os.path.basename(f))
                continue
            raise FileExistsError(link)
        os.symlink(os.path.abspath(f), link)
        names.append(os.path.basename(f))
    return names


def write_merged_index(model_dir: str, wmap: dict) -> str:
    """Write model.safetensors.index.json (total_size = bytes of the tensors listed; header-only)."""
    total, hdrs = 0, {}
    for k, f in wmap.items():
        if f not in hdrs:
            hdrs[f] = read_header(os.path.join(model_dir, f))[0]
        a, b = hdrs[f][k]["data_offsets"]
        total += b - a
    path = os.path.join(model_dir, "model.safetensors.index.json")
    json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(wmap.items()))}, open(path + ".tmp", "w"), indent=1)
    os.replace(path + ".tmp", path)
    return path
