"""Page-cache hygiene of the safetensors loader (G1b).

Problem (measured 15/11 on the 169 GB GLM-5.3 pack, 256 GiB box): the loader read every expert tensor through an mmap view, so each byte it consumed stayed in
the page cache (file-backed, "active" because just touched) while the resident weights it built (anonymous Metal buffers, idle until the first token) grew to
~175 GB. With a warm cache, cache + weights > RAM, and macOS then compresses the idle *anonymous* weights instead of dropping the clean file pages:
compressor +136 GiB in 60 s, which is what the host memory guard (+8 GiB / 60 s) kills. FP4 data is incompressible, so the compressor only burns CPU.

Fix: the loader copies a weight out of a shard with a pread on an F_NOCACHE descriptor (cold: 11 GiB/s on this SSD against 2.6 GiB/s for faulting the mmap in;
warm: served from the cache at memcpy speed) and then drops the pages it just consumed (msync MS_INVALIDATE on the page-aligned range: clean pages of a
read-only file are discarded, nothing is written). The cache a load leaves behind is thereby about the size of what it has not read yet, never cache + weights.
Nothing here changes a byte of what is loaded; `KYOJIN_LOAD_NOCACHE=0` restores the plain mmap copy.
"""
from __future__ import annotations

import bisect
import ctypes
import mmap
import os
import sys
import threading
import weakref

import numpy as np

F_NOCACHE = 48            # <sys/fcntl.h> (macOS)
MS_INVALIDATE = 2         # <sys/mman.h>
_PAGE = mmap.PAGESIZE
_LOCK = threading.Lock()
_MAPS: list = []          # sorted by base address: (base, size, weakref(SafetensorsFile), fd)
_libc = None


def enabled() -> bool:
    return sys.platform == "darwin" and os.environ.get("KYOJIN_LOAD_NOCACHE", "1") != "0"


def _lc():
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL("/usr/lib/libc.dylib", use_errno=True)
        _libc.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    return _libc


def register(ck) -> int:
    """Index the mmap of every file of the vendor_input.Checkpoint `ck` (arrays that are views of it can then be copied with `copy_into`). Returns the file count."""
    if not enabled():
        return 0
    n = 0
    with _LOCK:
        dead = [m for m in _MAPS if m[2]() is None or m[2]()._mm is None]
        _MAPS[:] = [m for m in _MAPS if m not in dead]
        for m in dead:
            try:
                os.close(m[3])
            except OSError:
                pass
        for sf in ck.files.values():
            mm = getattr(sf, "_mm", None)
            if mm is None or any(m[2]() is sf for m in _MAPS):
                continue
            base = np.frombuffer(mm, np.uint8, count=1).__array_interface__["data"][0]
            fd = os.open(sf.path, os.O_RDONLY)
            import fcntl
            fcntl.fcntl(fd, F_NOCACHE, 1)
            bisect.insort(_MAPS, (base, len(mm), weakref.ref(sf), fd), key=lambda m: m[0])
            n += 1
    return n


def _locate(a):
    """(fd, file offset, map base, map size) of a C-contiguous numpy view of a registered mmap, else None."""
    if not _MAPS or not a.flags.c_contiguous or a.nbytes == 0:
        return None
    p = a.__array_interface__["data"][0]
    i = bisect.bisect_right(_MAPS, p, key=lambda m: m[0]) - 1
    if i < 0:
        return None
    base, size, ref, fd = _MAPS[i]
    sf = ref()
    if sf is None or sf._mm is None or p + a.nbytes > base + size:
        return None
    return fd, p - base, base, size


def drop(base: int, off: int, n: int) -> None:
    """Discard the cached pages fully inside [off, off+n) of the map at `base` (partial edge pages stay: a neighbour tensor may still need them)."""
    lo = (base + off + _PAGE - 1) // _PAGE * _PAGE
    hi = (base + off + n) // _PAGE * _PAGE
    if hi > lo:
        _lc().msync(ctypes.c_void_p(lo), hi - lo, MS_INVALIDATE)


def copy_into(dst: np.ndarray, src: np.ndarray) -> None:
    """np.copyto(dst, src), except that when `src` is a registered mmap view it is read with pread (F_NOCACHE) and its pages are dropped afterwards."""
    loc = _locate(src) if (enabled() and dst.flags.c_contiguous and dst.dtype == src.dtype and dst.shape == src.shape) else None
    if loc is None:
        np.copyto(dst, src)
        return
    fd, off, base, _ = loc
    mv = memoryview(dst.reshape(-1).view(np.uint8))
    n, got = len(mv), 0
    while got < n:
        r = os.preadv(fd, [mv[got:]], off + got)
        if r <= 0:
            raise OSError(f"short read at {off + got} of {n} bytes")
        got += r
    drop(base, off, n)


def drop_all() -> int:
    """Discard the cached pages of every registered file (end of a load: what the dense / MTP tensors read through mmap or mx.load). Returns the file count."""
    if not enabled():
        return 0
    n = 0
    with _LOCK:
        for base, size, ref, _ in _MAPS:
            if ref() is not None:
                drop(base, 0, size)
                n += 1
    return n
