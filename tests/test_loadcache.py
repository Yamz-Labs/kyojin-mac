"""G1b: the loader's expert copies read with pread (F_NOCACHE) and leave no page cache behind; the bytes are the plain copy, bit for bit."""
import ctypes
import json
import mmap
import struct
import sys

import numpy as np
import pytest

from kyojin_mlx import foreign_linear as FL
from kyojin_mlx import loadcache as LC
from kyojin_mlx import vendor_input as VI

darwin = pytest.mark.skipif(sys.platform != "darwin", reason="F_NOCACHE / mincore semantics of macOS")
ROWS, COLS, N = 64, 4096, 24            # 24 tensors of 256 KiB (> a few pages each, with an odd 3 B offset between them)


def _write(path, rng):
    ts, hdr, off = [], {}, 0
    for i in range(N):
        t = rng.integers(0, 256, (ROWS, COLS), dtype=np.uint8)
        pad = 3 if i % 2 else 0                                          # tensors that do not start on a page boundary
        hdr[f"t{i}"] = {"dtype": "U8", "shape": [ROWS, COLS], "data_offsets": [off + pad, off + pad + t.nbytes]}
        off += pad + t.nbytes
        ts.append((pad, t))
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h)
        for pad, t in ts:
            f.write(b"\0" * pad + t.tobytes())
    return [t for _, t in ts]


def _resident(path):
    lc = ctypes.CDLL("/usr/lib/libc.dylib")
    lc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        a = np.frombuffer(mm, np.uint8)
        vec = (ctypes.c_char * ((len(mm) + mmap.PAGESIZE - 1) // mmap.PAGESIZE))()
        assert lc.mincore(ctypes.c_void_p(a.__array_interface__["data"][0]), len(mm), vec) == 0
        n = sum(1 for b in vec.raw if b & 1) * mmap.PAGESIZE
        del a
        mm.close()
    return n


@pytest.fixture()
def ckpt(tmp_path, monkeypatch):
    monkeypatch.setattr(LC, "_MAPS", [])
    rng = np.random.default_rng(1)
    ts = _write(tmp_path / "a.safetensors", rng)
    ck = VI.Checkpoint(str(tmp_path))
    return tmp_path / "a.safetensors", ck, ts


def test_copy_into_is_the_plain_stack(ckpt, monkeypatch):
    _, ck, ts = ckpt
    LC.register(ck)
    views = [ck.raw(f"t{i}")[2] for i in range(N)]
    for n in ("1", "8"):
        monkeypatch.setenv("KYOJIN_LOAD_THREADS", n)
        assert np.array_equal(FL._stack_par(views), np.stack(ts))


def test_views_of_views_and_dtype_reshapes(ckpt):
    _, ck, ts = ckpt
    LC.register(ck)
    v = ck.raw("t5")[2].view(np.uint32).reshape(ROWS, COLS // 4)         # what DevPack stacks: a uint32 reshape of the mmap bytes
    out = np.empty(v.shape, v.dtype)
    LC.copy_into(out, v)
    assert out.tobytes() == ts[5].tobytes()
    assert LC._locate(v) is not None


def test_unregistered_and_foreign_arrays_fall_back(ckpt):
    _, ck, ts = ckpt
    v = ck.raw("t1")[2]
    assert LC._locate(v) is None                                         # nothing registered: plain copy
    out = np.empty_like(v)
    LC.copy_into(out, v)
    assert np.array_equal(out, ts[1])
    LC.register(ck)
    heap = np.arange(100, dtype=np.uint8)
    assert LC._locate(heap) is None
    assert LC._locate(v[:, ::2]) is None                                 # not contiguous
    o2 = np.empty(heap.shape, heap.dtype)
    LC.copy_into(o2, heap)
    assert np.array_equal(o2, heap)


def test_flag_off_is_the_plain_copy(ckpt, monkeypatch):
    _, ck, ts = ckpt
    monkeypatch.setenv("KYOJIN_LOAD_NOCACHE", "0")
    assert LC.register(ck) == 0
    assert np.array_equal(FL._stack_par([ck.raw(f"t{i}")[2] for i in range(N)]), np.stack(ts))


@darwin
def test_pages_are_not_kept(ckpt):
    path, ck, ts = ckpt
    LC.register(ck)
    open(path, "rb").read()                                              # warm the whole file
    size = path.stat().st_size
    assert _resident(path) > size * 0.9
    views = [ck.raw(f"t{i}")[2] for i in range(N)]
    FL._stack_par(views)
    assert _resident(path) < size * 0.1                                  # only the partial edge pages (header, odd offsets) stay


@darwin
def test_plain_mmap_copy_keeps_the_pages(ckpt, monkeypatch):
    path, ck, ts = ckpt                                                  # the behaviour G1b removes (control of the test above)
    monkeypatch.setenv("KYOJIN_LOAD_NOCACHE", "0")
    open(path, "rb").read()
    FL._stack_par([ck.raw(f"t{i}")[2] for i in range(N)])
    assert _resident(path) > path.stat().st_size * 0.9


@darwin
def test_drop_all(ckpt):
    path, ck, _ = ckpt
    LC.register(ck)
    open(path, "rb").read()
    assert LC.drop_all() == 1
    assert _resident(path) < path.stat().st_size * 0.05
