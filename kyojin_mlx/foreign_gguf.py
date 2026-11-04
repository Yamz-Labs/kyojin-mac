# SPDX-License-Identifier: Apache-2.0
"""GGUF reader (numpy + mmap, no llama.cpp) and the block decoders of the types it serves.

Layouts are those of ggml/src/ggml-common.h + ggml-quants.c (llama.cpp, MIT; read from the reference text, see THIRD_PARTY_NOTICES.md).
`GgufFile` maps the file once; `tensor_raw` is a zero-copy view of a tensor's bytes. A tensor of ggml shape ne = [k, n, ...] (k fastest)
is `n` rows of `k` inputs, each row `k / block_elems` consecutive blocks: the [out, in] matrix of a linear.

`dequant(type, raw_blocks)` decodes every type listed in DEQUANT to float32 (vectorised, checked in tests against a literal transcription of the
reference C loops). `to_pack(...)` re-lays the 4-bit types the Metal kernels read into foreign_pack.Pack (a lossless re-ordering of the same bits).
A type outside DEQUANT raises GgufTypeError naming it: nothing is guessed."""
from __future__ import annotations

import mmap
import os
import struct

import numpy as np

from . import foreign_grids as G
from . import foreign_grids_gen as GG

MAGIC = b"GGUF"
ALIGN_DEFAULT = 32

# ggml_type: name, block elements, block bytes (ggml.h enum; ggml-common.h structs)
TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292), 16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66), 39: ("MXFP4", 32, 17), 40: ("NVFP4", 64, 36),
}
NAME_TO_ID = {v[0]: k for k, v in TYPES.items()}
DEQUANT = ("F32", "F16", "BF16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q4_K", "Q5_K", "Q6_K", "IQ3_XXS", "IQ3_S", "IQ4_NL", "IQ4_XS", "MXFP4", "NVFP4",
           "Q8_1", "Q2_K", "Q3_K", "Q8_K", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ1_S", "IQ1_M", "TQ1_0", "TQ2_0")
KERNEL_TYPES8 = ("Q8_0",)                                               # read as stored only with flag fp8stored (foreign_linear.fp8stored)
KERNEL_TYPESK = ("Q5_K", "Q6_K", "IQ4_XS", "Q3_K", "IQ3_S", "IQ3_XXS", "Q2_K", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ1_S", "IQ1_M")                              # read as stored only with flag kstored (foreign_linear.kstored)
KERNEL_TYPES = ("Q4_0", "Q4_K", "IQ4_NL", "MXFP4", "NVFP4")           # types the Metal kernels read as stored (after a bit re-ordering)
KVALUES_IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.float32)
KVALUES_FP4 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], np.float32)       # doubled E2M1 (ggml-common.h kvalues_fp4)


class GgufError(ValueError):
    pass


class GgufTypeError(GgufError):
    pass


# ---------------------------------------------------------------------------------------------------------------- container
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
MAX_STR = 1 << 24
MAX_ARR = 1 << 26


class _Cur:
    def __init__(self, buf, pos=0):
        self.b, self.p = buf, pos

    def take(self, n):
        if n < 0 or self.p + n > len(self.b):
            raise GgufError("truncated GGUF header")
        v = self.b[self.p:self.p + n]
        self.p += n
        return v

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        n = self.u64()
        if n > MAX_STR:
            raise GgufError(f"string of {n} bytes")
        return bytes(self.take(n)).decode("utf-8", "replace")

    def value(self, t):
        if t in _SCALAR:
            f = _SCALAR[t]
            return struct.unpack(f, self.take(struct.calcsize(f)))[0]
        if t == 8:
            return self.string()
        if t == 9:
            et, n = self.u32(), self.u64()
            if n > MAX_ARR:
                raise GgufError(f"array of {n} elements")
            if et in _SCALAR:
                f = _SCALAR[et]
                sz = struct.calcsize(f)
                return list(struct.unpack("<" + f[1:] * n, self.take(sz * n))) if n else []
            return [self.value(et) for _ in range(n)]
        raise GgufError(f"unknown GGUF metadata value type {t}")


class TensorInfo:
    __slots__ = ("name", "ne", "type", "offset", "nbytes", "tname")

    def __init__(self, name, ne, type_, offset):
        self.name, self.ne, self.type, self.offset = name, tuple(ne), type_, offset
        if type_ not in TYPES:
            raise GgufTypeError(f"tensor {name}: ggml type id {type_} is not in the type table of this reader")
        self.tname, be, bb = TYPES[type_]
        n = int(np.prod(self.ne, dtype=np.int64))
        if n % be:
            raise GgufError(f"tensor {name}: {n} elements is not a multiple of the {self.tname} block of {be}")
        if self.ne[0] % be:
            raise GgufError(f"tensor {name}: row length {self.ne[0]} is not a multiple of the {self.tname} block of {be}")
        self.nbytes = n // be * bb

    @property
    def shape(self):
        """numpy order (slowest first): [.., n, k]."""
        return tuple(reversed(self.ne))


class GgufFile:
    def __init__(self, path):
        self.path = str(path)
        self._f = open(self.path, "rb")
        size = os.fstat(self._f.fileno()).st_size
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ) if size else None
        if size < 24 or self._mm[:4] != MAGIC:
            raise GgufError(f"{path}: not a GGUF file")
        c = _Cur(memoryview(self._mm), 4)
        self.version = c.u32()
        if self.version not in (2, 3):
            raise GgufError(f"{path}: GGUF version {self.version} (this reader: 2 and 3, little endian)")
        nt, nkv = c.u64(), c.u64()
        if nt > 1 << 22 or nkv > 1 << 22:
            raise GgufError("implausible tensor / key count")
        self.kv = {}
        for _ in range(nkv):
            k = c.string()
            self.kv[k] = c.value(c.u32())
        self.tensors = {}
        for _ in range(nt):
            name = c.string()
            nd = c.u32()
            if nd > 8:
                raise GgufError(f"tensor {name}: {nd} dimensions")
            ne = [c.u64() for _ in range(nd)]
            t = c.u32()
            off = c.u64()
            if name in self.tensors:
                raise GgufError(f"duplicate tensor {name}")
            self.tensors[name] = TensorInfo(name, ne, t, off)
        al = int(self.kv.get("general.alignment", ALIGN_DEFAULT))
        if al <= 0 or al & (al - 1):
            raise GgufError(f"general.alignment {al}")
        self.data_start = (c.p + al - 1) // al * al
        for ti in self.tensors.values():
            if self.data_start + ti.offset + ti.nbytes > size:
                raise GgufError(f"tensor {ti.name}: data past the end of the file")

    def close(self):
        self._mm = None
        self._f.close()

    def raw(self, name) -> np.ndarray:
        """uint8 view of the tensor's bytes, shape [rows.., row_bytes] (zero copy)."""
        ti = self.tensors[name]
        _, be, bb = TYPES[ti.type]
        a = np.frombuffer(self._mm, dtype=np.uint8, count=ti.nbytes, offset=self.data_start + ti.offset)
        return a.reshape(ti.shape[:-1] + (ti.ne[0] // be * bb,))

    def tensor_f32(self, name) -> np.ndarray:
        ti = self.tensors[name]
        return dequant(ti.tname, self.raw(name), ti.ne[0]).reshape(ti.shape)


def find_gguf(path):
    """The single .gguf of a directory (or the file itself). Split shards (-0000x-of-0000y) are refused: one file per model here."""
    p = str(path)
    if os.path.isfile(p) and p.endswith(".gguf"):
        return p
    fs = sorted(f for f in os.listdir(p) if f.endswith(".gguf")) if os.path.isdir(p) else []
    fs = [f for f in fs if not f.startswith("mmproj")]
    if len(fs) > 1:
        raise GgufError(f"{p}: {len(fs)} .gguf files ({fs[:3]}...): split / multi-file GGUF is not read; point at one file")
    return os.path.join(p, fs[0]) if fs else ""


# ---------------------------------------------------------------------------------------------------------------- block decoders
def _f16(b2):
    return np.ascontiguousarray(b2).view("<f2").astype(np.float32)[..., 0]


def _blocks(raw, be, bb, k):
    """raw [.., row_bytes] -> [rows, nb, bb] uint8."""
    r = np.ascontiguousarray(raw)
    return r.reshape(-1, k // be, bb)


def _scale_min_k4(s):
    """12-byte K-quant scale block [.., 12] -> (sc, mn) uint8 [.., 8] (get_scale_min_k4)."""
    sc, mn = np.empty(s.shape[:-1] + (8,), np.uint8), np.empty(s.shape[:-1] + (8,), np.uint8)
    sc[..., :4], mn[..., :4] = s[..., 0:4] & 63, s[..., 4:8] & 63
    sc[..., 4:] = (s[..., 8:12] & 15) | ((s[..., 0:4] >> 6) << 4)
    mn[..., 4:] = (s[..., 8:12] >> 4) | ((s[..., 4:8] >> 6) << 4)
    return sc, mn


def ue4m3(b):
    """A plain unsigned E4M3 value (ggml_ue4m3_to_fp32 returns half of it, because its kvalues_fp4 are doubled); 0 and 0x7F decode to 0."""
    b = np.asarray(b, np.uint8)
    e, m = ((b >> 3) & 15).astype(np.int32), (b & 7).astype(np.float32)
    v = np.where(e == 0, np.ldexp(m, -9), np.ldexp(1.0 + m / 8.0, e - 7)).astype(np.float32)
    return np.where((b == 0) | (b == 0x7F), 0.0, v).astype(np.float32)


def e8m0(b):
    return np.ldexp(np.float32(1.0), np.asarray(b, np.int32) - 127).astype(np.float32)


_CHUNK_ELEMS = 1 << 22            # rows per decode chunk: bounds the float32 temporaries (about 6 x chunk) and gives the thread pool independent pieces


def dequant(tname: str, raw, k: int) -> np.ndarray:
    """raw uint8 [.., row_bytes] of rows of `k` elements -> float32 [rows.., k].  Rows are independent: big inputs are decoded in row chunks on a
    thread pool (numpy releases the GIL in its array loops); the values are the same as one single call (every row is a function of its own bytes)."""
    r = np.asarray(raw)
    lead = r.shape[:-1]
    nrow = int(np.prod(lead, dtype=np.int64)) if lead else 1
    step = max(1, _CHUNK_ELEMS // max(k, 1))
    if tname in ("F32", "F16", "BF16") or nrow <= step:
        return _dequant_rows(tname, raw, k)
    flat = r.reshape(nrow, r.shape[-1])
    out = np.empty((nrow, k), np.float32)

    def one(i):
        out[i:i + step] = _dequant_rows(tname, flat[i:i + step], k)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max(1, min(8, os.cpu_count() or 1))) as ex:
        list(ex.map(one, range(0, nrow, step)))
    return out.reshape(lead + (k,))


_NO_FRONT_D = frozenset(("MXFP4", "NVFP4", "Q6_K", "Q2_K", "Q3_K", "Q8_K", "IQ1_M", "TQ2_0", "TQ1_0"))


def _dequant_rows(tname: str, raw, k: int) -> np.ndarray:
    if tname not in TYPES_BY_NAME:
        raise GgufTypeError(f"GGUF type {tname}: unknown to this reader")
    if tname not in DEQUANT:
        raise GgufTypeError(f"GGUF type {tname}: its layout is not implemented in kyojin_mlx.foreign_gguf (supported: {', '.join(DEQUANT)}); "
                            f"not guessed. Re-quantise this tensor to one of those types.")
    lead = np.asarray(raw).shape[:-1]
    _, be, bb = TYPES_BY_NAME[tname]
    if tname in ("F32", "F16", "BF16"):
        r = np.ascontiguousarray(raw)
        if tname == "F32":
            return r.view("<f4").reshape(lead + (k,)).astype(np.float32)
        if tname == "F16":
            return r.view("<f2").reshape(lead + (k,)).astype(np.float32)
        return (r.view("<u2").astype(np.uint32) << 16).view(np.float32).reshape(lead + (k,))
    B = _blocks(raw, be, bb, k)
    nr, nb = B.shape[:2]
    d = None if tname in _NO_FRONT_D else _f16(B[..., 0:2])[..., None]     # these types keep other bytes there (Q6_K: ql), read as fp16 they gave NaN warnings
    lo = lambda q: (q & 15).astype(np.float32)
    hi = lambda q: (q >> 4).astype(np.float32)
    if tname == "Q4_0":
        q = B[..., 2:18]
        y = np.concatenate([(lo(q) - 8) * d, (hi(q) - 8) * d], -1)
    elif tname == "Q4_1":
        m = _f16(B[..., 2:4])[..., None]
        q = B[..., 4:20]
        y = np.concatenate([lo(q) * d + m, hi(q) * d + m], -1)
    elif tname in ("Q5_0", "Q5_1"):
        o = 2 if tname == "Q5_0" else 4
        m = _f16(B[..., 2:4])[..., None] if tname == "Q5_1" else 0.0
        qh = np.ascontiguousarray(B[..., o:o + 4]).view("<u4")[..., 0].astype(np.uint32)       # [nr, nb]
        q = B[..., o + 4:o + 20]
        j = np.arange(16, dtype=np.uint32)
        xh0 = (((qh[..., None] >> j) << 4) & 0x10).astype(np.float32)
        xh1 = ((qh[..., None] >> (j + 12)) & 0x10).astype(np.float32)
        x0, x1 = lo(q) + xh0, hi(q) + xh1
        if tname == "Q5_0":
            x0, x1 = x0 - 16, x1 - 16
        y = np.concatenate([x0 * d + m, x1 * d + m], -1) if tname == "Q5_1" else np.concatenate([x0 * d, x1 * d], -1)    # Q5_0: no `+ 0.0`, it would turn -0.0 into +0.0
    elif tname == "Q8_0":
        y = B[..., 2:34].view(np.int8).astype(np.float32) * d
    elif tname == "IQ4_NL":
        q = B[..., 2:18]
        y = np.concatenate([KVALUES_IQ4NL[q & 15], KVALUES_IQ4NL[q >> 4]], -1) * d
    elif tname == "MXFP4":
        dd = e8m0(B[..., 0])[..., None] * 0.5
        q = B[..., 1:17]
        y = np.concatenate([KVALUES_FP4[q & 15], KVALUES_FP4[q >> 4]], -1) * dd
    elif tname == "NVFP4":
        sc = ue4m3(B[..., 0:4]) * 0.5                                    # [nr, nb, 4]
        q = B[..., 4:36].reshape(nr, nb, 4, 8)
        v = np.concatenate([KVALUES_FP4[q & 15], KVALUES_FP4[q >> 4]], -1)       # [nr, nb, 4, 16]: byte j -> element j and j + 8 of its 16-sub-block
        y = v * sc[..., None]
    elif tname == "Q4_K":
        dm = _f16(B[..., 2:4])[..., None, None, None]
        sc, mn = _scale_min_k4(B[..., 4:16])
        q = B[..., 16:144].reshape(nr, nb, 4, 32)
        qq = np.stack([q & 15, q >> 4], 3).astype(np.float32)             # [nr, nb, 4, 2, 32]
        s = sc.reshape(nr, nb, 4, 2, 1).astype(np.float32)
        m = mn.reshape(nr, nb, 4, 2, 1).astype(np.float32)
        y = d[..., None, None] * s * qq - dm * m
    elif tname == "Q5_K":
        dm = _f16(B[..., 2:4])[..., None, None, None]
        sc, mn = _scale_min_k4(B[..., 4:16])
        qh = B[..., 16:48]                                                # [nr, nb, 32]
        q = B[..., 48:176].reshape(nr, nb, 4, 32)
        bit = np.arange(8, dtype=np.uint8)                                # u1 = 1 << 2g, u2 = 2 << 2g
        hb = ((qh[..., None, :] >> bit[:, None]) & 1).reshape(nr, nb, 4, 2, 32).astype(np.float32) * 16.0
        qq = np.stack([q & 15, q >> 4], 3).astype(np.float32) + hb
        s = sc.reshape(nr, nb, 4, 2, 1).astype(np.float32)
        m = mn.reshape(nr, nb, 4, 2, 1).astype(np.float32)
        y = d[..., None, None] * s * qq - dm * m
    elif tname == "Q6_K":
        ql, qh = B[..., 0:128].reshape(nr, nb, 2, 64), B[..., 128:192].reshape(nr, nb, 2, 32)
        sc = B[..., 192:208].view(np.int8).reshape(nr, nb, 2, 8).astype(np.float32)
        dd = _f16(B[..., 208:210])[..., None, None, None]
        q1 = ((ql[..., 0:32] & 15) | (((qh >> 0) & 3) << 4)).astype(np.float32) - 32
        q2 = ((ql[..., 32:64] & 15) | (((qh >> 2) & 3) << 4)).astype(np.float32) - 32
        q3 = ((ql[..., 0:32] >> 4) | (((qh >> 4) & 3) << 4)).astype(np.float32) - 32
        q4 = ((ql[..., 32:64] >> 4) | (((qh >> 6) & 3) << 4)).astype(np.float32) - 32
        isx = np.arange(32) // 16
        out = []
        for i, qv in enumerate((q1, q2, q3, q4)):
            out.append(dd[..., 0] * sc[..., 2 * i + isx] * qv)
        y = np.concatenate(out, -1)                                       # [nr, nb, 2, 128]
    elif tname == "IQ4_XS":
        dd = d[..., 0:1]
        sh = np.ascontiguousarray(B[..., 2:4]).view("<u2")[..., 0].astype(np.uint32)
        sl = B[..., 4:8]
        qs = B[..., 8:136].reshape(nr, nb, 8, 16)
        ib = np.arange(8)
        ls = ((sl[..., ib // 2] >> (4 * (ib % 2))) & 15).astype(np.int32) | (((sh[..., None] >> (2 * ib)) & 3).astype(np.int32) << 4)
        dl = (dd * (ls - 32).astype(np.float32))[..., None]               # [nr, nb, 8, 1]
        y = np.concatenate([KVALUES_IQ4NL[qs & 15], KVALUES_IQ4NL[qs >> 4]], -1) * dl
    elif tname == "IQ3_XXS":
        grid = _grid_bytes(G.IQ3XXS_GRID)
        qs = B[..., 2:66].reshape(nr, nb, 8, 8)
        sas = np.ascontiguousarray(B[..., 66:98]).view("<u4").reshape(nr, nb, 8)
        db = (d * (0.5 + (sas >> 28).astype(np.float32)) * 0.5)[..., None]
        y = np.empty((nr, nb, 8, 4, 8), np.float32)
        ks = np.array(G.KSIGNS_IQ2XS, np.uint8)
        kmask = np.array([1, 2, 4, 8, 16, 32, 64, 128], np.uint8)
        for l in range(4):
            sg = ks[(sas >> (7 * l)) & 127]                                        # [nr, nb, 8]
            g1, g2 = grid[qs[..., 2 * l]], grid[qs[..., 2 * l + 1]]                # [nr, nb, 8, 4]
            gg = np.concatenate([g1, g2], -1).astype(np.float32)
            neg = (sg[..., None] & kmask) != 0
            y[..., l, :] = db * gg * np.where(neg, -1.0, 1.0)
    elif tname == "IQ3_S":
        grid = _grid_bytes(G.IQ3S_GRID)
        qs = B[..., 2:66].reshape(nr, nb, 8, 8)
        qh = B[..., 66:74].reshape(nr, nb, 8)
        sg = B[..., 74:106].reshape(nr, nb, 8, 4)
        sc = B[..., 106:110]
        db = d * (1 + 2 * np.stack([sc & 15, sc >> 4], -1).reshape(nr, nb, 8).astype(np.float32))       # ib32 = 2i -> low, 2i+1 -> high
        db = db[..., None]
        y = np.empty((nr, nb, 8, 4, 8), np.float32)
        kmask = np.array([1, 2, 4, 8, 16, 32, 64, 128], np.uint8)
        for l in range(4):
            i1 = qs[..., 2 * l].astype(np.int32) | ((qh.astype(np.int32) << (8 - 2 * l)) & 256)
            i2 = qs[..., 2 * l + 1].astype(np.int32) | ((qh.astype(np.int32) << (7 - 2 * l)) & 256)
            gg = np.concatenate([grid[i1], grid[i2]], -1).astype(np.float32)
            neg = (sg[..., l, None] & kmask) != 0
            y[..., l, :] = db * gg * np.where(neg, -1.0, 1.0)
    elif tname == "Q8_1":                                                 # d, s (= d * sum q, unused by a decode), 32 x int8
        y = B[..., 4:36].view(np.int8).astype(np.float32) * d
    elif tname == "Q8_K":
        dk = np.ascontiguousarray(B[..., 0:4]).view("<f4")                # [nr, nb, 1]
        y = B[..., 4:260].view(np.int8).astype(np.float32) * dk
    elif tname == "Q2_K":
        d, dm = _f16(B[..., 80:82])[..., None, None, None, None], _f16(B[..., 82:84])[..., None, None, None, None]
        sc = B[..., 0:16].reshape(nr, nb, 2, 4, 2, 1)                     # scale index = 8 * half128 + 2 * j + sub16
        q = B[..., 16:80].reshape(nr, nb, 2, 1, 2, 16)
        sh = (2 * np.arange(4, dtype=np.uint8)).reshape(1, 1, 1, 4, 1, 1)
        qq = ((q >> sh) & 3).astype(np.float32)                           # [nr, nb, 2, 4, 2, 16]
        y = (d * (sc & 15).astype(np.float32)) * qq - dm * (sc >> 4).astype(np.float32)
    elif tname == "Q3_K":
        dall = _f16(B[..., 108:110])[..., None, None, None, None]
        s12 = B[..., 96:108]
        lo, hi = s12[..., 0:8], s12[..., 8:12]
        sc = np.concatenate([(lo[..., 0:4] & 15) | ((hi & 3) << 4), (lo[..., 4:8] & 15) | (((hi >> 2) & 3) << 4),
                             (lo[..., 0:4] >> 4) | (((hi >> 4) & 3) << 4), (lo[..., 4:8] >> 4) | (((hi >> 6) & 3) << 4)], -1)
        dl = dall * (sc.astype(np.int32) - 32).astype(np.float32).reshape(nr, nb, 2, 4, 2, 1)       # scale index = 8 * half128 + 2 * j + sub16
        q = B[..., 32:96].reshape(nr, nb, 2, 1, 2, 16)
        sh = (2 * np.arange(4, dtype=np.uint8)).reshape(1, 1, 1, 4, 1, 1)
        hm = B[..., 0:32].reshape(nr, nb, 1, 1, 2, 16)
        hb = (hm >> np.arange(8, dtype=np.uint8).reshape(1, 1, 2, 4, 1, 1)) & 1                   # mask bit 4 * half128 + j
        qq = ((q >> sh) & 3).astype(np.int32) - np.where(hb != 0, 0, 4)
        y = dl * qq.astype(np.float32)
    elif tname in ("IQ2_XXS", "IQ2_XS", "IQ2_S"):
        kmask = np.array([1, 2, 4, 8, 16, 32, 64, 128], np.uint8)
        ks = np.array(G.KSIGNS_IQ2XS, np.uint8)
        grid = GG.iq2_grid({"IQ2_XXS": "xxs", "IQ2_XS": "xs", "IQ2_S": "s"}[tname])
        if tname == "IQ2_XXS":
            qs = B[..., 2:66].reshape(nr, nb, 8, 8)
            sas = np.ascontiguousarray(qs[..., 4:8]).view("<u4")[..., 0]                          # [nr, nb, 8]
            db = (d * (0.5 + (sas >> 28).astype(np.float32)) * 0.25)[..., None, None]            # [nr, nb, 8, 1, 1]
            gi = qs[..., 0:4].astype(np.int64)
            sg = ks[(sas[..., None] >> (7 * np.arange(4, dtype=np.uint32))) & 127]
        else:
            if tname == "IQ2_XS":
                q16 = np.ascontiguousarray(B[..., 2:66]).view("<u2").reshape(nr, nb, 8, 4)
                gi, sg = (q16 & 511).astype(np.int64), ks[q16 >> 9]
                scb = B[..., 66:74]
            else:
                qs, qh = B[..., 2:34].reshape(nr, nb, 8, 4), B[..., 66:74]
                gi = qs.astype(np.int64) | ((qh[..., None].astype(np.int64) << (8 - 2 * np.arange(4))) & 0x300)
                sg = B[..., 34:66].reshape(nr, nb, 8, 4)
                scb = B[..., 74:82]
            sl = np.stack([scb & 15, scb >> 4], -1).astype(np.float32)                           # [nr, nb, 8, 2]
            db = (d[..., None] * (0.5 + sl) * 0.25)[..., None]                                   # [nr, nb, 8, 2, 1]
            db = np.repeat(db, 2, 3)                                                             # [nr, nb, 8, 4, 1]: l // 2
        neg = (sg[..., None] & kmask) != 0
        y = db * grid[gi].astype(np.float32) * np.where(neg, np.float32(-1.0), np.float32(1.0))
    elif tname == "IQ1_S":
        qh = np.ascontiguousarray(B[..., 34:50]).view("<u2").astype(np.int64)                  # [nr, nb, 8]
        gi = B[..., 2:34].reshape(nr, nb, 8, 4).astype(np.int64) | (((qh[..., None] >> (3 * np.arange(4))) & 7) << 8)
        dl = (d * (2 * ((qh >> 12) & 7) + 1).astype(np.float32))[..., None, None]
        delta = np.where((qh & 0x8000) != 0, np.float32(-0.125), np.float32(0.125))[..., None, None]
        y = dl * (GG.iq1_grid()[gi].astype(np.float32) + delta)
    elif tname == "IQ1_M":
        sc = np.ascontiguousarray(B[..., 48:56]).view("<u2").astype(np.uint32)                  # [nr, nb, 4]
        u16 = ((sc[..., 0] >> 12) | ((sc[..., 1] >> 8) & 0x00f0) | ((sc[..., 2] >> 4) & 0x0f00) | (sc[..., 3] & 0xf000)).astype(np.uint16)
        dm = u16.view("<f2").astype(np.float32)[..., None, None, None]
        ib = np.arange(8)
        scb = sc[..., ib // 2]                                                                    # [nr, nb, 8]
        lsh = (6 * (ib % 2)).astype(np.uint32)
        dl = np.stack([(scb >> lsh) & 7, (scb >> (lsh + 3)) & 7], -1)                              # [nr, nb, 8, 2]
        dl = np.repeat(dm[..., 0] * (2 * dl + 1).astype(np.float32), 2, -1).reshape(nr, nb, 8, 4, 1)       # l // 2
        qs = B[..., 0:32].reshape(nr, nb, 8, 4).astype(np.int64)
        qh = B[..., 32:48].reshape(nr, nb, 8, 2).astype(np.int64)
        h0, h1 = qh[..., 0], qh[..., 1]
        gi = qs | np.stack([(h0 << 8) & 0x700, (h0 << 4) & 0x700, (h1 << 8) & 0x700, (h1 << 4) & 0x700], -1)
        neg = np.stack([h0 & 0x08, h0 & 0x80, h1 & 0x08, h1 & 0x80], -1) != 0
        delta = np.where(neg, np.float32(-0.125), np.float32(0.125))[..., None]
        y = dl * (GG.iq1_grid()[gi].astype(np.float32) + delta)
    elif tname == "TQ2_0":
        dt = _f16(B[..., 64:66])[..., None, None, None]
        q = B[..., 0:64].reshape(nr, nb, 2, 1, 32)
        sh = (2 * np.arange(4, dtype=np.uint8)).reshape(1, 1, 1, 4, 1)
        y = (((q >> sh) & 3).astype(np.int32) - 1).astype(np.float32) * dt
    elif tname == "TQ1_0":
        dt = _f16(B[..., 52:54])[..., None]
        p3 = np.array([1, 3, 9, 27, 81], np.uint8).reshape(5, 1)
        tern = lambda q: ((q.astype(np.uint16) * 3) >> 8).astype(np.int32) - 1                    # q: uint8 already wrapped mod 256
        a = tern(B[..., 0:32].reshape(nr, nb, 1, 32) * p3).reshape(nr, nb, 160)
        b = tern(B[..., 32:48].reshape(nr, nb, 1, 16) * p3).reshape(nr, nb, 80)
        c = tern(B[..., 48:52].reshape(nr, nb, 1, 4) * np.array([1, 3, 9, 27], np.uint8).reshape(4, 1)).reshape(nr, nb, 16)
        y = np.concatenate([a, b, c], -1).astype(np.float32) * dt
    else:
        raise GgufTypeError(f"GGUF type {tname}: no decoder")
    return np.ascontiguousarray(y.reshape(lead + (k,)), dtype=np.float32)


TYPES_BY_NAME = {v[0]: v for v in TYPES.values()}


def _grid_bytes(tab):
    return np.array(tab, "<u4").view(np.uint8).reshape(-1, 4)


# ---------------------------------------------------------------------------------------------------------------- repack for the Metal kernels
def to_pack(tname: str, raw, k: int):
    """raw blocks of a KERNEL_TYPES tensor [n, row_bytes] -> foreign_pack.Pack (the same bits, in the order the kernels read)."""
    from . import foreign_pack as FP
    if tname not in KERNEL_TYPES + KERNEL_TYPES8 + KERNEL_TYPESK:
        raise GgufTypeError(f"GGUF type {tname} has no Metal kernel (kernel types: {', '.join(KERNEL_TYPES)})")
    _, be, bb = TYPES_BY_NAME[tname]
    n = np.asarray(raw).shape[0]
    B = _blocks(raw, be, bb, k)                                                       # [n, nb, bb]
    if tname == "Q8_0":                                                                # d (fp16) + 32 int8 in element order: the bytes as stored
        d = np.ascontiguousarray(B[..., 0:2]).view("<f2")[..., 0]
        return FP.Pack("q8_0", n, k, np.ascontiguousarray(B[..., 2:34]).reshape(n, k), d.astype(np.float16))
    if tname in ("Q4_0", "IQ4_NL"):
        d = np.ascontiguousarray(B[..., 0:2]).view("<f2")[..., 0]                      # [n, nb] (one block = one 32-unit)
        qs = np.ascontiguousarray(B[..., 2:18]).reshape(n, k // 2)
        return FP.Pack(tname.lower(), n, k, qs, d.astype(np.float16))
    if tname == "Q4_K":
        sc12 = np.ascontiguousarray(B[..., 4:16]).view("<u4").reshape(n, -1, 3)
        dd = np.ascontiguousarray(B[..., 0:4]).view("<u4").reshape(n, -1, 1)
        meta = np.concatenate([sc12, dd], -1).reshape(n, -1).astype(np.uint32)
        return FP.Pack("q4_k", n, k, np.ascontiguousarray(B[..., 16:144]).reshape(n, k // 2), meta)
    if tname in KERNEL_TYPESK:
        return _to_pack_k(tname, B, n, k)
    if tname == "MXFP4":
        q = B[..., 1:17]                                                               # byte j: element j (low) and j + 16 (high)
        codes = np.concatenate([q & 15, q >> 4], -1).reshape(n, k)                      # element order
        return FP.Pack("mxfp4", n, k, FP.pack_native(codes), np.ascontiguousarray(B[..., 0]).reshape(n, k // 32))
    if tname == "NVFP4":
        q = B[..., 4:36].reshape(n, -1, 4, 8)                                          # byte j of sub-block: element j (low) and j + 8 (high)
        codes = np.concatenate([q & 15, q >> 4], -1).reshape(n, k)
        sc = np.ascontiguousarray(B[..., 0:4]).reshape(n, k // 16)                      # UE4M3 per 16 (ggml halves it and doubles kvalues: net a plain E4M3)
        return FP.Pack("nvfp4", n, k, FP.pack_native(codes), np.ascontiguousarray(sc).view("<u2").reshape(n, k // 32), 1.0)
    raise AssertionError(tname)


def _to_pack_k(tname, B, n, k):
    """Q5_K / Q6_K / IQ4_XS blocks [n, nb, bb] -> Pack of the same bytes (plus 2 padding bytes per Q6_K block) laid out for the stored kernels: the low nibbles
    of a 32-unit are 16 contiguous bytes (Q5_K keeps Q4_K's order, the others byte j = element j | element j + 16 << 4), the extra bits of a unit sit in their own
    32-bit words (hi), the block scales in uint32 words (meta)."""
    from . import foreign_pack as FP
    nb = k // 256
    if tname in ("Q3_K", "IQ3_S", "IQ3_XXS"):
        return _to_pack_k3(tname, B, n, k)
    if tname in ("Q2_K", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ1_S", "IQ1_M"):
        return _to_pack_k2(tname, B, n, k)
    if tname == "IQ4_XS":                                                               # nibbles already in unit order; d + scales_h and scales_l are 2 words
        meta = np.ascontiguousarray(B[..., 0:8]).view("<u4").reshape(n, nb * 2)
        return FP.Pack("iq4_xs", n, k, np.ascontiguousarray(B[..., 8:136]).reshape(n, k // 2), meta.astype(np.uint32))
    if tname == "Q5_K":
        sc12 = np.ascontiguousarray(B[..., 4:16]).view("<u4").reshape(n, nb, 3)
        dd = np.ascontiguousarray(B[..., 0:4]).view("<u4").reshape(n, nb, 1)
        meta = np.concatenate([sc12, dd], -1).reshape(n, nb * 4).astype(np.uint32)
        qh = B[..., 16:48]                                                              # [n, nb, 32]; bit 2g of qh[l] = bit 4 of the low-nibble element 64g + l, bit 2g + 1 the high-nibble one
        g = np.arange(4, dtype=np.uint8)
        bit = lambda sh: ((qh[..., None, :] >> sh[:, None]) & 1).astype(np.uint32)      # [n, nb, 4, 32]
        lo, hi = bit(2 * g), bit(2 * g + 1)
        e = np.arange(16, dtype=np.uint32)
        word = lambda a: (a.reshape(n, nb, 4, 2, 16) << e).sum(-1, dtype=np.uint32)     # [n, nb, 4 g, 2 hf]
        hw = word(lo) | (word(hi) << 16)
        return FP.Pack("q5_k", n, k, np.ascontiguousarray(B[..., 48:176]).reshape(n, k // 2), meta, extra={"hi": np.ascontiguousarray(hw.reshape(n, nb * 8)).astype(np.uint32)})
    ql, qh = B[..., 0:128].reshape(n, nb, 2, 64), B[..., 128:192].reshape(n, nb, 2, 32)    # Q6_K: back to the 6-bit value of each element in natural order, then re-split
    q = np.concatenate([(ql[..., 0:32] & 15) | (((qh >> 0) & 3) << 4), (ql[..., 32:64] & 15) | (((qh >> 2) & 3) << 4),
                        (ql[..., 0:32] >> 4) | (((qh >> 4) & 3) << 4), (ql[..., 32:64] >> 4) | (((qh >> 6) & 3) << 4)], -1).reshape(n, k // 32, 2, 16)
    lo4 = ((q[:, :, 0] & 15) | ((q[:, :, 1] & 15) << 4)).astype(np.uint8)                # [n, nu, 16]: byte j = element j | element j + 16 << 4
    h2 = (q >> 4).astype(np.uint32)
    hw = (h2 << (2 * np.arange(16, dtype=np.uint32))).sum(-1, dtype=np.uint32)         # [n, nu, 2]
    meta = np.concatenate([np.ascontiguousarray(B[..., 192:208]).view("<u4").reshape(n, nb, 4),
                           np.ascontiguousarray(B[..., 208:210]).view("<u2").astype(np.uint32).reshape(n, nb, 1)], -1).reshape(n, nb * 5).astype(np.uint32)
    return FP.Pack("q6_k", n, k, np.ascontiguousarray(lo4).reshape(n, k // 2), meta, extra={"hi": np.ascontiguousarray(hw.reshape(n, k // 32 * 2)).astype(np.uint32)})


def _to_pack_k3(tname, B, n, k):
    """Q3_K / IQ3_S / IQ3_XXS blocks [n, nb, bb] -> Pack of 3-bit-class stored arrays: qs = 8 bytes per 32-unit (2-bit codes, or the 8 grid indices), hi = one uint32
    per unit (Q3_K high bits, IQ3_S signs, IQ3_XXS signs + scale word), meta per 256 (Q3_K 16 int8 scales + d, IQ3_S d + qh + scales, IQ3_XXS d)."""
    from . import foreign_pack as FP
    nb, nu = k // 256, k // 32
    u32 = lambda a: np.ascontiguousarray(a).view("<u4")
    d = np.ascontiguousarray(B[..., 0:2]).view("<u2").astype(np.uint32)[..., 0]                        # IQ3_*: d at the front
    if tname == "IQ3_XXS":
        return FP.Pack("iq3_xxs", n, k, np.ascontiguousarray(B[..., 2:66]).reshape(n, k // 4), d.reshape(n, nb).astype(np.uint32),
                       extra={"hi": u32(B[..., 66:98]).reshape(n, nu).astype(np.uint32)})
    if tname == "IQ3_S":
        meta = np.concatenate([d[..., None], u32(B[..., 66:74]).reshape(n, nb, 2), u32(B[..., 106:110]).reshape(n, nb, 1)], -1).reshape(n, nb * 4)
        return FP.Pack("iq3_s", n, k, np.ascontiguousarray(B[..., 2:66]).reshape(n, k // 4), meta.astype(np.uint32),
                       extra={"hi": u32(B[..., 74:106]).reshape(n, nu).astype(np.uint32)})
    hm, q = B[..., 0:32], B[..., 32:96].reshape(n, nb, 2, 32)                                           # Q3_K: unit u = 4 * half + j, element e of it: bits 2j of q[half][e], bit 4 half + j of hm[e]
    j = np.arange(4, dtype=np.uint8)
    codes = ((q[:, :, :, None, :] >> (2 * j)[:, None]) & 3).astype(np.uint32).reshape(n, nu, 32)       # [n, nb, half, j, 32]
    hb = ((hm[:, :, None, :] >> np.arange(8, dtype=np.uint8)[:, None]) & 1).astype(np.uint32).reshape(n, nu, 32)
    e = np.arange(16, dtype=np.uint32)
    qw = np.stack([(codes[..., :16] << (2 * e)).sum(-1, dtype=np.uint32), (codes[..., 16:] << (2 * e)).sum(-1, dtype=np.uint32)], -1)       # [n, nu, 2]
    hw = (hb << np.arange(32, dtype=np.uint32)).sum(-1, dtype=np.uint32)
    s12 = B[..., 96:108]
    lo, hi = s12[..., 0:8], s12[..., 8:12]
    sc = np.concatenate([(lo[..., 0:4] & 15) | ((hi & 3) << 4), (lo[..., 4:8] & 15) | (((hi >> 2) & 3) << 4),
                         (lo[..., 0:4] >> 4) | (((hi >> 4) & 3) << 4), (lo[..., 4:8] >> 4) | (((hi >> 6) & 3) << 4)], -1)
    sc8 = np.ascontiguousarray((sc.astype(np.int16) - 32).astype(np.int8)).view("<u4")                  # [n, nb, 4]
    dd = np.ascontiguousarray(B[..., 108:110]).view("<u2").astype(np.uint32).reshape(n, nb, 1)
    meta = np.concatenate([sc8, dd], -1).reshape(n, nb * 5).astype(np.uint32)
    return FP.Pack("q3_k", n, k, np.ascontiguousarray(qw.astype("<u4")).view(np.uint8).reshape(n, k // 4), meta, extra={"hi": hw.reshape(n, nu).astype(np.uint32)})


def _to_pack_k2(tname, B, n, k):
    """Q2_K / IQ2_* / IQ1_* blocks [n, nb, bb] -> Pack of the 2-bit-class stored arrays (see foreign_pack.K_*): qs = the grid indices (or the 2-bit codes) per 32-unit,
    hi = one uint32 per unit (IQ2_XXS signs + scale word, IQ2_S sign bytes), meta per 256 (d first as an fp16 word, then the scale / high-bit bytes as words)."""
    from . import foreign_pack as FP
    nb, nu = k // 256, k // 32
    u32 = lambda a: np.ascontiguousarray(a).view("<u4")
    h16 = lambda a: np.ascontiguousarray(a).view("<u2").astype(np.uint32)
    if tname == "Q2_K":                                                                 # unit u = 4 * half + j: element e = bits 2j of q[half][e]; 16 scale|min bytes as 4 words, d | dmin << 16
        q = B[..., 16:80].reshape(n, nb, 2, 32)
        j = np.arange(4, dtype=np.uint8)
        codes = ((q[:, :, :, None, :] >> (2 * j)[:, None]) & 3).astype(np.uint32).reshape(n, nu, 32)
        e = np.arange(16, dtype=np.uint32)
        qw = np.stack([(codes[..., :16] << (2 * e)).sum(-1, dtype=np.uint32), (codes[..., 16:] << (2 * e)).sum(-1, dtype=np.uint32)], -1)
        meta = np.concatenate([u32(B[..., 0:16]).reshape(n, nb, 4), h16(B[..., 80:82]).reshape(n, nb, 1) | (h16(B[..., 82:84]).reshape(n, nb, 1) << 16)], -1)
        return FP.Pack("q2_k", n, k, np.ascontiguousarray(qw.astype("<u4")).view(np.uint8).reshape(n, k // 4), meta.reshape(n, nb * 5).astype(np.uint32))
    if tname == "IQ1_M":                                                                # d is spread over the top nibbles of the four scale words
        sc = h16(B[..., 48:56])
        dh = (sc[..., 0] >> 12) | ((sc[..., 1] >> 8) & 0x00f0) | ((sc[..., 2] >> 4) & 0x0f00) | (sc[..., 3] & 0xf000)
        meta = np.concatenate([dh[..., None], u32(B[..., 48:56]).reshape(n, nb, 2), u32(B[..., 32:48]).reshape(n, nb, 4)], -1)
        return FP.Pack("iq1_m", n, k, np.ascontiguousarray(B[..., 0:32]).reshape(n, k // 8), meta.reshape(n, nb * 7).astype(np.uint32))
    d = h16(B[..., 0:2])[..., 0]
    if tname == "IQ1_S":
        meta = np.concatenate([d[..., None], u32(B[..., 34:50]).reshape(n, nb, 4)], -1)
        return FP.Pack("iq1_s", n, k, np.ascontiguousarray(B[..., 2:34]).reshape(n, k // 8), meta.reshape(n, nb * 5).astype(np.uint32))
    if tname == "IQ2_XXS":
        q = B[..., 2:66].reshape(n, nb, 8, 8)
        return FP.Pack("iq2_xxs", n, k, np.ascontiguousarray(q[..., 0:4]).reshape(n, k // 8), d.reshape(n, nb).astype(np.uint32),
                       extra={"hi": u32(q[..., 4:8]).reshape(n, nu).astype(np.uint32)})
    if tname == "IQ2_XS":
        meta = np.concatenate([d[..., None], u32(B[..., 66:74]).reshape(n, nb, 2)], -1)
        return FP.Pack("iq2_xs", n, k, np.ascontiguousarray(B[..., 2:66]).reshape(n, k // 4), meta.reshape(n, nb * 3).astype(np.uint32))
    meta = np.concatenate([d[..., None], u32(B[..., 66:74]).reshape(n, nb, 2), u32(B[..., 74:82]).reshape(n, nb, 2)], -1)         # IQ2_S: qh (8 bytes) then scales (8 bytes)
    return FP.Pack("iq2_s", n, k, np.ascontiguousarray(B[..., 2:34]).reshape(n, k // 8),
                   meta.reshape(n, nb * 5).astype(np.uint32), extra={"hi": u32(B[..., 34:66]).reshape(n, nu).astype(np.uint32)})
