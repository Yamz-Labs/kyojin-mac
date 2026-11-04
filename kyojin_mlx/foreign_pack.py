# SPDX-License-Identifier: Apache-2.0
"""Device layout of foreign 4-bit weights (numpy only, no mlx): what the Metal kernels of foreign_metal read, how a source file's bits map into it,
and a CPU decoder / kernel-logic emulation used by the tests.

Pack(fmt, n, k, qs, meta, gscale): n output rows, k inputs, rows contiguous.
  qs    uint8 [n, k/2]; a 16-byte unit = 32 consecutive inputs (k % 32 == 0).
  fmt   qs nibble order inside a unit               meta (per unit unless noted)
  q4_0     H: byte j = element j | element j+16 << 4   float16 d [n, k/32];                     w = (q - 8) d
  iq4_nl   H                                           float16 d [n, k/32];                     w = KVALUES_IQ4NL[q] d
  q4_k     Q4K: 256 super-block, 4 groups of 64, byte l of group g = element 64g+l | 64g+32+l << 4;
                                                       uint32 [n, k/256*4]: 3 words of the 12-byte 6-bit scale/min block, then half2(d, dmin)
  nvfp4    N: byte b = element 2b | element 2b+1 << 4  uint16 [n, k/32]: two UE4M3 bytes (inputs 0-15, 16-31) ; gscale multiplies the row result
  mxfp4    N                                           uint8 [n, k/32]: one E8M0 byte
  dense    w float16 [n, k] in `qs` (types without a kernel; the loader says so)
All of them: 0.5625 bytes / weight except dense."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], np.float32)
KVALUES_IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.float32)
FORMATS = ("q4_0", "iq4_nl", "q4_k", "nvfp4", "mxfp4")
FORMATSK = ("q5_k", "q6_k", "iq4_xs", "q3_k", "iq3_s", "iq3_xxs", "q2_k", "iq2_xxs", "iq2_xs", "iq2_s", "iq1_s", "iq1_m")           # flag kstored: GGUF K / IQ blocks read as stored (same bytes, bit fields split per 32-unit; see the K section of foreign_metal)
FORMATS8 = ("fp8", "q8_0")                      # flag fp8stored: one byte per weight, natural order (see foreign_metal 8-bit section); not in FORMATS
LAYOUT = {"q4_0": "H", "iq4_nl": "H", "q4_k": "Q4K", "nvfp4": "N", "mxfp4": "N"}
UNIT = {"q4_0": 32, "iq4_nl": 32, "q4_k": 32, "nvfp4": 32, "mxfp4": 32}
KALIGN = {"q4_0": 32, "iq4_nl": 32, "q4_k": 256, "nvfp4": 32, "mxfp4": 32}      # k must be a multiple of this


K_META_WORDS = {"q5_k": 4, "q6_k": 5, "iq4_xs": 2, "q3_k": 5, "iq3_s": 4, "iq3_xxs": 1, "q2_k": 5, "iq2_xxs": 1, "iq2_xs": 3, "iq2_s": 5, "iq1_s": 5, "iq1_m": 7}     # uint32 per 256-super-block in `meta`
K_HI_WORDS = {"q5_k": 1, "q6_k": 2, "iq4_xs": 0, "q3_k": 1, "iq3_s": 1, "iq3_xxs": 1, "q2_k": 0, "iq2_xxs": 1, "iq2_xs": 0, "iq2_s": 1, "iq1_s": 0, "iq1_m": 0}   # uint32 per 32-unit in extra["hi"]
K_QS_BYTES = {"q5_k": 16, "q6_k": 16, "iq4_xs": 16, "q3_k": 8, "iq3_s": 8, "iq3_xxs": 8, "q2_k": 8, "iq2_xxs": 4, "iq2_xs": 8, "iq2_s": 4, "iq1_s": 4, "iq1_m": 4}   # bytes of qs per 32-unit


@dataclass
class Pack:
    fmt: str
    n: int
    k: int
    qs: np.ndarray
    meta: np.ndarray | None = None
    gscale: float | np.ndarray = 1.0            # per-tensor (or per-expert array) output scale; nvfp4 only
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.fmt == "dense":
            return
        if self.fmt in FORMATS8:
            if self.k % 32 or self.qs.shape[-2:] != (self.n, self.k) or self.qs.dtype != np.uint8:
                raise ValueError(f"{self.fmt}: qs {self.qs.shape} {self.qs.dtype} != uint8 [{self.n}, {self.k}] with k % 32 == 0")
            return
        if self.fmt in FORMATSK:
            nsb, nu = self.k // 256, self.k // 32
            hi = self.extra.get("hi")
            if (self.k % 256 or self.qs.shape != (self.n, self.k // 32 * K_QS_BYTES[self.fmt]) or self.qs.dtype != np.uint8 or self.meta.shape != (self.n, nsb * K_META_WORDS[self.fmt])
                    or self.meta.dtype != np.uint32 or (K_HI_WORDS[self.fmt] and (hi is None or hi.shape != (self.n, nu * K_HI_WORDS[self.fmt]) or hi.dtype != np.uint32))):
                raise ValueError(f"{self.fmt}: bad shapes for n={self.n} k={self.k} (k % 256 == 0, qs uint8 [n, k/32*{K_QS_BYTES[self.fmt]}], meta uint32 [n, k/256*{K_META_WORDS[self.fmt]}], hi uint32 [n, k/32*{K_HI_WORDS[self.fmt]}])")
            return
        if self.fmt not in FORMATS:
            raise ValueError(f"unknown foreign format {self.fmt}")
        if self.k % KALIGN[self.fmt]:
            raise ValueError(f"{self.fmt}: k={self.k} is not a multiple of {KALIGN[self.fmt]}")
        if self.qs.shape[-2:] != (self.n, self.k // 2) or self.qs.dtype != np.uint8:
            raise ValueError(f"{self.fmt}: qs {self.qs.shape} {self.qs.dtype} != uint8 [{self.n}, {self.k // 2}]")

    @property
    def nbytes(self):
        return self.qs.nbytes + (0 if self.meta is None else self.meta.nbytes) + (self.extra["hi"].nbytes if "hi" in self.extra else 0)


def _e4m3_tab():
    b = np.arange(256)
    e, m = (b >> 3) & 15, (b & 7).astype(np.float32)
    v = np.where(e == 0, np.ldexp(m, -9), np.ldexp(1.0 + m / 8.0, e - 7)).astype(np.float32)
    v = np.where((b & 127) == 127, 0.0, v)
    return np.where(b & 128, -v, v).astype(np.float32)


E4M3_TAB = _e4m3_tab()                           # E4M3FN codes -> float32, the NaN codes 0x7F / 0xFF -> 0 (the kernel table)


def pack_native(codes):
    """4-bit codes [.., n] (element order) -> bytes [.., n/2], element 2b low nibble, 2b+1 high."""
    c = np.asarray(codes, np.uint8)
    return np.ascontiguousarray(c[..., 0::2] | (c[..., 1::2] << 4))


def unpack_native(qs):
    q = np.asarray(qs, np.uint8)
    out = np.empty(q.shape[:-1] + (q.shape[-1] * 2,), np.uint8)
    out[..., 0::2], out[..., 1::2] = q & 15, q >> 4
    return out


def _e4m3(b):
    b = np.asarray(b, np.uint8) & 127
    e, m = (b >> 3).astype(np.int32), (b & 7).astype(np.float32)
    v = np.where(e == 0, np.ldexp(m, -9), np.ldexp(1.0 + m / 8.0, e - 7)).astype(np.float32)
    return np.where(b == 127, 0.0, v).astype(np.float32)            # same as the kernel: 0x7F (the NaN code) scales by 0


def _e8m0(b):
    return np.ldexp(np.float32(1.0), np.asarray(b, np.int32) - 127).astype(np.float32)


def q4k_unpack_scales(s):
    sc, mn = np.empty(s.shape[:-1] + (8,), np.uint8), np.empty(s.shape[:-1] + (8,), np.uint8)
    sc[..., :4], mn[..., :4] = s[..., 0:4] & 63, s[..., 4:8] & 63
    sc[..., 4:] = (s[..., 8:12] & 15) | ((s[..., 0:4] >> 6) << 4)
    mn[..., 4:] = (s[..., 8:12] >> 4) | ((s[..., 4:8] >> 6) << 4)
    return sc, mn

def _h(a):
    """float32 -> fp16 saturated at +-65504: a scale or bias product past fp16 must not become inf (same clamp as the kernels)."""
    return np.clip(a, -65504.0, 65504.0).astype(np.float16)


def mq_arrays(p: Pack):
    """MLX-native quantized form of a pack, mapped code for code (no re-quantization): (wq uint32 [n, k bits/32], scales, biases or None, mode, group_size, bits).
    q4_0 / q4_k / q5_k: affine group 32, w = scale q + bias (q4_0: d, -8 d; q4_k, q5_k: d sc, -dmin mn, products rounded to fp16; q5_k codes are 5 bits, packed
    LSB first as one bit stream, MLX's layout for bits=5). nvfp4 / mxfp4: MLX modes of the same bits (the nvfp4 global scale stays outside, in gscale). None for other formats."""
    n, k = p.n, p.k
    bits = 4
    if p.fmt == "q5_k":
        nsb = k // 256
        q = p.qs.reshape(n, nsb, 4, 32)
        hw = p.extra["hi"].reshape(n, nsb, 4, 2, 1)                                     # unit words [g, hf]; bit e = low-nibble element e of the unit, bit 16 + e = high-nibble one
        hb = ((hw >> np.arange(32, dtype=np.uint32)) & 1).astype(np.uint8).reshape(n, nsb, 4, 2, 2, 16).transpose(0, 1, 2, 4, 3, 5).reshape(n, nsb, 4, 2, 32)
        codes = (np.stack([q & 15, q >> 4], 3) | (hb << 4)).reshape(n, k)
        m4 = np.ascontiguousarray(p.meta).reshape(n, nsb, 4)
        sc, mn = q4k_unpack_scales(np.ascontiguousarray(m4[..., :3]).view(np.uint8))
        hd = np.ascontiguousarray(m4[..., 3]).view(np.float16).reshape(n, nsb, 2).astype(np.float32)
        sca = _h(hd[..., :1] * sc.astype(np.float32)).reshape(n, k // 32)
        bia = _h(-hd[..., 1:] * mn.astype(np.float32)).reshape(n, k // 32)
        stream = np.packbits(((codes[..., None] >> np.arange(5, dtype=np.uint8)) & 1).reshape(n, k * 5), axis=-1, bitorder="little")
        return np.ascontiguousarray(stream).view(np.uint32).reshape(n, k * 5 // 32), sca, bia, "affine", 32, 5
    if p.fmt == "q4_0":
        q = p.qs.reshape(n, k // 32, 16)
        codes = np.concatenate([q & 15, q >> 4], -1).reshape(n, k)
        d = np.ascontiguousarray(p.meta).astype(np.float16)
        s, b, mode, g = d, _h(np.float32(-8) * d.astype(np.float32)), "affine", 32
    elif p.fmt == "q4_k":
        nsb = k // 256
        q = p.qs.reshape(n, nsb, 4, 32)
        codes = np.stack([q & 15, q >> 4], 3).reshape(n, k)
        m4 = np.ascontiguousarray(p.meta).reshape(n, nsb, 4)
        sc, mn = q4k_unpack_scales(np.ascontiguousarray(m4[..., :3]).view(np.uint8))
        hd = np.ascontiguousarray(m4[..., 3]).view(np.float16).reshape(n, nsb, 2).astype(np.float32)
        s = _h(hd[..., :1] * sc.astype(np.float32)).reshape(n, k // 32)
        b = _h(-hd[..., 1:] * mn.astype(np.float32)).reshape(n, k // 32)
        mode, g = "affine", 32
    elif p.fmt == "nvfp4":
        codes, s, b, mode, g = unpack_native(p.qs), np.ascontiguousarray(p.meta).view(np.uint8).reshape(n, k // 16), None, "nvfp4", 16
    elif p.fmt == "mxfp4":
        codes, s, b, mode, g = unpack_native(p.qs), np.ascontiguousarray(p.meta).reshape(n, k // 32), None, "mxfp4", 32
    else:
        return None
    return np.ascontiguousarray(pack_native(codes)).view(np.uint32).reshape(n, k // 8), s, b, mode, g, bits


def dequant(p: Pack, g=True) -> np.ndarray:
    """float32 [n, k] of the pack's weights (semantic reference of the layout; gscale applied when g). Leading expert dims are not handled here."""
    n, k = p.n, p.k
    if p.fmt == "dense":
        return p.qs.astype(np.float32)
    if p.fmt == "fp8":
        w = E4M3_TAB[p.qs]
        return w * np.broadcast_to(np.asarray(p.gscale, np.float32).reshape(-1, 1), (n, 1)) if g else w
    if p.fmt == "q8_0":
        return (p.qs.view(np.int8).astype(np.float32).reshape(n, k // 32, 32) * p.meta.astype(np.float32)[..., None]).reshape(n, k)
    if p.fmt in FORMATSK:
        return _dequant_k(p)
    if p.fmt in ("q4_0", "iq4_nl"):
        q = p.qs.reshape(n, k // 32, 16)
        q = np.concatenate([q & 15, q >> 4], -1)
        d = p.meta.astype(np.float32)[..., None]
        w = (q.astype(np.float32) - 8) * d if p.fmt == "q4_0" else KVALUES_IQ4NL[q] * d
        return w.reshape(n, k)
    if p.fmt == "q4_k":
        nsb = k // 256
        q = p.qs.reshape(n, nsb, 4, 32)
        q = np.stack([q & 15, q >> 4], 3).astype(np.float32)
        m4 = np.ascontiguousarray(p.meta).reshape(n, nsb, 4)
        sc, mn = q4k_unpack_scales(np.ascontiguousarray(m4[..., :3]).view(np.uint8))
        hd = np.ascontiguousarray(m4[..., 3]).view(np.float16).reshape(n, nsb, 2).astype(np.float32)
        w = hd[..., 0, None, None, None] * sc.reshape(n, nsb, 4, 2).astype(np.float32)[..., None] * q - hd[..., 1, None, None, None] * mn.reshape(n, nsb, 4, 2).astype(np.float32)[..., None]
        return w.reshape(n, k)
    codes = unpack_native(p.qs)
    if p.fmt == "nvfp4":
        s = _e4m3(np.ascontiguousarray(p.meta).view(np.uint8).reshape(n, k // 16))
        w = (E2M1[codes].reshape(n, k // 16, 16) * s[..., None]).reshape(n, k)
        return w * np.float32(p.gscale) if g else w
    s = _e8m0(p.meta.reshape(n, k // 32))
    return (E2M1[codes].reshape(n, k // 32, 32) * s[..., None]).reshape(n, k)


def _dequant_k(p: Pack) -> np.ndarray:
    """float32 [n, k] of a K-format pack, from the device layout (independent of the GGUF reader: tests compare the two)."""
    n, k = p.n, p.k
    nsb, nu = k // 256, k // 32
    if p.fmt == "q5_k":
        q = p.qs.reshape(n, nsb, 4, 32)
        m4 = np.ascontiguousarray(p.meta).reshape(n, nsb, 4)
        sc, mn = q4k_unpack_scales(np.ascontiguousarray(m4[..., :3]).view(np.uint8))
        hd = np.ascontiguousarray(m4[..., 3]).view(np.float16).reshape(n, nsb, 2).astype(np.float32)
        hw = p.extra["hi"].reshape(n, nsb, 4, 2, 1)                                     # [g, hf] unit words; bit e = low-nibble element, bit 16 + e = high-nibble element
        bits = ((hw >> np.arange(32, dtype=np.uint32)) & 1).reshape(n, nsb, 4, 2, 2, 16)      # [g, hf, sel, e]
        hb = bits.transpose(0, 1, 2, 4, 3, 5).reshape(n, nsb, 4, 2, 32).astype(np.float32) * 16.0
        qq = np.stack([q & 15, q >> 4], 3).astype(np.float32) + hb
        w = hd[..., 0, None, None, None] * sc.reshape(n, nsb, 4, 2).astype(np.float32)[..., None] * qq - hd[..., 1, None, None, None] * mn.reshape(n, nsb, 4, 2).astype(np.float32)[..., None]
        return w.reshape(n, k)
    if p.fmt in ("q3_k", "iq3_s", "iq3_xxs"):
        return _dequant_k3(p)
    if p.fmt in ("q2_k", "iq2_xxs", "iq2_xs", "iq2_s", "iq1_s", "iq1_m"):
        return _dequant_k2(p)
    qu = p.qs.reshape(n, nu, 16)
    lo = np.concatenate([qu & 15, qu >> 4], -1).astype(np.uint32)                       # [n, nu, 32] natural element order
    if p.fmt == "iq4_xs":
        m = p.meta.reshape(n, nsb, 2)
        d = (m[..., 0] & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float32)
        sh, sl = m[..., 0] >> 16, m[..., 1]
        ib = np.arange(8, dtype=np.uint32)
        ls = ((sl[..., None] >> (8 * (ib >> 1) + 4 * (ib & 1))) & 15) | (((sh[..., None] >> (2 * ib)) & 3) << 4)
        dl = d[..., None] * (ls.astype(np.int32) - 32).astype(np.float32)               # [n, nsb, 8]
        return (KVALUES_IQ4NL[lo].reshape(n, nsb, 8, 32) * dl[..., None]).reshape(n, k)
    m = p.meta.reshape(n, nsb, 5)
    sc = np.ascontiguousarray(m[..., :4]).view(np.int8).reshape(n, nsb, 16).astype(np.float32)
    d = (m[..., 4] & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float32)
    hw = p.extra["hi"].reshape(n, nu, 2, 1)
    h2 = (hw >> (2 * np.arange(16, dtype=np.uint32))) & 3                               # [n, nu, 2, 16]
    q = (lo | (h2.reshape(n, nu, 32) << 4)).astype(np.float32) - 32.0
    return (q.reshape(n, nsb, 16, 16) * (d[..., None] * sc)[..., None]).reshape(n, k)


def _dequant_k3(p: Pack) -> np.ndarray:
    """float32 [n, k] of a Q3_K / IQ3_S / IQ3_XXS pack from the device layout (see foreign_gguf._to_pack_k3)."""
    from . import foreign_grids as G
    n, k = p.n, p.k
    nsb, nu = k // 256, k // 32
    hi = p.extra["hi"].reshape(n, nu)
    if p.fmt == "q3_k":
        qw = np.ascontiguousarray(p.qs).view("<u4").reshape(n, nu, 2, 1)
        codes = ((qw >> (2 * np.arange(16, dtype=np.uint32))) & 3).reshape(n, nu, 32).astype(np.int32)
        hb = ((hi[..., None] >> np.arange(32, dtype=np.uint32)) & 1).astype(np.int32)
        m = p.meta.reshape(n, nsb, 5)
        sc = np.ascontiguousarray(m[..., :4]).view(np.int8).reshape(n, nsb, 16).astype(np.float32)
        d = (m[..., 4] & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float32)
        return ((codes + 4 * hb - 4).astype(np.float32).reshape(n, nsb, 16, 16) * (d[..., None] * sc)[..., None]).reshape(n, k)
    idx = p.qs.reshape(n, nu, 8)
    bit = np.arange(8, dtype=np.uint32)
    if p.fmt == "iq3_xxs":
        grid = np.array(G.IQ3XXS_GRID, "<u4").view(np.uint8).reshape(-1, 4).astype(np.float32)
        d = (p.meta.reshape(n, nsb) & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float32)
        db = (np.repeat(d[..., None], 8, -1).reshape(n, nu) * (0.5 + (hi >> 28).astype(np.float32)) * 0.5)
        ks = np.array(G.KSIGNS_IQ2XS, np.uint8)
        sg = ks[(hi[..., None] >> (7 * np.arange(4, dtype=np.uint32))) & 127]                        # [n, nu, 4]
        gg = grid[idx].reshape(n, nu, 4, 8)
    else:
        grid = np.array(G.IQ3S_GRID, "<u4").view(np.uint8).reshape(-1, 4).astype(np.float32)
        m = p.meta.reshape(n, nsb, 4)
        d = (m[..., 0] & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float32)
        qh = np.ascontiguousarray(m[..., 1:3]).view(np.uint8).reshape(n, nu).astype(np.int32)
        ls = np.ascontiguousarray(m[..., 3]).view(np.uint8).reshape(n, nsb, 4)
        ls = np.stack([ls & 15, ls >> 4], -1).reshape(n, nu).astype(np.float32)
        db = np.repeat(d[..., None], 8, -1).reshape(n, nu) * (1 + 2 * ls)
        l = np.arange(4)
        i1 = idx[..., 0::2].astype(np.int32) | ((qh[..., None] << (8 - 2 * l)) & 256)
        i2 = idx[..., 1::2].astype(np.int32) | ((qh[..., None] << (7 - 2 * l)) & 256)
        gg = np.concatenate([grid[i1], grid[i2]], -1)
        sg = ((hi[..., None] >> (8 * np.arange(4, dtype=np.uint32))) & 255).astype(np.uint8)
    neg = ((sg[..., None] >> bit.astype(np.uint8)) & 1) != 0
    return (db[..., None, None] * gg * np.where(neg, -1.0, 1.0)).reshape(n, k).astype(np.float32)


def _dequant_k2(p: Pack) -> np.ndarray:
    """float32 [n, k] of a Q2_K / IQ2_* / IQ1_* pack from the device layout (see foreign_gguf._to_pack_k2)."""
    from . import foreign_grids as G, foreign_grids_gen as GG
    n, k = p.n, p.k
    nsb, nu = k // 256, k // 32
    u16 = lambda a: np.ascontiguousarray(a).view("<u2")
    half = lambda w: (np.asarray(w) & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float32)
    m = p.meta.reshape(n, nsb, K_META_WORDS[p.fmt])
    ub = lambda x: np.repeat(x, 8, 1).reshape(n, nu) if x.shape[1] == nsb else x           # per-block value -> per unit
    if p.fmt == "q2_k":
        qw = np.ascontiguousarray(p.qs).view("<u4").reshape(n, nu, 2, 1)
        codes = ((qw >> (2 * np.arange(16, dtype=np.uint32))) & 3).reshape(n, nu, 2, 16).astype(np.float32)
        sb = np.ascontiguousarray(m[..., :4]).view(np.uint8).reshape(n, nu, 2)
        d = np.repeat(half(m[..., 4])[..., None], 8, -1).reshape(n, nu, 1)
        dmin = np.repeat(half(m[..., 4] >> 16)[..., None], 8, -1).reshape(n, nu, 1)
        w = d[..., None] * (sb & 15).astype(np.float32)[..., None] * codes - dmin[..., None] * (sb >> 4).astype(np.float32)[..., None]
        return w.reshape(n, k).astype(np.float32)
    d = half(m[..., 0])
    dbu = np.repeat(d[..., None], 8, -1).reshape(n, nu)
    bit = np.arange(8, dtype=np.uint8)
    sign = lambda sg: np.where(((sg[..., None] >> bit) & 1) != 0, np.float32(-1.0), np.float32(1.0))
    ks = np.array(G.KSIGNS_IQ2XS, np.uint8)
    if p.fmt in ("iq2_xxs", "iq2_xs", "iq2_s"):
        grid = GG.iq2_grid({"iq2_xxs": "xxs", "iq2_xs": "xs", "iq2_s": "s"}[p.fmt]).astype(np.float32)
        if p.fmt == "iq2_xxs":
            hi = p.extra["hi"].reshape(n, nu)
            gi = p.qs.reshape(n, nu, 4).astype(np.int64)
            sg = ks[(hi[..., None] >> (7 * np.arange(4, dtype=np.uint32))) & 127]
            db = (dbu * (0.5 + (hi >> 28).astype(np.float32)) * 0.25)[..., None, None]
        else:
            if p.fmt == "iq2_xs":
                q16 = u16(p.qs).reshape(n, nu, 4)
                gi, sg = (q16 & 511).astype(np.int64), ks[q16 >> 9]
                scb = np.ascontiguousarray(m[..., 1:3]).view(np.uint8).reshape(n, nu)
            else:
                qh = np.ascontiguousarray(m[..., 1:3]).view(np.uint8).reshape(n, nu).astype(np.int64)
                gi = p.qs.reshape(n, nu, 4).astype(np.int64) | ((qh[..., None] << (8 - 2 * np.arange(4))) & 0x300)
                sg = np.ascontiguousarray(p.extra["hi"].reshape(n, nu)).view(np.uint8).reshape(n, nu, 4)
                scb = np.ascontiguousarray(m[..., 3:5]).view(np.uint8).reshape(n, nu)
            sl = np.stack([scb & 15, scb >> 4], -1).astype(np.float32)
            db = np.repeat((dbu[..., None] * (0.5 + sl) * 0.25), 2, 2)[..., None]            # [n, nu, 4, 1]: l // 2
        return (db * grid[gi] * sign(sg)).reshape(n, k).astype(np.float32)
    grid = GG.iq1_grid().astype(np.float32)
    if p.fmt == "iq1_s":
        qh = u16(m[..., 1:5]).reshape(n, nu).astype(np.int64)
        gi = p.qs.reshape(n, nu, 4).astype(np.int64) | (((qh[..., None] >> (3 * np.arange(4))) & 7) << 8)
        dl = (dbu * (2 * ((qh >> 12) & 7) + 1).astype(np.float32))[..., None, None]
        delta = np.where((qh & 0x8000) != 0, np.float32(-0.125), np.float32(0.125))[..., None, None]
        return (dl * (grid[gi] + delta)).reshape(n, k).astype(np.float32)
    sc = u16(m[..., 1:3]).astype(np.uint32)                                                   # [n, nsb, 4]
    ib = np.arange(8)
    scb = sc[..., ib // 2]
    lsh = (6 * (ib % 2)).astype(np.uint32)
    dl = np.stack([(scb >> lsh) & 7, (scb >> (lsh + 3)) & 7], -1)
    dl = np.repeat(dbu.reshape(n, nsb, 8)[..., None] * (2 * dl + 1).astype(np.float32), 2, -1).reshape(n, nu, 4, 1)
    qhb = np.ascontiguousarray(m[..., 3:7]).view(np.uint8).reshape(n, nu, 2).astype(np.int64)
    h0, h1 = qhb[..., 0], qhb[..., 1]
    gi = p.qs.reshape(n, nu, 4).astype(np.int64) | np.stack([(h0 << 8) & 0x700, (h0 << 4) & 0x700, (h1 << 8) & 0x700, (h1 << 4) & 0x700], -1)
    neg = np.stack([h0 & 0x08, h0 & 0x80, h1 & 0x08, h1 & 0x80], -1) != 0
    delta = np.where(neg, np.float32(-0.125), np.float32(0.125))[..., None]
    return (dl * (grid[gi] + delta)).reshape(n, k).astype(np.float32)


# ------------------------------------------------------------------------------------------------------------ MMA kernel input permutation
def k_unit(fmt, u):
    """k of the first element of the lane mapping of unit u (matches kUnit in foreign_metal)."""
    if LAYOUT[fmt] == "Q4K":
        u8 = u & 7
        return 256 * (u >> 3) + 64 * (u8 >> 1) + 16 * (u8 & 1)
    return 32 * u


def k_off(fmt, t, s):
    """k offset (from k_unit) of A tile t, k-slot s = 2c + e (matches kOff in foreign_metal)."""
    L = LAYOUT[fmt]
    if L == "N":
        return 8 * (s >> 1) + 2 * t + (s & 1)
    if L == "H":
        return 16 * (t >> 1) + 4 * (s >> 1) + 2 * (t & 1) + (s & 1)
    return 32 * (t >> 1) + 4 * (s >> 1) + 2 * (t & 1) + (s & 1)


def xperm_index(fmt, k):
    """int32 [k]: x_perm[r][u*32 + s*4 + t] = x[r][idx[u*32 + s*4 + t]], the activations in the order the B fragments are loaded."""
    u = np.arange(k // 32)[:, None, None]
    s = np.arange(8)[None, :, None]
    t = np.arange(4)[None, None, :]
    ku = np.vectorize(lambda v: k_unit(fmt, int(v)))(np.arange(k // 32))[:, None, None]
    ko = np.vectorize(lambda ss, tt: k_off(fmt, int(tt), int(ss)))(s, t)
    return (ku + ko).reshape(-1).astype(np.int32)


def lane_weights(p: Pack, n, u):
    """What lane (fm, c) of a simdgroup computes for row n, unit u: array [4 lanes c][4 tiles t][2 e] of float32, mirroring the MSL (bit tricks included
    as their closed forms). Used by emulate_mma."""
    fmt = p.fmt
    qs32 = np.ascontiguousarray(p.qs[n, u * 16:(u + 1) * 16]).view("<u4")           # 4 words, word c = lane c
    out = np.zeros((4, 4, 2), np.float32)
    if fmt == "q4_k":
        m = np.ascontiguousarray(p.meta).reshape(p.n, -1, 4)[n, u >> 3]
        sc, mn = q4k_unpack_scales(np.ascontiguousarray(m[:3]).view(np.uint8))
        hd = m[3:4].view(np.float16).astype(np.float32)
        ja = ((u & 7) >> 1) * 2
        S = [(hd[0] * sc[ja], hd[1] * mn[ja]), (hd[0] * sc[ja + 1], hd[1] * mn[ja + 1])]
    for c in range(4):
        wq = int(qs32[c])
        for t in range(4):
            for e in range(2):
                L = LAYOUT[fmt]
                if L == "N":
                    code = (wq >> (8 * t + 4 * e)) & 15
                else:
                    code = (wq >> (8 * ((t & 1) * 2 + e) + 4 * (t >> 1))) & 15
                if fmt == "q4_0":
                    v = (np.float32(code) - 8) * np.float32(p.meta[n, u])
                elif fmt == "iq4_nl":
                    v = KVALUES_IQ4NL[code] * np.float32(p.meta[n, u])
                elif fmt == "q4_k":
                    Sd, Mn = S[t >> 1]
                    v = np.float32(code) * Sd - Mn
                elif fmt == "nvfp4":
                    mc = int(p.meta[n, u])
                    v = E2M1[code] * _e4m3((mc >> (8 * (c >> 1))) & 255)
                else:
                    v = E2M1[code] * _e8m0(int(p.meta[n, u]))
                out[c, t, e] = v
    return out


def emulate_mma(p: Pack, x: np.ndarray) -> np.ndarray:
    """y [R, n] float64 by the fragment algebra of the MMA kernel (A[fm][2c+e] = lane weights, B[s][row] = x_perm), slow python: small shapes."""
    R, n, k = x.shape[0], p.n, p.k
    xp = x.astype(np.float64)[:, xperm_index(p.fmt, k)].reshape(R, k // 32, 8, 4)
    y = np.zeros((R, n))
    for n0 in range(0, n, 8):
        for u in range(k // 32):
            A = np.zeros((4, 8, 8))                                                  # [tile][n = fm][slot = 2c + e]
            for fm in range(8):
                lw = lane_weights(p, n0 + fm, u)
                for c in range(4):
                    for t in range(4):
                        A[t, fm, 2 * c:2 * c + 2] = lw[c, t]
            for t in range(4):
                B = xp[:, u, :, t].T                                                 # [slot][row]
                y[:, n0:n0 + 8] += (A[t] @ B).T
    return y * (float(p.gscale) if p.fmt == "nvfp4" else 1.0)


# ------------------------------------------------------------------------------------------------------------ safetensors (NVFP4 / MXFP4) sources
def pack_from_safetensors(ck, c: dict) -> Pack:
    """ck: vendor_input.Checkpoint, c: vendor_input.classify_linear(...) of an nvfp4 / mxfp4 linear -> Pack (native E2M1 bytes as stored)."""
    tag, shape, w = ck.raw(c["weight"])
    if len(shape) != 2:
        raise ValueError(f"{c['weight']}: expected a 2D packed weight, got {shape}")
    n, kh = shape
    k = kh * 2
    qs = np.ascontiguousarray(w).view(np.uint8)
    stag, sshape, sc = ck.raw(c["scale"])
    sc = np.ascontiguousarray(sc).view(np.uint8)
    if c["fmt"] == "nvfp4":
        if sshape[-1] * 16 != k:
            raise ValueError(f"{c['scale']}: NVFP4 needs one scale per 16 inputs, got {sshape} for k={k}")
        if c["scale2"] is None:
            raise ValueError(f"{c['weight']}: NVFP4 without a global scale tensor")
        g2 = np.asarray(ck.f32(c["scale2"]))
        if g2.size != 1:                                      # one scalar per weight matrix; a per-row / per-group global would be applied wrongly as element 0
            raise ValueError(f"{c['scale2']}: NVFP4 global scale has {g2.size} values {tuple(g2.shape)}, this reader serves one per matrix")
        g = float(g2.reshape(-1)[0])
        if not np.isfinite(g) or g == 0.0:
            raise ValueError(f"{c['scale2']}: NVFP4 global scale {g} is not usable")
        g = 1.0 / g if c["inverse_global"] else g
        return Pack("nvfp4", n, k, qs, np.ascontiguousarray(sc).view("<u2").reshape(n, k // 32), g)
    if sshape[-1] * 32 != k:
        raise ValueError(f"{c['scale']}: this reader serves MXFP4 with one E8M0 scale per 32 inputs, got scale {sshape} for k={k} ({c['detail']})")
    return Pack("mxfp4", n, k, qs, sc.reshape(n, k // 32))


# ------------------------------------------------------------------------------------------------------------ same-input groups (flag cfmtspeed)
def concat_rows(packs):
    """Several linears that read the SAME input as one matrix of n = sum(n_i) rows: (Pack, rowgs float32 [n]). The bytes of qs / meta are the members' bytes
    one after the other (rows are independent in every layout: nothing is re-ordered, re-scaled or re-quantised); the per-tensor global scale of each member
    becomes the per-row array rowgs that the `rowgs` kernels multiply with where the plain kernels multiply by gs[0]. The returned Pack has gscale 1.0, so
    dequant(pack) * rowgs[:, None] is, row for row, dequant(member_i) bit for bit."""
    p0 = packs[0]
    if p0.fmt == "dense" or any((p.fmt, p.k) != (p0.fmt, p0.k) for p in packs):
        raise ValueError("concat_rows: members must share one non-dense format and k")
    rg = np.concatenate([np.full(p.n, np.float32(p.gscale), np.float32) for p in packs])
    q = np.concatenate([np.ascontiguousarray(p.qs) for p in packs], 0)
    m = None if p0.meta is None else np.concatenate([np.ascontiguousarray(p.meta) for p in packs], 0)
    return Pack(p0.fmt, int(rg.shape[0]), p0.k, q, m, 1.0), rg


def fp4v_bits(c):
    """numpy transcription of the fp4v bit trick of foreign_metal.HDR (uint codes -> float32); the cfmtspeed table E2M1T must equal it for all 16 codes."""
    c = np.asarray(c, np.uint32)
    v = c & 7
    f = ((v << 22) + np.uint32(0x3F000000)).astype(np.uint32).view(np.float32)
    f = np.where(v < 2, np.float32(0.5) * v.astype(np.float32), f).astype(np.float32)
    return (f.view(np.uint32) | ((c & 8) << 28)).astype(np.uint32).view(np.float32)
