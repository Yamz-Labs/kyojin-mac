"""Pure-numpy reference for EXL3 `mul1` linears (no torch, no engine import, runs anywhere).

Format (one linear, in = k, out = n, both multiples of 128):
  trellis int16 [k/16, n/16, 16*K]   16x16 tiles, K bits per weight, tail-biting bit stream per tile
  suh fp16 [k]  svh fp16 [n]         input / output sign-scale vectors (suh carries the codebook scale)
Stream: the 16*K int16 of a tile are viewed as 8*K little-endian uint32 words; stream bit b lives in
word b // 32 at bit position 31 - (b % 32) (MSB first). Position t (0..255) of a tile has the 16-bit state
made of stream bits [(t+1)K - 16, (t+1)K), cyclic modulo 256K, read MSB first.
Codebook mul1: p = s * 0x83DCD12D (mod 2^32); bsum = sum of the 4 bytes of p;
  value = fp16( fma_fp16( fp16(1024 + bsum), fp16(0x1EEE), fp16(0xC931) ) )  (one rounding)
Tile layout: run g (positions 32g..32g+31) covers tile columns g and g+8; position p of a run sits at
  row kk = 2*(p // 8) + (p % 2) + 8*((p % 4) // 2), column g (p % 8 < 4) or g + 8 (p % 8 >= 4).
Layer:  W[k, n] = diag(suh) . H . Wr . H . diag(svh), H = Sylvester Hadamard 128 (block diagonal) / sqrt(128).
GEMV:   y = H( (H(x * suh) / sqrt(128)) @ Wr ) / sqrt(128) * svh
"""
from __future__ import annotations

import numpy as np

HAD = 128
INV_SQRT_HAD = np.float32(1.0 / np.sqrt(128.0))   # 0.08838834764831845 as fp32
MUL1_MULT = 0x83DCD12D
K_INV_BITS = 0x1EEE
K_BIAS_BITS = 0xC931


def trellis_words(trellis: np.ndarray) -> np.ndarray:
    """int16 [ti, tj, 16K] -> uint32 [ti*tj, 8K] (little-endian view, the layout the Metal kernel reads)."""
    ti, tj, w = trellis.shape
    t = np.ascontiguousarray(trellis.astype("<i2"))
    return t.view("<u4").reshape(ti * tj, w // 2)


def decode_states(words: np.ndarray, K: int) -> np.ndarray:
    """uint32 [T, 8K] -> uint16 [T, 256]: funnel shift of one or two words per position."""
    T, nw = words.shape
    assert nw == 8 * K
    out = np.empty((T, 256), np.uint16)
    w64 = words.astype(np.uint64)
    for t in range(256):
        e = (t + 1) * K                      # exclusive end bit of the window
        ih = (e - 1) // 32                   # word holding the last bit
        s = (ih + 1) * 32 - e                # LSB offset of the window inside that word
        if s <= 16:
            v = (w64[:, ih] >> np.uint64(s)) & np.uint64(0xFFFF)
        else:
            prev = w64[:, (ih - 1) % nw]
            v = ((prev << np.uint64(32 - s)) | (w64[:, ih] >> np.uint64(s))) & np.uint64(0xFFFF)
        out[:, t] = v.astype(np.uint16)
    return out


def mul1_values(states: np.ndarray) -> np.ndarray:
    """uint16 states -> fp16 codebook values (single-rounding fp16 fma)."""
    p = (states.astype(np.uint64) * np.uint64(MUL1_MULT)) & np.uint64(0xFFFFFFFF)
    bsum = (p & 255) + ((p >> 8) & 255) + ((p >> 16) & 255) + (p >> 24)
    r = (np.uint16(0x6400) + bsum.astype(np.uint16)).view(np.float16)      # 1024 + bsum, exact in fp16
    k_inv = np.array([K_INV_BITS], np.uint16).view(np.float16)[0].astype(np.float64)
    k_bias = np.array([K_BIAS_BITS], np.uint16).view(np.float16)[0].astype(np.float64)
    return (r.astype(np.float64) * k_inv + k_bias).astype(np.float16)     # exact in fp64, one rounding to fp16


def mcg_values(states: np.ndarray, three_inst: bool = False) -> np.ndarray:
    """Codebooks mcg (flag `mcg`, published packs) and 3inst (default): half2 add of the two fp16 halves."""
    x = states.astype(np.uint64)
    x = (x * np.uint64(89226354) + np.uint64(64248484) if three_inst else x * np.uint64(0xCBAC1FED)) & np.uint64(0xFFFFFFFF)
    x = ((x & np.uint64(0x8FFF8FFF)) ^ np.uint64(0x3B603B60)).astype(np.uint32)
    h = x.view(np.float16).reshape(x.shape + (2,))
    return h[..., 0] + h[..., 1]                       # fp16 add, one rounding (matches __hadd)


def codebook_values(states: np.ndarray, cb: str = "mul1") -> np.ndarray:
    return {"mul1": mul1_values, "mcg": mcg_values, "3inst": lambda s: mcg_values(s, True)}[cb](states)


def run_position(p: int) -> tuple[int, int]:
    """(row kk, column half 0/1) of position p (0..31) inside a run."""
    kk = 2 * (p // 8) + (p % 2) + 8 * ((p % 4) // 2)
    return kk, (p % 8) // 4


def tile_perm() -> np.ndarray:
    """perm[q] = row-major index (row * 16 + col) of tile position q (q = 32 g + p)."""
    perm = np.empty(256, np.int64)
    for g in range(8):
        for p in range(32):
            kk, hi = run_position(p)
            perm[32 * g + p] = kk * 16 + g + 8 * hi
    return perm


def decode_tiles(trellis: np.ndarray, cb: str = "mul1") -> np.ndarray:
    """int16 [ti, tj, 16K] -> Wr fp16 [ti*16, tj*16] (regularised weight, before the Hadamards)."""
    ti, tj, w = trellis.shape
    K = w // 16
    vals = codebook_values(decode_states(trellis_words(trellis), K), cb)            # [T, 256] trellis order
    tiles = np.empty_like(vals)
    tiles[:, tile_perm()] = vals
    return tiles.reshape(ti, tj, 16, 16).transpose(0, 2, 1, 3).reshape(ti * 16, tj * 16)


def fwht128(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Unnormalised Sylvester Walsh-Hadamard transform over blocks of 128 along `axis` (butterfly, in x's dtype)."""
    x = np.moveaxis(x, axis, -1)
    shp = x.shape
    y = x.reshape(shp[:-1] + (shp[-1] // HAD, HAD)).copy()
    h = 1
    while h < HAD:
        y = y.reshape(y.shape[:-1] + (HAD // (2 * h), 2, h))
        a, b = y[..., 0, :].copy(), y[..., 1, :].copy()
        y[..., 0, :], y[..., 1, :] = a + b, a - b
        y = y.reshape(y.shape[:-3] + (HAD,))
        h *= 2
    return np.moveaxis(y.reshape(shp), -1, axis)


def reconstruct(trellis: np.ndarray, suh: np.ndarray, svh: np.ndarray, cb: str = "mul1") -> np.ndarray:
    """Dense W [k, n] float32 (what y = x @ W multiplies with)."""
    wr = decode_tiles(trellis, cb).astype(np.float32)
    w = fwht128(wr, axis=0) * INV_SQRT_HAD
    w = w * suh.astype(np.float32)[:, None]
    w = fwht128(w, axis=1) * INV_SQRT_HAD
    return w * svh.astype(np.float32)[None, :]


def gemv(x: np.ndarray, trellis: np.ndarray, suh: np.ndarray, svh: np.ndarray, dtype=np.float64, cb: str = "mul1") -> np.ndarray:
    """y [rows, n] = the layer applied to x [rows, k], computed in `dtype` from the fp16 Wr values."""
    wr = decode_tiles(trellis, cb).astype(dtype)
    xh = fwht128(x.astype(dtype) * suh.astype(dtype)[None, :], axis=1) * dtype(INV_SQRT_HAD) if dtype is np.float32 \
        else fwht128(x.astype(dtype) * suh.astype(dtype)[None, :], axis=1) / np.sqrt(128.0)
    inner = xh @ wr
    inner = fwht128(inner, axis=1) * (dtype(INV_SQRT_HAD) if dtype is np.float32 else 1.0 / np.sqrt(128.0))
    return inner * svh.astype(dtype)[None, :]


def random_linear(k: int, n: int, K: int, seed: int = 0):
    """Random but valid linear: every bit stream decodes, so random int16 words are a legal trellis."""
    rng = np.random.default_rng(seed)
    trellis = rng.integers(-32768, 32768, size=(k // 16, n // 16, 16 * K), dtype=np.int16)
    suh = (rng.choice([-1.0, 1.0], size=k) * rng.uniform(0.5, 1.5, size=k)).astype(np.float16)
    svh = (rng.choice([-1.0, 1.0], size=n) * rng.uniform(0.5, 1.5, size=n)).astype(np.float16)
    return trellis, suh, svh


def fma32(a: np.ndarray, b: np.ndarray, d: np.ndarray) -> np.ndarray:
    """fp32 fused multiply-add, correctly rounded once (IEEE fma). The fp32 product is exact in fp64; the fp64 sum is made round-to-odd with the TwoSum
    error term, so the final cast to fp32 cannot double-round (a plain fp64 sum then cast can, on ~2^-29 of the inputs)."""
    p = a.astype(np.float64) * b.astype(np.float64)
    d = d.astype(np.float64)
    s = p + d
    bb = s - p
    e = (p - (s - bb)) + (d - bb)                                          # exact: p + d = s + e
    odd = (e != 0) & ((s.view(np.uint64) & 1) == 0) & np.isfinite(s)
    s = np.where(odd, np.nextafter(s, np.where(e > 0, np.inf, -np.inf)), s)
    return s.astype(np.float32)


def rope_pairs_f32(x1: np.ndarray, x2: np.ndarray, c: np.ndarray, s: np.ndarray, form: int = 0):
    """(r1, r2) fp32 of the rope pair for the three forms (see rope_pairs_ref); all inputs fp32."""
    x1, x2, c, s = (np.asarray(v, np.float32) for v in (x1, x2, c, s))
    if form == 0:
        return x1 * c - x2 * s, x1 * s + x2 * c
    if form == 1:                                                          # the shipped MLX: fmuladd(x1, c, (-s) * x2), fmuladd(x1, s, c * x2)
        return fma32(x1, c, (-s) * x2), fma32(x1, s, c * x2)
    if form == 2:
        return fma32(-x2, s, x1 * c), fma32(x2, c, x1 * s)
    raise ValueError(form)


def rope_pairs_ref(x: np.ndarray, tab: np.ndarray, form: int = 0) -> np.ndarray:
    """The rope-tail pair rotation of the fused attention kernels in numpy fp32, rounded to fp16 like `static_cast<T>`.
    x [..., 2p] fp16 (interleaved pairs, mx.fast.rope traditional=True), tab [..., 2p] fp32 interleaved (cos0, sin0, cos1, ...) broadcastable to x.
    form 0: two roundings of the products (`x1 * c - x2 * s` under fp contract(off)); 1: `fma(x1, c, -(x2 * s))`, `fma(x1, s, x2 * c)`;
    2: `fma(-x2, s, x1 * c)`, `fma(x2, c, x1 * s)` (fma32: one rounding)."""
    x1, x2 = x[..., 0::2].astype(np.float32), x[..., 1::2].astype(np.float32)
    c, s = tab[..., 0::2].astype(np.float32), tab[..., 1::2].astype(np.float32)
    r1, r2 = rope_pairs_f32(x1, x2, c, s, form)
    out = np.empty(np.broadcast(x1, c).shape[:-1] + (x.shape[-1],), np.float16)
    out[..., 0::2], out[..., 1::2] = r1.astype(np.float16), r2.astype(np.float16)
    return out
