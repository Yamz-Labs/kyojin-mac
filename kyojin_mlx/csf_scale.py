"""Loader-side decoder for NVIDIA's lossless compressed NVFP4 block scales (`weight_scale_encoding` `csf` and `csf_32x2_v1`,
modelopt checkpoints such as nvidia/GLM-5.3-Flash-FP4-Local).

Adapted from vLLM, vllm/model_executor/layers/quantization/utils/nvfp4_utils.py (Apache-2.0, Copyright contributors to the
vLLM project): `_decode_csf`, `_csf32_blocks`, `_decode_csf_32x2`, `decode_csf_scale`. Ported from torch to numpy; the stream
validation is kept. See THIRD_PARTY_NOTICES.md.

Formats (scale matrix `[rows, cols]` of E4M3 bytes, cols = in_features / 16):
- `csf`: `nvfp4_csf_fixed` (uint8, `[rows // 16, 16 * (1 + cols // 2)]`) holds, per 16-row slab, one base byte per row followed
  by the rows' 4-bit codes, two per byte, low nibble first; a scale byte is `base + code`. `nvfp4_csf_exceptions` (uint32) holds
  sorted words: low 24 bits = flat `row * cols + col`, high 8 bits = the replacement byte.
- `csf_32x2_v1`: `csf32_bases` (uint8, one per row) and `csf32_codes` (uint8, 4-bit codes) cover 32-row x 2-column blocks,
  grouped into 128 x 4 tiles, 32 tiles to a page. `csf32_exceptions` (uint16): in-block position in the low byte, scale byte in
  the high byte. `csf32_page_bases` (uint32, byte offset of each page's first exception) and `csf32_offsets` (uint16, per page a 0
  then each block's exception end) index them.

`CsfCheckpoint` is a vendor_input.Checkpoint that shows the streams of `<p>.weight_scale.<stream>` as one plain F8_E4M3
`<p>.weight_scale`, so the existing NVFP4 path (classify_linear / load_linear) consumes it unchanged.
"""
from __future__ import annotations

import functools

import numpy as np

from . import vendor_input as vi


def _decode_csf(fixed, exceptions):
    if fixed.dtype != np.uint8 or fixed.ndim != 2 or fixed.shape[1] % 16:
        raise ValueError(f"Invalid CSF fixed stream {fixed.dtype} {fixed.shape}")
    if exceptions.dtype != np.uint32:
        raise ValueError(f"Invalid CSF exceptions dtype {exceptions.dtype}")
    rows = fixed.shape[0] * 16
    cols = (fixed.shape[1] // 16 - 1) * 2
    if rows * cols > 1 << 24:
        raise ValueError("CSF scale exceeds 24-bit exception positions")
    bases = fixed[:, :16].reshape(rows, 1)
    if (bases > 240).any():
        raise ValueError("CSF row bases must be in 0..240")
    packed = fixed[:, 16:].reshape(rows, cols // 2)
    codes = np.stack((packed & 0xF, packed >> 4), axis=-1).reshape(rows, cols)
    out = (bases + codes).astype(np.uint8).reshape(-1)
    words = exceptions.reshape(-1)
    positions = words & 0xFFFFFF
    if positions.size and (positions[-1] >= rows * cols or (positions[1:] <= positions[:-1]).any()):
        raise ValueError("CSF exception positions must be increasing and in range")
    out[positions] = words >> 24
    return out.reshape(rows, cols)


@functools.lru_cache(maxsize=8)
def _csf32_blocks(rows, cols):
    """Origin, width and page of each 32x2 block in storage order, and the flat `[rows, cols]` position of each code."""
    tile_r, tile_c, pair, block = np.meshgrid(
        np.arange(-(-rows // 128)), np.arange(-(-cols // 4)), np.arange(2), np.arange(4), indexing="ij")
    r0 = (128 * tile_r + 32 * block).ravel()
    c0 = (4 * tile_c + 2 * pair).ravel()
    page = (tile_r * tile_c.shape[1] + tile_c).ravel() // 32
    keep = (r0 < rows) & (c0 < cols)
    r0, c0, page = r0[keep], c0[keep], page[keep]
    width = np.minimum(cols - c0, 2)
    i, j = np.meshgrid(np.arange(32), np.arange(2), indexing="ij")
    cells = (r0[:, None, None] + i) * cols + c0[:, None, None] + j
    return r0, c0, width, page, cells[j < width[:, None, None]]


@functools.lru_cache(maxsize=8)
def _csf32_rows(rows, cols):
    """Row of each code cell of _csf32_blocks (the flat position // cols), kept so the per-tensor decode does not redo 0.5 M int64 divisions."""
    return _csf32_blocks(rows, cols)[4] // cols


def _decode_csf_32x2(bases, codes, page_bases, offsets, exceptions):
    dtypes = (np.uint8, np.uint8, np.uint32, np.uint16, np.uint16)
    streams = (bases, codes, page_bases, offsets, exceptions)
    rows = bases.size
    cols = 2 * codes.size // max(rows, 1)
    if any(t.dtype != d for t, d in zip(streams, dtypes)) or rows % 32 or rows * cols != 2 * codes.size or cols == 0:
        raise ValueError("Invalid csf_32x2_v1 streams")
    r0, c0, width, page, cells = _csf32_blocks(rows, cols)
    num_pages = int(page[-1]) + 1
    base, packed, origins, index, words = (t.reshape(-1) for t in streams)
    if origins.size != num_pages or index.size != r0.size + num_pages:
        raise ValueError("Invalid csf_32x2_v1 index sizes")
    if (base > 240).any():
        raise ValueError("CSF row bases must be in 0..240")
    out = np.empty(rows * cols, dtype=np.uint8)
    out[cells] = np.stack((packed & 0xF, packed >> 4), -1).ravel() + base[_csf32_rows(rows, cols)]

    first = np.flatnonzero(np.diff(page, prepend=-1))
    ends = origins[page].astype(np.int64) // 2 + index[np.arange(r0.size) + page + 1]
    starts = np.concatenate(([0], ends[:-1]))
    if ((index[first + np.arange(num_pages)] != 0).any() or (origins != 2 * starts[first]).any()
            or (ends < starts).any() or (ends - starts > 32 * width).any() or ends[-1] != words.size):
        raise ValueError("Invalid csf_32x2_v1 exception index")
    block = np.repeat(np.arange(r0.size), ends - starts)
    pos = (words & 0xFF).astype(np.int64)
    if ((pos >= 64).any() or (pos % 2 >= width[block]).any()
            or ((pos[1:] <= pos[:-1]) & (block[1:] == block[:-1])).any()):
        raise ValueError("CSF exception positions must be increasing and in range")
    out[(r0[block] + pos // 2) * cols + c0[block] + pos % 2] = words >> 8
    return out.reshape(rows, cols)


# weight_scale_encoding -> (stream suffixes in decoder argument order, decoder)
CSF_SCALE_ENCODINGS = {
    "csf": (("nvfp4_csf_fixed", "nvfp4_csf_exceptions"), _decode_csf),
    "csf_32x2_v1": (tuple(f"csf32_{n}" for n in ("bases", "codes", "page_bases", "offsets", "exceptions")), _decode_csf_32x2),
}


def decode_csf_scale(encoding, streams):
    """Rebuild one `[rows, cols]` uint8 matrix of E4M3 scale bytes from `{stream suffix: numpy array}`."""
    names, decode = CSF_SCALE_ENCODINGS[encoding]
    return decode(*(np.asarray(streams[n]) for n in names))


def csf_prefixes(info):
    """{scale name (`<p>.weight_scale`): encoding} for every complete set of CSF streams in a header map."""
    out = {}
    for enc, (names, _) in CSF_SCALE_ENCODINGS.items():
        for n in info:
            if n.endswith("." + names[0]):
                base = n[: -len(names[0]) - 1]
                if all(f"{base}.{s}" in info for s in names):
                    out[base] = enc
    return out


def csf_stream_names(info):
    """Every tensor name that belongs to a CSF stream set (to drop from a converter's passthrough keys)."""
    return {f"{b}.{s}" for b, e in csf_prefixes(info).items() for s in CSF_SCALE_ENCODINGS[e][0]}


class CsfCheckpoint(vi.Checkpoint):
    """Checkpoint whose CSF-encoded `<p>.weight_scale` appears as an ordinary F8_E4M3 tensor, decoded on demand."""

    def info(self):
        if getattr(self, "_csf_info", None) is None:
            base = dict(super().info())
            self._csf = csf_prefixes(base)
            for b, enc in self._csf.items():
                rows = base[f"{b}.{CSF_SCALE_ENCODINGS[enc][0][0]}"][1]
                if enc == "csf_32x2_v1":
                    n = int(np.prod(rows))
                    shape = (n, 2 * int(np.prod(base[f"{b}.csf32_codes"][1])) // max(n, 1))
                else:
                    f = base[f"{b}.nvfp4_csf_fixed"][1]
                    shape = (f[0] * 16, (f[1] // 16 - 1) * 2)
                base[b] = ("F8_E4M3", shape)
            self._csf_info = base
        return self._csf_info

    def raw(self, name):
        self.info()
        enc = self._csf.get(name)
        if enc is None:
            return super().raw(name)
        a = decode_csf_scale(enc, {s: super(CsfCheckpoint, self).raw(f"{name}.{s}")[2] for s in CSF_SCALE_ENCODINGS[enc][0]})
        return "F8_E4M3", a.shape, a
