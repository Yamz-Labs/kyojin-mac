# SPDX-License-Identifier: Apache-2.0
"""Synthetic foreign-format files for tests and the gate tool (numpy only): random but valid GGUF blocks, a GGUF writer, safetensors NVFP4 / MXFP4
writers. The blocks are random bit patterns with sane fp16 scales (the decoders are checked against the reference decode of those bytes, not
against a quantiser), so every layout path is exercised."""
from __future__ import annotations

import json
import struct

import numpy as np

from . import foreign_gguf as GF

# byte offsets of the fp16 fields of a block (random bytes there would be inf / nan)
_FP16_AT = {"Q4_0": (0,), "Q4_1": (0, 2), "Q5_0": (0,), "Q5_1": (0, 2), "Q8_0": (0,), "Q4_K": (0, 2), "Q5_K": (0, 2), "Q6_K": (208,), "Q3_K": (108,), "IQ3_XXS": (0,),
            "IQ3_S": (0,), "IQ4_NL": (0,), "IQ4_XS": (0,), "Q2_K": (80, 82), "IQ2_XXS": (0,), "IQ2_XS": (0,), "IQ2_S": (0,), "IQ1_S": (0,)}


def set_scales(tname: str, b: np.ndarray, vals: np.ndarray) -> None:
    """Writes the fp16 scale(s) of every block of b [n, nb, bb] (IQ1_M spreads its fp16 d over the top nibbles of the four scale words)."""
    v = vals.astype("<f2")
    for off in _FP16_AT.get(tname, ()):
        b[..., off:off + 2] = v.view(np.uint8).reshape(*vals.shape, 2)
    if tname == "IQ1_M":
        dv = v.view("<u2").astype(np.uint32)
        sc = np.ascontiguousarray(b[..., 48:56]).view("<u2").astype(np.uint32)
        for i in range(4):
            sc[..., i] = (sc[..., i] & 0x0fff) | (((dv >> (4 * i)) & 15) << 12)
        b[..., 48:56] = sc.astype("<u2").view(np.uint8)


def random_blocks(tname: str, n: int, k: int, rng) -> np.ndarray:
    """uint8 [n, row_bytes]: n rows of k elements in GGUF type `tname`."""
    _, be, bb = GF.TYPES_BY_NAME[tname]
    if tname in ("F32", "F16", "BF16"):
        w = rng.standard_normal((n, k)).astype(np.float32)
        return {"F32": w.view(np.uint8), "F16": w.astype("<f2").view(np.uint8), "BF16": (w.view(np.uint32) >> 16).astype("<u2").view(np.uint8)}[tname].reshape(n, -1)
    b = rng.integers(0, 256, (n, k // be, bb), dtype=np.uint8)
    set_scales(tname, b, rng.uniform(0.01, 0.5, (n, k // be)) * rng.choice([-1, 1], (n, k // be)))
    if tname == "MXFP4":
        b[..., 0] = rng.integers(118, 130, (n, k // be), dtype=np.uint8)
    if tname == "NVFP4":
        b[..., 0:4] = rng.integers(0x28, 0x46, (n, k // be, 4), dtype=np.uint8)
    return np.ascontiguousarray(b.reshape(n, -1))


def _s(x: str) -> bytes:
    e = x.encode()
    return struct.pack("<Q", len(e)) + e


def _kv(v):
    if isinstance(v, bool):
        return struct.pack("<I?", 7, v)
    if isinstance(v, int):
        return struct.pack("<IQ", 10, v) if v >= 2 ** 31 else struct.pack("<II", 4, v)
    if isinstance(v, float):
        return struct.pack("<If", 6, v)
    if isinstance(v, str):
        return struct.pack("<I", 8) + _s(v)
    if isinstance(v, list):
        if all(isinstance(e, str) for e in v):
            return struct.pack("<IIQ", 9, 8, len(v)) + b"".join(_s(e) for e in v)
        if v and all(isinstance(e, float) for e in v):
            return struct.pack("<IIQ", 9, 6, len(v)) + b"".join(struct.pack("<f", e) for e in v)
        return struct.pack("<IIQ", 9, 5, len(v)) + b"".join(struct.pack("<i", e) for e in v)
    raise TypeError(v)


def write_gguf(path, kv: dict, tensors: list, align: int = 32, version: int = 3):
    """tensors: [(name, ne (k fastest), type name, uint8 bytes)]. Data is aligned like llama.cpp."""
    kv = dict(kv)
    if align != 32:
        kv["general.alignment"] = align
    head = b"GGUF" + struct.pack("<IQQ", version, len(tensors), len(kv))
    for k, v in kv.items():
        head += _s(k) + _kv(v)
    off, infos, datas = 0, b"", []
    for name, ne, tname, data in tensors:
        data = np.ascontiguousarray(data).reshape(-1).view(np.uint8)
        infos += _s(name) + struct.pack("<I", len(ne)) + b"".join(struct.pack("<Q", d) for d in ne) + struct.pack("<IQ", GF.NAME_TO_ID[tname], off)
        pad = (-data.size) % align
        datas.append(data.tobytes() + b"\0" * pad)
        off += data.size + pad
    head += infos
    head += b"\0" * ((-len(head)) % align)
    with open(path, "wb") as f:
        f.write(head + b"".join(datas))


def write_safetensors(path, tensors: dict, metadata=None):
    """tensors: {name: (dtype tag, shape, uint8 bytes)}."""
    hdr, off, parts = {}, 0, []
    if metadata:
        hdr["__metadata__"] = metadata
    for k, (tag, shape, data) in tensors.items():
        data = np.ascontiguousarray(data).reshape(-1).view(np.uint8)
        hdr[k] = {"dtype": tag, "shape": list(shape), "data_offsets": [off, off + data.size]}
        parts.append(data.tobytes())
        off += data.size
    h = json.dumps(hdr).encode()
    h += b" " * ((-len(h)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(parts))


def nvfp4_linear(rng, n, k, modelopt=True, group_scale=None):
    """safetensors tensors of one NVFP4 linear: weight U8 [n, k/2], weight_scale F8_E4M3 [n, k/16], global scale F32 [] (ModelOpt: weight_scale_2, multiplied)
    or F32 [1] (compressed-tensors: weight_global_scale, divided)."""
    w = rng.integers(0, 256, (n, k // 2), dtype=np.uint8)
    sc = rng.integers(0x28, 0x46, (n, k // 16), dtype=np.uint8)
    g = np.array(rng.uniform(0.002, 0.02), np.float32) if modelopt else np.array([rng.uniform(40.0, 400.0)], np.float32)
    names = ("weight", "weight_scale", "weight_scale_2") if modelopt else ("weight_packed", "weight_scale", "weight_global_scale")
    return {names[0]: ("U8", (n, k // 2), w), names[1]: ("F8_E4M3", (n, k // 16), sc), names[2]: ("F32", g.shape, np.atleast_1d(g).view(np.uint8))}


def mxfp4_linear(rng, n, k):
    return {"weight": ("U8", (n, k // 2), rng.integers(0, 256, (n, k // 2), dtype=np.uint8)),
            "weight_scale": ("U8", (n, k // 32), rng.integers(118, 130, (n, k // 32), dtype=np.uint8))}
