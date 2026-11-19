# SPDX-License-Identifier: Apache-2.0
"""Header-only GGUF read: kv + tensor infos from the first bytes of a file (a range read or a sparse head copy), no data-bound check.

foreign_gguf.GgufFile needs the whole file (it checks every tensor's bytes are there); this reads only the header so structural checks
(tensor -> skeleton slot maps, second-model header proofs) run on a few MB of a 85 GiB shard. Raises GgufError when the bytes end inside the header."""
from __future__ import annotations

from .foreign_gguf import MAGIC, GgufError, _Cur


def read_header(src, max_bytes: int = 64 << 20):
    """src: path or bytes -> (kv dict, {tensor name: (ne list, ggml type id)}) in file order. A path is read up to `max_bytes` only (never the data)."""
    if not isinstance(src, (bytes, bytearray, memoryview)):
        with open(src, "rb") as f:
            src = f.read(max_bytes)
    b = memoryview(src)
    if len(b) < 24 or bytes(b[:4]) != MAGIC:
        raise GgufError("not a GGUF header")
    c = _Cur(b, 4)
    if c.u32() not in (2, 3):
        raise GgufError("GGUF version (2 and 3 only)")
    nt, nkv = c.u64(), c.u64()
    kv = {}
    for _ in range(nkv):
        k = c.string()
        kv[k] = c.value(c.u32())
    tensors = {}
    for _ in range(nt):
        name = c.string()
        nd = c.u32()
        ne = [c.u64() for _ in range(nd)]
        t = c.u32()
        c.u64()
        tensors[name] = (ne, t)
    return kv, tensors
