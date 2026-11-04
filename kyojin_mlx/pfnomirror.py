# SPDX-License-Identifier: Apache-2.0
"""Prefill without the pool mirror (KYOJIN_PF_NO_MIRROR, default 0; track pfmirror).

Where the 3.54 GB goes. `depthflat.mirror_read` keeps one dense fp16 row per pooled row (1024 B) and one per index key (256 B) for the life of the
session, plus the 1/8 capacity slack. DS4.1 at 983K tokens has four kv-source layers (2, 8, 14 at compress_ratio 2, 20 at ratio 1), i.e. 2.5 pool rows +
2.5 key rows per token: 2.5 x 1024 x 9/8 + 2.5 x 256 x 9/8 = 3600 B/token = 3.54 GB (2.82 GB pool, 0.71 GB key). The first prefill chunk builds it and
it is the largest item of a live 1M session beside the codes themselves.

Who reads the dense form inside one prefill chunk (bytes are what the rows cost, not what is gathered):

  reader (file:line)                                   rows it reads                            bytes at 983040 tokens
  attn_prefill._K1, also the K1O body, :69              T x (W + topk) gathered pool rows of      the 2.82 GB pool mirror is the table
    (4 halves per lane per 16-block)                    every query, per kv-source layer;         it addresses
                                                        1.3 GB per 8192-row chunk per layer
  attn_prefill._K3 :227 / _K3O :291                     the same rows again, 8 dims a copy        same table
    (one uint4 staged per 8 dims)
  fp4pool.append_rows :422 -> _dec :198 ->              the rows just appended                    O(new rows) per chunk
    depthflat.mirror_read :39 / :56
  ds_pf_idx._impl_indexer :362 -> depthflat.mirror_read  the whole key buffer of a plain /         128 MB per key layer per chunk,
                                                       source layer                             0.71 GB mirror in all
  kvledger._slot_rows :50                               metadata only (mirror.nbytes)            0

Which of them can read the codes instead:

  K1 / K3 (the 2.82 GB pool column)   YES. The row address is `id - Tw`; 4 keys decode from 2 code bytes + 1 scale byte (K1) and 8 keys from 4 + 1
                                      (K3), exactly as dsa_attend.attend_packed decodes a pool row in decode. THIS LEVER.
  the append read-back                It is only the writer: `defer_dense` drops it and nothing reads the layer there.
  ds_pf_idx (the 0.71 GB key column)  YES in principle: pf_flat's first pass already ranks blocks on int8 codes, so a packed first pass + a packed
                                      gathered rescore (`_GSCORE`) would feed itself from the codes. Not this lever (see the last paragraph).

This lever removes the pool column (2.82 GB of the 3.54 GB). Its reader is the one that needs the pool row-addressable over the whole width, so the K1
/ K3 gather copies decode the fp4 codes of a row in the copy that reads it, exactly as `dsa_attend.attend_packed` does on the decode side: the mirror is
dropped at the layer call (`prefill_pool`), the append is opened with `defer_dense` so the write-back never reads the layer back dense, and
`PackedSlots.lazy` hands `_compress_part` a `fp4pool.PackedView` (the contract the decode path already carries) instead of a dense array. K1 holds four
key ids per lane, so it decodes the four block scales of each once per 64-dim step instead of once per read. A layer whose codes are not one fp16
session row (off the grid, a batch) keeps the dense path and the mirror, and `.dense()` still serves every reader without a packed form (it rebuilds a
mirror, as at flag 0).

Exactness. Every fp4 row is `LUT[nibble] * scale` (e4m3 scale, block 16) and `fp4pack.unpack` writes exactly `half(fp32(val) * fp32(scale))`, the same
single fp32 multiply then fp32 -> fp16 rounding the kernels do (`dsa_attend.fp4_row8` is bit-identical to `unpack` under
tests/test_fp4pack.py::test_attend_packed_pool_is_bit_identical). The table the attention reads is therefore the same array the mirror held, element for
element, and tests/test_pf_no_mirror.py asserts the prefill output equal bit for bit (not a tolerance).

Cost. The gather reads 3.5x fewer bytes (72 B against 256 B per row) and pays the decode: MEASURED(mac) the isolated K1/K3 gather of a 1024-row chunk
at 65536 keys is 2.43 ms dense against 2.95 ms packed (+21 %), i.e. MODEL +4 ms per kv-source layer per 8192-row chunk, +0.4 % of the ~4.0 s chunk at
2030 tok/s.

What it does not do. (1) The key column stays dense here: its reader is `ds_pf_idx`, and `cpackscan` (pf_idx_cpack) already decodes the packed keys into
a transient array for the plain layers; a packed first pass in `pf_flat` is a separate lever. (2) It does not remove the mirror rebuild of a join: a
`stack_rows` with 2 or more sessions (or the first stacked decode) reads `cache.compress[l]` dense, which builds the layer's mirror again; at flag 0 that
read copies a mirror the prefill already held. mirrorq removes that rebuild on the decode side (one-session only).
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import mlx.core as mx

ENABLED = os.environ.get("KYOJIN_PF_NO_MIRROR", "0") not in ("0", "")
HITS = {"layers": 0, "views": 0, "attention": 0, "mirror_dropped": 0, "drop_bytes": 0, "skip": 0, "dense_fallback": 0}

# The decode of one code byte group: the 16 fp4 magnitudes as the exact fp32 literals `fp4pack._LUT` holds, the e4m3 scale as `mx.from_fp8(..., float32)`
# (sign 1, exp 4 bias 7, mantissa 3; exp 0 is the subnormal 2^-6 * m / 8). Both are powers of two / m/8 * 2^k, so `half(val * scale)` is the one fp32
# multiply and one rounding `unpack` performs.
_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_LIT = ", ".join(repr(float(v)) + "f" for v in _LUT)

HEAD = """
constant float PNM_LUT[8] = {@LUT@};
inline float pnm_e4m3(uint b) {                             // the fp32 value `mx.from_fp8(b, float32)` gives (pool scales, block 16)
    const float m = float(b & 7u) * 0.125f;
    const uint e = (b >> 3) & 15u;
    const float v = e == 0u ? m * 0.015625f : (1.0f + m) * metal::exp2(float(int(e) - 7));
    return (b & 128u) ? -v : v;
}
// 4 dims [o, o + 4) of a packed row (o within one 16-value scale block: 2 code bytes + 1 scale byte) -> the 4 halves `unpack` writes.
inline uint2 pnm_row4(const device uchar* codes, const device uchar* scales, ulong row, uint o, uint D) {
    const ushort w = *(const device ushort*)(codes + row * (D / 2) + (o >> 1));
    const float sc = pnm_e4m3(uint(scales[row * (D / 16) + (o >> 4)]));
    half h[4];
    for (uint i = 0; i < 4; ++i) { const uint c = (w >> (4 * i)) & 15u; const float mag = PNM_LUT[c & 7u]; h[i] = static_cast<half>(((c & 8u) ? -mag : mag) * sc); }
    return *reinterpret_cast<thread uint2*>(h);
}
// the same 4 halves with the block scale decoded by the caller (K1 reads it once per 64-dim step for each of its four rows).
inline uint2 pnm_row4s(const device uchar* codes, uint row, uint o, uint D, float sc) {
    const ushort w = *(const device ushort*)(codes + ulong(row) * (D / 2) + (o >> 1));
    half h[4];
    for (uint i = 0; i < 4; ++i) { const uint c = (w >> (4 * i)) & 15u; const float mag = PNM_LUT[c & 7u]; h[i] = static_cast<half>(((c & 8u) ? -mag : mag) * sc); }
    return *reinterpret_cast<thread uint2*>(h);
}
// 8 dims [o, o + 8), o a multiple of 8 (one 16-value scale block, 4 code bytes + 1 scale byte) -> the 8 halves `unpack` writes.
inline uint4 pnm_row8(const device uchar* codes, const device uchar* scales, ulong row, uint o, uint D) {
    const uint w = *(const device uint*)(codes + row * (D / 2) + (o >> 1));
    const float sc = pnm_e4m3(uint(scales[row * (D / 16) + (o >> 4)]));
    half h[8];
    for (uint i = 0; i < 8; ++i) { const uint c = (w >> (4 * i)) & 15u; const float mag = PNM_LUT[c & 7u]; h[i] = static_cast<half>(((c & 8u) ? -mag : mag) * sc); }
    return *reinterpret_cast<thread uint4*>(h);
}
""".replace("@LUT@", _LIT)


def on() -> bool:
    """The flag, read per call (POST /admin/flags -> flags.set writes KYOJIN_PF_NO_MIRROR back)."""
    return ENABLED


def is_packed(pool) -> bool:
    """`pool` is a `fp4pool.PackedView` of a pool layer (block 16 / e4m3 codes): a reader can decode its rows instead of holding a dense copy."""
    return (hasattr(pool, "codes") and hasattr(pool, "scales") and getattr(pool, "block", 0) == 16
            and bool(getattr(pool, "e4m3", False)) and pool.dtype == mx.float16 and int(pool.D) % 128 == 0)


def pool_codes(pool):
    """(codes [cap, D / 2] uint8, scales [cap, D / 16] uint8, n) of a packed pool, or None (flag off, dense pool, another codec)."""
    if not on() or not is_packed(pool):
        return None
    return pool.codes[0], pool.scales[0], int(pool.n)


def dense_of(pool):
    """The dense rows of `pool` for a reader with no packed form (a `PackedView` unpacks; it may rebuild the layer's mirror, as at flag 0)."""
    if not hasattr(pool, "codes"):
        return pool
    HITS["dense_fallback"] += 1
    return pool.dense()


def drop_mirror(e) -> int:
    """Free the dense mirror / memo of a packed layer; returns the bytes freed. The codes stay the ground truth."""
    nb = 0 if e.mirror is None else int(e.mirror.nbytes)
    if e.mirror is not None or e.mn or e.memo is not None:
        e.mirror, e.mn, e.memo = None, 0, None
        HITS["mirror_dropped"] += 1
    HITS["drop_bytes"] += nb
    return nb


def prefill_ok(cache, layer: int) -> bool:
    """The pool layer of `cache` is one fp16 session row of fp4 codes with the pool codec: the packed prefill path can serve it."""
    slots = getattr(cache, "compress", None)
    if not on() or slots is None or not hasattr(slots, "raw") or type(slots).BLOCK != 16:
        return False
    try:
        e = slots.raw(layer)
    except (IndexError, AttributeError):
        return False
    from .fp4pool import _Pk
    return bool(isinstance(e, _Pk) and e.codes is not None and e.codes.shape[0] == 1 and e.dtype == mx.float16)


@contextmanager
def prefill_pool(cache, layer: int):
    """Around the `_compress_part` of a prompt chunk: the pool layer reads back as its own codes (`PackedSlots.lazy`, no unpack of the whole pool), an
    append does not read the layer back dense (`defer_dense`), and a dense mirror the session still holds is dropped. Yields True when the layer is packed,
    False when the dense path (and its mirror) stays. Both class attributes are restored on the way out, so a layer that is not packed and the flag 0 code
    are unchanged."""
    slots = getattr(cache, "compress", None)
    if not prefill_ok(cache, layer):
        HITS["skip"] += 1
        yield False
        return
    e = slots.raw(layer)
    freed = drop_mirror(e)
    old = (slots.lazy, slots.defer_dense)
    slots.lazy, slots.defer_dense = True, True
    HITS["layers"] += 1
    if freed:
        HITS["views"] += 1
    try:
        yield True
    finally:
        slots.lazy, slots.defer_dense = old


def mirror_bytes(cache) -> int:
    """Bytes of the dense mirrors the compress slots of a cache still hold (what this lever removes; read only)."""
    from .fp4pool import _Pk
    slots = getattr(cache, "compress", None)
    if not hasattr(slots, "raw"):
        return 0
    return sum(int(e.mirror.nbytes) for e in list.__iter__(slots) if isinstance(e, _Pk) and e.mirror is not None)
