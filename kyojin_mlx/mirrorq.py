# SPDX-License-Identifier: Apache-2.0
"""mirrorq: the depth-flat mirror stops being a per-session dense copy of the whole pool / index keys (KYOJIN_MIRRORQ, default 0; track mirrorq).

Where the bytes are. `depthflat.mirror_read` keeps one dense fp16 row per pooled row and one per index key for the whole session: 1024 B per pool row,
256 B per key row, plus the 1/8 capacity slack. DS4.1 at 983K tokens has four kv-source layers (`is_kv_source`: 2, 8, 14 at compress_ratio 2, 20 at 1),
so 2.5 pool rows and 2.5 key rows per token: 2.5 x 1024 x 9/8 + 2.5 x 256 x 9/8 = **3540 B/token**, i.e. the 3.54 GB `kv_roles` reports (2.82 GB of pool
mirror, 0.71 GB of key mirror). It is the largest item of a 1M-token session.

Who reads the dense form, and which rows (decode, one `dsv4_core.core` call over n rows):

  reader                                         rows it reads                         bytes/step             needs the dense form?
  dsv4_core._layer 1698 -> attend 715 / _ATTN    W window rows (the ring, not the pool)  K x 1024 B per row     NO: indexed rows only, so a
      pool[0, idx[u, k]], K = index_topk 512     + K selected pool rows of the row      per kv-source layer    slab of the named rows is
                                                                                                               enough
  dsv4_core._layer 1616-1670 -> the index        every key of the buffer for the 4      n x 2.5L x 132 B       NO: the int8 first pass
      scores (dec_idx_i8 -> dsa_decode           plain layers (int8 codes), then the    (int8 codes)           reads the codes; the exact
      .scores_rows), then the exact rescore      topk+MARGIN 8-key blocks (5120 rows)                          rescore reads picked rows
                                                 of the buffer per layer
  dsv4_core._stack_ragged 1280 (stack_rows)      all rows of every layer, once per       n x D x 2 per row      NO: it is a copy into the
                                                 stacked decode batch                                          stacked buffer the core reads

Prefill (`attn_prefill.sparse_attention` gathers T x K rows of the pool, `ds_pf_idx` scans the whole key buffer) uses the dense form row-addressable over
the whole width, built once per prompt.

What this flag does.
  keys     `stack_dense` builds the dense stacked key buffer the core reads by unpacking the codes CHUNK rows at a time straight into it, and drops each
           session entry's mirror afterwards. Same array, same bits, same per-step read: the mirror is redundant because the stack holds the same rows.
  pool     `pool_view` hands the stacked decode the session's OWN packed pool entry (a `fp4pool.PackedView`, the contract every view branch of dsv4_core
           already handles: `_put` -> `cdecpack.put`, `commit_layer` -> `cdecpack.settle`, `sync_rows` -> nothing to write back) instead of a dense copy,
           and drops the pool mirror. The attention then reads the rows its top-k names through `decode_rows`: one fused kernel per layer decodes those
           rows from the codes into a slab of idx.size rows and the indices are remapped to it. The kernel reads 288 B per selected row instead of the
           1024 B the attention reads, and writes the 1024 B the attention then reads: ~0.15 MB read + 0.5 MB written per kv-source layer per row, against
           ~1 MB the attention reads either way.
  prefill  unchanged: the first prefill chunk still builds the dense form (a whole-width gather there), and the next `stack_rows` drops it. A session
           that only decodes afterwards never builds it again.
  freeze   appended rows are written into the packed codes (`cdecpack.put`), so no read-back is taken (`defer_dense` around the key write-back): an append
           costs its own rows and never rebuilds a mirror.

Exactness. Rows are independent in the fp4 codec (`fp4pack.unpack` of a row subset equals the row subset of the whole unpack; depthflat's premise), so a
decoded row is bit-identical to the dense row it replaces, `decode_rows` returns the same rows and remap as `cdecpack.compact`, and the kernels that read
the slab use an index purely as a row address (`0 <= idx < C`, `dsv4_core._ATTN`, `fuse_attn._ATTEND_ROWS_M`), so a remapped index reads the same row in
the same order. The index-key readers see the same dense key stack as at flag 0. `tests/test_mirrorq.py` checks rows, slab, stack, append, rollback, and
the served core at flag 0 against flag 1 on Metal.

The fused kernel computes `fp16(LUT16[nibble] * scale32)`: the LUT holds the exact fp32 literals `fp4pack.unpack` produces (sign bit = nibble 8), the
scale is the same `from_fp8` / `2^(e-127)` value, and the same single fp32 multiply then fp32 -> fp16 rounding is applied, under `fp contract(off)` so
nothing fuses. `tests/test_mirrorq.py::test_kernel_equals_unpack` compares every element against `fp4pack.unpack`.
"""
from __future__ import annotations

import functools
import os
from contextlib import contextmanager

import mlx.core as mx

from . import fp4pack
from .fp4pool import PackedView, _Pk

ENABLED = os.environ.get("KYOJIN_MIRRORQ", "0") not in ("0", "")
CHUNK = int(os.environ.get("KYOJIN_MIRRORQ_CHUNK", "32768"))           # rows unpacked per step of a whole-layer read (bounds the fp32 temporaries)
SLAB_ROWS = int(os.environ.get("KYOJIN_MIRRORQ_SLAB_ROWS", "16384"))    # rows of one decode slab; a wider index set is decoded in steps
HITS = {"pool_view": 0, "pool_drop": 0, "keys_stack": 0, "keys_drop": 0, "keys_copy": 0, "drop_bytes": 0,
        "decode": 0, "decode_rows": 0, "decode_kernel": 0, "decode_fallback": 0,
        "skip": 0, "stack_skip": 0, "free": 0, "no_readback": 0}
_DTYPES = (mx.float16, mx.bfloat16, mx.float32)


def on() -> bool:
    """The flag, read per call (POST /admin/flags -> flags.set writes KYOJIN_MIRRORQ back)."""
    return ENABLED


def is_view(x) -> bool:
    from .cdecpack import is_view as v
    return v(x)


def free(e) -> int:
    """Drop the dense mirror / memo of a packed entry; returns the bytes freed."""
    if not isinstance(e, _Pk):
        return 0
    nb = 0 if e.mirror is None else int(e.mirror.nbytes)
    if e.mirror is not None or e.mn or e.memo is not None:
        e.mirror, e.mn, e.memo = None, 0, None
        HITS["free"] += 1
    HITS["drop_bytes"] += nb
    return nb


@contextmanager
def writeback(slots):
    """Around the write-back of a layer whose dense form this flag moved into the stack: `append_rows` does not read the layer back dense, so an append
    costs its own rows and never rebuilds the mirror that was dropped."""
    if slots is None or not hasattr(slots, "defer_dense"):
        yield
        return
    slots.defer_dense = True
    HITS["no_readback"] += 1
    try:
        yield
    finally:
        slots.defer_dense = False


# ----------------------------------------------------------------------------------------------------------------- the pool: the session's own codes
def pool_view(slots, l: int):
    """Layer `l` of a session's pool slots as a `fp4pool.PackedView` over its own codes, its dense mirror dropped, or None when the layer has no packed
    form (dense off the grid, more than one batch row, above the depthflat budget, not a packed slots object). The view is the contract `cdecpack.view`
    establishes: every view branch of `dsv4_core` (`_put` -> put, `commit_layer` -> settle, `sync_rows` -> nothing) already handles it."""
    e = slots.raw(l) if hasattr(slots, "raw") else None
    if not isinstance(e, _Pk) or e.codes is None or e.codes.shape[0] != 1 or e.dtype not in _DTYPES:
        HITS["skip"] += 1
        return None
    from . import depthflat as DF
    D = e.codes.shape[2] * 2
    if e.codes.shape[0] * e.n * D * (4 if e.dtype == mx.float32 else 2) > DF.MIRROR_BYTES:     # depthflat keeps the stock path above the budget
        HITS["skip"] += 1
        return None
    free(e)
    HITS["pool_view"] += 1
    return PackedView(e.codes, e.scales, e.n, e.dtype, (slots, e), type(slots).BLOCK, type(slots).E4M3)


# ----------------------------------------------------------------------------------------------------------------- the keys: the stack, not a mirror
def stack_dense(slot_lists: list, l: int, step: int):
    """(dense [S, cap, D], cnt) of layer `l` for the stacked decode, unpacked from the codes CHUNK rows at a time straight into the buffer `_stack_ragged`
    would allocate (same cap, same dtype, same rows), each session entry's mirror dropped afterwards: the stack holds those rows now. None when a session
    holds the layer dense or off the packed form (the caller stacks it dense as before)."""
    raws = []
    for sl in slot_lists:
        if not hasattr(sl, "raw"):
            HITS["stack_skip"] += 1
            return None
        raws.append(sl.raw(l))
    have = [e for e in raws if e is not None]
    if not have or any(not isinstance(e, _Pk) or e.codes is None or e.codes.shape[0] != 1 for e in have):
        HITS["stack_skip"] += 1
        return None
    e0 = have[0]
    if any(e.dtype != e0.dtype or e.codes.shape[2] != e0.codes.shape[2] or e.scales.shape[2] != e0.scales.shape[2] for e in have):
        HITS["stack_skip"] += 1
        return None
    cnt = [0 if e is None else e.n for e in raws]
    if max(cnt) == 0:
        HITS["stack_skip"] += 1
        return None
    S, D = len(raws), e0.codes.shape[2] * 2
    cap = -(-(max(cnt) + 1) // step) * step
    buf = mx.zeros((S, cap, D), dtype=e0.dtype)
    for u, e in enumerate(raws):
        if not cnt[u]:
            continue
        if e.mirror is not None and e.mn == e.n:                       # a mirror the prefill read already built: copy it (as `_stack_ragged`) and drop it
            buf[u:u + 1, :cnt[u]] = e.mirror[:, :cnt[u]]
            HITS["keys_copy"] += 1
        else:
            blk, e4 = type(slot_lists[u]).BLOCK, type(slot_lists[u]).E4M3
            for a in range(0, cnt[u], CHUNK):                          # rows are independent in the codec: any row split is the same unpack
                b = min(cnt[u], a + CHUNK)
                buf[u:u + 1, a:b] = fp4pack.unpack(e.codes[:, a:b], e.scales[:, a:b], blk, e4, e.dtype)
        HITS["keys_drop"] += 1 if free(e) else 0
    HITS["keys_stack"] += 1
    return buf, cnt


# ----------------------------------------------------------------------------------------------------------------- the slab: only the rows a reader names
_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_LUT16 = _LUT + tuple(-v for v in _LUT)                                  # nibble 0..15: the sign bit is 8 (negative from 8), the values unpack builds
_LIT = ", ".join(repr(float(v)).replace("inf", "INFINITY") + "f" for v in _LUT16)
_LUTF = mx.array(_LUT, dtype=mx.float32)

# The scale byte of a row, decoded in the kernel: `mx.from_fp8(s, float32)` for e4m3 (pool rows, block 16) and `2^(e - 127)` for the ue8m0 exponent
# byte (index keys, block 32). Both are exact: an e4m3 value is m/8 * 2^k, the exponent byte is a power of two.
_HEAD = r"""#pragma METAL fp contract(off)
constant float MQ_LUT[16] = {@LUT@};
inline float mq_e4m3(uint b) {{                             // OCP e4m3: sign 1, exp 4 (bias 7), mantissa 3; exp 0 = subnormal * 2^-6
    const float m = float(b & 7u) * 0.125f;
    const uint e = (b >> 3) & 15u;
    float v = e == 0u ? m * 0.015625f : (1.0f + m) * exp2(float(int(e) - 7));
    return (b & 128u) ? -v : v;
}}
inline float mq_ue8m0(uint b) {{ return exp2(float(int(b)) - 127.0f); }}
"""

_DEC = r"""
    const uint m = threadgroup_position_in_grid.y;
    const uint e = threadgroup_position_in_grid.x * 128 + thread_index_in_threadgroup;
    const int row = rows[m];
    const uint ok = (row >= 0 && uint(row) < nrow[0]) ? 1u : 0u;
    const uint s = ok != 0u ? uint(row) : 0u;
    const uint b = uint(codes[ulong(s) * (@DD@ / 2) + (e >> 1)]);
    const uint nib = (e & 1u) ? (b >> 4) : (b & 15u);
    const float sc = @SCALE@(uint(scales[ulong(s) * (@DD@ / @BK@) + e / @BK@]));
    out[ulong(m) * @DD@ + e] = T(MQ_LUT[nib] * sc);
    if (e == 0u) remap[m] = ok != 0u ? int(m) : -1;
"""


@functools.lru_cache(maxsize=None)
def _dec_kernel(D: int, block: int, e4m3: bool):
    """One launch: `idx` -> the dense fp16 rows of the codes (`fp4pack.unpack` of exactly those rows) and the slab slot of each entry (-1 outside
    `[0, n)`). `out` is written flat, so an output shaped like `idx` is laid out exactly like `idx.reshape(-1)`: no separate remap op."""
    return mx.fast.metal_kernel(name=f"kyojin_mq_rows_{D}_{block}_{int(e4m3)}",
                                header=_HEAD.replace("@LUT@", _LIT),
                                input_names=["codes", "scales", "rows", "nrow"], output_names=["out", "remap"],
                                source=_DEC.replace("@DD@", str(D)).replace("@BK@", str(block)).replace("@SCALE@", "mq_e4m3" if e4m3 else "mq_ue8m0"))


def _scales32(e, block: int, e4m3: bool, rows):                          # [M, D / block] fp32, the same value `unpack` multiplies by
    s = e.scales[0][rows]
    return mx.from_fp8(s, mx.float32) if e4m3 else mx.power(2.0, s.astype(mx.float32) - 127.0)


def _rows_mlx(e, block: int, e4m3: bool, rows, dtype):
    """`fp4pack.unpack`'s own ops over the named rows (all in range): the no-Metal form, and the reference the kernel is checked against."""
    D = e.codes.shape[2] * 2
    c = e.codes[0][rows]
    v = mx.stack([c & 0xF, c >> 4], axis=-1).reshape(rows.size, D)
    mag = _LUTF[(v & 7).astype(mx.int32)]
    val = mx.where((v & 8) > 0, -mag, mag)
    sc = _scales32(e, block, e4m3, rows)
    return ((val.reshape(rows.size, -1, block) * sc[..., None]).reshape(rows.size, D).astype(dtype))[None]


def decode_rows(view, idx):
    """(rows [1, M, D] fp16, remap) for `view` a `PackedView`: the dense rows of the entries `idx` names, decoded from the codes, and the slot of each
    entry in `rows` (-1 where `idx` is outside `[0, view.n)`). M = idx.size, so the slab holds one row per selected entry (a duplicate is decoded twice: a
    slab row is an address the attention reads, not a cache). Same rows and the same remap as `cdecpack.compact`; ONE launch (the scale decode and the
    remap ride in the same kernel), against the ~10 elementwise launches of the ops form."""
    shape = idx.shape
    M, D = idx.size, view.D
    e = view.owner[1] if view.owner is not None else None
    block, e4m3 = getattr(view, "block", 16), getattr(view, "e4m3", True)
    fused = e is not None and view.dtype == mx.float16 and D % 128 == 0 and bool(mx.metal.is_available()) and M >= 1
    if fused:
        HITS["decode_kernel"] += 1
        k = _dec_kernel(D, block, bool(e4m3))
        nrow = mx.array([view.n], dtype=mx.uint32)
        flat = idx.reshape(-1)
        outs = None
        for a in range(0, M, SLAB_ROWS):
            b = min(M, a + SLAB_ROWS)
            one = k(inputs=[view.codes, view.scales, flat[a:b], nrow], template=[("T", view.dtype)],
                    grid=(D, b - a, 1), threadgroup=(128, 1, 1),
                    output_shapes=[(b - a, D), (b - a,)], output_dtypes=[view.dtype, mx.int32])
            outs = one if outs is None else (mx.concatenate([outs[0], one[0]], axis=0), mx.concatenate([outs[1], one[1]], axis=0))
        rows, remap = outs[0][None], outs[1].reshape(shape)
    else:
        HITS["decode_fallback"] += 1
        valid = (idx >= 0) & (idx < view.n)
        safe = mx.where(valid, idx, 0).reshape(-1)
        if e is None:                                                  # no codec entry behind the view: the stock unpack of the whole layer
            rows = fp4pack.unpack(view.codes[:, :view.n], view.scales[:, :view.n], block, e4m3, view.dtype)[:, safe]
        else:
            parts = [_rows_mlx(e, block, e4m3, safe[a:min(M, a + SLAB_ROWS)], view.dtype) for a in range(0, M, SLAB_ROWS)]
            rows = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)
        remap = mx.where(valid, mx.arange(M, dtype=mx.int32).reshape(shape), -1)
    HITS["decode"] += 1
    HITS["decode_rows"] += M
    return rows, remap


def slab_bytes(idx, D: int) -> int:
    """Bytes one decode slab holds: `idx.size` rows of width D (the bound this lever puts on the decoded rows of a step)."""
    return int(idx.size) * int(D) * 2
