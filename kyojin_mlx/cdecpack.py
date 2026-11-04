# SPDX-License-Identifier: Apache-2.0
"""cdecpack: the served decode reads the fp4-packed pool and index keys, and gathers the selected pool rows only.

The dense copy is not built by a kernel that needs it: it is built by the READ of the packed layer. `fp4pool.PackedSlots._dec` hands a long layer back as the
depthflat mirror (`depthflat.mirror_read`: n x D x 2 bytes kept for the life of the session, 1 KiB per pool row and 256 B per key row, ~3.1 GiB at 1M tokens),
and `dsv4_core.stack_rows` then copies it into the dense stack the batched core reads (`_stack_ragged`), whose `_put` scatter buffer is a second dense copy.
So a hook on the kernel input (the old `compact` sites: the chained verify of a reuse layer and the `fuse_attn.attend` fallback) never fires on the served
decode, whose core reads the stacked pool through `dsv4_core.attend`.

KYOJIN_CDECPACK=1 (default 0, exact) hooks the read:
    view      a single-session decode stacks the session's OWN packed entries (`view`): the codes and scales, no unpack, no mirror, no copy. Multi-session
              batches stay dense here (`stackpack` is the packed stack for 2+ sessions).
    put       the rows an emitting pass adds are packed and scattered into the codes of that same entry (`put`); nothing is unpacked, copied back or
              re-packed, and the entry's valid count follows the commit (`settle`). The producers (`fake_quant_fp4_e4m3`, `rot_fq4`) round the rows onto the
              codec's grid, so the pack is exact; tests/test_decpack.py checks it against the dense scatter row for row.
    compact   the attention reads the K rows its top-k names: `compact` gathers those rows of codes, unpacks only them and remaps the indices to the slab.
              The kernels use an index purely as a row address with the bound test `0 <= idx < N` (dsv4_core._ATTN, fuse_attn._ATTEND_ROWS), so a row
              gathered by `unpack(codes[i]) == unpack(codes)[i]` (rows and blocks are independent, the depthflat premise) and addressed by its slot in the
              slab is read with the same bits, in the same order, as the dense row it replaces; an index outside [0, n) stays -1.
    keys      the index score kernels (idxfuse.fused / scores_fast) already have a packed-key load; a reader without one gets a transient unpack for that
              call (`dense_of`), never a mirror.
HITS counts every path, so a flag that never reached the served path is visible. No kernel here writes a hidden stream, so the fp16 saturating clamp does not
apply."""
from __future__ import annotations

import os

import mlx.core as mx

from . import fp4pack

ENABLED = os.environ.get("KYOJIN_CDECPACK", "0") not in ("0", "")
HITS = {"read": 0, "drop": 0, "put": 0, "compact": 0, "rows": 0, "key_packed": 0, "key_dense": 0, "skip": 0, "settle": 0}


def is_view(x) -> bool:
    from .fp4pool import PackedView
    return isinstance(x, PackedView)


def view(slots, l):
    """Layer `l` of the packed slots `slots` as a `fp4pool.PackedView` (codes and scales of the session's own entry, no unpack, no mirror), or None when the
    layer has no packed form (dense off the grid, more than one batch row, not a packed slots object). A depthflat mirror the layer already keeps (the prefill
    read built it) is dropped here: the decode reads the codes from now on, and that buffer is the ~3 GiB this lever is for."""
    from .fp4pool import _Pk, PackedView
    e = slots.raw(l) if hasattr(slots, "raw") else None
    if not isinstance(e, _Pk) or e.codes is None or e.codes.shape[0] != 1 or e.dtype != mx.float16:
        HITS["skip"] += 1
        return None
    if e.mirror is not None or e.memo is not None:
        e.mirror, e.mn, e.memo = None, 0, None
        HITS["drop"] += 1
    HITS["read"] += 1
    return PackedView(e.codes, e.scales, e.n, e.dtype, (slots, e), type(slots).BLOCK, type(slots).E4M3)


def put(buf, l: int, em, vals, step: int, capacity):
    """The packed form of `dsv4_core._put` for a single-session view of layer `l`: `vals` [n_emit, D] (on the codec's grid) are packed and written at
    (row of the pass, write index) of the session's own codes, which grow by `capacity` when a row would overflow. Returns the view that holds them; the
    valid count is the commit's (`settle`), so rows a rejected pass wrote past it stay unread and unbilled."""
    slots, e = buf.owner
    codes, scales = e.codes, e.scales
    need = em.need
    if codes.shape[1] < need:                                        # grow once per capacity step, the same rule as the dense buffer
        cap = capacity(need, step)
        nc = mx.zeros((1, cap, codes.shape[2]), dtype=mx.uint8)
        ns = mx.zeros((1, cap, scales.shape[2]), dtype=mx.uint8)
        nc[:, :codes.shape[1]] = codes
        ns[:, :scales.shape[1]] = scales
        codes, scales = nc, ns
        e.mirror, e.mn, e.memo = None, 0, None                       # a mirror of the old buffer describes rows the new one holds too; drop it rather than alias
    pc, ps = fp4pack.pack(vals, buf.block, buf.e4m3)
    codes[:, em.widx_arr] = pc
    scales[:, em.widx_arr] = ps
    e.codes, e.scales = codes, scales
    slots.set_raw(l, e)
    HITS["put"] += 1
    HITS["rows"] += int(em.n)
    return type(buf)(codes, scales, max(buf.n, need), e.dtype, (slots, e), buf.block, buf.e4m3)


def settle(view_, counts) -> "object":
    """After a commit: the valid count of the session's own entry is the committed row count (the rows a rejected pass wrote past it are ignored), and the
    stack keeps a view with that bound."""
    slots, e = view_.owner
    e.n = int(counts[0])
    HITS["settle"] += 1
    return type(view_)(view_.codes, view_.scales, e.n, view_.dtype, view_.owner, view_.block, view_.e4m3)


def compact(view, idx):
    """(rows, remap): `view` a PackedView (any object with codes / scales / n / dtype), `idx` int32 of any shape (-1 = no entry).
    rows [1, M, D] with M = idx.size are the dense fp16 rows of the entries idx names (unpacked from the gathered codes only); remap has idx's shape and holds the
    slot of each entry in `rows`, -1 where idx is outside [0, view.n)."""
    shape = idx.shape
    flat = idx.reshape(-1).astype(mx.int32)
    valid = (flat >= 0) & (flat < view.n)
    safe = mx.where(valid, flat, 0)
    rows = fp4pack.unpack(view.codes[0][safe], view.scales[0][safe], view.block, view.e4m3, view.dtype)[None]
    remap = mx.where(valid, mx.arange(flat.size, dtype=mx.int32), -1).reshape(shape)
    HITS["compact"] += 1
    HITS["rows"] += int(flat.size)
    return rows, remap


def dense_of(x):
    """`x` as a dense array for a reader with no packed form. A PackedView is unpacked for this call only: no depthflat mirror is built, so the bytes do not
    stay for the life of the session (the readers that matter, the score kernels, take the codes themselves)."""
    if is_view(x):
        HITS["key_dense"] += 1
        return fp4pack.unpack(x.codes[:, :x.n], x.scales[:, :x.n], x.block, x.e4m3, x.dtype)
    from . import stackpack as SP
    return SP.dense_of(x)


def packed_keys(x) -> bool:
    """True when a score kernel can load the keys from the codes: idxfuse's packed variant does, for a view and for a stackpack stack alike."""
    return is_view(x) or hasattr(x, "codes")


def keys_hit(x) -> None:
    """Count a score call that reads the key codes (the packed load of idxfuse.fused / scores_fast)."""
    if packed_keys(x):
        HITS["key_packed"] += 1


def rows(x) -> int:
    """Valid rows of a pool / key buffer (the readable bound)."""
    return x.n if is_view(x) else x.shape[1]

