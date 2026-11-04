# SPDX-License-Identifier: Apache-2.0
"""DS4.1 cache that OWNS the compressed storage of the fp4 pool (kyojin_mlx.fp4pack).

The vendor model rounds every pooled KV row (blocks of 16, e4m3 scale) and every index key (blocks of 32, power-of-two scale) to the fp4 grid before it enters
the cache, so the values are exactly representable in 4.5 / 4.25 bits. `PackedDsCache` keeps `compress` and `keys` as `PackedSlots`: a list of per-layer
entries stored as (codes, scales) that reads back, on `slots[l]` / iteration, as the dense fp16 array the readers (prefill, ced, dsv4_core, spec, kvdisk
checkpoints) already expect, and packs again on `slots[l] = dense`. Reads are bit-identical to the dense cache by construction; a write whose rows are NOT on the
grid (checked on every write) keeps that layer dense instead of rounding it, so the cache can never lose a bit.

KYOJIN_KVSEG (segments, default 8192 rows = on): a prefix snapshot (`rows.snapshot`) stops aliasing the live buffer. An alias makes the live session's next append copy the
whole capacity-sized buffer (the buffer is shared, so the slice update cannot run in place) and pins that buffer in the snapshot, so a snapshot costs O(rows) bytes and
the copy grows with the depth. With KYOJIN_KVSEG=N the rows already full (every N rows) are copied out ONCE into immutable segments that every later snapshot of the
session, and every session forked from one, shares by identity; a snapshot adds the segments completed since the last one plus a copy of the open tail (< N rows).
The live `_Pk` keeps its single contiguous `codes` / `scales`, so every kernel (dsa_attend.attend_packed, fuse_index, dsa_decode.scores_packed) binds the same one buffer
as with the flag off; segments exist only in snapshots (`SegSnapPk`, held by the `Seg*Slots` classes) and a fork thaws one into a private contiguous buffer.

KYOJIN_PF_POOL_LAZY (pf_pool_lazy, default 0 = off): the on-grid check of a pool append (`_try_pack`: pack, unpack, array_equal, read back) is a host sync, so it moves to
a lazy bool and is resolved at the next read / append / snapshot of the layer (`_settle`). The appended rows are written into the dense mirror as they were written, so
every read is exact before the verdict; a layer whose check comes back off the grid goes dense exactly as the checked append makes it. Without the depthflat mirror
the checked append runs (the lazy path needs an existing mirror). Pool only: the index keys keep the checked append.

A pending layer (`ok` not None) may not be copied, thawed into a segment or snapshotted: `copy`, `segment_snapshot` and `raw` resolve it first, because the codes hold
the round trip's bits, not necessarily the rows the layer reads back.

KYOJIN_PF_GROW (pf_grow, default 0 = off): a restored layer takes its next appends in place. Every restore ends with a code buffer of exactly its rows — a
disk checkpoint (`PackedDsCache.state` -> `room`), a compacted prefix snapshot (`rows.compact`), a thawed segment — or with a view of a stored buffer
(`rows.snapshot` -> `fork` -> `cow`), so the first append after it lands past the capacity: `append_rows` allocates `mx.zeros` of the whole next capacity
step and copies every row (985 MB of zeros + the copy of 0.83 GiB of codes on the served lane at 983040 rows, per turn). With the flag on, the restore copies
the rows ONCE into a buffer of `longctx.room(n)` rows (rows + 1/16 + one step, 6.25 % at 983040: 0.83 -> 0.88 GiB at rest) and the append writes its rows where
they go; past the room the `capacity` growth rule is the one of today. `rows.restore` is the hit path (`Engine._begin_prefill`), `PackedDsCache.state` the
disk one. Reads are the same rows of the same codes: only the capacity (the row stride the kernels address with) changes.

Readers that must not unpack: `serve.backends.arrays_of` (state evaluation) reads `raw_arrays()`, `spec.shell` / `spec.copy_layer` and `rows.fork` use `copy()` / `raw` /
`set_raw`."""
from __future__ import annotations

from . import archs as _A
import os

import mlx.core as mx

from . import fp4pack


SEG = int(os.environ.get("KYOJIN_KVSEG", "8192"))                                  # rows per immutable snapshot segment (0 = off, snapshots alias the live buffer); read per snapshot
LAZY_POOL = os.environ.get("KYOJIN_PF_POOL_LAZY", "0") not in ("0", "")           # pf_pool_lazy: the grid check of a pool append does not read back on the host (PoolSlots only)
LAZY_KEYS = os.environ.get("KYOJIN_PF_KEYS_LAZY", "0") not in ("0", "")           # pf_keys_lazy: the same deferred grid check for the index keys (KeySlots); needs the depthflat mirror
RESTORE_ROOM = os.environ.get("KYOJIN_PF_GROW", "0") not in ("0", "")             # pf_grow: a restored layer takes a private code buffer with room for its next appends (room_cap)
HITS = {"sync_tail": 0, "seal": 0, "snap": 0, "thaw": 0, "reset": 0,                               # segments copied out / segmented snapshots taken / snapshot layers thawed / seal restarts
        "lazy": 0, "settle": 0, "settle_dense": 0, "room": 0,                                 # deferred grid checks taken / resolved / resolved off the grid (pf_pool_lazy)
        "tail_rows": 0, "write_bytes": 0}                                          # write_back: rows packed into an existing entry / bytes of a whole-layer write
MEMO_BYTES = int(float(os.environ.get("KYOJIN_FP4_MEMO_MB", "8")) * 2 ** 20)       # a layer whose dense rows fit this stays unpacked in memory after its first read (0 = always packed)


def own(a):
    """A copy of `a` in a buffer of exactly its own size, same bits. A slice of an evaluated array (`buf[:, :n]`, `rows[:, -w:]`) keeps the whole parent buffer alive
    while `nbytes` counts only the slice; a gather by arange is never a view, so the parent can go. Lazy: evaluate the result."""
    if a is None or a.ndim == 0 or a.size == 0:
        return a
    ax = max(a.ndim - 2, 0)
    return mx.take(a, mx.arange(a.shape[ax], dtype=mx.int32), axis=ax)


def roomed(a, cap: int):
    """`own(a)` with `cap` rows of capacity: `a`'s rows and a zero tail in ONE new buffer (the copy a restore already pays, with room for the next appends)."""
    if a is None or a.ndim == 0 or a.size == 0:
        return a
    ax = max(a.ndim - 2, 0)
    n = a.shape[ax]
    if cap <= n:
        return own(a)
    shape = list(a.shape)
    shape[ax] = cap - n
    return mx.concatenate([a, mx.zeros(shape, dtype=a.dtype)], axis=ax)


def room_cap(n: int, step: int | None = None) -> int:
    """Capacity a restored packed layer of `n` rows takes: `longctx.room` with pf_grow on, `n` (no room: the next append grows) with it off."""
    if not RESTORE_ROOM:
        return n
    from . import longctx as LC
    return LC.room(n, PackedSlots.STEP if step is None else step)


def _with_room(e, cap: int):
    """Layer `e`'s rows [0, n) in private code buffers of `cap` rows (one copy), the rest of the entry as it was (dense mirror, memo, sealed segments)."""
    c = _Pk(roomed(e.codes[:, :e.n], cap), roomed(e.scales[:, :e.n], cap), e.n, e.dtype)
    c.mirror, c.mn, c.memo = e.mirror, e.mn, e.memo
    c.segs, c.sealed, c.sr, c.sb = e.segs, e.sealed, e.sr, e.sb
    c.owned = True
    return c


class _Pk:
    """One layer: codes uint8 [B, cap, D / 2], scales uint8 [B, cap, D / block], n valid rows, the dtype of the rows it replaces."""
    __slots__ = ("codes", "scales", "n", "dtype", "memo", "mirror", "mn", "segs", "sealed", "sr", "sb", "ok", "owned")

    def __init__(self, codes, scales, n, dtype):
        self.codes, self.scales, self.n, self.dtype = codes, scales, n, dtype
        self.sb = 0                                         # bytes of the sealed segments (they stay resident in the live session while any snapshot or hit shares them)
        self.segs, self.sealed, self.sr = None, 0, 0        # KYOJIN_KVSEG: tuple of immutable (codes, scales) segments of `sr` rows covering rows [0, sealed) (copies, never views of codes)
        self.memo = None                                    # dense rows of a short layer (speed); long layers stay packed at rest
        self.mirror, self.mn = None, 0                      # KYOJIN_DEPTHFLAT: dense buffer [B, cap, D], first `mn` rows valid (see depthflat.mirror_read)
        self.ok = None                                      # pf_pool_lazy: lazy bool, "every row appended since the last settle is on the grid"; the mirror holds the written rows themselves
        self.owned = False                                  # pf_grow: the codes are a buffer this entry made for itself (a thaw, a restore), not a view of another's


class PackedView:
    """A packed pool layer handed to the decode attention without unpacking (`PackedSlots.lazy`, or `cdecpack.view`): codes uint8 [1, cap, D / 2], scales
    uint8 [1, cap, D / block], n valid rows (the readable bound; `stride` is the capacity of the code buffers). Supports the slices the vendor code takes
    (`[:batch, :n]`), `.shape`, and `.dense()` for every reader that needs the fp16 rows."""
    __slots__ = ("codes", "scales", "n", "dtype", "D", "owner", "block", "e4m3")

    def __init__(self, codes, scales, n, dtype, owner=None, block=16, e4m3=True):
        self.codes, self.scales, self.n, self.dtype, self.D = codes, scales, n, dtype, codes.shape[-1] * 2
        self.owner = owner                                  # (slots, _Pk) the view was cut from: its dense mirror serves `.dense()` (KYOJIN_DEPTHFLAT)
        self.block, self.e4m3 = block, e4m3                  # the codec of the layer (pool 16 / e4m3, index keys 32 / ue8m0)

    @property
    def shape(self):
        return (1, self.n, self.D)

    @property
    def stride(self):
        """Rows per code buffer (the capacity, >= n): the row stride a packed reader has to address with, where `shape[1]` is the valid count."""
        return self.codes.shape[1]

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2 and all(isinstance(k, slice) for k in key) and key[0].stop in (None, 1) and key[1].start in (None, 0):
            return PackedView(self.codes, self.scales, self.n if key[1].stop is None else min(self.n, key[1].stop), self.dtype, self.owner,
                              self.block, self.e4m3)
        return self.dense()[key]

    def dense(self):
        if self.owner is not None:
            from . import depthflat as DF
            if DF.ENABLED:
                d = DF.mirror_read(*self.owner)                     # rows < self.n never change (append-only): a view older than the owner's n slices the mirror
                if d is not None and d.shape[1] >= self.n:
                    DF.HITS["view"] += 1
                    return d[:, :self.n]
        return fp4pack.unpack(self.codes[:, :self.n], self.scales[:, :self.n], self.block, self.e4m3, self.dtype)


def stride(x) -> int:
    """Row stride of a packed buffer (its code capacity), `shape[1]` of a dense one: what a kernel that addresses row `i` by `i * stride` needs."""
    return x.stride if isinstance(x, PackedView) else x.shape[1]


SYNC_TAIL = os.environ.get("KYOJIN_SYNC_TAIL", "1") not in ("0", "")


class PackedSlots(list):
    BLOCK = 16
    E4M3 = True
    STEP = 256
    lazy = False                                            # True: a long packed layer reads back as a PackedView (pool only; set around the decode attention core)
    defer_dense = False                                     # True (KYOJIN_PF_IDX_CPACK, around an index-key publish): `append_rows` does not read the layer back dense, so no mirror is built or grown

    def __init__(self, items=()):
        super().__init__()
        for v in items:
            super().append(self._enc(v))

    # ------------------------------------------------------------ codec
    def _try_pack(self, v):
        codes, scales = fp4pack.pack(v, self.BLOCK, self.E4M3)
        ok = bool(mx.array_equal(fp4pack.unpack(codes, scales, self.BLOCK, self.E4M3, v.dtype), v))
        return (codes, scales) if ok else None

    def _enc(self, v):
        if v is None or isinstance(v, _Pk):
            return v
        if v.size == 0:
            return None
        if v.shape[-1] % self.BLOCK or v.dtype not in (mx.float16, mx.bfloat16, mx.float32):
            return v
        p = self._try_pack(v)
        return v if p is None else _Pk(p[0], p[1], v.shape[1], v.dtype)

    lazy_check = False                                      # PoolSlots: True while pf_pool_lazy is on

    def _settle(self, i) -> None:
        """Resolve the deferred grid check of layer i (pf_pool_lazy). On the grid: nothing to do (the codes are right). Off the grid: the layer goes dense, as the
        checked append would have made it: the mirror holds the rows exactly as they were written, so it is that dense layer."""
        e = list.__getitem__(self, i)
        if not isinstance(e, _Pk) or e.ok is None:
            return
        ok, e.ok = e.ok, None
        HITS["settle"] += 1
        if not bool(ok):
            HITS["settle_dense"] += 1
            list.__setitem__(self, i, e.mirror[:, :e.n])

    def _settle_all(self) -> None:
        for i in range(len(self)):
            self._settle(i)

    def _dec(self, e, i=None):
        if isinstance(e, _Pk):
            if e.ok is not None:                            # a pending layer is read through its mirror only (exact rows); anything else needs the verdict first
                from . import depthflat as DF
                if DF.ENABLED and e.mirror is not None and e.mn == e.n and not self.lazy:
                    return e.mirror[:, :e.n]
                i = self._index_of(e) if i is None else i
                self._settle(i)
                return self._dec(list.__getitem__(self, i), i)
            if e.memo is not None:
                return e.memo
            if self.lazy and e.dtype == mx.float16 and e.codes.shape[0] == 1 and e.n * e.codes.shape[2] * 4 > MEMO_BYTES:
                return PackedView(e.codes, e.scales, e.n, e.dtype, (self, e), self.BLOCK, self.E4M3)
            from . import depthflat as DF
            if DF.ENABLED:
                d = DF.mirror_read(self, e)
                if d is not None:
                    return d
            d = fp4pack.unpack(e.codes[:, :e.n], e.scales[:, :e.n], self.BLOCK, self.E4M3, e.dtype)
            if d.size * 2 <= MEMO_BYTES:
                e.memo = d
            return d
        return e

    def _index_of(self, e) -> int:
        for j, x in enumerate(list.__iter__(self)):
            if x is e:
                return j
        raise ValueError("entry not in slots")

    # ------------------------------------------------------------ list API (dense view)
    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self._dec(e) for e in super().__getitem__(i)]
        return self._dec(super().__getitem__(i), i if i >= 0 else len(self) + i)

    def __iter__(self):
        for e in list.__iter__(self):
            yield self._dec(e)

    def __setitem__(self, i, v):
        if isinstance(i, slice):
            super().__setitem__(i, [self._enc(x) for x in v])
        else:
            super().__setitem__(i, self._enc(v))

    # ------------------------------------------------------------ packed access
    def raw(self, i):
        self._settle(i if i >= 0 else len(self) + i)
        return list.__getitem__(self, i)

    def set_raw(self, i, e):
        list.__setitem__(self, i, e)

    def raw_arrays(self) -> list:
        """Arrays that hold the state, for evaluation. A pending layer adds its deferred verdict: it is evaluated with the codes, no host read."""
        out = []
        for e in list.__iter__(self):
            if isinstance(e, _Pk):
                out += [e.codes[:, :e.n], e.scales[:, :e.n]]
                if e.ok is not None:
                    out.append(e.ok)
            elif e is not None:
                out.append(e)
        return out

    @staticmethod
    def cow(e):
        if isinstance(e, _Pk):
            c = _Pk(e.codes[...], e.scales[...], e.n, e.dtype)
            if e.mirror is not None:                                # an alias (new array object): an append to either side never shows in the other
                c.mirror, c.mn = e.mirror[...], e.mn
            if e.segs is not None:                                  # sealed rows never change (append-only): the copy shares the segments
                c.segs, c.sealed, c.sr, c.sb = e.segs, e.sealed, e.sr, e.sb
            return c
        return None if e is None else e[...]

    def seg_bytes(self) -> int:
        """Bytes of sealed segments this live layer set keeps resident beside its contiguous buffer (KYOJIN_KVSEG): resident whether or not a prefix entry still holds them, so a
        session's byte count (`Backend.nbytes`) includes them; the prefix cache bills the same segments once for its entries (a conservative overlap, never an under-count)."""
        return sum(e.sb for e in list.__iter__(self) if isinstance(e, _Pk))

    def retained(self, seen: set, billed: dict | None = None) -> dict:
        """Read-only ledger of what these layers keep resident (flag-independent, never builds a mirror: raw entries only). `seen` holds ids of entries already counted
        (one entry reached twice counts once; `cow` aliases a mirror under a new array object, so a forked pair can still count a mirror twice: an upper bound).
        counted = what `raw_arrays` / `nbytes` bills (the n valid rows); slack = the rest of the code buffers; mirror / memo = dense buffers `nbytes` never sees;
        seg_overlap = sealed segments of a live layer that the prefix ledger (`billed`: {id(segment): bytes}) already bills."""
        out = dict.fromkeys(("counted", "slack", "mirror", "memo", "seg", "seg_overlap"), 0)
        for e in list.__iter__(self):
            if not isinstance(e, _Pk) or id(e) in seen:
                continue
            seen.add(id(e))
            n_c = e.codes.shape[0] * e.n * e.codes.shape[2] + e.scales.shape[0] * e.n * e.scales.shape[2]       # shape arithmetic: no slice op, safe beside the engine thread (uint8: 1 B per element)
            out["counted"] += n_c
            out["slack"] += e.codes.nbytes + e.scales.nbytes - n_c
            out["mirror"] += 0 if e.mirror is None else e.mirror.nbytes
            out["memo"] += 0 if e.memo is None else e.memo.nbytes
            out["seg"] += e.sb
            if billed and e.segs:
                out["seg_overlap"] += sum(b for b in (billed.get(id(sg)) for sg in e.segs) if b)
        return out

    def drop_mirrors(self):
        """For a snapshot: packed codes only, a dense read rebuilds the mirror."""
        self._settle_all()
        for e in list.__iter__(self):
            if isinstance(e, _Pk):
                e.mirror, e.mn, e.memo = None, 0, None

    def compacted(self):
        """For a long-lived snapshot (flag prefix_compact): the valid rows only, each layer in buffers of exactly n rows (no capacity slack, no dense mirror or memo). Lazy arrays."""
        new = type(self)()
        for e in list.__iter__(self):
            if isinstance(e, SegSnapPk):                             # sealed segments and tail are already copies of exactly their size, the segments stay shared
                list.append(new, e)
                continue
            list.append(new, _Pk(own(e.codes[:, :e.n]), own(e.scales[:, :e.n]), e.n, e.dtype) if isinstance(e, _Pk) else own(e))
        return new

    def copy(self):
        self._settle_all()
        new = type(self)()
        for e in list.__iter__(self):
            list.append(new, self.cow(e))
        return new

    # ------------------------------------------------------------ pf_grow (a restored layer has room)
    def _rooms(self, force: bool) -> None:
        """Give every packed layer that needs one a private code buffer of `room_cap` rows, evaluated at the end: `force` = the layer is a view of a stored buffer
        (`restore`), else only a layer whose buffer has no room is touched (`room`)."""
        parts = []
        for i, e in enumerate(list.__iter__(self)):
            if isinstance(e, _Pk) and e.codes is not None:
                self._settle(i)                                     # a pending lazy check is the rows' verdict: resolve it before the codes are copied
                e = list.__getitem__(self, i)
                if not isinstance(e, _Pk):
                    continue
                cap = room_cap(e.n)
                if e.owned and cap <= e.codes.shape[1]:
                    parts += [e.codes, e.scales]                    # a private buffer with room already (a thaw): only make sure it exists
                    continue
                if force or cap > e.codes.shape[1]:
                    HITS["room"] += 1
                    e = _with_room(e, cap)
                    list.__setitem__(self, i, e)
                    parts += [e.codes, e.scales]
        if parts:
            mx.eval(*parts)                                         # the padded buffers exist once, at the restore: a lazy tail would make the first append copy it

    def room(self):
        """A layer restored into a code buffer of its own (a disk checkpoint, through `PackedDsCache.state`): every packed layer that has no room for its next
        appends takes a buffer of `room_cap` rows, in one copy of its rows (pf_grow). A layer that already has room is left alone: the codes and the rows are
        the ones it had."""
        if RESTORE_ROOM:
            self._rooms(force=False)
        return self

    def restore(self):
        """A cache restored from a stored snapshot (a hit: `copy` / `fork` leaves every packed layer a view of the stored buffer, so its first append copies the
        whole buffer to write in place, and an exact snapshot — prefix_compact, a disk checkpoint — grows it with a zero buffer of the next capacity step): each
        layer takes a private code buffer with room for its next appends, in one copy of its rows (pf_grow). The first append after the restore then writes its
        rows where they go. Called on the fork (`rows.restore`), never on a live layer."""
        if RESTORE_ROOM:
            self._rooms(force=True)
        return self

    # ------------------------------------------------------------ KYOJIN_KVSEG
    def segment_snapshot(self):
        """A snapshot of these slots that shares the full segments and copies only the open tail (see the module doc). The live entries are not aliased: their buffers stay
        donatable, so the next append writes in place. Layers that are not packed (None, dense off-grid) are aliased as `cow` does. The sealed copies are evaluated here
        (a lazy copy would keep the live buffer alive until it ran). A pending pf_pool_lazy check is resolved first: until then the codes are not the rows the layer reads back."""
        self._settle_all()
        new = seg_class(type(self))()
        pend, todo = [], []
        for e in list.__iter__(self):
            if isinstance(e, _Pk) and e.codes is not None and SEG > 0:
                if e.sr != SEG or e.sealed > e.n:                   # the segment size changed, or the layer was rebuilt below its sealed rows: start over
                    if e.segs:
                        HITS["reset"] += 1
                    e.segs, e.sealed, e.sr, e.sb = (), 0, SEG, 0
                full = e.n // SEG * SEG
                fresh = []
                for a in range(e.sealed, full, SEG):
                    fresh.append((_own(e.codes[:, a:a + SEG]), _own(e.scales[:, a:a + SEG])))
                tail = None if e.n == full else (_own(e.codes[:, full:e.n]), _own(e.scales[:, full:e.n]))
                for c, sc in fresh:
                    pend += [c, sc]
                if tail is not None:
                    pend += list(tail)
                todo.append((e, fresh, full))
                list.append(new, SegSnapPk(e.segs + tuple(fresh), tail, e.n, e.dtype, SEG, e.sb + sum(c.nbytes + sc.nbytes for c, sc in fresh)))
            else:
                list.append(new, self.cow(e))
        if pend:
            mx.eval(*pend)
        for e, fresh, full in todo:                                 # commit after the copies ran: a failed eval leaves the live entry as it was
            if fresh:
                e.segs, e.sealed, e.sb = e.segs + tuple(fresh), full, e.sb + sum(c.nbytes + sc.nbytes for c, sc in fresh)
                HITS["seal"] += len(fresh)
        HITS["snap"] += 1
        return new

    # ------------------------------------------------------------ pf_pool_lazy
    def _lazy_ok(self, e, values) -> bool:
        """pf_pool_lazy applies: a packed layer that already has an exact dense mirror of all its rows, the same dtype, inside the mirror budget."""
        if not (self.lazy_check and values.dtype == e.dtype):
            return False
        from . import depthflat as DF
        return (DF.ENABLED and e.mirror is not None and e.mn == e.n and e.dtype in DF._DTYPES
                and e.codes.shape[0] * (e.n + values.shape[1]) * e.codes.shape[2] * 2 * (4 if e.dtype == mx.float32 else 2) <= DF.MIRROR_BYTES)

    def _mirror_write(self, e, values) -> None:
        """Write the appended rows themselves into the mirror tail (what the checked append's unpack of the codes gives when the rows are on the grid)."""
        from . import depthflat as DF
        n, end = e.n, e.n + values.shape[1]
        m, B, D = e.mirror, e.codes.shape[0], e.codes.shape[2] * 2
        if m.shape[1] < end:                                        # grow: copy the valid prefix once per n/8 rows (as depthflat.mirror_read)
            DF.HITS["grow"] += 1
            cap = -(-(end + (end >> DF.SLACK_SHIFT)) // self.STEP) * self.STEP
            m = mx.concatenate([m[:, :n], mx.zeros((B, cap - n, D), dtype=e.dtype)], axis=1)
        DF.HITS["tail"] += 1
        m[:, n:end] = values
        e.mirror, e.mn = m, end

    def append_rows(self, i, values):
        """Append `values` [B, m, D] to layer i; returns the dense layer."""
        e = list.__getitem__(self, i)
        if e is None or not isinstance(e, _Pk):
            cur = e
            new = values if cur is None else mx.concatenate([cur, values], axis=1)
            list.__setitem__(self, i, self._enc(new) if cur is None else new)
            return None if self.defer_dense else self[i]
        lazy = self._lazy_ok(e, values)
        if lazy:
            codes_new, scales_new = fp4pack.pack(values, self.BLOCK, self.E4M3)
            rt = fp4pack.unpack(codes_new, scales_new, self.BLOCK, self.E4M3, values.dtype)
            ok = mx.array_equal(rt, values)                         # not read here (array_equal: -0.0 == +0.0, so the mirror takes the unpacked bits, see _mirror_write)
            p = (codes_new, scales_new)
            HITS["lazy"] += 1
        else:
            self._settle(i)
            e = list.__getitem__(self, i)
            if not isinstance(e, _Pk):                              # the pending check came back off the grid: the layer is dense now
                list.__setitem__(self, i, mx.concatenate([e, values], axis=1))
                return None if self.defer_dense else self[i]
            p = self._try_pack(values) if values.dtype == e.dtype else None
        if p is None:                                               # off the grid: the layer goes dense (never rounded)
            list.__setitem__(self, i, mx.concatenate([self._dec(e, i), values], axis=1))
            return self[i]
        end = e.n + values.shape[1]
        if end > e.codes.shape[1]:
            from . import longctx as LC
            cap = LC.capacity(end, self.STEP)
            B = e.codes.shape[0]
            codes = mx.zeros((B, cap, e.codes.shape[2]), dtype=mx.uint8)
            scales = mx.zeros((B, cap, e.scales.shape[2]), dtype=mx.uint8)
            codes[:, :e.n], scales[:, :e.n] = e.codes[:, :e.n], e.scales[:, :e.n]
            old = e
            e = _Pk(codes, scales, e.n, e.dtype)
            e.mirror, e.mn, e.ok = old.mirror, old.mn, old.ok     # the dense mirror survives the code buffer growth (it has its own capacity)
            e.segs, e.sealed, e.sr, e.sb = old.segs, old.sealed, old.sr, old.sb  # so do the sealed segments (rows below `sealed` are copied out, the new buffer holds the same rows)
            list.__setitem__(self, i, e)
        if lazy:
            e.ok = ok if e.ok is None else mx.logical_and(e.ok, ok)
            # The checked path reads an on-grid layer back as unpack(codes) (bits: -0.0 comes back +0.0) and a layer that went dense as the written rows. The mirror gets
            # the same bits before the verdict is known: the unpacked rows while every append since the last settle is on the grid, else the rows as written.
            self._mirror_write(e, mx.where(e.ok, rt, values))
        e.codes[:, e.n:end], e.scales[:, e.n:end] = p
        e.n = end
        e.memo = None
        return None if self.defer_dense else self[i]

    def write_back(self, i, dense, n) -> None:
        """Layer i takes rows [0, n) of `dense` [1, >=n, D], the stack that was copied from it and that only appends (flag sync_tail). A packed entry that still holds the
        first e.n rows packs only the rows after them: the codes of the older rows stay, the same codes a whole re-pack gives. Anything else takes the whole write."""
        e = list.__getitem__(self, i)
        if SYNC_TAIL and n and isinstance(e, _Pk) and 0 < e.n <= n and e.codes.shape[0] == 1 and dense.dtype == e.dtype:
            if e.n < n:
                self.append_rows(i, dense[:, e.n:n][...])
                HITS["tail_rows"] += n - e.n
            HITS["sync_tail"] += 1
            return
        HITS["write_bytes"] += dense[:, :n].nbytes if n else 0
        self[i] = dense[:, :n][...] if n else None


def _own(x):
    """An independent copy of an array slice (a real op: a view would keep the live buffer referenced and force the live session's next append to copy it whole)."""
    return x + mx.zeros((1,), dtype=x.dtype)


class SegSnapPk:
    """One packed layer of a segmented snapshot: sealed segments (shared by identity with the live session and every other snapshot of it), a private copy of the open
    tail, and no contiguous buffer. `[...]` (what `PackedSlots.cow` does) thaws it into a private `_Pk` with its own contiguous buffer: one O(rows) concatenation, the
    copy a hit pays today at its first append. The thawed `_Pk` keeps the segments, so its own snapshots stay O(new rows)."""
    __slots__ = ("segs", "tail", "n", "dtype", "sr", "sb")

    def __init__(self, segs, tail, n, dtype, sr, sb):
        self.segs, self.tail, self.n, self.dtype, self.sr, self.sb = segs, tail, n, dtype, sr, sb

    def arrays(self) -> list:
        out = [a for seg in self.segs for a in seg]
        return out + list(self.tail or ())

    def __getitem__(self, key):
        assert key is Ellipsis, key
        HITS["thaw"] += 1
        codes = [seg[0] for seg in self.segs] + ([self.tail[0]] if self.tail else [])
        scales = [seg[1] for seg in self.segs] + ([self.tail[1]] if self.tail else [])
        from . import longctx as LC
        cap = LC.capacity(self.n, PackedSlots.STEP)
        if cap > self.n:
            B = codes[0].shape[0]
            codes.append(mx.zeros((B, cap - self.n, codes[0].shape[2]), dtype=mx.uint8))
            scales.append(mx.zeros((B, cap - self.n, scales[0].shape[2]), dtype=mx.uint8))
        e = _Pk(mx.concatenate(codes, axis=1) if len(codes) > 1 else codes[0][...], mx.concatenate(scales, axis=1) if len(scales) > 1 else scales[0][...], self.n, self.dtype)
        e.segs, e.sealed, e.sr, e.sb = self.segs, len(self.segs) * self.sr, self.sr, self.sb
        e.owned = cap > self.n                                      # the tail's zeros made it a concatenation, not a view of the last segment
        return e


_SEG_CLASSES: dict = {}


def seg_class(base):
    """The slots class of a segmented snapshot of `base` (PoolSlots / KeySlots): every reader that a snapshot meets (dense read, copy / fork, cow, state arrays, append)
    works on it, whatever KYOJIN_KVSEG is now; the live classes carry none of these overrides, so their read paths are unchanged."""
    cls = _SEG_CLASSES.get(base)
    if cls is None:
        class SegSlots(base):
            seg_base = base

            def _dec(self, e, i=None):
                return base._dec(self, e[...] if isinstance(e, SegSnapPk) else e, i)

            def copy(self):
                self._settle_all()                           # a pending pf_pool_lazy check is resolved before the codes are copied
                new = base()
                for e in list.__iter__(self):
                    list.append(new, base.cow(e))               # a SegSnapPk thaws through `e[...]`, the others alias as always
                return new

            def raw_arrays(self) -> list:
                out = []
                for e in list.__iter__(self):
                    if isinstance(e, SegSnapPk):
                        out += e.arrays()
                    elif isinstance(e, _Pk):
                        out += [e.codes[:, :e.n], e.scales[:, :e.n]]
                    elif e is not None:
                        out.append(e)
                return out

            def seg_parts(self) -> dict:
                """{id(segment): bytes} of the shared segments, for the prefix cache's byte ledger."""
                return {id(seg): seg[0].nbytes + seg[1].nbytes for e in list.__iter__(self) if isinstance(e, SegSnapPk) for seg in e.segs}

            def drop_mirrors(self):
                pass

            def seg_bytes(self) -> int:                         # a snapshot lists its segments in raw_arrays: nothing extra
                return 0

            def retained(self, seen: set, billed: dict | None = None) -> dict:
                """A snapshot lists its segments in `raw_arrays`: nothing is hidden. Layers that were thawed into a `_Pk` are counted as live ones."""
                out = dict.fromkeys(("counted", "slack", "mirror", "memo", "seg", "seg_overlap"), 0)
                for e in list.__iter__(self):
                    if isinstance(e, _Pk):
                        one = base.retained(self._one(e), seen, billed)
                        for k, v in one.items():
                            out[k] += v
                return out

            @staticmethod
            def _one(e):
                s = base()
                list.append(s, e)
                return s

            def append_rows(self, i, values):                   # a snapshot that is written to becomes a live layer first
                e = list.__getitem__(self, i)
                if isinstance(e, SegSnapPk):
                    list.__setitem__(self, i, e[...])
                return base.append_rows(self, i, values)

            def segment_snapshot(self):
                return base.segment_snapshot(self.copy())

        SegSlots.__name__ = SegSlots.__qualname__ = "Seg" + base.__name__
        cls = _SEG_CLASSES[base] = SegSlots
    return cls


def retained_walk(obj, seen: set | None = None, billed: dict | None = None, acc: dict | None = None) -> dict:
    """Sum `PackedSlots.retained` over every packed slots object reachable from a cache (lists, tuples, dicts, objects), each once. Read only."""
    seen = set() if seen is None else seen
    acc = dict.fromkeys(("counted", "slack", "mirror", "memo", "seg", "seg_overlap"), 0) if acc is None else acc
    if obj is None or isinstance(obj, (mx.array, int, float, str, bytes)) or id(obj) in seen:
        return acc
    seen.add(id(obj))
    if hasattr(obj, "retained") and isinstance(obj, list):
        for k, v in obj.retained(seen, billed).items():
            acc[k] += v
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            retained_walk(v, seen, billed, acc)
    elif isinstance(obj, dict):
        for v in obj.values():
            retained_walk(v, seen, billed, acc)
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        for v in list(vars(obj).values()):
            retained_walk(v, seen, billed, acc)
    return acc


class PoolSlots(PackedSlots):
    BLOCK, E4M3 = 16, True

    @property
    def lazy_check(self):                                   # KeySlots keep the checked append (index keys: track cpackscan)
        return LAZY_POOL


class KeySlots(PackedSlots):
    BLOCK, E4M3 = 32, False

    @property
    def lazy_check(self):                                   # pf_keys_lazy (default 0): the index-key append defers its on-grid host sync like the pool's
        return LAZY_KEYS


def enabled() -> bool:
    return os.environ.get("KYOJIN_FP4_POOL", "1") != "0"


def make_packed_class(base):
    """DeepseekV41Cache -> the subclass whose compress / keys slots are packed (built once, on the vendor class of the loaded model)."""
    class PackedDsCache(base):
        def __init__(self, n_layers: int = 0):
            super().__init__(n_layers)
            self.compress, self.keys = PoolSlots(self.compress), KeySlots(self.keys)

        def _append(self, arrays, buffers, layer_idx, values):
            if isinstance(arrays, PackedSlots):
                return arrays.append_rows(layer_idx, values)
            return super()._append(arrays, buffers, layer_idx, values)

        def extract(self, index):
            """Flag extract_raw: a one-session cache leaves for a batch with its packed layers as they are. The vendor's extract reads every layer back dense and
            `_try_pack` packs and verifies it again: O(prompt) on the first decode round after every prefill, for the same codes."""
            if os.environ.get("KYOJIN_EXTRACT_RAW", "1") in ("0", "") or index != 0:
                return super().extract(index)
            for layers in (self.compress, self.keys):
                for e in list.__iter__(layers):
                    if isinstance(e, SegSnapPk) or (e is not None and (e.codes if isinstance(e, _Pk) else e).shape[0] != 1):
                        return super().extract(index)
            pools = self.compress, self.keys
            self.compress, self.keys = [None] * self.n_layers, [None] * self.n_layers       # the vendor's extract copies everything else
            try:
                cache = super().extract(index)
            finally:
                self.compress, self.keys = pools
            cache.compress, cache.keys = pools[0].copy(), pools[1].copy()
            return cache

        @property
        def state(self):
            return base.state.fget(self)

        @state.setter
        def state(self, value):
            base.state.fset(self, value)
            self.compress, self.keys = PoolSlots(self.compress), KeySlots(self.keys)
            self._compress_buffers = [None] * self.n_layers
            self._key_buffers = [None] * self.n_layers
            for slots in (self.compress, self.keys):                 # pf_grow: a restored layer's code buffers are exact: give them room (one copy, no per-turn growth)
                slots.room()

    return PackedDsCache


def wrap_cache(cache: list) -> list:
    """Swap the model's DeepseekV41Cache entry for the packed subclass (empty cache from make_cache)."""
    if not cache or type(cache[0]).__name__ != _A.vendor().cache or not enabled():
        return cache
    cls = make_packed_class(type(cache[0]))
    return [cls(cache[0].n_layers)]
