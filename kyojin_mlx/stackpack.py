# SPDX-License-Identifier: Apache-2.0
"""stackpack: the served batched decode keeps the pool and the index keys of its sessions as fp4 codes + scales, not as dense fp16.

`dsv4_core.stack_rows` used to turn each session's packed layer (`fp4pool._Pk`: codes uint8 [1, n, D / 2], scales uint8 [1, n, D / block]) into a dense
[S, cap, D] buffer, and the session kept its own packed copy and its depthflat mirror besides: 2.56 KB (pool) + 0.64 KB (keys) per token dense, 288 B / 68 B per
row packed. KYOJIN_STACKPACK=1 (default 0, only for 2 or more sessions) stacks the codes and the scales instead:

* `stack`: codes / scales of the sessions copied into [S, cap, ...] buffers (the same `cap` as `_stack_ragged`, so every capacity-dependent choice of the core is
  the same). The sessions' own entries are released while stacked (`sync_rows` hands the codes back), so the batch holds one packed copy and no mirror.
* `put`: the rows an emitting pass adds are packed (`fp4pack.pack`, block 16 e4m3 for the pool, block 32 ue8m0 for the keys) and scattered into the buffers. The
  producers (`fake_quant_fp4_e4m3`, `rot_fq4`) round the rows onto that grid, so the pack is exact; it is NOT re-checked at run time (a check is a host sync per
  layer and step), tests/test_stackpack.py checks it on the producers' output instead.
* `gather_slab`: the attention reads the top set only. The K rows each row names are gathered from the codes, unpacked, and placed in a slab of the row's own
  session; the index is remapped to the slab. The kernels use an index as a row address with the test `0 <= idx < C` (dsv4_core._ATTN / _ATTN_PART,
  fuse_attn._ATTEND_ROWS_M), in the order of the list, so the rows read are the same bits in the same order.
* The index score kernels (idxfuse.fused / scores_fast) have a variant that loads each key from the codes (same value as `fp4pack.unpack`, one cast to the
  key dtype); any other key reader (index_scores, dec_filter) gets a transient dense unpack of that layer (HITS["key_dense"]).
HITS makes a flag that never reached the served path visible."""
from __future__ import annotations

import os

import mlx.core as mx

from . import fp4pack

ENABLED = os.environ.get("KYOJIN_STACKPACK", "0") not in ("0", "")
HITS = {"stack": 0, "dense_layer": 0, "put": 0, "gather": 0, "gather_rows": 0, "key_packed": 0, "key_dense": 0, "released": 0, "writeback": 0}


class PackedStack:
    """One stacked layer, packed: codes uint8 [S, cap, D / 2], scales uint8 [S, cap, D / block]. `shape` / `dtype` are those of the dense buffer it replaces.
    `segs[u]` keeps the sealed kvseg segment list of session u (or None) so the commit-back can give it back: the rows it covers are the same, append-only."""
    __slots__ = ("codes", "scales", "dtype", "block", "e4m3", "segs")

    def __init__(self, codes, scales, dtype, block: int, e4m3: bool, segs=None):
        self.codes, self.scales, self.dtype, self.block, self.e4m3 = codes, scales, dtype, block, e4m3
        self.segs = segs

    @property
    def shape(self):
        return (self.codes.shape[0], self.codes.shape[1], self.codes.shape[2] * 2)

    def arrays(self) -> list:
        return [self.codes, self.scales]

    raw_arrays = arrays                                        # serve.backends.arrays_of counts the stored bytes

    def dense(self):
        """The whole dense buffer [S, cap, D] (what the dense stack held): a transient for the readers that have no packed form."""
        return fp4pack.unpack(self.codes, self.scales, self.block, self.e4m3, self.dtype)


def is_packed(x) -> bool:
    return isinstance(x, PackedStack)


def arrays_of(x) -> list:
    """The arrays a stacked layer is made of (for `mx.depends` and the byte ledger): the codes and scales of a packed stack or of a `fp4pool.PackedView`."""
    if isinstance(x, PackedStack):
        return x.arrays()
    if hasattr(x, "codes") and hasattr(x, "scales"):           # cdecpack: a single-session view keeps its rows in the same two arrays
        return [x.codes, x.scales]
    return [] if x is None else [x]


def dense_of(x):
    """`x` as a dense array for a reader without a packed form."""
    if isinstance(x, PackedStack):
        HITS["key_dense"] += 1
        return x.dense()
    return x


def stack(slot_lists: list, l: int, step: int, S: int | None = None, slots: list | None = None):
    """Layer `l` of the sessions' packed slots (one PoolSlots / KeySlots per session) -> (PackedStack, cnt) with the cap `_stack_ragged` would pick, or None when
    the layer cannot be stacked packed (a session holds it dense, off the grid, or the slots have no packed form): the caller then stacks it dense as before.
    `S` / `slots` (flag mu_keep): a session axis of `S` slots with session u in `slots[u]`, so a later join writes a free slot. `cnt` is per slot then.
    Reads `.raw(l)` only: nothing is unpacked and no mirror is built."""
    raws = []
    for sl in slot_lists:
        if not hasattr(sl, "raw"):
            return None
        raws.append(sl.raw(l))
    from .fp4pool import _Pk
    have = [e for e in raws if e is not None]
    if not have or any(not isinstance(e, _Pk) or e.codes.shape[0] != 1 for e in have):
        return None
    e0 = have[0]
    cls = type(slot_lists[0])
    if any(type(sl) is not cls for sl in slot_lists) or any(
            e.dtype != e0.dtype or e.codes.shape[2] != e0.codes.shape[2] or e.scales.shape[2] != e0.scales.shape[2] for e in have):
        return None
    cnt = [0 if e is None else e.n for e in raws]
    if max(cnt) == 0:
        return None
    cap = -(-(max(cnt) + 1) // step) * step
    N = len(raws) if slots is None else S
    codes = mx.zeros((N, cap, e0.codes.shape[2]), dtype=mx.uint8)
    scales = mx.zeros((N, cap, e0.scales.shape[2]), dtype=mx.uint8)
    for u, e in enumerate(raws):
        if cnt[u]:
            s = u if slots is None else slots[u]
            codes[s:s + 1, :cnt[u]] = e.codes[:, :cnt[u]]
            scales[s:s + 1, :cnt[u]] = e.scales[:, :cnt[u]]
    HITS["stack"] += 1
    segs = [(e.segs, e.sealed, e.sr, e.sb) if isinstance(e, _Pk) and e.segs else None for e in raws]      # kvseg: give the sealed list back at the commit
    if slots is not None:                                                    # mu_keep: per slot, 0 for a slot no session owns
        per = [0] * max(N, max(slots) + 1)
        for u, s in enumerate(slots):
            per[s] = cnt[u]
        cnt = per
    return PackedStack(codes, scales, e0.dtype, cls.BLOCK, cls.E4M3, segs), cnt


def release(slot_lists: list, l: int) -> None:
    """Drop the sessions' own entries of layer l (packed codes and any mirror): the batch holds the only copy until `sync_rows`."""
    for sl in slot_lists:
        if sl.raw(l) is not None:
            sl.set_raw(l, None)
            HITS["released"] += 1


def put(buf, S: int, D: int, dtype, em, vals, step: int, block: int, e4m3: bool, capacity, canon: bool):
    """The packed form of dsv4_core._put: buf is None or a PackedStack; vals [n_emit, D] on the fp4 grid; returns the PackedStack that holds them at
    (em.sess_arr, em.widx_arr). Same growth rule (`capacity`), same canonical gather copy."""
    need = em.need
    if buf is not None and canon:
        codes, scales = buf.codes[mx.arange(S)], buf.scales[mx.arange(S)]
    elif buf is not None:
        codes, scales = buf.codes, buf.scales
    if buf is None:
        cap = -(-need // step) * step
        codes = mx.zeros((S, cap, D // 2), dtype=mx.uint8)
        scales = mx.zeros((S, cap, D // block), dtype=mx.uint8)
    elif need > codes.shape[1]:
        cap = capacity(need, step)
        nc = mx.zeros((S, cap, codes.shape[2]), dtype=mx.uint8)
        ns = mx.zeros((S, cap, scales.shape[2]), dtype=mx.uint8)
        nc[:, :codes.shape[1]] = codes
        ns[:, :scales.shape[1]] = scales
        codes, scales = nc, ns
    pc, ps = fp4pack.pack(vals, block, e4m3)
    codes[em.sess_arr, em.widx_arr] = pc
    scales[em.sess_arr, em.widx_arr] = ps
    HITS["put"] += 1
    return PackedStack(codes, scales, dtype, block, e4m3, None if buf is None else buf.segs)


def session_entry(ps: PackedStack, u: int, n: int):
    """The packed layer of session u (first n rows) as a `fp4pool._Pk`, a copy of just its rows: the commit-back, no unpack and no repack. A kvseg sealed list the
    session held goes back with it when its sealed rows are still covered by n (append-only: the rows below `sealed` never changed)."""
    from .fp4pool import _Pk
    HITS["writeback"] += 1
    e = _Pk(mx.contiguous(ps.codes[u:u + 1, :n]), mx.contiguous(ps.scales[u:u + 1, :n]), n, ps.dtype)
    if ps.segs is not None and ps.segs[u] is not None and ps.segs[u][1] <= n:
        e.segs, e.sealed, e.sr, e.sb = ps.segs[u]
    return e


def gather_slab(ps: PackedStack, idx, sess, rows_in_sess, S: int, maxc: int):
    """The rows the top-k names, dense, from the codes only. idx int32 [n, K] (-1 = none), sess int32 [n] (session of each row), rows_in_sess int32 [n]
    (position of the row among its session's rows of the pass), maxc = most rows of one session. Returns (pool [S, maxc * K, D], idx' [n, K]): session s's slab
    holds the K rows of each of its rows in turn and idx' addresses them (-1 where idx was outside [0, cap), the kernels' own bound). With one row per session
    the slab is the gather itself."""
    n, K = idx.shape
    cap = ps.codes.shape[1]
    flat = idx.astype(mx.int32)
    valid = (flat >= 0) & (flat < cap)
    safe = mx.where(valid, flat, 0)
    g = fp4pack.unpack(ps.codes[sess[:, None], safe], ps.scales[sess[:, None], safe], ps.block, ps.e4m3, ps.dtype)         # [n, K, D]
    cols = mx.arange(K, dtype=mx.int32)[None, :]
    if maxc == 1 and S == n:                                   # one row per session, identity slots: the slab is the gather itself
        pool = g
        slot = cols
    else:
        pool = mx.zeros((S * maxc, K, g.shape[-1]), dtype=g.dtype)
        pool[sess * maxc + rows_in_sess] = g
        pool = pool.reshape(S, maxc * K, g.shape[-1])
        slot = rows_in_sess[:, None] * K + cols
    HITS["gather"] += 1
    HITS["gather_rows"] += n * K
    return pool, mx.where(valid, slot, -1).astype(mx.int32)
