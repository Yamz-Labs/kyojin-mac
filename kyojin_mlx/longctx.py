# SPDX-License-Identifier: Apache-2.0
"""Long-context levers for DSA-style indexer models (DS4.1; generic over the indexer shapes): remove the terms of a decode step / prefill chunk
that grow with the past and are not the model's own arithmetic. Every lever here is exact (bit-identical outputs).

    rope    `_index_cos_sin` / `_cached_index_cos_sin` of the vendor model rebuild the whole float64 numpy table every 256 positions (lru of 64
            tables): at 1M past that is ~0.3-0.5 s of host work every 256 decode steps and up to 64 tables (16 GB) alive. Rows do not depend
            on the table length (row i = cos(float32(i) * inv_freq) in float64, rounded to float32), so ONE table per (rope_dim, theta, yarn)
            grown to the next power of two (>= 4096) and sliced gives the same values: O(log n) rebuilds over a whole context.
    rope_pf the rebuild that remains (one per doubling) still lands inside a decode step; ROPE_PF builds the next power-of-two table on a
            background thread once the current one is 3/4 used (numpy releases the GIL inside the trig), so the step that crosses the edge finds
            it ready and joins a finished thread. One extra table in memory (2x the current, 512 MB at 1M with rope_dim 64).
    grow    pool / key buffers grow by a 1/8 slack (rounded to the step) instead of one step (256 rows): a full copy of every layer's pool every
            256 rows becomes a copy every n/8 rows (amortized O(1) per token, no multi-ms spikes at long context). Capacity is never read as a
            length (the published arrays are prefix slices; the runtime-width selection masks t >= lens).
    room    (not a switch: `room(need, step)` is the capacity a RESTORED buffer takes, for flag pf_grow) 1/16 of the rows plus one step, so a pool
            restored from a prefix snapshot or a disk checkpoint takes its next appends in place instead of growing on its first append.
    pftopk  prefill indexer (any layer that is not the candidate source): the stock path builds the [T, Tc] float32 scores of the whole chunk
            against the whole past (16 GB at T = 8192, Tc = 512K) then a full argsort over Tc per row. Here keys are scored in tiles of
            KYOJIN_LONGCTX_PF_TILE columns (same matmul + post kernel per column, see ds_pf_idx.scores_mm), each tile keeps its exact top-k
            (idxfast.topk_mt) and a stable merge keeps the running top-k: the selected set is the same (value desc, lower index first on ties),
            memory is O(T * tile), and no sort over Tc is done.

Switches (rope default on, grow / pftopk / rope_pf default off): KYOJIN_LONGCTX=1 turns the three long-context levers on; KYOJIN_LONGCTX=0 turns them off;
KYOJIN_LONGCTX_ROPE / _GROW / _PFTOPK / _PF = 0|1 override one. The decode top-k lever lives in idxfast (KYOJIN_IDX_FAST=1, engaged from KYOJIN_IDX_FAST_MIN
keys) and the register-blocked decode score kernel in dsv4_core (flags.py iscore_kb, from iscore_kb_min keys) - both have their own flag now.
HITS counts the calls that took each path (the gate prints it, so a flag that never reached the served path is visible)."""
from __future__ import annotations

from . import archs as _A
import os
import threading

import mlx.core as mx


def _flag(name: str, default: str = "0") -> bool:
    v = os.environ.get(f"KYOJIN_LONGCTX_{name}")
    if v is None:
        v = os.environ.get("KYOJIN_LONGCTX", default)
    return v not in ("0", "")


ROPE = _flag("ROPE", "1")                                                   # flags.py longctx_rope
ROPE_PF = _flag("ROPE_PF")                                                  # flags.py longctx_rope_pf
GROW = _flag("GROW")
PFTOPK = _flag("PFTOPK")
PF_TILE = int(os.environ.get("KYOJIN_LONGCTX_PF_TILE", "16384"))            # key columns per scored tile (a multiple of 256)
PF_MIN = int(os.environ.get("KYOJIN_LONGCTX_PF_MIN", str(2 * PF_TILE)))     # past keys from which the tiled path is taken (>= 2 tiles)
PF_EVAL_ELEMS = 1 << 28                                                     # T * Tc above which each tile is evaluated before the next (bounded peak)
GROW_SHIFT = 3                                                              # slack = need >> GROW_SHIFT
HITS = {"rope": 0, "grow": 0, "pftopk": 0, "dec_fast": 0, "dec_fast_solo": 0, "dec_fuse": 0, "dec_score_fast": 0, "iscore_kb": 0, "rope_pf": 0, "dec_score_mma": 0}


# ----------------------------------------------------------------------------------------------------------------- grow
def capacity(need: int, step: int) -> int:
    """Buffer capacity for `need` rows: one step multiple (stock), plus the 1/8 slack when GROW is on (growth only)."""
    from . import depthflat as DF
    if DF.ENABLED:                                       # KYOJIN_DEPTHFLAT carries the same slack
        return DF.capacity(need, step)
    if GROW:
        HITS["grow"] += 1
        need = need + (need >> GROW_SHIFT)
    return -(-need // step) * step


RESTORE_SHIFT = 4                                        # pf_grow: room of a restored buffer = need >> 4 (6.25 % of the rows), and never less than one step


def room(need: int, step: int) -> int:
    """Capacity a restored (not grown) pool / key buffer takes: `need` rows, one step of room at least plus `need >> RESTORE_SHIFT`, rounded up to the step.
    A layer restored from a prefix snapshot or a disk checkpoint is a buffer of exactly its rows (or a view of a stored one), so its first append lands past the
    capacity and `fp4pool.append_rows` allocates a zero buffer of the whole next capacity step and copies every row. With the room the append writes its rows
    where they go; the buffer grows by the `capacity` rule again once the room is used up. Bytes at rest: +6.25 % at 983040 rows (docs/reports/pfgrow_REPORT.md)."""
    return -(-(need + max(need >> RESTORE_SHIFT, step)) // step) * step


# ----------------------------------------------------------------------------------------------------------------- rope
_tables: dict = {}
_orig: dict = {}
_pf: dict = {}                                                              # key -> (cap, thread, box) while a bigger table is built ahead of use


def _build(length: int, rope_dim: int, theta: float, yarn: tuple):
    """The vendor table (row i of a table of any length): the float64 numpy work, through the unwrapped build (no lru retention)."""
    build = getattr(_orig["cached"], "__wrapped__", _orig["cached"])
    return build(length, rope_dim, theta, yarn)


def _ahead(key, cap: int, length: int, rope_dim: int, theta: float, yarn: tuple) -> None:
    """ROPE_PF: from 3/4 of a table's capacity, build the next power of two on a background thread (numpy's trig releases the GIL): the decode position
    that outgrows the table finds the rows ready instead of paying the whole build. One build at a time per key."""
    if not ROPE_PF or key in _pf:
        return
    nxt = cap << 1
    box: list = []
    thread = threading.Thread(target=lambda: box.append(_build(nxt, rope_dim, theta, yarn)), name="kyojin-rope-pf", daemon=True)
    _pf[key] = (nxt, thread, box)
    thread.start()


def _table(length: int, rope_dim: int, theta: float, yarn: tuple):
    key = (rope_dim, theta, yarn)
    t = _tables.get(key)
    if t is not None and t[0] >= length:
        if length * 4 >= t[0] * 3:
            _ahead(key, t[0], length, rope_dim, theta, yarn)
        return t
    p = _pf.pop(key, None)
    if p is not None:                                            # a build ahead of use is running for this key: take it when it covers the request
        cap, thread, box = p
        thread.join()
        if cap >= length and box:
            HITS["rope_pf"] += 1                                         # the step joined a table built ahead of use
            t = _tables[key] = (cap,) + tuple(box[0])
            if length * 4 >= cap * 3:
                _ahead(key, cap, length, rope_dim, theta, yarn)
            return t
    cap = max(4096, 1 << max(0, length - 1).bit_length())
    t = _tables[key] = (cap,) + tuple(_build(cap, rope_dim, theta, yarn))
    if length * 4 >= cap * 3:
        _ahead(key, cap, length, rope_dim, theta, yarn)
    return t


def index_cos_sin(length: int, rope_dim: int, theta: float, yarn: tuple):
    """Same rows as the vendor `_index_cos_sin(length, ...)` (and `_cached_index_cos_sin`): [length, rope_dim // 2] float32 cos, sin."""
    HITS["rope"] += 1
    _, cos, sin = _table(length, rope_dim, theta, yarn)
    return cos[:length], sin[:length]


def install_rope(L) -> int:
    if not ROPE or getattr(L, "_kyojin_longctx_rope", False):
        return 0
    _orig["cached"] = L._cached_index_cos_sin
    _orig["index"] = L._index_cos_sin
    L._index_cos_sin = index_cos_sin                     # module globals: the vendor code and every kyojin caller look it up at call time
    L._cached_index_cos_sin = index_cos_sin
    L._kyojin_longctx_rope = True
    return 1


def _append(self, arrays, buffers, layer_idx, values):
    """DeepseekV41Cache._append (vendor) with the capacity from `capacity` (same published prefix, same values)."""
    previous = arrays[layer_idx]
    start = 0 if previous is None else previous.shape[1]
    end = start + values.shape[1]
    buffer = buffers[layer_idx]
    if buffer is None:
        buffer = previous
    if buffer is None or end > buffer.shape[1]:
        cap = capacity(end, self.step) if buffer is not None else ((end + self.step - 1) // self.step) * self.step
        shape = (values.shape[0], cap, *values.shape[2:])
        dtype = values.dtype if buffer is None else buffer.dtype
        buffer = mx.zeros(shape, dtype=dtype)
        if previous is not None:
            buffer[:, :start] = previous
    buffer[:, start:end] = values
    buffers[layer_idx] = buffer
    arrays[layer_idx] = buffer[:, :end]
    return arrays[layer_idx]


def install_grow(L) -> int:
    cls = getattr(L, _A.vendor().cache, None)
    if not GROW or cls is None or getattr(cls, "_kyojin_longctx_grow", False):
        return 0
    cls._append = _append
    cls._kyojin_longctx_grow = True
    return 1


# ----------------------------------------------------------------------------------------------------------------- prefill top-k
def pf_topk(q, keys, w, start: int, ratio: int, k: int, cand=None, tile: int | None = None):
    """Exact top-k columns of the prefill indexer scores without the [T, Tc] matrix.
    q [1, T, NH, 128] half (rotated), keys [1, Tc, 128], w [1, T, NH], cand [1, T, Tc] bool or None (uses_candidates layers).
    Returns int32 [T, k] ascending: the set `mx.sort(mx.argsort(-S)[:, :k])` of the stock chain, S = scores_mm(...) masked by cand."""
    from . import ds_pf_idx as PI
    from . import idxfast as IX
    T, Tc = q.shape[1], keys.shape[1]
    tile = tile or PF_TILE
    assert Tc > k and tile >= k
    HITS["pftopk"] += 1
    run_i = run_v = None
    lazy_ok = T * Tc <= PF_EVAL_ELEMS
    for a in range(0, Tc, tile):
        b = min(Tc, a + tile)
        s = PI.scores_mm(q, keys[:, a:b], w, start, ratio, n0=a)                       # [T, b - a] float32, columns a .. b - 1
        if cand is not None:
            s = mx.where(cand[0, :, a:b], s, -mx.inf)
        kk = min(k, b - a)
        ti = IX.topk_mt(s, mx.full((T,), b - a, dtype=mx.int32), kk)                    # ascending, no -1 tail (kk <= width)
        tv = mx.take_along_axis(s, ti, axis=-1)
        ti = ti + a
        if run_i is None:
            run_i, run_v = ti, tv
        else:
            ci = mx.concatenate([run_i, ti], axis=-1)                                     # index-ascending: ties keep the lower index (stable sort)
            cv = mx.concatenate([run_v, tv], axis=-1)
            o = mx.argsort(-cv, axis=-1)[:, :k]
            si, sv = mx.take_along_axis(ci, o, axis=-1), mx.take_along_axis(cv, o, axis=-1)
            p = mx.argsort(si, axis=-1)
            run_i, run_v = mx.take_along_axis(si, p, axis=-1), mx.take_along_axis(sv, p, axis=-1)
        if not lazy_ok:
            mx.eval(run_i, run_v)
    return run_i.astype(mx.int32)


def install(model=None) -> dict:
    """Install the vendor-side patches (rope table, cache growth); the kyojin-side switches are read where they are used."""
    import importlib
    try:
        L = _A.lang()
    except Exception:
        return {"rope": 0, "grow": 0}
    return {"rope": install_rope(L), "grow": install_grow(L)}
