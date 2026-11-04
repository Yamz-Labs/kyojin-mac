# SPDX-License-Identifier: Apache-2.0
"""Live replay of the verify forward of a one-session pass (rows.verify, N = 1) as a parametrised ICB region (replay/rp.mm, rp_region_fit).

The command stream of the forward is fixed for a given (rows, state shapes, pooled-count bucket); only the position p of its first row moves some constants, grids and tiny host buffers,
and those are periodic-affine in p. The first calls of a key run PROBES: the forward at positions p .. p + M - 1 (state untouched: rows past the committed counts are never read), recorded;
rp_region_fit turns them into one table, rp_region_build into one ICB. Later passes of the key are one trigger op: graph build + encode become a few microseconds. Inputs of the region
are the stacked state arrays (rebound at every call) and the position; outputs are the logits, the pending state and the taps.

KYOJIN_VERIFY_REPLAY=1 (needs KYOJIN_REPLAY=1 for the Metal hooks). Anything unexpected (fit fails, shapes move, shadow check differs) marks the key dead: every call goes through the eager forward."""
from __future__ import annotations

import ctypes
import os
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

ENABLED = os.environ.get("KYOJIN_VERIFY_REPLAY", "0") not in ("0", "") and os.environ.get("KYOJIN_REPLAY", "0") not in ("0", "")
SAMPLES = int(os.environ.get("KYOJIN_VERIFY_REPLAY_SAMPLES", "20"))
MAXL = int(os.environ.get("KYOJIN_VERIFY_REPLAY_MAXL", "10"))
AUDIT = int(os.environ.get("KYOJIN_VERIFY_REPLAY_AUDIT", "64"))       # one replay in AUDIT also runs the eager forward and compares bitwise (0 = never)
VERBOSE = os.environ.get("KYOJIN_VERIFY_REPLAY_VERBOSE", "0") not in ("0", "")
WARM = int(os.environ.get("KYOJIN_VERIFY_REPLAY_WARM", "3"))
SWALLOW = os.environ.get("KYOJIN_VERIFY_REPLAY_SWALLOW", "1") not in ("0", "")
RETRIES = int(os.environ.get("KYOJIN_VERIFY_REPLAY_RETRIES", "3"))
import atexit
STATS = dict(eager=0, probe=0, replay=0, audit=0, mismatch=0, dead=0)
_SLOT0 = 40
_entries: dict = {}
_LAST = ["start"]
_lib = None


def _addr(x: mx.array) -> int:
    return ctypes.addressof(ctypes.c_char.from_buffer(memoryview(x)))


def _L():
    global _lib
    if _lib is None:
        from . import replay
        lib = replay._load()
        C = ctypes
        lib.rp_region_fit.argtypes = [C.c_int, C.c_int, C.c_void_p, C.c_void_p, C.c_int, C.c_int]
        lib.rp_region_fit.restype = C.c_int
        _lib = lib
    return _lib


class _Entry:
    def __init__(self):
        self.state, self.calls, self.replays = "new", 0, 0


def _state(st) -> list:
    from . import stackpack as SP
    return [a for lst in (st.win, st.cmp, st.keys, st.kvs, st.scs) for v in lst if v is not None for a in SP.arrays_of(v)]


_RATIOS: dict = {}
_WIN: list = []


def _wblock(lm, plan, last: int | None = None) -> tuple:
    """Window ring block of the rows: the slot of a position is pos % W, so a table fitted inside one block of W positions does not hold across a multiple of W."""
    _ratios(lm)
    W = _WIN[0] if _WIN else 0
    if not W:
        return (0, 0)
    p_last = plan.pos[-1] if last is None else last
    return (plan.pos[0] // W, p_last // W)


def _ratios(lm) -> dict:
    if not _RATIOS:
        for m in lm.modules():
            if hasattr(m, "layer_idx") and getattr(m, "compress_ratio", 0):
                _RATIOS[m.layer_idx] = m.compress_ratio
            if hasattr(m, "layer_idx") and getattr(m, "window_size", 0) and not _WIN:
                _WIN.append(int(m.window_size))
        _RATIOS.setdefault(-1, 0)
    return _RATIOS


def _bucket(lm, plan, last: int | None = None) -> tuple:
    """Pooled-count bucket per compressed layer after the pass (its last row at position `last`): the sort / select kernels change variant when the count crosses a power of two."""
    p_last = plan.pos[-1] if last is None else last
    from . import dsv4_core as DC
    tb = DC.TMAX_BUCKET
    return tuple((((p_last + 1) // r).bit_length(), -(-((p_last + 1) // r) // tb), l) for l, r in sorted(_ratios(lm).items()) if r > 1)    # sort variant class + top-k width bucket (a regime boundary of the bucketed width)


def _stride(lm, rows: int = 0) -> int:
    """Phase period of the stream structure. The number of emitting rows of a pass depends on pos mod ratio only when the pass has fewer rows than a ratio (1-row pass, ratio 2: every other
    position emits); then probes and keys step by the lcm of the small compress ratios, otherwise (4 rows, ratios 2 / 4: always the same emitters) the stream is phase-free: stride 1."""
    import math
    S = 1
    for r in _ratios(lm).values():
        if 1 < r <= 8:
            S = S * r // math.gcd(S, r)
    if S == 1 or rows <= 0:
        return S
    sigs = {_emit_sig(lm, [q + i for i in range(rows)]) for q in range(S)}
    return 1 if len(sigs) == 1 else S


def _emit_sig(lm, positions) -> tuple:
    """Emitting rows of a pass per compress ratio: the command stream has one structure per signature (ratio-128 layers emit once per 128 positions)."""
    return tuple(sum(1 for q in positions if (q + 1) % r == 0) for r in sorted({r for r in _ratios(lm).values() if r > 1}))


def _key(lm, st, plan):
    return (tuple(plan.counts), tuple((tuple(a.shape), tuple(memoryview(a).strides)) for a in _state(st)), _bucket(lm, plan), _wblock(lm, plan), plan.pos[0] % _stride(lm, len(plan.pos)), _emit_sig(lm, plan.pos))


def _pend_arrays(st) -> list:
    """(layer, field, array) of the pending state, in a fixed order."""
    out = []
    for l in sorted(st.pend):
        for name in ("win", "kvs", "scs", "cmp", "keys"):
            a = getattr(st.pend[l], name, None)
            if isinstance(a, mx.array):
                out.append((l, name, a))
    return out


def _discard(st) -> None:
    st.pend = {}
    st.taps = {}


# ---------------------------------------------------------------------------------------------------------------- Engram
# The Engram layer reads its rows from a disk table: ParallelRowEmbedding does np.asarray(indices) (a host sync) and builds arrays whose sizes follow the number of distinct rows, which no
# recorded stream can follow. A replayed pass uses the staged form instead: the rows of every Engram layer are pread into STABLE slot buffers (ring of two per row count, the slots of
# engram_mlx.StagedRowEmbedding) before the trigger, the stream reads the slots as region inputs and dequantizes them on the device (same values as the parallel embed: a row
# is dequantized row by row, deduplication only saved reads). The engram hashes of the pass are a region input too.
_ENG = {"staged": False, "slots": [], "embeds": None}


def _wrap_engram(lm):
    """Wrap engram.embed of every Engram layer (once). Returns the wrappers in layer order, or None when this model's Engram is not a parallel / staged disk table."""
    if _ENG["embeds"] is not None:
        return _ENG["embeds"] or None
    import mlx.nn as nn
    from . import engram_mlx as EM
    embeds = []
    for l in getattr(lm, "layers", []):
        eng = getattr(l, "engram", None)
        if eng is None:
            continue
        emb = eng.embed
        if isinstance(emb, _ReplayEmbed):
            embeds.append(emb)
        elif hasattr(emb, "_tables") and hasattr(emb, "_pos"):
            w = _ReplayEmbed(emb)
            eng.embed = w
            embeds.append(w)
        else:
            _ENG["embeds"] = []
            return None
    _ENG["embeds"] = embeds
    return embeds or None


def _make_replay_embed():
    import mlx.nn as nn
    from . import engram_mlx as EM

    class ReplayEmbed(nn.Module):
        RING = 2

        def __init__(self, orig):
            super().__init__()
            object.__setattr__(self, "_orig", orig)
            object.__setattr__(self, "_tables", orig._tables)
            object.__setattr__(self, "_pos", orig._pos)
            object.__setattr__(self, "_slots", {})
            object.__setattr__(self, "_next", {})

        @property
        def reads(self):
            return self._orig.reads

        def slot(self, n: int):
            ring = self._slots.get(n)
            if ring is None:
                ring = self._slots[n] = [EM._Slot(n, self._tables.spec.head_dim) for _ in range(self.RING)]
                self._next[n] = 0
            i = self._next[n]
            self._next[n] = (i + 1) % self.RING
            return ring[i]

        def __call__(self, indices):
            if not _ENG["staged"]:
                return self._orig(indices)
            n = 1
            for d in indices.shape:
                n *= d
            sl = self.slot(n)
            _ENG["slots"].append((self, sl))
            rows = EM._dequant(sl.w, sl.s)
            return rows.reshape(*indices.shape, rows.shape[-1])
    return ReplayEmbed


_ReplayEmbed = _make_replay_embed()


def _stage_rows(embeds, eh) -> list:
    """Fill a slot per Engram layer from the hashes `eh` [rows, 1, layers, cols] and return the slot arrays in the order the stream reads them."""
    host = np.asarray(eh)
    out = []
    reqs = []
    for emb in embeds:
        idx = host[:, :, emb._pos, :].reshape(-1)
        sl = emb.slot(idx.size)
        reqs.append((emb._pos, idx, sl.w_np, sl.s_np))
        out += [sl.w, sl.s]
    embeds[0]._tables.read_rows_into(reqs)
    return out


def forward(lm, st, plan, ids: mx.array, kw: dict, eager):
    """Logits [rows, V] fp32 of the pass; st.pend / st.taps are filled as the eager forward would. `eager()` runs the normal forward."""
    if not ENABLED or plan.S != 1 or st._counts is None:
        return eager()
    if kw and (set(kw) != {"engram_hashes"} or _wrap_engram(lm) is None):
        return eager()
    key = _key(lm, st, plan)
    e = _entries.get(key)
    if e is None:
        e = _entries[key] = _Entry()
    e.calls += 1
    if e.state == "dead":
        STATS["eager"] += 1
        _LAST[0] = "eager"
        return eager()
    if e.state == "new":
        if e.calls < 3:                                              # warm: kernels built, caches settled
            STATS["eager"] += 1
            _LAST[0] = "eager"
            return eager()
        try:
            _fit(lm, st, plan, ids, e, eager, kw)
        except Exception as ex:                                      # noqa: BLE001 - any failure leaves the eager path
            e.fails = getattr(e, "fails", 0) + 1
            if e.fails < RETRIES:                                    # the stream may still be settling (lazily built fused weights, first-use caches): try again later
                e.state, e.calls = "new", -8
            else:
                e.state = "dead"; STATS["dead"] += 1
            if VERBOSE:
                print(f"verify_replay: fit failed ({ex!r}): eager for key", key, flush=True)
            _discard(st)
            return eager()
        if e.state != "ready":
            _discard(st)
            return eager()
    r_ = _replay(lm, st, plan, ids, e, eager, kw)
    _LAST[0] = f"replay{e.h}"
    return r_


def _fit(lm, st, plan, ids, e, eager, kw=None) -> None:
    lib = _L()
    C = ctypes
    M = SAMPLES
    row = st.rows[0]
    p0 = row.offset
    S = _stride(lm, len(ids))
    last = p0 + len(ids) - 1 + S * (M - 1)
    sig0 = _emit_sig(lm, plan.pos)
    if any(_emit_sig(lm, [p0 + S * k + i for i in range(len(ids))]) != sig0 for k in range(M)):     # a ratio-128 layer would emit inside the probe window: other structure, wait
        e.calls = 0
        return
    if _bucket(lm, plan, last) != _bucket(lm, plan) or _wblock(lm, plan, last)[1] != _wblock(lm, plan)[1]:         # the probes would cross a count bucket or a window block: wait for the next key
        e.calls = 0
        return
    shapes0 = [(tuple(a.shape), tuple(memoryview(a).strides)) for a in _state(st)]
    ins_state = list(_state(st))
    mx.eval(ids, *ins_state)
    outs0 = None
    try:                                                                    # warm pass at p0: lru tables (rope cos / sin of this capacity) and per-plan constants exist before the first probe,
        for _w in range(WARM):
            st._plan = None; mx.eval(eager()); mx.synchronize(); _discard(st)    # so the recorded stream is the steady-state one (a cold pass also records the table build)
    finally:
        row.offset = p0
    ins_state = list(_state(st))                                            # the inputs of probe 0: read AFTER the warm pass (it adopted new pool buffers into the state)
    shapes0 = [(tuple(a.shape), tuple(memoryview(a).strides)) for a in ins_state]
    mx.eval(ids, *ins_state)
    eh = kw["engram_hashes"] if kw else None
    if eh is not None:
        mx.eval(eh)
    saved = {n_: [list(x_) if isinstance(x_, list) else x_ for x_ in getattr(st, n_)] for n_ in ("win", "cmp", "keys", "kvs", "scs", "cnt", "kcnt")}
    old = mx.set_cache_limit(0)
    if SWALLOW:
        lib.rp_set_swallow.argtypes = [ctypes.c_int]
    try:
        for k in range(M):
            mx.synchronize(); mx.clear_cache()
            row.offset = p0 + S * k
            st._plan = None                                                    # every probe builds its own Plan (per-plan caches of the warm pass must not shorten probe 0)
            if os.environ.get('VR_HANDDBG'): st.hand.dbg_o = []
            lib.rp_rec_begin(_SLOT0 + k); lib.rp_rec_snap_born(_SLOT0 + k, 1)
            if SWALLOW: lib.rp_set_swallow(1)
            if kw:
                _ENG["staged"] = True; _ENG["slots"] = []
            out = eager()
            pend = _pend_arrays(st)
            taps = [(l, st.taps[l]) for l in sorted(st.taps)]
            mx.eval(out, *[a for _, _, a in pend], *[t for _, t in taps])
            mx.synchronize()
            lib.rp_rec_end()
            if SWALLOW: lib.rp_set_swallow(0)
            if os.environ.get('VR_PROBEDUMP'): lib.rp_dump(_SLOT0 + k, f"/tmp/vrp_{k}.txt".encode())
            STATS["probe"] += 1
            if k == 0:
                outs0 = (out, pend, taps)
                slots0 = list(_ENG["slots"]) if kw else []
                hand_dbg = getattr(st.hand, 'topk_idxs', None) if os.environ.get('VR_HANDDBG') else None
                if hand_dbg is not None:
                    mx.eval(hand_dbg)
                    dbg_o = list(st.hand.dbg_o); mx.eval(*dbg_o)
                meta = {l: getattr(pd, "r", 0) for l, pd in st.pend.items()}
            else:
                _discard(st)
            if [(tuple(a.shape), tuple(memoryview(a).strides)) for a in _state(st)] != shapes0:
                raise RuntimeError("state shapes moved during the probes (capacity growth)")
    finally:
        row.offset = p0
        _ENG["staged"] = False
        mx.set_cache_limit(old)
        if SWALLOW:
            lib.rp_set_swallow(0)
            for n_, v_ in saved.items():                                      # nothing ran: the pools the probes adopted are uninitialised, put the committed state back
                setattr(st, n_, v_)
            st.hand.compress_kv = st.hand.index_k = st.hand.candidates = st.hand.topk_idxs = None
    out0, pend0, taps0 = outs0
    # outputs: logits, then every pending / tap array that is not already (part of) an input buffer
    spans = [(_addr(a), _addr(a) + max(a.nbytes, 1)) for a in ins_state]

    def alias_of(a):
        if os.environ.get('VR_NOALIAS'):
            return -1
        ad = _addr(a)
        for j, (s, t) in enumerate(spans):
            if s <= ad and ad + a.nbytes <= t and a.nbytes == t - s:
                return j
        return -1
    flat = [out0]
    e.dbg_k = None
    if os.environ.get('VR_HANDDBG') and hand_dbg is not None:
        e.dbg_k = len(flat); flat.append(hand_dbg)
        e.dbg_o0 = len(flat)
        for o_ in dbg_o: flat.append(o_)
        e.dbg_no = len(dbg_o)
    pend_plan, tap_plan = [], []
    for l, name, a in pend0:
        j = alias_of(a)
        pend_plan.append((l, name, j, None if j >= 0 else len(flat)))
        if j < 0:
            flat.append(a)
    for l, t in taps0:
        j = alias_of(t)
        tap_plan.append((l, j, None if j >= 0 else len(flat)))
        if j < 0:
            flat.append(t)
    sl = (C.c_int * M)(*range(_SLOT0, _SLOT0 + M)); ps = (C.c_int * M)(*range(p0 // S, p0 // S + M))
    h = lib.rp_region_new()
    if VERBOSE:
        lib.rp_dump(_SLOT0, f"/tmp/vr_probe0_{h}.txt".encode())
    L = lib.rp_region_fit(h, M, sl, ps, 0, MAXL)
    if L <= 0:
        raise RuntimeError("fit: no periodic-affine table")
    par = mx.array([p0 // S], dtype=mx.int32); mx.eval(par)
    slot_arrays = [a_ for _, sl_ in slots0 for a_ in (sl_.w, sl_.s)]
    ins = [par, ids] + ([eh] if kw else []) + ins_state + slot_arrays
    e.n_extra = len(slot_arrays)
    sp = sorted((_addr(x), _addr(x) + max(x.nbytes, 1), k) for k, x in enumerate(ins))
    ov = [(a, b) for a, b in zip(sp, sp[1:]) if b[0] < a[1]]
    if ov:
        raise RuntimeError(f"inputs share memory: {[(a[2], b[2]) for a, b in ov]}")
    ip = (C.c_size_t * len(ins))(*[_addr(x) for x in ins]); il = (C.c_size_t * len(ins))(*[x.nbytes for x in ins])
    op = (C.c_size_t * len(flat))(*[_addr(x) for x in flat]); ol = (C.c_size_t * len(flat))(*[x.nbytes for x in flat])
    rc = lib.rp_region_build(h, _SLOT0, len(ins), ip, il, len(flat), op, ol, 3)
    if rc != 0:
        raise RuntimeError(f"region build rc={rc}")
    e.h, e.p0, e.L, e.S = h, p0, L, S
    e.shapes = [tuple(x.shape) for x in flat]; e.dtypes = [x.dtype for x in flat]
    e.pend_plan, e.tap_plan, e.meta = pend_plan, tap_plan, meta
    e.trig = mx.fast.metal_kernel(name=f"kyojin_region_trigger_{h}", input_names=[f"i{k}" for k in range(len(ins))], output_names=[f"o{k}" for k in range(len(flat))],
                                  source="if (thread_position_in_grid.x == 0) { o0[0] = o0[0]; }")
    e.n_state = len(ins_state)
    e.state = "ready"
    if VERBOSE:
        print(f"verify_replay: key ready, p0 {p0}, period {L}, {len(flat)} outputs ({sum(1 for x in pend_plan if x[2] >= 0)} pend aliased), ICB {lib.rp_region_stat(h, 3) / 1e6:.0f} MB", flush=True)
    _discard(st)


def _replay(lm, st, plan, ids, e, eager, kw=None):
    p = plan.pos[0]
    if p < e.p0:                                                     # a lower position than the samples (new sequence): the table does not cover it
        STATS["eager"] += 1
        return eager()
    state = list(_state(st))
    if os.environ.get('VR_EVALIDS'):
        mx.eval(ids)
    if os.environ.get('VR_PRESYNC'):
        mx.eval(ids, *state); mx.synchronize()
    before = [np.array(a).copy() for a in state] if os.environ.get('VR_HASH') else None
    if kw and os.environ.get('VR_ENG_SYNC'):
        mx.synchronize()
    slots = _stage_rows(_ENG["embeds"], kw["engram_hashes"]) if kw else []
    ins = [mx.array([p // getattr(e, 'S', 1)], dtype=mx.int32), ids] + ([kw["engram_hashes"]] if kw else []) + state + slots
    e._last_slots = slots
    res = e.trig(inputs=ins, grid=(1, 1, 1), threadgroup=(1, 1, 1), output_shapes=e.shapes, output_dtypes=e.dtypes)
    STATS["replay"] += 1
    e.replays += 1
    if os.environ.get("VR_PEEK") and e.replays in (2, 5):
        mx.eval(*res); mx.synchronize()
        buf = (ctypes.c_ubyte * 65536)(); n_ = _L().rp_region_peek(e.h, 0, buf, 65536)
        arr = np.frombuffer(bytes(buf)[:max(n_, 0)], dtype=np.uint8)
        print("  PEEK scratch0 replay", e.replays, "bytes", n_, "0xFF bytes", int((arr == 255).sum()), "0x00 bytes", int((arr == 0).sum()), "first 48 nonff idx", np.nonzero(arr != 255)[0][:8].tolist(), "last nonff", int(np.nonzero(arr != 255)[0].max()) if (arr != 255).any() else -1, flush=True)
    if before is not None:
        mx.eval(*res); mx.synchronize()
        for j, (a, b) in enumerate(zip(state, before)):
            now = np.array(a)
            if not np.array_equal(now, b):
                nz = np.nonzero((now != b).reshape(now.shape[0], now.shape[1], -1).any(-1))
                print("  STATE INPUT CHANGED by the replay: input", j, "shape", tuple(a.shape), "rows(axis1) changed", sorted(set(nz[1].tolist()))[:12], "p", p, flush=True)
    if os.environ.get('VR_SYNC'):
        mx.eval(*res); mx.synchronize()
    st.pend = {}
    for l, name, j, k in e.pend_plan:
        pd = st.pend.get(l)
        if pd is None:
            r = e.meta.get(l, 0)
            pd = st.pend[l] = SimpleNamespace(win=None, kvs=None, scs=None, cmp=None, keys=None, r=r, cnt=None, kcnt=None)
        setattr(pd, name, mx.depends(state[j], [res[0]]) if j >= 0 else res[k])
    st.taps = {}
    for l, j, k in e.tap_plan:
        st.taps[l] = mx.depends(state[j], [res[0]]) if j >= 0 else res[k]
    if os.environ.get("VR_EMB"):
        mx.eval(*res); mx.synchronize()
        pw = [a for l_, n_, a in _pend_arrays(st) if l_ == 0 and n_ == "win"]
        if pw:
            win_in = np.array(state[0]).astype(np.float32)[0]; win_out = np.array(pw[0]).astype(np.float32)
            written = {q % 128 for q in plan.pos}
            other = [sl for sl in range(128) if sl not in written]
            bad_old = int(sum(not np.array_equal(win_out[r][other], win_in[other]) for r in range(win_out.shape[0])))
            print("  WIN base slots preserved in rows:", [bool(np.array_equal(win_out[r][other], win_in[other])) for r in range(win_out.shape[0])], "nan", int(np.isnan(win_out).sum()), flush=True)
        buf = (ctypes.c_ubyte * 65536)(); n_ = _L().rp_region_peek(e.h, 0, buf, 65536)
        emb = next(m_ for n_, m_ in lm.named_modules() if n_.endswith("embed_tokens"))
        got = np.frombuffer(bytes(buf)[:4 * len(ids) * 0 + len(ids) * 10240], dtype=np.float16).reshape(len(ids), -1)
        exp = np.array(emb(mx.array(np.array(ids)))).astype(np.float16).reshape(len(ids), -1)
        ok_ = [bool(np.array_equal(got[i], exp[i])) for i in range(len(ids))]
        print("  EMB rows equal:", ok_, "p", p, "ids", np.array(ids).tolist(), flush=True)
    out = res[0]
    e_dbg = res[e.dbg_k] if getattr(e, 'dbg_k', None) is not None else None
    if AUDIT and e.replays % AUDIT == (1 % AUDIT):
        STATS["audit"] += 1
        mine = np.array(out)
        if os.environ.get("VR_CMP"):
            mx.synchronize()
            lib_ = _L(); lib_.rp_region_cmp.argtypes = [ctypes.c_int, ctypes.c_int]
            t_save = dict(st.taps); pd_save = dict(st.pend)
            state_before = list(_state(st))                                    # the inputs of the replayed call (the diagnostic eager run below adopts new pools into st)
            _discard(st); mx.clear_cache()
            lib_.rp_rec_begin(95); lib_.rp_rec_snap_born(95, 1)
            o_ = eager(); pend_ = _pend_arrays(st); tp_ = [st.taps[l_] for l_ in sorted(st.taps)]
            mx.eval(o_, *[a_ for _, _, a_ in pend_], *tp_); mx.synchronize(); lib_.rp_rec_end()
            if not np.array_equal(mine, np.array(o_)):
                print("  CMP at DIFF: eager run recorded; building R2 from it and comparing private buffers with the key region", flush=True)
                lib_.rp_dump(95, f"/tmp/vr_eager95_{e.h}.txt".encode())
                C_ = ctypes
                h2 = lib_.rp_region_new()
                par2 = mx.array([p // getattr(e, 'S', 1)], dtype=mx.int32); mx.eval(par2)
                ins2 = [par2, ids] + state_before
                ip2 = (C_.c_size_t * len(ins2))(*[_addr(x) for x in ins2]); il2 = (C_.c_size_t * len(ins2))(*[x.nbytes for x in ins2])
                op2 = (C_.c_size_t * 1)(_addr(o_)); ol2 = (C_.c_size_t * 1)(o_.nbytes)
                rc2 = lib_.rp_region_build(h2, 95, len(ins2), ip2, il2, 1, op2, ol2, 1)
                print("  R2 build rc", rc2, flush=True)
                if rc2 == 0:
                    trig2 = mx.fast.metal_kernel(name=f"kyojin_region_trigger_{h2}", input_names=[f"i{k}" for k in range(len(ins2))], output_names=["o0"], source="if (thread_position_in_grid.x == 0) { o0[0] = o0[0]; }")
                    # key region again at the same state, then R2, both synchronous
                    rk = e.trig(inputs=[mx.array([p // getattr(e, 'S', 1)], dtype=mx.int32), ids] + list(_state(st)), grid=(1, 1, 1), threadgroup=(1, 1, 1), output_shapes=e.shapes, output_dtypes=e.dtypes); mx.eval(rk[0]); mx.synchronize()
                    r2 = trig2(inputs=ins2, grid=(1, 1, 1), threadgroup=(1, 1, 1), output_shapes=[o_.shape], output_dtypes=[mx.float32]); mx.eval(r2[0]); mx.synchronize()
                    print("  R2 (exact replay of the eager stream) equals eager:", np.array_equal(np.array(r2[0]), np.array(o_)), "| key region equals eager:", np.array_equal(np.array(rk[0]), np.array(o_)), flush=True)
                    lib_.rp_region_cmp2.argtypes = [C_.c_int, C_.c_int, C_.c_int]
                    lib_.rp_set_prefix.argtypes = [C_.c_int]; lib_.rp_set_poison.argtypes = [C_.c_int]
                    def run_both(n_):
                        lib_.rp_set_prefix(n_); lib_.rp_set_poison(2)
                        a_ = e.trig(inputs=[mx.array([p // getattr(e, 'S', 1)], dtype=mx.int32), ids] + list(_state(st)), grid=(1, 1, 1), threadgroup=(1, 1, 1), output_shapes=e.shapes, output_dtypes=e.dtypes); mx.eval(a_[0]); mx.synchronize()
                        b_ = trig2(inputs=ins2, grid=(1, 1, 1), threadgroup=(1, 1, 1), output_shapes=[o_.shape], output_dtypes=[mx.float32]); mx.eval(b_[0]); mx.synchronize()
                        return lib_.rp_region_cmp2(e.h, h2, 1)
                    def outs_under(poison):
                        lib_.rp_set_prefix(-1); lib_.rp_set_poison(poison)
                        a_ = e.trig(inputs=[mx.array([p // getattr(e, 'S', 1)], dtype=mx.int32), ids] + list(_state(st)), grid=(1, 1, 1), threadgroup=(1, 1, 1), output_shapes=e.shapes, output_dtypes=e.dtypes); mx.eval(a_[0]); mx.synchronize()
                        b_ = trig2(inputs=ins2, grid=(1, 1, 1), threadgroup=(1, 1, 1), output_shapes=[o_.shape], output_dtypes=[mx.float32]); mx.eval(b_[0]); mx.synchronize()
                        lib_.rp_set_poison(-1)
                        return np.array(a_[0]), np.array(b_[0])
                    eag = np.array(o_)
                    ka_, rb_ = outs_under(-1)
                    bufk = (ctypes.c_ubyte * 65536)(); bufr = (ctypes.c_ubyte * 65536)()
                    nk_ = lib_.rp_region_out_src(e.h, 0, bufk, 65536); nr_ = lib_.rp_region_out_src(h2, 0, bufr, 65536)
                    srck = np.frombuffer(bytes(bufk)[:nk_], dtype=np.float32).reshape(ka_.shape); srcr = np.frombuffer(bytes(bufr)[:nr_], dtype=np.float32).reshape(rb_.shape)
                    print("  OUTSRC key: copy==its source", np.array_equal(srck, ka_), "| R2: copy==its source", np.array_equal(srcr, rb_), "| key source==R2 source", np.array_equal(srck, srcr), "| key source==eager", np.array_equal(srck, eag), "| R2 source==eager", np.array_equal(srcr, eag), flush=True)
                    k1, _ = outs_under(0); k2, _ = outs_under(0); k3, _ = outs_under(-1); k4, _ = outs_under(-1)
                    print("  KEY determinism: poison0 run1==run2", np.array_equal(k1, k2), "| nopoison run1==run2", np.array_equal(k3, k4), "| key rows maxdiff vs eager", np.abs(k3 - eag).max(-1).tolist(), flush=True)
                    for pz in (-1, 0, 1, 2):
                        ka, rb = outs_under(pz)
                        print(f"  OUT poison={pz}: key==eager {np.array_equal(ka, eag)} | R2==eager {np.array_equal(rb, eag)} | key==R2 {np.array_equal(ka, rb)}", flush=True)
                    lib_.rp_region_tablecmp.argtypes = [C_.c_int, C_.c_int, C_.c_int]
                    lib_.rp_region_tablecmp(e.h, h2, p)
                    lib_.rp_set_zero_range.argtypes = [C_.c_int, C_.c_int]; lib_.rp_region_npriv.argtypes = [C_.c_int]; lib_.rp_region_priv_info.argtypes = [C_.c_int, C_.c_int, C_.c_char_p, C_.c_int]
                    def key_out(lo, hi):
                        lib_.rp_set_zero_range(lo, hi)
                        a_ = e.trig(inputs=[mx.array([p // getattr(e, 'S', 1)], dtype=mx.int32), ids] + list(_state(st)), grid=(1, 1, 1), threadgroup=(1, 1, 1), output_shapes=e.shapes, output_dtypes=e.dtypes); mx.eval(a_[0]); mx.synchronize()
                        lib_.rp_set_zero_range(0, 0)
                        return np.array(a_[0])
                    base_ = key_out(0, 0); npr = lib_.rp_region_npriv(e.h)
                    allz = key_out(0, npr)
                    print(f"  ZERO-RANGE: leftover run vs zero-all-private run equal: {np.array_equal(base_, allz)} ({npr} private buffers); leftover==eager {np.array_equal(base_, eag)}", flush=True)
                    if not np.array_equal(base_, allz):
                        lo_, hi_ = 0, npr
                        while hi_ - lo_ > 1:
                            mid_ = (lo_ + hi_) // 2
                            if not np.array_equal(key_out(lo_, mid_), base_): hi_ = mid_
                            elif not np.array_equal(key_out(mid_, hi_), base_): lo_ = mid_
                            else: break
                        inf_ = ctypes.create_string_buffer(256)
                        for k_ in range(lo_, min(hi_ + 1, npr)):
                            lib_.rp_region_priv_info(e.h, k_, inf_, 256)
                            print("  buffer whose leftover content matters:", k_, inf_.value.decode(), flush=True)
                    lib_.rp_set_hb_zero.argtypes = [C_.c_int, C_.c_int]; lib_.rp_region_nhb.argtypes = [C_.c_int]; lib_.rp_region_hb_info.argtypes = [C_.c_int, C_.c_int, C_.c_char_p, C_.c_int]
                    def key_out_hb(lo, hi):
                        lib_.rp_set_hb_zero(lo, hi)
                        a_ = e.trig(inputs=[mx.array([p // getattr(e, 'S', 1)], dtype=mx.int32), ids] + list(_state(st)), grid=(1, 1, 1), threadgroup=(1, 1, 1), output_shapes=e.shapes, output_dtypes=e.dtypes); mx.eval(a_[0]); mx.synchronize()
                        lib_.rp_set_hb_zero(0, 0)
                        return np.array(a_[0])
                    nhb = lib_.rp_region_nhb(e.h)
                    hz = key_out_hb(0, nhb)
                    print(f"  HB-ZERO: key output with all host buffers zeroed equals the normal key output: {np.array_equal(hz, base_)} ({nhb} host buffers)", flush=True)
                    if not np.array_equal(hz, base_):
                        lo_, hi_ = 0, nhb
                        while hi_ - lo_ > 1:
                            mid_ = (lo_ + hi_) // 2
                            if not np.array_equal(key_out_hb(lo_, mid_), base_): hi_ = mid_
                            elif not np.array_equal(key_out_hb(mid_, hi_), base_): lo_ = mid_
                            else: break
                        inf_ = ctypes.create_string_buffer(512)
                        for k_ in range(lo_, min(hi_ + 1, nhb)):
                            lib_.rp_region_hb_info(e.h, k_, inf_, 512)
                            print("  host buffer whose content matters:", inf_.value.decode(), flush=True)
                    lib_.rp_region_hb_vs.argtypes = [C_.c_int, C_.c_int, C_.c_int, C_.c_char_p, C_.c_int]
                    # slot 0 now holds the table values for p (one apply_params ran in the last key run); judge only the host buffers whose content changes the key output
                    key_out_hb(0, 0)
                    rel = []
                    for q_ in range(nhb):
                        if not np.array_equal(key_out_hb(q_, q_ + 1), base_): rel.append(q_)
                    key_out_hb(0, 0)
                    bufv = ctypes.create_string_buffer(512); nbad = 0
                    for q_ in rel:
                        r_ = lib_.rp_region_hb_vs(e.h, q_, h2, bufv, 512)
                        if r_ != 1:
                            nbad += 1
                            lib_.rp_region_hb_info(e.h, q_, inf2 := ctypes.create_string_buffer(512), 512)
                            print("  RELEVANT host buffer DIFFERS from the exact recording:", inf2.value.decode()[:110], "|", bufv.value.decode(), flush=True)
                    print(f"  RELEVANT host buffers: {len(rel)} of {nhb}; differing from exact: {nbad}", flush=True)
                    lib_.rp_region_patchcmp.argtypes = [C_.c_int, C_.c_int]
                    lib_.rp_region_patchcmp(e.h, h2)
                    ncm = 161
                    full = run_both(ncm)
                    lo_, hi_ = 0, ncm
                    if full > 0:
                        while hi_ - lo_ > 1:
                            mid_ = (lo_ + hi_) // 2
                            if run_both(mid_) > 0: hi_ = mid_
                            else: lo_ = mid_
                        print(f"  PREFIX BISECT (both regions poisoned to 0 first): first N with a differing private buffer = {hi_} (command index {hi_ - 1})", flush=True)
                        lib_.rp_set_prefix(hi_); lib_.rp_set_poison(2)
                        a_ = e.trig(inputs=[mx.array([p // getattr(e, 'S', 1)], dtype=mx.int32), ids] + list(_state(st)), grid=(1, 1, 1), threadgroup=(1, 1, 1), output_shapes=e.shapes, output_dtypes=e.dtypes); mx.eval(a_[0]); mx.synchronize()
                        b_ = trig2(inputs=ins2, grid=(1, 1, 1), threadgroup=(1, 1, 1), output_shapes=[o_.shape], output_dtypes=[mx.float32]); mx.eval(b_[0]); mx.synchronize()
                        lib_.rp_region_cmp2(e.h, h2, 0)
                    else:
                        print("  PREFIX: after poisoning both regions agree on every private buffer (full run)", flush=True)
                    lib_.rp_set_prefix(-1); lib_.rp_set_poison(-1)
            st.pend = pd_save; st.taps = t_save
        mine_pend = {(l, n): np.array(a) for l, n, a in _pend_arrays(st)} if VERBOSE else {}
        mine_taps = {l_: np.array(t_).astype(np.float32) for l_, t_ in st.taps.items()} if VERBOSE else {}
        _discard(st)
        if os.environ.get('VR_HANDDBG'): st.hand.dbg_o = []
        ref = eager()
        if e_dbg is not None and not np.array_equal(mine, np.array(ref)):
            pass
        if VERBOSE and kw and not np.array_equal(mine, np.array(ref)):
            from . import engram_mlx as EM
            eh_ = np.asarray(kw['engram_hashes'])
            _ENG['staged'] = True; _ENG['slots'] = []
            try:
                for emb_ in _ENG['embeds']:                                     # fill the slots the staged eager pass will take next
                    pass
                sl_arrays = _stage_rows(_ENG['embeds'], kw['engram_hashes'])
                # rewind the rings so the eager pass takes exactly the slots just filled
                for emb_ in _ENG['embeds']:
                    n_ = int(np.prod(np.asarray(kw['engram_hashes'])[:, :, emb_._pos, :].shape))
                    emb_._next[n_] = (emb_._next[n_] - 1) % emb_.RING
                _discard(st)
                r_staged = eager(); mx.eval(r_staged)
            finally:
                _ENG['staged'] = False
            _eh = np.asarray(kw['engram_hashes']); print('   DIFF-STATE eh shape', _eh.shape, 'min', int(_eh.min()), 'max', int(_eh.max()), 'distinct', len(np.unique(_eh)), 'ids', np.array(ids).tolist(), 'p', plan.pos, flush=True)
            for _r in range(_eh.shape[0]): print('     row', _r, _eh[_r, 0, 0, :].tolist(), flush=True)
            slots2 = _stage_rows(_ENG['embeds'], kw['engram_hashes'])
            res2_ = e.trig(inputs=[mx.array([plan.pos[0] // getattr(e, 'S', 1)], dtype=mx.int32), ids, kw['engram_hashes']] + list(_state(st)) + slots2, grid=(1, 1, 1), threadgroup=(1, 1, 1), output_shapes=e.shapes, output_dtypes=e.dtypes)
            mx.eval(res2_[0]); mx.synchronize()
            print('   SECOND replay (fresh slots) equals eager:', np.array_equal(np.array(res2_[0]), np.array(ref)), '| equals first replay:', np.array_equal(np.array(res2_[0]), mine), flush=True)
            print('   eager STAGED vs eager PARALLEL (same state) equal:', np.array_equal(np.array(r_staged), np.array(ref)), '| replay vs staged-eager equal:', np.array_equal(np.array(r_staged), mine), flush=True)
            for i_, emb in enumerate(_ENG['embeds']):
                idx_ = eh_[:, :, emb._pos, :]
                a_ = np.array(emb._orig(mx.array(idx_)).astype(mx.float32))
                b_ = np.array(EM._dequant(e._last_slots[2 * i_], e._last_slots[2 * i_ + 1]).astype(mx.float32)).reshape(a_.shape)
                print('   ENGRAM rows staged vs parallel equal:', np.array_equal(a_, b_), 'idx rows', idx_.shape, 'distinct', len(np.unique(idx_)), flush=True)
            if e.dbg_no:
                ev_ = [np.array(x_) for x_ in st.hand.dbg_o[-e.dbg_no:]] if getattr(st.hand, 'dbg_o', None) else []
                names_ = ['o', 'q', 'win', 'pool', 'idx', 'pos']
                for li_ in range(min(e.dbg_no, len(ev_))):
                    rp_ = np.array(res[e.dbg_o0 + li_]).astype(np.float32); eg_ = ev_[li_].astype(np.float32)
                    if rp_.shape != eg_.shape:
                        print('   ATTEND dbg', li_ // 6, names_[li_ % 6], 'shape differs', rp_.shape, eg_.shape, '(compare flattened prefix:', np.array_equal(rp_.reshape(-1), eg_.reshape(-1)[:rp_.size]), ')', flush=True); continue
                    print('   ATTEND dbg layer', li_ // 6, names_[li_ % 6], 'equal' if np.array_equal(rp_, eg_) else 'DIFFERS maxdiff %.5f' % float(np.abs(rp_ - eg_).max()), flush=True)
            if os.environ.get('VR_SHOW'):
                ht3 = st.hand.topk_idxs; mx.eval(ht3); lb_ = _L(); lb_.rp_region_scan.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]; lb_.rp_region_show_cmd.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
                lb_.rp_region_scan(e.h, p, -12345)
                print('   EXPECTED: topk width K =', ht3.shape[-1], 'pool capacity', st.cmp[0].shape[1], 'positions', plan.pos, flush=True)
                lb_.rp_region_show_cmd(e.h, b'dsv4_attn', 0)
            if os.environ.get('VR_SCAN'):
                ht2 = st.hand.topk_idxs; mx.eval(ht2); w_ = ht2.shape[-1]; _L().rp_region_scan.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
                for tgt in (w_ - 2, w_ - 1):
                    _L().rp_region_scan(e.h, p, tgt)
            mx.eval(ref); ht_ = st.hand.topk_idxs; mx.eval(ht_)
            print('   HAND topk_idxs replay vs eager equal:', np.array_equal(np.array(e_dbg), np.array(ht_)), 'shape', tuple(ht_.shape), flush=True)
            print('   HAND flattened prefix equal (width-independent):', np.array_equal(np.array(e_dbg).reshape(-1), np.array(ht_).reshape(-1)[:e_dbg.size]), 'replay shape', tuple(e_dbg.shape), flush=True)
            if not np.array_equal(np.array(e_dbg), np.array(ht_)):
                print('     replay', np.array(e_dbg)[:, 0, :16].tolist()[:4], flush=True); print('     eager ', np.array(ht_)[:, 0, :16].tolist()[:4], flush=True)
        mx.eval(ref)
        if VERBOSE and not np.array_equal(mine, np.array(ref)):
            for l_, t_ in sorted(st.taps.items()):
                dd_ = np.abs(np.array(t_).astype(np.float32) - mine_taps[l_]).reshape(mine_taps[l_].shape[0], -1).max(-1)
                print('   TAP layer', l_, 'row maxdiff', [round(float(x), 4) for x in dd_], flush=True)
            for l, n, a in _pend_arrays(st):
                d = np.abs(np.array(a).astype(np.float32) - mine_pend[(l, n)].astype(np.float32))
                print("   pend", l, n, "maxdiff", float(d.max()), "rows differing", np.nonzero(d.reshape(d.shape[0], -1).max(-1))[0].tolist(), flush=True)
                if n == 'win' and l == 1:
                    dd_ = d.reshape(d.shape[0], d.shape[1], -1).max(-1)
                    print('     layer1 win differing slots per row:', [np.nonzero(dd_[r_])[0].tolist()[:12] for r_ in range(dd_.shape[0])], 'positions', plan.pos, flush=True)
        if VERBOSE:
            if kw:
                _ehh = np.asarray(kw['engram_hashes']); print('  audit-eh distinct', len(np.unique(_ehh)), 'of', _ehh.size, 'ids', np.array(ids).tolist(), 'result', 'ok' if np.array_equal(mine, np.array(ref)) else 'DIFF', flush=True)
            print("  audit p", p, "prev", _LAST[0], "this", e.h, "k", p - e.p0, "rows", plan.counts, "state rows", tuple(st.win[0].shape[:1]), "ok" if np.array_equal(mine, np.array(ref)) else "DIFF", flush=True)
        if not np.array_equal(mine, np.array(ref)):
            STATS["mismatch"] += 1
            if not os.environ.get("VR_KEEP"):
                e.state = "dead"; STATS["dead"] += 1
            if VERBOSE:
                r_ = np.array(ref)
                res2 = e.trig(inputs=[mx.array([p // getattr(e, 'S', 1)], dtype=mx.int32), ids] + list(_state(st)), grid=(1, 1, 1), threadgroup=(1, 1, 1), output_shapes=e.shapes, output_dtypes=e.dtypes)
                try:
                    _L().rp_region_ext_changed.restype = ctypes.c_int
                    print("  ext buffers changed since build:", _L().rp_region_ext_changed(e.h), flush=True)
                except Exception as ex_:
                    print("  ext_changed unavailable", ex_, flush=True)
                try:
                    lib_ = _L(); names = []
                    for t_ in range(3):
                        mx.synchronize(); lib_.rp_rec_begin(70 + t_); lib_.rp_rec_snap_born(70 + t_, 1)
                        o_ = eager(); mx.eval(o_); mx.synchronize(); n_ = lib_.rp_rec_end(); _discard(st)
                        lib_.rp_dump(70 + t_, f"/tmp/vr_dump{t_}.txt".encode()); names.append(n_)
                    print("  same-input eager passes, command counts:", names, flush=True)
                    ev = []
                    for t_ in range(3):
                        o_ = eager(); mx.eval(o_); ev.append(np.array(o_)); _discard(st)
                    print("  eager passes equal each other:", np.array_equal(ev[0], ev[1]), np.array_equal(ev[1], ev[2]), "| replay == eager[0]:", np.array_equal(mine, ev[0]), "| replay == audit ref:", np.array_equal(mine, r_), flush=True)
                except Exception as ex_:
                    print("  dump failed", ex_, flush=True)
                try:
                    e2 = _Entry(); _discard(st); _fit(lm, st, plan, ids, e2, eager); _discard(st)
                    r4 = e2.trig(inputs=[mx.array([p // getattr(e, 'S', 1)], dtype=mx.int32), ids] + list(_state(st)), grid=(1, 1, 1), threadgroup=(1, 1, 1), output_shapes=e2.shapes, output_dtypes=e2.dtypes)
                    print("  FRESH FIT at the failing state: replay equals eager:", np.array_equal(np.array(r4[0]), r_), "state", e2.state, flush=True)
                except Exception as ex_:
                    print("  fresh fit failed", ex_, flush=True)
                sweep = []
                row0 = st.rows[0]; off0 = row0.offset
                for q in range(e.p0, e.p0 + 14):
                    row0.offset = q
                    r1 = e.trig(inputs=[mx.array([q], dtype=mx.int32), ids] + list(_state(st)), grid=(1, 1, 1), threadgroup=(1, 1, 1), output_shapes=e.shapes, output_dtypes=e.dtypes)
                    mx.eval(r1[0]); a1 = np.array(r1[0]); _discard(st); r2 = eager(); mx.eval(r2); _discard(st)
                    sweep.append((q - e.p0, bool(np.array_equal(a1, np.array(r2)))))
                row0.offset = off0
                print("  sweep k -> equal:", sweep, flush=True)
                ids2 = mx.array(np.array(ids).tolist(), dtype=mx.int32); mx.eval(ids2)
                res3 = e.trig(inputs=[mx.array([p // getattr(e, 'S', 1)], dtype=mx.int32), ids2] + list(_state(st)), grid=(1, 1, 1), threadgroup=(1, 1, 1), output_shapes=e.shapes, output_dtypes=e.dtypes)
                print("  third replay (fresh ids) equals eager:", np.array_equal(np.array(res3[0]), r_), "ids dtype", ids.dtype, "shape", ids.shape, flush=True)
                print("  second replay equals eager:", np.array_equal(np.array(res2[0]), r_), "| ids", np.array(ids).tolist(), "| cnt", st.cnt, "kcnt", st.kcnt, "| offsets", [r.offset for r in st.rows], "| plan.pos", plan.pos,
                      "| win shapes", [tuple(a.shape) for a in st.win], flush=True)
                print("  replay vs eager: p", p, "row maxdiff", np.abs(mine - r_).max(-1), "p0", e.p0, "k", p - e.p0, flush=True)
            print("verify_replay: SHADOW CHECK DIFFERS, key dead (eager result used)", flush=True)
        return ref
    return out


if VERBOSE:
    atexit.register(lambda: print('verify_replay STATS', STATS, flush=True))
