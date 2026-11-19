# SPDX-License-Identifier: Apache-2.0
"""Block N3 of the decoder skeleton: kpool DSA selection. Fires on tensors (`indexer.attn_q_b` + `indexer_compressor_{ape,gate}`) and the keys
`attention.indexer.{kpool, kpool_select_tail, types}`; no architecture name appears here.

The op (read off hy-v4-flash.cpp of the hy4-flash-mac-minimal patch, line numbers of that patch file):
  index query  iq = indexer.attn_q_b(q_lora residual) [H, d], rope on the trailing n_rot dims at the token position          M:5619-5621, 5585-5600
  index key    ik = rms(indexer.attn_k(x)) ; value = compressor_value(ik), gate = compressor_gate(ik)                       M:5625-5632
  pool         groups of `kpool` consecutive positions; per channel softmax over the group of (gate + ape[slot]), sum of value;
               rms(compressor_norm), rope on the tail at the group's FIRST position                                         M:5646-5670, 4282-4287
  score        sum_h relu(iq_h . pool) * w_h, w = indexer.proj(x) / sqrt(d * H)                                             M:5677-5703
  visible      kpool_select_tail: pools that end before the token (ends < p); else pools that end at or before it           M:4325-4334
  keep         top (top_k / kpool - select_tail) visible pools by score, plus the token's own group up to itself (the tail)  M:5707-5713, 4351-4378

Rows contract (rows.py of the DS4.1 engine): every function takes a per-row cache offset `off` (int, or int array [B]). The cache is the mlx-lm
batch layout: row b's keys are RIGHT-aligned in the S slots, so slot s holds position s - pad[b] with pad[b] = S - off[b] - L (pad = 0 for a solo
cache). Stage 1 recomputes the pools from the cached value / gate rows at each call (a pooled-key cache is a later speed step); the result is the
bool `keep` [B, 1, L, S] the skeleton's `_attend_plain` already takes.
"""
from __future__ import annotations

import os
import sys
import weakref

import numpy as np
import mlx.core as mx

KEYCACHE = os.environ.get("KYOJIN_KPOOL_KEYCACHE", "0") not in ("0", "")      # flag kpool_keycache: env at import, then bound by flags.set, read per call
PATHS = {"keycache_hit": 0, "keycache_new": 0, "keycache_fallback": 0}
# Side buffers of the pooled keys, one per cache object (the cache classes are not edited): cache -> [pk [1, J, d], n_seen, shape key, the cache's `keys` array it was built on]. Never copied by
# fork_cache (a copy is a new object: empty state, one full recompute), so a forked / restored cache can never see another cache's pools.
_STATE = weakref.WeakKeyDictionary()

# tensor kinds (after `blk.N.`, without .weight) this block claims -> role (module path under model.layers.N.)
KINDS = {
    "indexer.attn_q_b": "self_attn.kpool.q_b_proj", "indexer.attn_k": "self_attn.kpool.k_proj", "indexer.k_norm": "self_attn.kpool.k_norm",
    "indexer.proj": "self_attn.kpool.weights_proj", "indexer_compressor_value": "self_attn.kpool.value_proj",
    "indexer_compressor_gate": "self_attn.kpool.gate_proj", "indexer_compressor_ape": "self_attn.kpool.ape",
    "indexer_compressor_norm": "self_attn.kpool.pool_norm",
}
# metadata keys (after `<arch>.`) it places beyond the skeleton's own indexer keys (head_count, key_length, top_k, types are foreign_skeleton._KNOWN)
KEYS = {"attention.indexer.kpool", "attention.indexer.kpool_select_tail", "attention.indexer.index_share_mtp"}
# predicate key: True = the nextn (MTP) draft layer reuses the main stack's indexer selection instead of an indexer of its own (glm5-next writes
# True); absent or False = no sharing (Hy4-flash). Header-mapped only: the nextn draft path is stage 5, so this form is mapped, not run.
SHARE_MTP = "attention.indexer.index_share_mtp"


def shares_mtp(kv: dict) -> bool:
    a = kv.get("general.architecture", "")
    return bool(kv.get(f"{a}.{SHARE_MTP}", False))


def fires(layer_tensors: dict) -> bool:
    return all(k + ".weight" in layer_tensors for k in ("indexer.attn_q_b", "indexer_compressor_ape", "indexer_compressor_gate"))


def row_pad(off, L: int, S: int, B: int):
    """-> int32 [B]: slots before row b's first position (right-aligned batch cache)."""
    off = mx.array(off, dtype=mx.int32) if not isinstance(off, mx.array) else off.astype(mx.int32)
    off = mx.broadcast_to(off.reshape(-1), (B,)) if off.size == 1 else off
    return (S - L) - off, off


_AR = {}


def ar(n):
    """int32 arange(n) as a cached constant (a numpy-backed leaf): inside a compiled decode step it is an input, not one launch per layer per call."""
    a = _AR.get(n)
    if a is None:
        a = _AR[n] = mx.array(np.arange(n, dtype=np.int32))
    return a


def pool_keys(val, gate, ape, norm_w, eps, pad, n_pos, kp, rope, J=None, p0=0):
    """val / gate [B, S, d] cached rows (right-aligned), ape [kp, d], pad / n_pos [B] -> pooled keys [B, J, d], valid [B, J] (complete pools).
    `rope(x [B, 1, J, d], scale)` rotates the tail at positions j * scale."""
    B, S, d = val.shape
    J = J if J is not None else max(int(mx.max(n_pos).item()) // kp, 1)      # J given: static pool count, no device read (fixed-state cache)
    idx = pad[:, None, None] + (ar(J)[None, :, None] * kp + ar(kp)[None, None, :])
    valid = (ar(J)[None, :] + 1) * kp <= n_pos[:, None]
    idx = mx.clip(idx, 0, S - 1).reshape(B, J * kp, 1)
    v = mx.take_along_axis(val.astype(mx.float32), idx, axis=1).reshape(B, J, kp, d)
    g = mx.take_along_axis(gate.astype(mx.float32), idx, axis=1).reshape(B, J, kp, d) + ape.astype(mx.float32)[None, None]
    pk = (mx.softmax(g, axis=2, precise=True) * v).sum(axis=2)
    pk = mx.fast.rms_norm(pk, norm_w.astype(mx.float32), eps)
    pk = (rope(pk[:, None], kp) if p0 == 0 else rope(pk[:, None], kp, p0=p0))[:, 0]
    return pk, valid


def pool_rows(val, gate, ape, norm_w, eps, kp, rope, p0):
    """val / gate [nj * kp, d]: the rows of nj consecutive complete groups -> pooled keys [1, nj, d] at pool indexes p0 .. p0 + nj - 1 (p0 an int or a 0-d array). The ops of pool_keys on
    the same rows (a contiguous row range instead of a gather), so the values are bit-equal."""
    nj, d = val.shape[0] // kp, val.shape[-1]
    v = val.astype(mx.float32).reshape(1, nj, kp, d)
    g = gate.astype(mx.float32).reshape(1, nj, kp, d) + ape.astype(mx.float32)[None, None]
    pk = (mx.softmax(g, axis=2, precise=True) * v).sum(axis=2)
    pk = mx.fast.rms_norm(pk, norm_w.astype(mx.float32), eps)
    return rope(pk[:, None], kp, p0=p0)[:, 0]


def pools_valid(J, n_pos, kp):
    """valid [B, J]: the pools whose kp rows are all among the first n_pos [B]."""
    return (ar(J)[None, :] + 1) * kp <= n_pos[:, None]


def select_pools(iq, w, pk, valid, off, S, kp, top_k, select_tail, left=False):
    """THE selection (shared by the mask path and the gather path). iq [B, H, L, d] (roped at the token positions), w [B, L, H], pk [B, J, d],
    valid [B, J] -> dict(sel bool [B, L, J] chosen pools, top int [B, L, n_top] their indices (sel is true at top where the pool is visible),
    pad [B], qpos [B, L], gstart [B, L, 1] first position of the forced tail group)."""
    B, H, L, _ = iq.shape
    J = pk.shape[1]
    from . import kpool_score as KS
    if B == 1 and L > 8 and KS.use(L) and J > min(J, top_k // kp - int(bool(select_tail))):       # flag kpool_score: prompt chunks on the fused scorer + exact multi-threadgroup top-k
        return KS.select_pools_fast(iq, w, pk, valid, off, S, kp, top_k, select_tail, left)
    pad, off = row_pad(off, L, S, B)
    if left:                                                                                      # left-aligned fixed buffer: slot == position
        pad = mx.zeros_like(pad)
    qpos = off[:, None] + ar(L)[None, :]                                                   # [B, L]
    kq = mx.maximum(iq.astype(mx.float32) @ pk.astype(mx.float32)[:, None].swapaxes(-1, -2), 0.0)   # [B, H, L, J]
    score = (kq * w.astype(mx.float32).transpose(0, 2, 1)[..., None]).sum(axis=1)               # [B, L, J]
    j = ar(J)[None, None, :]
    nvis = qpos // kp if select_tail else (qpos + 1) // kp                                       # pools visible to the token
    vis = (j < nvis[..., None]) & valid[:, None, :]
    score = mx.where(vis, score, -mx.inf)
    n_top = min(J, top_k // kp - int(bool(select_tail)))
    top = mx.argsort(-score, axis=-1)[..., :n_top]
    sel = mx.put_along_axis(mx.zeros(score.shape, dtype=mx.bool_), top, mx.ones(top.shape, dtype=mx.bool_), axis=-1) & vis    # [B, L, J]
    q = qpos[..., None]
    gstart = (q // kp) * kp if select_tail else q - (q + 1) % kp + 1                              # first position of the tail
    return dict(sel=sel, top=top, pad=pad, qpos=qpos, gstart=gstart, J=J, S=S, kp=kp)


_VALID = {}


def _ones_valid(J):
    """valid [1, J] all True (every pool of a cached_keys result is complete); one array per J."""
    v = _VALID.get(J)
    if v is None:
        v = _VALID[J] = mx.ones((1, J), dtype=mx.bool_)
    return v


def keep_mask(s):
    """The bool keep mask [B, 1, L, S] of a select_pools result."""
    sel, pad, qpos, gstart, J, S, kp = s["sel"], s["pad"], s["qpos"], s["gstart"], s["J"], s["S"], s["kp"]
    B, L, _ = sel.shape
    kpos = ar(S)[None, :] - pad[:, None]                                                  # [B, S] position held by each slot
    kj = mx.clip(kpos // kp, 0, J - 1)
    in_pool = mx.take_along_axis(sel, mx.broadcast_to(kj[:, None, :], (B, L, S)), axis=-1) & ((kpos // kp) < J)[:, None, :]
    q = qpos[..., None]
    tail = (kpos[:, None, :] >= gstart) & (kpos[:, None, :] <= q)
    keep = (in_pool | tail) & (kpos[:, None, :] >= 0) & (kpos[:, None, :] <= q)
    return keep[:, None]


def gather_slots(s, sort: bool = True):
    """The same keep set as keep_mask as a fixed-width per-row slot list: -> slots int32 [B, L, W] ascending (the masked path's summation order),
    ok bool [B, L, W]; W = n_top * kp + kp <= top_k (+ kp when select_tail is off). Not-kept entries are slot 0 with ok = False."""
    sel, top, pad, qpos, gstart, S, kp = s.get("sel"), s["top"], s["pad"], s["qpos"], s["gstart"], s["S"], s["kp"]
    B, L = top.shape[0], top.shape[1]
    ok_pool = s["ok_pool"] if "ok_pool" in s else mx.take_along_axis(sel, top, axis=-1)           # [B, L, n_top]  (ok_pool given: the top-k came from a selection that marks its own empty slots)
    ar = mx.arange(kp)
    pos = (top[..., None] * kp + ar).reshape(B, L, -1)                                           # [B, L, n_top * kp] positions
    ok = mx.broadcast_to(ok_pool[..., None], (B, L, top.shape[-1], kp)).reshape(B, L, -1)
    tpos = gstart + ar                                                                           # [B, L, kp] tail group positions
    tok = (tpos <= qpos[..., None]) & (tpos >= 0)
    pos = mx.concatenate([pos, tpos], axis=-1)
    ok = mx.concatenate([ok, tok], axis=-1) & (pos >= 0) & (pos <= qpos[..., None])
    slot = pos + pad[:, None, None]
    ok = ok & (slot >= 0) & (slot < S)
    if not sort:                                                                                  # decode / verify rows (KYOJIN_HY4_GATHER_NOSORT=1): the attend masks by ok, so the list needs no order (one sort launch less per DSA layer)
        return mx.where(ok, slot, 0).astype(mx.int32), ok
    slot = mx.sort(mx.where(ok, slot, S), axis=-1)                                                # invalid last, kept slots ascending
    ok = slot < S
    return mx.where(ok, slot, 0).astype(mx.int32), ok


def select(iq, w, pk, valid, off, S, kp, top_k, select_tail):
    """iq [B, H, L, d] (roped at the token positions), w [B, L, H], pk [B, J, d], valid [B, J] -> keep bool [B, 1, L, S]."""
    return keep_mask(select_pools(iq, w, pk, valid, off, S, kp, top_k, select_tail))


def make_rope(n_rot, base, traditional, rope_scale=1.0):
    """Rope on the trailing n_rot dims of each head (M:5585-5600), positions = offset + index times `scale`."""
    def rope_at(x, off_or_scale, offset=None, p0=0):
        nope = x.shape[-1] - n_rot
        if offset is None:                         # pooled keys: index j -> position j * scale
            back = mx.fast.rope(x[..., nope:], n_rot, traditional=traditional, base=base, scale=rope_scale * off_or_scale, offset=p0)      # p0: index of the first pool of x
        else:
            back = mx.fast.rope(x[..., nope:], n_rot, traditional=traditional, base=base, scale=rope_scale, offset=offset)
        return mx.concatenate([x[..., :nope], back.astype(x.dtype)], axis=-1)
    return rope_at


class KpoolSelect:
    """The per-layer state of N3 (weights as arrays; projections stay the skeleton's Linears once N1 wires the q_lora residual)."""

    def __init__(self, ape, norm_w, eps, kp, top_k, select_tail, n_rot, base, traditional, rope_scale=1.0):
        self.ape, self.norm_w, self.eps, self.kp, self.top_k, self.select_tail = ape, norm_w, eps, kp, top_k, select_tail
        self.rope = make_rope(n_rot, base, traditional, rope_scale)

    def keep(self, iq, w, val, gate, off):
        """iq [B, H, L, d] NOT yet roped, w [B, L, H], val / gate [B, S, d] the cached rows (this step's appended) -> keep [B, 1, L, S]."""
        return keep_mask(self.pools(iq, w, val, gate, off))

    def keep_fused(self, iq, w, val, gate, off):
        """Flag kpool_fuse, decode rows: the keep mask [B, 1, L, S] of keep() from ONE selection launch (kpool_fuse.select); pooled keys as pools()."""
        from . import kpool_fuse as KF
        B, _, L, _ = iq.shape
        S = val.shape[1]
        pad, offv = row_pad(off, L, S, B)
        iq = self.rope(iq, None, offset=offv if offv.size > 1 else int(offv[0].item()))
        pk, valid = pool_keys(val, gate, self.ape, self.norm_w, self.eps, pad, offv + L, self.kp, self.rope)
        return KF.select(iq, w, pk, valid, offv, S, self.kp, self.top_k, self.select_tail)[0]

    def cached_keys(self, cache, val, gate, off, L):
        """Flag kpool_keycache: (pk [1, J, d], valid [1, J], offset int) = the pooled keys of the J = n // kp complete groups of the first n = off + L cached
        rows, from a per-cache buffer extended only by the groups this call completes (llama.cpp keeps a pool in the cells once per kp tokens; here every
        token re-ran the gather + f32 cast + softmax + norm + rope over ALL rows). Same ops on the same rows per pool, pools being independent
        along J: bit-equal to pool_keys. None = not applicable (stock path): array offset, rows not right-aligned from 0, fewer than one group, a cache
        object that cannot be keyed. A rewound cache (off < rows seen) drops the pools that end after `off`."""
        if cache is None or not isinstance(off, int) or val.shape[0] != 1 or val.shape[1] != off + L or (off + L) // self.kp < 1:
            return None
        kp, n = self.kp, off + L
        J = n // kp
        key = (kp, self.ape.shape, val.shape[-1])
        try:
            st = _STATE.get(cache)
        except TypeError:
            return None
        pk0, have = None, 0
        if st is not None and st[2] == key and st[3] is getattr(cache, "keys", None):   # same buffer object: restored / re-grown caches miss
            have = min(st[0].shape[1], off // kp)            # pools that end at or before the rows still there; those rows are untouched
            pk0 = st[0][:, :have] if have < st[0].shape[1] else st[0]
        if have == J:
            PATHS["keycache_hit"] += 1
            pk = pk0
        else:
            new, _ = pool_keys(val[:, have * kp:J * kp], gate[:, have * kp:J * kp], self.ape, self.norm_w, self.eps, mx.zeros((1,), mx.int32),
                               mx.array([(J - have) * kp]), kp, self.rope, J=J - have, p0=have)
            pk = new if pk0 is None or have == 0 else mx.concatenate([pk0, new], axis=1)
            PATHS["keycache_new"] += 1
        try:
            _STATE[cache] = [pk, n, key, getattr(cache, "keys", None)]
        except TypeError:
            return None
        return pk, _ones_valid(J), off

    def keep_cached(self, iq, w, cache, val, gate, off, fused):
        """keep [B, 1, L, S] through cached_keys (None when it does not apply); fused = the kpool_fuse launch, else select_pools + keep_mask."""
        B, _, L, _ = iq.shape
        r = self.cached_keys(cache, val, gate, off, L)
        if r is None:
            PATHS["keycache_fallback"] += 1
            return None
        pk, valid, _ = r
        S = val.shape[1]
        offv = mx.array([off], dtype=mx.int32)
        iq = self.rope(iq, None, offset=off)
        if fused:
            from . import kpool_fuse as KF
            return KF.select(iq, w, pk, valid, offv, S, self.kp, self.top_k, self.select_tail)[0]
        return keep_mask(select_pools(iq, w, pk, valid, offv, S, self.kp, self.top_k, self.select_tail))

    def pools(self, iq, w, val, gate, off):
        """The shared selection (select_pools dict) behind keep() and the gather path."""
        B, _, L, _ = iq.shape
        S = val.shape[1]
        pad, offv = row_pad(off, L, S, B)
        off_int = int(off) if isinstance(off, int) else None                                          # a python offset (one session): no device read below
        iq = self.rope(iq, None, offset=off_int if off_int is not None else (offv if offv.size > 1 else int(offv[0].item())))
        pk, valid = pool_keys(val, gate, self.ape, self.norm_w, self.eps, pad, offv + L, self.kp, self.rope,
                              J=None if off_int is None else max((off_int + L) // self.kp, 1))
        return select_pools(iq, w, pk, valid, offv, S, self.kp, self.top_k, self.select_tail)

    def keep_fused_fixed(self, iq, w, val, gate, off, pk=None):
        """Flag kpool_fuse on the left-aligned fixed-state buffer: keep [B, 1, L, S] of keep_mask(pools_fixed(...)) from ONE selection launch (no device read, static shapes)."""
        from . import kpool_fuse as KF
        B, _, L, _ = iq.shape
        S = val.shape[1]
        offv = off.astype(mx.int32).reshape(-1)
        iq = self.rope(iq, None, offset=offv.reshape(()))
        pk, valid = self._pk(val, gate, offv, L, S, B, pk)
        return KF.select(iq, w, pk, valid, offv, S, self.kp, self.top_k, self.select_tail, left=True)[0]

    def update_pk(self, pk, rows, off, L, a, dx):
        """Flag KYOJIN_HY4_FIXED_PKCACHE: pk [1, Jc, d] (cache[2]) with the pools this call completes. rows [1, 1, C, D] = the buffer AFTER the L new rows were written at `off` (int32 [1]).
        The groups that can complete are the nj = ceil(L / kp) ones from off // kp on (the group of the first new row, consecutive); the rest of the candidates is written with the pool of
        still-partial rows, which nothing reads (visibility needs a complete group) and which the call that completes the group overwrites."""
        kp = self.kp
        nj = -(-L // kp)
        j0 = (off // kp).astype(mx.int32)
        r = mx.slice(rows, j0 * kp, (2,), (1, 1, nj * kp, rows.shape[3]))[0, 0]
        new = pool_rows(r[:, a:a + dx], r[:, a + dx:a + 2 * dx], self.ape, self.norm_w, self.eps, kp, self.rope, j0.reshape(()))
        return mx.slice_update(pk, new.astype(pk.dtype), j0, (1,))

    def pools_fixed(self, iq, w, val, gate, off, pk=None):
        """pools() for the left-aligned fixed-state buffer (slot == position, `off` an int32 [1] array, S the static buffer / view length): no device read,
        every shape static, so the whole layer can be one compiled program. Rows past the write offset are masked by the validity / visibility terms."""
        B, _, L, _ = iq.shape
        S = val.shape[1]
        offv = off.astype(mx.int32).reshape(-1)
        iq = self.rope(iq, None, offset=offv.reshape(()))
        pk, valid = self._pk(val, gate, offv, L, S, B, pk)
        return select_pools(iq, w, pk, valid, offv, S, self.kp, self.top_k, self.select_tail, left=True)

    def _pk(self, val, gate, offv, L, S, B, pk):
        """The pooled keys [B, J, d] and valid [B, J] of the J = S // kp pools of a left-aligned fixed buffer: the cache's own (pk [1, Jc, d], flag KYOJIN_HY4_FIXED_PKCACHE) or computed from the rows."""
        J = max(S // self.kp, 1)
        if pk is not None:
            return pk[:, :J], pools_valid(J, offv + L, self.kp)
        return pool_keys(val, gate, self.ape, self.norm_w, self.eps, mx.zeros((B,), dtype=mx.int32), offv + L, self.kp, self.rope, J=J)


    def pools_fixed_top(self, iq, w, val, gate, off, block: int = 256, pk=None):
        """The select_pools dict of pools_fixed (left-aligned fixed buffer) for the GATHER attend, without the [B, H, L, J] product and the full-row argsort:
        the rows are scored `block` at a time (relu(iq . pk) * w, the fp32 matmul of
        select_pools) and take the exact multi-threadgroup radix top-k of idxfast.topk_mt (the same pools, ties to the lowest pool index). -> dict with
        top [B, L, n_top] (ascending pool index, empty slots 0) and ok_pool [B, L, n_top], pad / qpos / gstart / S / kp as select_pools."""
        from . import kpool_fuse as KF
        B, _, L, _ = iq.shape
        S = val.shape[1]
        n_top = min(max(S // self.kp, 1), self.top_k // self.kp - int(bool(self.select_tail)))
        from . import kpool_hisa as HI
        J0 = max(S // self.kp, 1)
        hisa = HI.wanted(L, B, J0, n_top) if n_top >= 1 else None                                      # KYOJIN_HY4_HISA=1: prompt chunks score block bounds, then only the candidate pools exactly
        if n_top < 1 or (hisa is None and (not mx.metal.is_available() or mx.default_device() != mx.gpu)):       # CPU / no Metal: the stock selection (its argsort), same gather downstream
            return self.pools_fixed(iq, w, val, gate, off, pk)
        offv = off.astype(mx.int32).reshape(-1)
        iq = self.rope(iq, None, offset=offv.reshape(()))
        J = max(S // self.kp, 1)
        pk, valid = self._pk(val, gate, offv, L, S, B, pk)
        kp = self.kp
        qpos = offv[:, None] + ar(L)[None, :]                                                    # [B, L]
        q = qpos[..., None]
        gstart = (q // kp) * kp if self.select_tail else q - (q + 1) % kp + 1
        if hisa is not None:
            t, ok_pool = HI.top(iq, w, pk, valid, offv, kp, n_top, self.select_tail, hisa)
            if os.environ.get("KYOJIN_HY4_HISA_AUDIT", "0") == "1":
                try:
                    HI.audit(iq, w, pk, valid, offv, kp, n_top, self.select_tail, t, ok_pool)
                except Exception as e:                                                            # inside a compiled program (no host sync): the audit is skipped
                    HI.STATS["errors"] = HI.STATS.get("errors", 0) + 1
                    if HI.STATS["errors"] == 1:
                        print(f"HISA audit skipped: {e!r}"[:200], file=sys.stderr, flush=True)
            return dict(top=t, ok_pool=ok_pool, pad=mx.zeros((B,), dtype=mx.int32), qpos=qpos, gstart=gstart, J=J, S=S, kp=kp)
        from . import idxfast as IX
        nvis = qpos // kp if self.select_tail else (qpos + 1) // kp                           # pools visible to each token
        from . import kpool_score as KS
        if KS.use(L):                                                                             # flag kpool_score: one fused scorer launch for all rows, no [B, H, n, J] product
            lens = mx.minimum(nvis, J).astype(mx.int32).reshape(-1)
            t = IX.topk_mt(KS.scores(iq, w, pk, valid, offv, kp, self.select_tail).reshape(-1, J), lens, n_top).reshape(B, L, n_top)
            ok_pool = t >= 0
            return dict(top=mx.where(ok_pool, t, 0), ok_pool=ok_pool, pad=mx.zeros((B,), dtype=mx.int32), qpos=qpos, gstart=gstart, J=J, S=S, kp=kp)
        outs = []
        pkT = pk.astype(mx.float32)[:, None].swapaxes(-1, -2)                                      # one cast of the pooled keys for every row block (fp16 pk: not one cast per block)
        for a in range(0, L, block):
            b = min(L, a + block)
            kq = mx.maximum(iq[:, :, a:b].astype(mx.float32) @ pkT, 0.0)       # [B, H, n, J]
            sc = (kq * w[:, a:b].astype(mx.float32).transpose(0, 2, 1)[..., None]).sum(axis=1)                          # [B, n, J]
            lens = mx.minimum(nvis[:, a:b], J).astype(mx.int32).reshape(-1)
            outs.append(IX.topk_mt(sc.reshape(-1, J), lens, n_top).reshape(B, b - a, n_top))
        t = mx.concatenate(outs, axis=1) if len(outs) > 1 else outs[0]
        ok_pool = t >= 0
        top = mx.where(ok_pool, t, 0)
        return dict(top=top, ok_pool=ok_pool, pad=mx.zeros((B,), dtype=mx.int32), qpos=qpos, gstart=gstart, J=J, S=S, kp=kp)
