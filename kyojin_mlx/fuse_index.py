# SPDX-License-Identifier: Apache-2.0
"""DS4.1 indexer for one decode token (batch 1, one query): the same arithmetic as `Indexer.__call__` with the dead work removed and the elementwise chains compiled.

Removed, with no change of any value: in decode the number of index keys is n = (start_pos + 1) // ratio, `compress_lens` is n for the single query and
the keys are positions 0 .. n - 1, so the mask `position < compress_lens` is all true and the final `idxs < compress_lens` is all true (idxs are positions
of the n keys): the arange / compare / where nodes are not built. Compiled (mx.compile, replay instead of a python graph build): the rotary of the query
rows with its fp4 round trip (about 28 launches -> 2). The score sum, the sort and the candidate selection stay stock ops (their shapes follow the
number of keys: a compile would trace again every second token; a top-k kernel is the next cut).

Switch: module attribute ENABLED (env KYOJIN_FUSE_INDEX=0). PACKED_KEYS (env KYOJIN_IDX_KEYS_PACKED, E8): the score kernel reads the fp4 codes of the
KeySlots entry (4.25 bits, 68 B per key) and unpacks them in the kernel, instead of the fp16 mirror (256 B): the same values by construction, 3.8x
less traffic on the one decode term that still grows with the context."""
from __future__ import annotations

from . import archs as _A
import os

import mlx.core as mx

from . import fired as FD

ENABLED = os.environ.get("KYOJIN_FUSE_INDEX", "1") != "0"
DSA = os.environ.get("KYOJIN_DSA_INDEX", "1") != "0"          # score + select kernels (dsa_decode.py) instead of the stock einsum / sort chain
IDX_FAST = os.environ.get("KYOJIN_IDX_FAST", os.environ.get("KYOJIN_LONGCTX", "0")) not in ("0", "")     # multi-threadgroup exact top-k (+ fused candidate mask) on the DSA scores
IDX_FAST_MIN = int(os.environ.get("KYOJIN_IDX_FAST_MIN", "16384"))     # from this many keys (fewer launches win below, see dsv4_core.IDX_FAST_MIN)
PACKED_KEYS = os.environ.get("KYOJIN_IDX_KEYS_PACKED", "0") not in ("0", "")    # E8: score the fp4 codes of the key slots, not their fp16 mirror
_fns: dict = {}


def _packed(cache, layer_idx: int, D: int, n: int):
    """(codes [cap, D / 2], scales [cap, D / 32]) of the KeySlots entry of `layer_idx` when it is the vendor fp4 grid (BLOCK 32, power-of-two scale) and
    holds at least `n` rows, else None: the exact values of the dense rows after `fp4pack.unpack` of the same codes."""
    slots = getattr(cache, "keys", None)
    raw = getattr(slots, "raw", None)                                               # fp4pool.PackedSlots (KV-free of the vendor cache, spec.shell, rows shells)
    if raw is None or getattr(slots, "BLOCK", 0) != 32 or getattr(slots, "E4M3", True):
        return None
    e = raw(layer_idx)
    if type(e).__name__ != "_Pk" or e.dtype not in (mx.float16, mx.bfloat16) or e.codes.shape[0] != 1 or e.codes.shape[2] * 2 != D or e.n < n:
        return None
    return e.codes[0], e.scales[0]


def _compiled(L, rope_dim: int):
    """Rotary + fq4 of the query rows as a compiled function for one rope width."""
    key = ("q", rope_dim)
    if key not in _fns:
        def qrot(q, cos, sin):
            return L.fake_quant_fp4_ue8m0(L._apply_index_rotary(q, cos, sin, rope_dim))

        cq = mx.compile(qrot)
        from . import dsv4_core as DC

        def rot(q, cos, sin):                         # fp16 / 128-wide heads: one kernel (ds_pf_idx.fq4_rot, bitwise equal); else the compiled chain
            if DC.IDX_FUSE and q.dtype == mx.float16 and q.shape[-1] == 128:
                return DC.rot_fq4(q, cos, sin, rope_dim)
            return cq(q, cos, sin)

        _fns[key] = rot
    return _fns[key]


def _publish_keys(L, self, latent, start_pos, cache):
    """Indexer._publish_keys with the rotary + fq4 of the new keys through _compiled (same bits)."""
    ratio, n_latent = self.compress_ratio, latent.shape[1]
    base = start_pos // ratio
    positions = (base + mx.arange(n_latent)) * ratio
    cos, sin = L._index_cos_sin(max((base + n_latent - 1) * ratio + 1, start_pos + n_latent + 1), self.rope_head_dim, self.rope_theta, self.yarn)
    k = _compiled(L, self.rope_head_dim)(self.k_norm(self.wk(latent)), cos[positions][None, :, :], sin[positions][None, :, :])
    return cache.append_index_keys(self.layer_idx, k)


_nn: dict = {}


def _n_array(n: int) -> mx.array:
    """int32 [1] holding the key count (a kernel input: the count changes every token, a template constant would recompile)."""
    a = _nn.get(n)
    if a is None:
        if len(_nn) > 4096:
            _nn.clear()
        a = _nn[n] = mx.array([n], dtype=mx.int32)
    return a


def _impl_indexer(orig):
    import importlib
    L = _A.lang()

    def __call__(self, x, qr, latent, start_pos, offset, cache):
        batch, seqlen = x.shape[0], x.shape[1]
        if not (ENABLED and batch == 1 and seqlen == 1 and x.dtype in (mx.float16, mx.bfloat16)):
            return orig(self, x, qr, latent, start_pos, offset, cache)
        ratio, end_pos = self.compress_ratio, start_pos + 1
        if self.n_heads * (end_pos // ratio) > L.INDEX_SCORE_TILE_MIN_ELEMS:        # the stock path tiles the key axis here: keep its exact chain
            return orig(self, x, qr, latent, start_pos, offset, cache)
        if self.owns_k:
            if latent is not None:
                _publish_keys(L, self, latent, start_pos, cache)
        n = end_pos // ratio
        qrot = _compiled(L, self.rope_head_dim)
        cos, sin = L._index_cos_sin(end_pos, self.rope_head_dim, self.rope_theta, self.yarn)
        q = self.wq_b(L.fake_quant_fp8_ue8m0(qr)).reshape(1, 1, self.n_heads, self.index_head_dim)
        q = qrot(q, cos[start_pos:end_pos][None, :, None, :], sin[start_pos:end_pos][None, :, None, :])
        weights = self.weights_proj(x) * (self.scale * self.n_heads ** -0.5)
        cand = cache.candidates.reshape(-1) if self.uses_candidates else None
        if self.owns_k:
            cache.index_k = cache.keys[self.layer_idx]                          # the dense handoff the consumer layers read (depth-flat: O(new rows))
        index_k = cache.index_k[:1, :n]
        pk = _packed(cache, self.layer_idx, self.index_head_dim, n) if (PACKED_KEYS and DSA and self.owns_k) else None
        if DSA and self.n_heads == 32 and self.index_head_dim == 128 and weights.dtype == q.dtype and (
                pk is not None or index_k.dtype == q.dtype):
            from . import dsa_decode as D
            nn = _n_array(n)
            if pk is not None:                                                  # E8: score the fp4 codes (68 B/key), not the fp16 mirror (256 B/key)
                FD.hit("idx_keys_packed")
                sc = D.scores_packed(q.reshape(32, 128), weights.reshape(32), pk[0], pk[1], nn, n, cand)
            else:
                sc = D.scores(q.reshape(32, 128), weights.reshape(32), cache.index_k[0, :n], nn, n, cand)
            if IDX_FAST and n >= max(IDX_FAST_MIN, self.index_topk):                 # topk_mt returns k columns; the stock select returns min(k, n)
                from . import idxfast as IX, longctx as LC
                LC.HITS["dec_fast_solo"] += 1
                FD.hit("idx_fast")
                sc1 = sc.reshape(1, -1)
                if self.is_candidate_source:
                    idxs, m = IX.select_fused(sc1, nn, self.index_topk, self.candidate_topk_blocks, self.candidate_block_size)
                    npad = sc1.shape[1]
                    cache.candidates = (m.reshape(1, 1, npad)[:, :, :n]).reshape(1, 1, n)
                    return idxs.reshape(1, 1, -1) + offset if offset else idxs.reshape(1, 1, -1)
                idxs = IX.topk_mt(sc1, nn, self.index_topk).reshape(1, 1, -1)
                return idxs + offset if offset else idxs
            if self.is_candidate_source:
                cache.candidates = D.candidate_mask(sc, nn, n, self.candidate_topk_blocks, self.candidate_block_size).reshape(1, 1, n)
            idxs = D.select(sc, nn, n, self.index_topk).reshape(1, 1, -1)
            return idxs + offset if offset else idxs
        tile = mx.einsum("bsnd,btd->bsnt", q, index_k)
        sc = (mx.maximum(tile, 0) * weights[..., None]).sum(axis=2)
        if self.uses_candidates:
            sc = mx.where(cache.candidates, sc, -mx.inf)
        if self.is_candidate_source:
            cache.candidates = L.select_candidate_blocks(sc, n, self.candidate_topk_blocks, self.candidate_block_size)
        topk = min(self.index_topk, n)
        idxs = mx.sort(mx.argsort(-sc, axis=-1)[..., :topk], axis=-1).astype(mx.int32)
        return idxs + offset if offset else idxs

    return __call__
