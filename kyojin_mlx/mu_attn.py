# SPDX-License-Identifier: Apache-2.0
"""Batched decode attention for N sessions of one model: one shared prefix KV (read ONCE for all sessions) + one pooled tail KV.

Op class: attention + KV read of a multi-session decode step (every model with full-attention layers, fp16 / bf16 KV). Before: one SDPA per session per
layer over the session's private copy of the whole KV (N launches, N x the prefix bytes, N x the prefix memory). Here, per layer:
  1. prefix: the split-K NAX flash-decode kernel of sdpa_dec.py with the N sessions' query heads folded into its row axis (rows = N x G per kv head,
     no causal tail): every prefix key is read once for all sessions; it returns the split partials (unnormalised O, max, sum).
  2. tails: the same kernel over the pooled tails [N, Hk, cap, D] (batch axis = session, per-session key count `lens`), one launch for all sessions.
  3. one merge launch: softmax merge of the prefix splits and the tail partial of each session -> [N, Hq, 1, D].
With no shared prefix (P = 0) steps 1 is skipped: still one launch for all sessions instead of N. Not bit-identical to the solo kernel (other split
points, fp32 merge): KL-gated (tools/mu95_kl.py). Needs the NAX tiles (sdpa_dec.nax_ok); multiuser.batch_caches falls back to RowsKV elsewhere.

    bc = multiuser.batch_caches([fork_cache(prefix) + own suffix, ...])   # shared prefix found from the fork records; KYOJIN_MU_POOL=0 keeps RowsKV"""
from __future__ import annotations

import os

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from . import sdpa_dec as SD

ENABLE = os.environ.get("KYOJIN_MU_POOL", "1") == "1"
STEP = 256

_kern: dict = {}


def _tail_kernel():
    """sdpa_dec's NAX kernel with a per-batch key count (lens[b]) instead of one N for all batches."""
    if "tail" not in _kern:
        src = SD._NAX_SRC.replace("const int N = info[0], T = info[1]", "const int N = lens[b], T = info[1]")
        assert src != SD._NAX_SRC
        _kern["tail"] = mx.fast.metal_kernel("kyojin_mu_tail", ["q", "k", "v", "info", "scale", "lens"], ["po", "pml"], src,
                                             header=SD._nax_header(), ensure_row_contiguous=False)
    return _kern["tail"]


# grid (D, G, Nu * Hk); z = u * Hk + hk. Prefix partials: rows (hk * NS1 + s) * RP1 + u * G + g. Tail partial: row z * RP2 + g. out [Nu, Hk, G, D].
_MRG = r"""
    const uint d = thread_position_in_grid.x, g = thread_position_in_grid.y, z = thread_position_in_grid.z;
    const int NS1 = info[0], RP1 = info[1], RP2 = info[2], G = info[3], Hk = info[4];
    const int u = int(z) / Hk, hk = int(z) % Hk;
    const int trow = int(z) * RP2 + int(g);
    const bool HS = info[5] != 0;
    const float SK = HS ? sinks[hk * G + int(g)] : 0.0f;                  // learned attention sink of query head hk * G + g (gpt-oss)
    float M = pml2[trow * 2];
    if (HS) M = max(M, SK);
    for (int s = 0; s < NS1; s++) M = max(M, pml1[((hk * NS1 + s) * RP1 + u * G + int(g)) * 2]);
    float w = fast::exp(pml2[trow * 2] - M);
    float acc = w * po2[trow * D + d], L = w * pml2[trow * 2 + 1];
    for (int s = 0; s < NS1; s++) {
        const int row = (hk * NS1 + s) * RP1 + u * G + int(g);
        w = fast::exp(pml1[row * 2] - M);
        acc += w * po1[row * D + d];
        L += w * pml1[row * 2 + 1];
    }
    if (HS) L += fast::exp(SK - M);                                     // the sink adds to the softmax denominator only
    out[(int(z) * G + int(g)) * D + d] = (T_)(acc / L);
"""


def _mrg_kernel():
    if "mrg" not in _kern:
        _kern["mrg"] = mx.fast.metal_kernel("kyojin_mu_mrg", ["po1", "pml1", "po2", "pml2", "info", "sinks"], ["out"], _MRG)
    return _kern["mrg"]


def _geom(R: int, D: int):
    TQ = 1
    WM, WN = -(-R // 16), max(2, D // (32 if R <= 16 else 64))
    return TQ, WM, WN, WM * TQ * 16


def pooled_attention(q, pk, pv, tk, tv, lens: mx.array, scale: float, sinks: mx.array | None = None):
    """q [Nu, Hq, 1, D]; pk / pv [1, Hk, P, D] shared prefix (or None); tk / tv [Nu, Hk, cap, D] pooled tails, lens [Nu] int32 keys per session
    (>= 1). Returns [Nu, Hq, 1, D]."""
    Nu, Hq, _, D = q.shape
    Hk, cap = tk.shape[1], tk.shape[2]
    G = Hq // Hk
    if pk is not None and Nu * G > SD.MAX_ROWS:                                         # more sessions than rows per threadgroup: chunk
        U = max(1, SD.MAX_ROWS // G)
        return mx.concatenate([pooled_attention(q[i:i + U], pk, pv, tk[i:i + U], tv[i:i + U], lens[i:i + U], scale, sinks) for i in range(0, Nu, U)])
    bk = SD.NAX_BK
    sc = mx.array([scale], dtype=mx.float32)
    q = mx.contiguous(q)
    # tails: batch = session, R = G rows per kv head, one split
    TQ, WM, WN, RP2 = _geom(G, D)
    po2, pml2 = _tail_kernel()(inputs=[q, tk, tv, mx.array([cap, 1, G, G, cap, 1], dtype=mx.int32), sc, lens],
                               template=[("D", D), ("BK", bk), ("TQ", TQ), ("WM", WM), ("WN", WN), ("T_", q.dtype)],
                               grid=(WM * WN * 32, Hk, Nu), threadgroup=(WM * WN * 32, 1, 1),
                               output_shapes=[(Nu * Hk * RP2, D), (Nu * Hk * RP2, 2)], output_dtypes=[mx.float32, mx.float32])
    if pk is not None:
        P = pk.shape[2]
        R = Nu * G
        TQ, WM, WN, RP1 = _geom(R, D)
        ns = SD.nsplit(P, Hk)
        C = max(bk, (P // ns) // bk * bk)
        ns = max(1, P // C)
        qp = mx.contiguous(q.reshape(Nu, Hk, G, D).transpose(1, 0, 2, 3))[None]          # [1, Hk, Nu * G, D]: row u * G + g of kv head hk
        po1, pml1 = SD._nax_kernel()(inputs=[qp, pk, pv, mx.array([P, 1, G, R, C, ns], dtype=mx.int32), sc],
                                     template=[("D", D), ("BK", bk), ("TQ", TQ), ("WM", WM), ("WN", WN), ("T_", q.dtype)],
                                     grid=(ns * WM * WN * 32, Hk, 1), threadgroup=(WM * WN * 32, 1, 1),
                                     output_shapes=[(Hk * ns * RP1, D), (Hk * ns * RP1, 2)], output_dtypes=[mx.float32, mx.float32])
    else:
        ns, RP1 = 0, 1
        po1, pml1 = po2[:1], pml2[:1]
    sk = sinks.astype(mx.float32) if sinks is not None else mx.zeros((1,), dtype=mx.float32)
    out = _mrg_kernel()(inputs=[po1, pml1, po2, pml2, mx.array([ns, RP1, RP2, G, Hk, int(sinks is not None)], dtype=mx.int32), sk], template=[("D", D), ("T_", q.dtype)],
                        grid=(D, G, Nu * Hk), threadgroup=(min(D, 256), 1, 1), output_shapes=[(Nu * Hk * G, D)], output_dtypes=[q.dtype])[0]
    return out.reshape(Nu, Hq, 1, D)


def supported(caches: list, prefix_len: int) -> bool:
    if not (ENABLE and SD.nax_ok()):
        return False
    c = caches[0]
    if not all(type(x) is KVCache and x.keys is not None and x.offset > prefix_len for x in caches):
        return False
    D = c.keys.shape[-1]
    return D in (64, 128, 256) and c.keys.dtype in (mx.float16, mx.bfloat16) and c.values.shape[-1] == D


class PoolKV:
    """Full-attention cache of a batch of sessions: shared prefix (one copy, read once per step) + pooled private tails. Decode only (one token per
    session per step). `offset` is the per-session rope position (vector)."""

    def __init__(self, caches: list[KVCache], prefix_len: int = 0):
        P = prefix_len
        c0 = caches[0]
        self.P = P
        self.pk = c0.keys[:, :, :P] if P else None                    # views of session 0's buffer (never written again)
        self.pv = c0.values[:, :, :P] if P else None
        lens = [c.offset - P for c in caches]
        cap = -(-(max(lens) + STEP) // STEP) * STEP
        _, Hk, _, D = c0.keys.shape
        Nu = len(caches)
        self.tk = mx.zeros((Nu, Hk, cap, D), dtype=c0.keys.dtype)
        self.tv = mx.zeros((Nu, Hk, cap, c0.values.shape[-1]), dtype=c0.values.dtype)
        for u, c in enumerate(caches):
            if lens[u]:
                self.tk[u:u + 1, :, :lens[u]] = c.keys[:, :, P:c.offset]
                self.tv[u:u + 1, :, :lens[u]] = c.values[:, :, P:c.offset]
        self.lens = lens
        chains = [dict(getattr(c, "mu_prefix", ())) for c in caches]           # fork records common to all users (<= P), kept by unbatch
        self.mu_prefix = tuple((t, min(ch[t] for ch in chains)) for t in chains[0] if all(t in ch for ch in chains))

    @property
    def offset(self) -> mx.array:
        return mx.array([self.P + n for n in self.lens])

    def make_mask(self, N: int, **kw):
        assert N == 1, "PoolKV is decode only (one token per session)"
        return None

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        Nu = keys.shape[0]
        assert keys.shape[2] == 1, "PoolKV is decode only (one token per session)"
        if max(self.lens) + 1 > self.tk.shape[2]:
            pad = lambda t: mx.concatenate([t, mx.zeros(t.shape[:2] + (STEP, t.shape[3]), dtype=t.dtype)], axis=2)
            self.tk, self.tv = pad(self.tk), pad(self.tv)
        idx = mx.array(self.lens, dtype=mx.int32).reshape(Nu, 1, 1, 1)
        self.tk = mx.put_along_axis(self.tk, idx, keys, axis=2)
        self.tv = mx.put_along_axis(self.tv, idx, values, axis=2)
        self.lens = [n + 1 for n in self.lens]
        return None, None

    def attend(self, q: mx.array, scale: float, sinks: mx.array | None = None) -> mx.array:
        return pooled_attention(q, self.pk, self.pv, self.tk, self.tv, mx.array(self.lens, dtype=mx.int32), scale, sinks)

    def unbatch(self) -> list[KVCache]:
        out = []
        for u, n in enumerate(self.lens):
            c = KVCache()
            tk, tv = self.tk[u:u + 1, :, :n], self.tv[u:u + 1, :, :n]
            c.keys = mx.concatenate([self.pk, tk], axis=2) if self.pk is not None else tk
            c.values = mx.concatenate([self.pv, tv], axis=2) if self.pv is not None else tv
            c.offset = self.P + n
            c.mu_prefix = self.mu_prefix                               # the next batch shares the prefix again
            out.append(c)
        return out

    @property
    def state(self):
        return [self.tk, self.tv] + ([self.pk, self.pv] if self.pk is not None else [])
