"""Engram n-gram hash of a decode step / verify row in ONE launch (macair40).

mlx-vlm's `NgramHashState.__call__` builds the hash ids of the positions of a call with about 50 small graph nodes (history concatenation, gathers, clip /
where chains, products, xor, remainders): 53 dispatches for one token, 55 more for every extra row of a chained verify. This replaces the call by one
Metal kernel for the calls it can serve (a history on the state with the same batch, at most `MAX_ROWS` new positions); everything else (the first call,
a prefill chunk, a batch change) runs the stock code. The kernel is integer only: the same tokens, the same int64 products, xors and remainders, so the
hash ids and the history written to `state.engram` are equal to the stock ones bit for bit (tests/test_engram_fuse.py). KYOJIN_ENGRAM_FUSE=0 off.
"""
import os

import mlx.core as mx

ENABLED = os.environ.get("KYOJIN_ENGRAM_FUSE", "1") != "0"
MAX_ROWS = 8
_KERNELS: dict = {}


def _source(n_layers: int, n_gram: int, n_heads: int, pad_id: int) -> str:
    cols = (n_gram - 1) * n_heads
    return f"""
    const int NL = {n_layers}, NG = {n_gram}, NH = {n_heads}, NC = {cols};
    const long DEAD = -1, PAD = {pad_id};
    uint gid = thread_position_in_grid.x;
    int B = ids_shape[0], L = ids_shape[1];
    int Lh = hist_shape[1];
    int start = sp[0];
    int need = start + L;
    int Lc = max(Lh, need);
    auto comp = [&](int b, int i) -> long {{
        long t = tmap[(long)ids[b * L + i]];
        if (HM && !msk[b * L + i]) t = DEAD;
        return t;
    }};
    if (gid < (uint)(B * Lc)) {{
        int b = gid / Lc, j = gid % Lc;
        long v = 0;
        if (j >= start && j < need) v = comp(b, j - start);
        else if (j < Lh) v = hist[b * Lh + j];
        newh[gid] = v;
    }}
    if (gid < (uint)(B * L * NL)) {{
        int ly = gid % NL;
        int s = (gid / NL) % L;
        int b = gid / (NL * L);
        int p = start + s;
        long tok[NG];
        bool blocked = false;
        for (int k = 0; k < NG; k++) {{
            int q = min(max(p - k, 0), Lc - 1);
            long src = (q >= start && q < need) ? comp(b, q - start) : (q < Lh ? hist[b * Lh + q] : 0);
            blocked = blocked || (p < k) || (src == DEAD);
            tok[k] = blocked ? PAD : src;
        }}
        long rolling = tok[0] * mult[ly * NG];
        for (int i = 1; i < NG; i++) {{
            rolling = rolling ^ (tok[i] * mult[ly * NG + i]);
            for (int h = 0; h < NH; h++) {{
                int c = (i - 1) * NH + h;
                out[gid * NC + c] = rolling % primes[(ly * (NG - 1) + (i - 1)) * NH + h] + offs[ly * NC + c];
            }}
        }}
    }}
    """


def _kernel(key):
    k = _KERNELS.get(key)
    if k is None:
        k = _KERNELS[key] = mx.fast.metal_kernel(
            name=f"engram_hash_{key[0]}_{key[1]}_{key[2]}_{key[3]}", input_names=["ids", "msk", "hist", "sp", "tmap", "mult", "primes", "offs"],
            output_names=["newh", "out"], source=_source(*key))
    return k


def hash_call(self, input_ids, start_pos, state, token_mask=None):
    """Drop-in for NgramHashState.__call__ (same arguments, same results and state update)."""
    cache = state.engram
    if not ENABLED or cache is None or input_ids.ndim != 2 or input_ids.shape[1] > MAX_ROWS or cache.shape[0] != input_ids.shape[0] or cache.dtype != mx.int64:
        return self._stock_call(input_ids, start_pos, state, token_mask)
    batch, seqlen = input_ids.shape
    lay = self.layout
    n_layers, n_gram, n_heads = len(lay.layer_ids), lay.max_ngram_size, lay.n_heads
    lc = max(cache.shape[1], start_pos + seqlen)
    key = (n_layers, n_gram, n_heads, int(self.pad_id))
    hm = token_mask is not None
    msk = token_mask if hm else mx.zeros((1,), dtype=mx.bool_)
    cols = (n_gram - 1) * n_heads
    newh, out = _kernel(key)(
        inputs=[input_ids, msk, cache, mx.array([start_pos], dtype=mx.int32), self._token_map, self._multipliers, self._primes, self._offsets],
        template=[("HM", hm)], grid=(max(batch * lc, batch * seqlen * n_layers), 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(batch, lc), (batch, seqlen, n_layers, cols)], output_dtypes=[mx.int64, mx.int64])
    state.engram = newh
    return out


def install() -> bool:
    """Replace NgramHashState.__call__ (idempotent); False when mlx-vlm has no Engram module."""
    try:
        from . import archs as _AR; E = _AR.sub("engram")
    except Exception:
        return False
    cls = E.NgramHashState
    if not hasattr(cls, "_stock_call"):
        cls._stock_call = cls.__call__
        cls.__call__ = hash_call
    return True
