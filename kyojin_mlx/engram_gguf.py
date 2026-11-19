# SPDX-License-Identifier: Apache-2.0
"""N4: an Engram (per-layer n-gram table) read from a GGUF, keyed on the `<arch>.ple.*` / `<arch>.engram.*` keys and the tensor kinds, never on a model name.

GGUF layout (DS4.1 UD GGUFs, llama.cpp DeepSeek-V4 converter; Hy4-flash uses the same keys):
    per_layer_token_embd.weight   ONE table for every Engram layer, [rows_total, head_dim]; type id 43 (F8_E4M3 plain, or F8_E4M3_B32 = e8m0 scale +
                                  32 e4m3 a block, see foreign_gguf); read by row only
    <arch>.ple.layers             Engram layer ids                      <arch>.ple.ngram_size        max n-gram size (window = ngram_size tokens)
    <arch>.ple.heads_per_ngram    hash heads per n-gram size            <arch>.embedding_length_per_layer_input   head_dim (table row length)
    <arch>.ple.layer_multipliers  [n_layers x ngram_size] odd int64     <arch>.ple.head_vocab_sizes  [n_layers x (ngram_size - 1) x heads] primes
    <arch>.ple.head_offsets       [n_layers x (ngram_size - 1) x heads] first table row of every (layer, n, head) bucket range, in the ONE table
    <arch>.engram.token_map / pad_token_id; compressed_vocab_size / vocab_size when present (DS4.1 has them, Hy4-flash not: then checks only)
    blk.N.engram_gate.weight      [hc, dim] F32 = q_weight * k_weight of the safetensors Engram (the product the gate uses), shown equal in f32
    blk.N.engram_kv.weight        [dim * (hc + 1), n_cols * head_dim] the wkv projection (DS4.1 UD: Q8_0 from the fp8 32x32 blocks)

Row ids (the same arithmetic as kyojin_mlx engram_rows.RowMap on a safetensors pack, shown equal on DS4.1 windows):
    window = token_map[ids] at pos, pos-1, .., pos-(ngram_size-1) (the pad's compressed id before the start)
    row(l, n, h) = (xor_{k<n} window[k] * mult[l][k]) % prime[l][n-1][h] + head_offset[l][n-1][h]     (uint64, as llama.cpp engram_hash)
The safetensors table of layer l is the GGUF table rows [head_offset[l][0][0], head_offset[l + 1][0][0]): GGUF row = per-layer row + that base.

Forward (`engram_forward`, the served DS4.1 Engram: mlx_vlm deepseek_v41 Engram with kyojin_mlx rows.py `_engram_rows`):
    emb   = rows [L, n_cols * head_dim], fp8 round trip with ue8m0 scales a 32-block (act_fp8)
    kv    = emb @ kv_w.T ; key [L, hc, dim], value [L, dim]
    score = signed sqrt(max(|sum(x * gate_w * key) * rstd(x) * rstd(key) * dim^-0.5|, clamp)) ; out = x + sigmoid(score) * value
eps is the caller's: DS4.1 safetensors takes rms_norm_eps; Hy4's llama.cpp patch takes FLT_EPSILON (HY4_BUILD.md), not read from a key."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import foreign_gguf as GF

TABLE = "per_layer_token_embd.weight"


def _is_prime(n: int) -> bool:
    if n < 2 or n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


@dataclass(frozen=True)
class EngramGgufSpec:
    layer_ids: tuple
    ngram_size: int
    n_heads: int
    head_dim: int
    multipliers: np.ndarray          # [nl, ngram_size] int64
    primes: np.ndarray               # [nl, ngram_size - 1, n_heads] int64
    offsets: np.ndarray              # [nl, n_cols] int64, rows of the ONE table
    token_map: np.ndarray            # [vocab] int64 -> compressed id
    pad_id: int                      # compressed id of the pad token
    table_rows: int
    table_type: str

    @property
    def n_cols(self) -> int:
        return (self.ngram_size - 1) * self.n_heads

    def layer_base(self, layer_pos: int) -> int:
        return int(self.offsets[layer_pos, 0])

    def layer_rows(self, layer_pos: int) -> int:
        end = int(self.offsets[layer_pos + 1, 0]) if layer_pos + 1 < len(self.layer_ids) else self.table_rows
        return end - self.layer_base(layer_pos)

    @classmethod
    def from_gguf(cls, g: "GF.GgufFile", arch: str | None = None) -> "EngramGgufSpec | None":
        """None when the file has no Engram table. Every key is checked against the others and the table shape (a mismatch raises GgufError)."""
        if TABLE not in g.tensors:
            return None
        kv = g.kv
        a = arch or kv["general.architecture"]
        k = lambda s: kv[f"{a}.{s}"]
        lids = tuple(int(x) for x in k("ple.layers"))
        ng, nh, hd = int(k("ple.ngram_size")), int(k("ple.heads_per_ngram")), int(k("embedding_length_per_layer_input"))
        nl = len(lids)
        mult = np.asarray(k("ple.layer_multipliers"), np.int64).reshape(nl, ng)
        primes = np.asarray(k("ple.head_vocab_sizes"), np.int64).reshape(nl, ng - 1, nh)
        offs = np.asarray(k("ple.head_offsets"), np.int64).reshape(nl, (ng - 1) * nh)
        tmap = np.asarray(k("engram.token_map"), np.int64)
        # predicate keys: present (DS4.1) = checked against token_map / the primes; absent (Hy4-flash) = token_map's range, no prime floor
        cvs = kv.get(f"{a}.engram.compressed_vocab_size")
        cvs = int(tmap.max()) + 1 if cvs is None else int(cvs)
        vs = kv.get(f"{a}.engram.vocab_size")
        vs = -1 if vs is None else int(vs)
        pad = int(k("engram.pad_token_id"))
        ti = g.tensors[TABLE]
        rows, width = ti.shape
        flat = primes.reshape(-1)
        err = []
        if width != hd:
            err.append(f"table row {width} != head_dim {hd}")
        if not np.array_equal(offs.reshape(-1), np.concatenate([[0], np.cumsum(flat)[:-1]])):
            err.append("head_offsets are not the running sum of head_vocab_sizes")
        if int(flat.sum()) != rows:
            err.append(f"head_vocab_sizes sum {int(flat.sum())} != table rows {rows}")
        if int(tmap.max()) + 1 != cvs:
            err.append(f"token_map max + 1 = {int(tmap.max()) + 1} != compressed_vocab_size {cvs}")
        if not 0 <= pad < len(tmap):
            err.append(f"engram.pad_token_id {pad} outside token_map [0, {len(tmap)})")
        if not (mult % 2 == 1).all():
            err.append("a layer multiplier is even")
        if not all(p > vs and _is_prime(int(p)) for p in flat) or len(set(flat.tolist())) != len(flat):
            err.append("head_vocab_sizes are not distinct primes (above engram.vocab_size when the file has it)")
        if int(mult.max()) * max(cvs - 1, 1) > np.iinfo(np.uint64).max:          # the hash is uint64 (Hy4-flash: 8.5e13 * 120831 > int64 max)
            err.append("token * multiplier can overflow uint64")
        if err:
            raise GF.GgufError("Engram GGUF keys: " + "; ".join(err))
        return cls(lids, ng, nh, hd, mult, primes, offs, tmap, int(tmap[pad]), rows, ti.tname)


class GgufRowMap:
    """Token ids -> rows of the ONE table [positions, n_layers, n_cols] int64. `history` carries the compressed ids of earlier positions."""

    def __init__(self, spec: EngramGgufSpec):
        self.spec = spec
        self.history = np.zeros(0, np.int64)

    def reset(self) -> None:
        self.history = np.zeros(0, np.int64)

    def __call__(self, token_ids) -> np.ndarray:
        return self.rows_valid(token_ids)[0]

    def rows_valid(self, token_ids):
        """-> (rows [L, nl, n_cols] int64, valid [L] bool). A token id outside token_map (or negative) is DEAD, as llama.cpp engram_hash
        (hy-v4-flash.cpp:353-374): its own window slot takes the pad's compressed id and the position is invalid (no Engram delta); as an earlier
        token of a later window it is the pad too (history keeps the pad's id)."""
        s = self.spec
        ids = np.asarray(token_ids, np.int64).reshape(-1)
        dead = (ids < 0) | (ids >= len(s.token_map))
        cids = np.where(dead, s.pad_id, s.token_map[np.where(dead, 0, ids)])
        start = len(self.history)
        self.history = np.concatenate([self.history, cids])
        pos = np.arange(start, len(self.history))
        toks = np.stack([np.where(pos - j < 0, s.pad_id, self.history[np.maximum(pos - j, 0)]) for j in range(s.ngram_size)], -1)   # [L, ng]
        # uint64 as llama.cpp engram_hash (M:5211-5216): equal to int64 while the products fit int64 (DS4.1), right past it (Hy4-flash)
        prod = toks.astype(np.uint64)[:, None, :] * s.multipliers.astype(np.uint64)[None]                                        # [L, nl, ng]
        rolling, out = prod[..., 0], []
        for n in range(1, s.ngram_size):
            rolling = np.bitwise_xor(rolling, prod[..., n])
            out.append((rolling[..., None] % s.primes[:, n - 1].astype(np.uint64)[None]).astype(np.int64))
        return np.concatenate(out, -1) + s.offsets[None], ~dead


def table_loc(g: "GF.GgufFile", spec: EngramGgufSpec) -> list:
    """Per Engram layer: (path, absolute file offset of that layer's first row, (rows, head_dim), stored type, row bytes) -- the `loc[lid]["w"]`
    entries of engram_rows.EngramTables, all pointing into the one table, no "s" entry (the scale, if any, is inside the row)."""
    ti = g.tensors[TABLE]
    src = g._owner.get(TABLE, g)
    _, be, bb = GF.TYPES_BY_NAME[ti.tname]
    rb = spec.head_dim // be * bb
    base = src.data_start + ti.offset
    return [(src.path, base + spec.layer_base(i) * rb, (spec.layer_rows(i), spec.head_dim), ti.tname, rb) for i in range(len(spec.layer_ids))]


def read_rows(g: "GF.GgufFile", spec: EngramGgufSpec, rows) -> np.ndarray:
    """float32 [.., head_dim] of table rows (any shape of ids), decoded by row (F8_E4M3 / F8_E4M3_B32)."""
    r = np.asarray(rows, np.int64)
    raw = g.raw(TABLE)
    return GF.dequant(spec.table_type, raw[r.reshape(-1)], spec.head_dim).reshape(r.shape + (spec.head_dim,))


def layer_weights(g: "GF.GgufFile", layer_id: int) -> tuple[np.ndarray, np.ndarray]:
    """(gate_w [hc, dim] f32, kv_w [dim * (hc + 1), n_cols * head_dim] f32) of blk.<layer_id>."""
    return g.tensor_f32(f"blk.{layer_id}.engram_gate.weight"), g.tensor_f32(f"blk.{layer_id}.engram_kv.weight")


# the Hy4 form's extra tensors (hy-v4-flash.cpp:293-298): present = that delta of the forward fires; DS4.1 has none of them
HY4_KINDS = {"engram_kv_b": "kv_b", "engram_lookup_norm": "lookup_norm", "engram_pre_gamma": "pre_gamma", "engram_hc_scale": "hc_scale"}
FLT_EPSILON = float(np.finfo(np.float32).eps)          # Hy4's Engram norms (HY_V4_FLASH_ENGRAM_EPS), not rms_norm_eps


def layer_extras(g: "GF.GgufFile", layer_id: int) -> dict:
    """{kv_b [hc + 1, dim], lookup_norm [n_cols, head_dim], pre_gamma [dim], hc_scale [hc, dim]} f32, the ones blk.<layer_id> has (engram_forward kwargs)."""
    return {k: g.tensor_f32(f"blk.{layer_id}.{kind}.weight") for kind, k in HY4_KINDS.items() if f"blk.{layer_id}.{kind}.weight" in g.tensors}


def fp8_roundtrip(x, block: int = 32):
    """mlx: e4m3 round trip with a power-of-two scale a block (ue8m0: 2^ceil(log2(amax / 448)), amax floored at 1e-4)."""
    import mlx.core as mx
    sh = x.shape
    xb = x.astype(mx.float32).reshape(*sh[:-1], sh[-1] // block, block)
    amax = mx.maximum(mx.max(mx.abs(xb), axis=-1, keepdims=True), 1e-4)
    bits = (amax * mx.array(1.0 / 448.0, mx.float32)).view(mx.uint32)                      # 2^ceil(log2(amax * (1/448))) through the IEEE bits
    e = ((bits >> 23) & 0xFF).astype(mx.int32) - 127 + ((bits & 0x7FFFFF) != 0).astype(mx.int32)
    s = ((e + 127) << 23).astype(mx.uint32).view(mx.float32)
    q = mx.from_fp8(mx.to_fp8(mx.clip(xb / s, -448.0, 448.0)), mx.float32) * s
    return q.reshape(sh).astype(x.dtype)


DEVICE = False   # KYOJIN_ENGRAM_DEVICE (flags.engram_device): the Hy4 Engram projections stay f32 mx arrays on the device (EngramParams), read per call


def _f32(a):
    """f32 mx array; an mx array already in f32 is used as is (no host copy), anything else goes through mx.array as before."""
    import mlx.core as mx
    return a if isinstance(a, mx.array) and a.dtype == mx.float32 else mx.array(a, mx.float32)


def engram_forward(x, rows, gate_w, kv_w, eps: float, clamp: float = 1e-6, act_fp8: bool = True, kv_dtype=None,
                   kv_b=None, lookup_norm=None, pre_gamma=None, hc_scale=None, valid=None):
    """x [L, hc, dim]; rows [L, n_cols, head_dim]; gate_w [hc, dim]; kv_w [dim * (hc + 1), n_cols * head_dim]. -> (score [L, hc], out [L, hc, dim]),
    float32 math, `out` in x.dtype. kv_dtype: the dtype of the kv projection (None: float32; the served mlx_vlm Engram runs it in x.dtype).
    Hy4 form (llama.cpp build_engram, hy-v4-flash.cpp:457-518), each delta on when its tensor is given (all None = the DS4.1 form above, unchanged):
      lookup_norm [n_cols, head_dim]  each row rms-normed on its own (eps) times its column's weight (times valid, when given)
      kv_b [hc + 1, dim]              bias of the key / value projection
      pre_gamma [dim]                 the delta is rms(value) * pre_gamma, not the raw value
      hc_scale [hc, dim]              the gate is gate + hc_scale (per stream and channel)
      valid [L] (or [L, n_cols])      a position with no valid column adds nothing (the bias would otherwise leak through)
    Hy4 passes eps = FLT_EPSILON and act_fp8 = False (its rows are F8 as stored, no activation round trip)."""
    import mlx.core as mx
    L, hc, dim = x.shape
    if lookup_norm is not None:
        r = mx.array(rows).astype(mx.float32)
        r = mx.fast.rms_norm(r, None, eps) * _f32(lookup_norm)
        if valid is not None:
            vm = mx.array(np.asarray(valid, np.float32))
            r = r * (vm.reshape(L, -1, 1) if vm.ndim > 1 else vm.reshape(L, 1, 1))
        emb = r.astype(x.dtype).reshape(L, -1)
    else:
        emb = mx.array(rows).astype(x.dtype).reshape(L, -1)
    if act_fp8:
        emb = fp8_roundtrip(emb)
    kd = mx.float32 if kv_dtype is None else kv_dtype
    kv = (emb.astype(kd) @ _f32(kv_w).T.astype(kd)).astype(mx.float32)
    if kv_b is not None:
        kv = kv + _f32(kv_b).reshape(1, -1)
    key, value = kv[:, :hc * dim].reshape(L, hc, dim), kv[:, hc * dim:]
    h = x.astype(mx.float32)
    rstd = mx.rsqrt(mx.mean(h * h, -1) + eps) * mx.rsqrt(mx.mean(key * key, -1) + eps)
    dot = mx.sum(h * _f32(gate_w) * key, -1) * rstd * (dim ** -0.5)
    mag = mx.sqrt(mx.maximum(mx.abs(dot), clamp))
    score = mx.where(dot >= 0, mag, -mag)
    if pre_gamma is None and hc_scale is None and valid is None:
        out = (h + mx.sigmoid(score)[..., None] * value[:, None, :]).astype(x.dtype)
        return score, out
    sig = mx.fast.rms_norm(value, _f32(pre_gamma), eps) if pre_gamma is not None else value
    sc = mx.sigmoid(score)[..., None]
    if hc_scale is not None:
        sc = sc + _f32(hc_scale)[None]
    delta = sig[:, None, :] * sc
    if valid is not None:
        va = np.asarray(valid, np.float32).reshape(L, -1).sum(-1) > 0
        delta = delta * mx.array(va.astype(np.float32)).reshape(L, 1, 1)
    return score, (h + delta).astype(x.dtype)
