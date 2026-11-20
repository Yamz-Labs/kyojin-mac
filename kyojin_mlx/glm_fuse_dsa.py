# SPDX-License-Identifier: Apache-2.0
"""GLM-5.x DSA decode on 2..4 rows of ONE session (flag `KYOJIN_GLM_FUSE_DSA`, G10c): the verify round of speculative decode (MTP / lookup drafts).

`glm_dsa` (flag DSA_FAST) serves ONE query row only; every verify round (2..4 rows) took the stock code in every one of the 11 DSA layers:
  * the stock indexer: ~72 launches per layer and call (masks, argpartition, argsort, gathers, cumsum over the key-valid state ...),
  * the stock `_attend`: the lazily dequantized DENSE latent of the whole context (tq_dequant of every row + a 256 x 256 rotation, O(context) per layer and step) plus
    ~15 launches per row.
Here the rows run the kernels of the one-row path: ONE scoring launch for all rows (the `glm_dsa.scores` kernel with a row axis on the grid, same arithmetic per row),
ONE radix select + expand + tail launch for all rows (`glm_dsa` kernel with a row axis), ONE gather launch for every selected row of every query row; the attention of each
row is the one-row sequence (embed_q, SDPA over its own gathered rows, unembed_out) so that row i of a verify round equals the solo decode of row i bit for bit (the
exactness contract of rows.py). The verify round runs under `start_speculative_cache` (the pooling caches record the round's inputs; the rows path only calls their stock methods). Everything else (batches, padding, few pools, > MAX_ROWS rows) takes the stock code.

Also here: `late_bind` (rows._override resolves the wrapped method when called, not when installed): `rows.enable` runs before the first cache is created, and
`latent_tq.install_attend` (class-level, at the first `wrap_latent_layers`) never reached the instances rows had already overridden (`Glm5NextAttention._attend`
is a rows site), so the served decode kept the stock dense dequant."""
from __future__ import annotations

import os

import mlx.core as mx

from . import fired as FD
from . import glm_mla as MLA
from . import dsa_attend_rows as AR2

ENABLED = os.environ.get("KYOJIN_GLM_FUSE_DSA", os.environ.get("KYOJIN_GLM_DSA_FAST", "0")) == "1"      # follows KYOJIN_GLM_DSA_FAST (G3d: the lane sets that one); KYOJIN_GLM_FUSE_DSA=0 opts out

IDX_SCALE_FUSE = os.environ.get("KYOJIN_GLM_IDX_SCALE_FUSE", "0") == "1"      # BP-TAIL: drop the no-op astype(f32) on `weights`
                                                                               # (mlx_vlm's stock _project_queries already returns float32, scaled by
                                                                               # n_heads**-0.5) before this module's own `* self.softmax_scale`. Value-identical
                                                                               # (an astype to the array's own dtype changes nothing); flagged off by default
                                                                               # per RULES_GPU_MERGE rule 2 pending the GPU-gated before/after launch count.

MAX_ROWS = int(os.environ.get("KYOJIN_GLM_FUSE_DSA_ROWS", "4"))
EVAL = int(os.environ.get("KYOJIN_GLM_FUSE_DSA_EVAL", "0"))      # DSA2: 0 = off; N >= 1: on a verify pass (L > 1) async_eval the attention output of every Nth DSA layer (the stock `_attend` does it for every row of
                                                                  # every layer: the GPU starts while the host still builds the rest of the pass; the fused path builds the pass 4x faster, so without this the
                                                                  # host build (~6 ms) is serial in front of the ~30 ms of GPU work instead of hidden behind it: dec6 pass 31.4 -> 36.2 ms)
_ORD: dict = {}                                                   # attention module id -> its ordinal among the DSA layers of a pass (first-call order = layer order)
STATS = {"ix_rows": 0, "ix_stock": 0, "ix_batch": 0, "att_rows": 0, "att_stock": 0, "att_batch": 0}
MAX_POOLS = 32768                                                # select_expand_rows bound (cpt <= 32 chunks of 1024): more pools in a row -> stock path, decided before any cache write
WHY: dict = {}                                                   # reason -> count of the calls that took the stock indexer after the cheap checks (G3e: counted fallbacks)


def fallback(reason: str, site: str = "fuse_dsa_rows") -> None:
    STATS["ix_stock"] += 1
    WHY[reason] = WHY.get(reason, 0) + 1
    FD.fb(site, reason)


def keys_ok(k, pool_cache) -> bool:
    """fp16 / bf16 keys and pool (glm_dsa.keys_ok): known BEFORE the caches are updated; the MTP head's fp32 indexer takes the stock call (G3e)."""
    from . import glm_dsa as G
    return G.keys_ok(k, pool_cache)


_k: dict = {}
_c: dict = {}


def _arr(key, make):
    a = _c.get(key)
    if a is None:
        a = _c[key] = make()
        mx.eval(a)
    return a


# ------------------------------------------------------------------------------------------------------------------------------ kernels with a row axis
def _score_src():
    from . import dsa_decode as D
    s = D._SCORE
    s = s.replace("const uint tg = threadgroup_position_in_grid.x;", "const uint tg = threadgroup_position_in_grid.x;\n    const uint u = threadgroup_position_in_grid.y;", 1)
    s = s.replace("Qs[i] = q[i];", "Qs[i] = q[u * 4096 + i];").replace("Wh[tid] = float(w[tid]);", "Wh[tid] = float(w[u * 32 + tid]);")
    s = s.replace("const uint nn = n[0];", "const uint nn = n[u];").replace("out[", "out[u * OSTR + ")
    s = s.replace("km + ulong(", "km + ulong(u / ip[1]) * ip[2] + ulong(")                   # query row u = session * ip[1] + row reads the keys of its session
    s = s.replace("OSTR", "ip[0]")                                                          # runtime integers (ip): a template value that changes per call would recompile the kernel
    assert s.count("out[u * ip[0] + ") >= 3 and "n[u]" in s and "w[u * 32" in s and s.count("u / ip[1]") >= 2
    return s


def _selx_src():
    from . import glm_dsa as G
    s = G._SELX
    s = "    const uint u = threadgroup_position_in_grid.y;\n" + s
    s = s.replace("const uint nn = n[0];", "const uint nn = n[u];").replace("sc[i]", "sc[u * ip[0] + i]")
    s = s.replace("out[pos * RATIO + e] = int(i) * RATIO + e;", "out[u * OTOT + pos * RATIO + e] = meta[3 * u + 2] + int(i) * RATIO + e;")
    s = s.replace("out[Kk * RATIO + tid]", "out[u * OTOT + Kk * RATIO + tid]")
    s = s.replace("meta[1]", "meta[3 * u + 1]").replace("meta[0]", "meta[3 * u]")
    assert "n[u]" in s and "ip[0]" in s and "u * OTOT + pos" in s and "u * OTOT + Kk" in s and "meta[3 * u]" in s and "meta[3 * u + 2] + int(i)" in s
    return s


def scores_rows(q, w, keys, n, n_host: int, lrows: int = 0):
    """q [Q, 32, 128], w fp32 [Q, 32] (head weights x scale), keys [P, 128] (or [S, P, 128] with Q = S * lrows: query row u reads the keys of session u // lrows),
    n int32 [Q] (valid keys per row) -> fp32 [Q, ntg * per]: row u holds `glm_dsa.scores(q[u], w[u], keys[session], n[u:u+1])` (the same kernel text; the row is the
    second grid axis)."""
    from . import dsa_decode as D, glm_dsa as G
    dt = q.dtype
    key = ("score_rows", str(dt))
    k = _k.get(key)
    if k is None:
        k = _k[key] = mx.fast.metal_kernel(name="kyojin_glm_score_rows", input_names=["q", "w", "km", "n", "cand", "ip"], output_names=["out"], source=_score_src(), header=D._HEADER)
    nsg, tpt = G.SC_NSG, G.SC_TPT
    per = nsg * tpt * 8
    ntg = -(-n_host // per)
    Q = int(q.shape[0])
    ip = mx.array([ntg * per, lrows or Q, int(keys.shape[-2]) * 128 if keys.ndim == 3 and lrows else 0], dtype=mx.int32)
    return k(inputs=[q, w, keys, n, D._nomask(), ip], template=[("T", dt), ("NSG", nsg), ("TPT", tpt), ("GT", tpt), ("HASMASK", False), ("F32", True), ("OT", mx.float32)],
             grid=(ntg * nsg * 32, Q, 1), threadgroup=(nsg * 32, 1, 1), output_shapes=[(Q, ntg * per)], output_dtypes=[mx.float32])[0]


def select_expand_rows(sc, n, n_host: int, topk: int, ratio: int, meta):
    """`glm_dsa.select_expand` for every row of sc [Q, S]: n int32 [Q] valid scores, meta int32 [3 Q] = (tail base, tail count, offset added to every selected position) per row -> int32 [Q, topk * ratio + ratio - 1]."""
    from . import dsa_decode as D, glm_dsa as G
    cpt = 1
    while cpt * 1024 < n_host:
        cpt *= 2
    if cpt > 32 or n_host <= topk:
        return None
    key = ("selx_rows", cpt, topk, ratio)
    k = _k.get(key)
    if k is None:
        src = _selx_src().replace("CPT", str(cpt)).replace("RATIO", str(ratio)).replace("uint(K)", str(topk))
        k = _k[key] = mx.fast.metal_kernel(name="kyojin_glm_select_expand_rows", input_names=["sc", "n", "meta", "ip"], output_names=["out"], header=D._HEADER, source=src)
    Q, S = int(sc.shape[0]), int(sc.shape[1])
    total = topk * ratio + ratio - 1
    return k(inputs=[sc, n, meta, mx.array([S], dtype=mx.int32)], template=[("OTOT", total)], grid=(1024, Q, 1), threadgroup=(1024, 1, 1), output_shapes=[(Q, total)], output_dtypes=[mx.int32])[0]


# ------------------------------------------------------------------------------------------------------------------------------ the indexer
def rows_why(self, x, padding_mask, cache, pool_cache):
    """None when the solo-cache rows indexer serves this call, else the reason (flag audit key; the guards of `rows_ok`, same order)."""
    from mlx_vlm.models.cache import KVCache, PoolingCache
    if self.training:
        return "training"
    if padding_mask is not None:
        return "padding_mask"
    if cache is None or pool_cache is None:
        return "no_cache"
    if x.shape[0] != 1:
        return f"batch={x.shape[0]}"
    if not (2 <= x.shape[1] <= MAX_ROWS):
        return f"rows={x.shape[1]}"
    if type(cache) is not KVCache or type(pool_cache) is not PoolingCache:
        return "cache=" + type(cache).__name__ + "/" + type(pool_cache).__name__     # a speculating PoolingCache is served: accumulate_windows / update_and_fetch are the stock calls, they record the round's inputs
    if pool_cache.pooled is None or pool_cache.pooled.shape[1] * self.index_kpool + pool_cache.remainder != cache.offset:
        return "pools_not_1to1"
    if not (self.n_heads == 32 and self.head_dim == 128 and self.always_select_tail and self.index_kpool >= 2):
        return "indexer_dims"
    return None


def rows_ok(self, x, padding_mask, cache, pool_cache) -> bool:
    return rows_why(self, x, padding_mask, cache, pool_cache) is None


def indexer_rows(self, stock, x, q_resid, padding_mask=None, cache=None, pool_cache=None, offset=0):
    from . import glm_dsa as G
    why = "flag_off" if not (ENABLED and G.ENABLED) else rows_why(self, x, padding_mask, cache, pool_cache)
    if why is not None:
        STATS["ix_stock"] += 1
        FD.fb("fuse_dsa_rows", why)
        return stock(self, x, q_resid, padding_mask, cache, pool_cache, offset)
    Lr, r = int(x.shape[1]), self.index_kpool
    sel_k = self.index_topk // r
    kv_after = cache.offset + Lr
    n_rows = [(kv_after - Lr + i + 1) // r for i in range(Lr)]          # pools complete at row i's position (pool p ends at 4p + 3 <= position)
    if n_rows[0] <= sel_k:                                               # fewer candidates than slots in some row: stock bookkeeping
        STATS["ix_stock"] += 1
        FD.fb("fuse_dsa_rows", "few_pools")
        return stock(self, x, q_resid, padding_mask, cache, pool_cache, offset)
    if n_rows[-1] > MAX_POOLS:                                           # select kernel bound: decided BEFORE _project_keys / any cache or pool write
        fallback("pools>32768", "fuse_dsa_rows")
        return stock(self, x, q_resid, padding_mask, cache, pool_cache, offset)
    k, gate = self._project_keys(x)
    if not keys_ok(k, pool_cache):
        fallback("pool_dtype_" + str(k.dtype).split(".")[-1], "fuse_dsa_rows")
        return stock(self, x, q_resid, padding_mask, cache, pool_cache, offset)
    cache.update_and_fetch(_arr(("ones", Lr), lambda: mx.ones((1, 1, Lr, 1), dtype=mx.bool_)), _arr(("zero", Lr), lambda: mx.zeros((1, 1, Lr, 0), dtype=mx.bool_)))
    ready_k, ready_gate, _ = pool_cache.accumulate_windows(k, gate.astype(k.dtype), offset)
    if ready_k.shape[1] == r and ready_k.dtype in (mx.float16, mx.bfloat16) and ready_gate.dtype == ready_k.dtype and G.POOLC_ENABLED:
        new_pool = G.pool_compress(ready_k, ready_gate, self.index_kpool_compress_ape)
    else:
        new_pool = self._compress_pools(ready_k, ready_gate)
    pool_keys = pool_cache.update_and_fetch(new_pool)
    P = pool_keys.shape[1]
    if pool_keys.dtype not in (mx.float16, mx.bfloat16) or P < n_rows[-1] or P * r + pool_cache.remainder != kv_after:
        raise RuntimeError(f"glm_fuse_dsa: unexpected pool state after the cache update: pool {pool_keys.dtype} P {P} rows {n_rows} remainder {pool_cache.remainder}")
    q, weights = self._project_queries(x, q_resid)
    q = q.astype(pool_keys.dtype)
    w = weights[0].astype(mx.float32) * self.softmax_scale
    n = mx.array(n_rows, dtype=mx.int32)
    meta = mx.array([v for i in range(Lr) for v in (r * n_rows[i], kv_after - Lr + i + 1 - r * n_rows[i], 0)], dtype=mx.int32)
    sc = scores_rows(q[0], w, pool_keys[0], n, P)
    topk = select_expand_rows(sc, n, P, sel_k, r, meta)
    if topk is None:
        STATS["ix_stock"] += 1
        raise RuntimeError("glm_fuse_dsa: select kernel out of range after the cache update")      # cannot happen: n_rows[0] > sel_k and P is bounded by the caller's context
    STATS["ix_rows"] += 1
    FD.fire("fuse_dsa_rows")
    return topk[None, :, :self.index_topk + r - 1]


# ------------------------------------------------------------------------------------------------------------------------------ the attention of the rows
def kick(self, res, Lr: int):
    """DSA2: the stock `_attend` evaluates every row output of a multi-row pass asynchronously (graph-build / GPU overlap); same value, only the host wait moves."""
    if EVAL > 0 and Lr > 1:
        o = _ORD.setdefault(id(self), len(_ORD))
        if o % EVAL == 0:
            FD.fire("fuse_dsa_eval")
            mx.async_eval(res[0])
    return res


def attend_rows(self, orig, q, latent, new_latent, topk, kv_cache, projected_cache, last_only=False):
    from . import glm_dsa as G, latent_tq as LT, archs as AR
    L = AR.lang(AR.GLM5)
    batch, _, length, _ = q.shape
    if not (ENABLED and G.ENABLED and 2 <= length <= MAX_ROWS and batch == 1 and not self.training and isinstance(kv_cache, LT.LatentTQCache) and kv_cache.words is not None
            and topk is not None and topk.shape[0] == 1 and topk.shape[1] == length and topk.dtype == mx.int32 and G.GATHER_ENABLED and kv_cache.qbits == 6 and kv_cache.dim == 512
            and kv_cache.words.shape[:2] == (1, 1) and kv_cache.sk is not None and kv_cache.rk is not None and new_latent.dtype in (mx.float16, mx.bfloat16)
            and length <= L.DECODE_BLOCK_SIZE and G.plain_latent_attn(self)):
        STATS["att_stock"] += 1
        FD.fb("fuse_dsa_att", "guard:" + type(kv_cache).__name__)
        return orig(self, q, latent, new_latent, topk, kv_cache, projected_cache, last_only)
    STATS["att_rows"] += 1
    FD.fire("fuse_dsa_att_rows")
    start = length - 1 if last_only else 0
    K = int(topk.shape[-1])
    if AR2.usable(self, q, topk, kv_cache, self.embed_q.weight, self.unembed_out.weight):      # V2-VERIFY: scores / softmax / PV / sum launches with thousands of simdgroups (KYOJIN_DSA_ATTEND_ROWS)
        return kick(self, (AR2.attend_rows(self, q, topk, kv_cache, start), (topk[:, -1:] if last_only else topk)), length)
    if MLA.usable(self, q, topk, kv_cache, self.embed_q.weight, self.unembed_out.weight):      # G10d: embed_q, attention, merge + unembed_out in 4 launches for all rows
        FD.fire("glm_mla")
        return kick(self, (MLA.attend_rows(self, q, topk, kv_cache, start), (topk[:, -1:] if last_only else topk)), length)
    rows, vm = G.gather_rows(kv_cache, topk[0, start:].reshape(-1), new_latent.dtype)           # every selected row of every query row: one launch
    outs = []
    for i in range(start, length):
        j = i - start
        sel, valid = rows[j * K:(j + 1) * K][None, None], vm[j * K:(j + 1) * K][None, None]
        query = self.embed_q(q[:, :, i:i + 1])
        out = L.scaled_dot_product_attention(query, sel, sel, cache=kv_cache, scale=self.scale, mask=valid[:, None])
        outs.append(self.unembed_out(out))
    out = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)
    if last_only:
        topk = topk[:, -1:]
    return kick(self, (out.transpose(0, 2, 1, 3).reshape(batch, out.shape[2], -1), topk), length)


# ------------------------------------------------------------------------------------------------------------------------------ batch caches (the served step)
# The engine decodes on the model's own BATCH caches even for one session (rows.batch -> CacheList.merge: LatentBatch (dense latent), BatchKVCache (valid state),
# BatchPoolingCache): `glm_dsa._fast_ok` wants plain KVCache / PoolingCache, so the one-row kernels never ran in the served step. Here the same kernels serve B sessions
# of L rows (left padding included): every position of a session is known on the host (`_idx` of the valid cache, `_processed` / `_pool_lengths` of the pooling
# cache), so no array is read back.
MAX_SESS = int(os.environ.get("KYOJIN_GLM_FUSE_DSA_SESS", "8"))

_POOLCB = r"""
    const uint d = thread_position_in_grid.x;
    const uint bi = thread_position_in_grid.y;
    const device T* kk = keys + bi * KP * HD;
    const device T* gg = gate + bi * KP * HD;
    float lg[KP];
    float top = -INFINITY;
    for (int j = 0; j < KP; ++j) {
        lg[j] = float(gg[j * HD + d]) + ape[j * HD + d];
        top = metal::max(top, lg[j]);
    }
    float tot = 0.0f;
    for (int j = 0; j < KP; ++j) { lg[j] = metal::precise::exp(lg[j] - top); tot += lg[j]; }
    const float inv = 1.0f / tot;
    T acc = static_cast<T>(0.0f);
    for (int j = 0; j < KP; ++j) {
        const T pr = static_cast<T>(lg[j] * inv);
        const T prod = static_cast<T>(float(pr) * float(kk[j * HD + d]));
        acc = j == 0 ? prod : static_cast<T>(float(acc) + float(prod));
    }
    out[bi * HD + d] = acc;
"""


def pool_compress_b(keys, gate, ape):
    """`glm_dsa.pool_compress` for B windows: keys / gate [B, KP, HD] -> [B, 1, HD] (same arithmetic; rows whose window is not complete hold unused values)."""
    b, kp, hd = int(keys.shape[0]), int(keys.shape[1]), int(keys.shape[2])
    key = ("poolcb", str(keys.dtype), kp, hd)
    k = _k.get(key)
    if k is None:
        k = _k[key] = mx.fast.metal_kernel(name="kyojin_glm_pool_compress_b", input_names=["keys", "gate", "ape"], output_names=["out"], source=_POOLCB)
    return k(inputs=[keys, gate, ape], template=[("T", keys.dtype), ("KP", kp), ("HD", hd)], grid=(hd, b, 1), threadgroup=(hd, 1, 1), output_shapes=[(b, 1, hd)], output_dtypes=[keys.dtype])[0]


_DGATHER = r"""
    const uint t = thread_position_in_grid.x;
    const uint row = t / 64, c = t % 64;
    const int idx = topk[row];
    const uint bs = row / ROWSPER;
    const bool ok = (idx >= 0) && (idx < ip[0]);
    if (c == 0) { vm[row] = ok; }
    device T* o = out + (size_t)row * 512 + c * 8;
    if (ok) {
        const device T* sp = lat + ((size_t)bs * ip[1] + (size_t)idx) * 512 + c * 8;
        for (int e = 0; e < 8; ++e) { o[e] = sp[e]; }
    } else {
        for (int e = 0; e < 8; ++e) { o[e] = (T)0; }
    }
"""


def dense_gather(buf, topk, rowsper: int, nt: int):
    """The selected rows of a dense latent buffer [B, 1, CAP, 512] (one launch for every row of every session): topk int32 [B * rowsper] (position, -1 = none) ->
    (rows [B * rowsper, 512] in the buffer dtype, zeros where the position is invalid; valid bool [B * rowsper]: -1 <= position < nt)."""
    key = ("dgather", str(buf.dtype), rowsper)
    k = _k.get(key)
    if k is None:
        k = _k[key] = mx.fast.metal_kernel(name="kyojin_glm_dense_gather", input_names=["lat", "topk", "ip"], output_names=["out", "vm"], source=_DGATHER)
    n = int(topk.shape[0])
    return k(inputs=[buf, topk, mx.array([nt, int(buf.shape[2])], dtype=mx.int32)], template=[("T", buf.dtype), ("ROWSPER", rowsper)], grid=(n * 64, 1, 1), threadgroup=(64, 1, 1),
             output_shapes=[(n, 512), (n,)], output_dtypes=[buf.dtype, mx.bool_])


_ACC = r"""
    const uint d = thread_position_in_grid.x, j = thread_position_in_grid.y, b = thread_position_in_grid.z;
    const int r = rem[b];
    const int t = r + LR;
    const size_t bo = ((size_t)b * RATIO + j) * HD + d;
    const bool comp = t >= RATIO;
    T rk = (T)0; G rg = (G)0, ng = (G)0; T nk = (T)0;
    if (comp) {
        if ((int)j < r) { rk = bk[bo]; rg = bg[bo]; }
        else { const size_t s = ((size_t)b * LR + ((int)j - r)) * HD + d; rk = kv[s]; rg = gt[s]; }
        const int nr = t - RATIO;
        if ((int)j < nr) { const size_t s = ((size_t)b * LR + (LR - nr + (int)j)) * HD + d; nk = kv[s]; ng = gt[s]; }
    } else {
        if ((int)j < r) { nk = bk[bo]; ng = bg[bo]; }
        else if ((int)j - r < LR) { const size_t s = ((size_t)b * LR + ((int)j - r)) * HD + d; nk = kv[s]; ng = gt[s]; }
        else if (!ANYC) { nk = bk[bo]; ng = bg[bo]; }
    }
    onk[bo] = nk; ong[bo] = ng; ork[bo] = rk; org[bo] = rg;
"""


def accumulate_fast(pc, kv, gate):
    """`BatchPoolingCache.accumulate_windows` for a decode call (every row L <= ratio tokens, no left padding left in the cache, no ragged lengths): the stock code is
    ~15 slice updates / copies (a python loop over the rows); here ONE launch writes the new remainder buffers and the completed windows. Same arrays, same host state."""
    B, Lr, D = (int(v) for v in kv.shape)
    r = pc.ratio
    pc._record_speculative_inputs(kv, gate)
    rem = list(pc.remainder)
    anyc = any(x + Lr >= r for x in rem)
    key = ("acc", str(kv.dtype), str(gate.dtype), r, D, Lr, anyc)
    k = _k.get(key)
    if k is None:
        k = _k[key] = mx.fast.metal_kernel(name="kyojin_glm_pool_acc", input_names=["kv", "gt", "bk", "bg", "rem"], output_names=["onk", "ong", "ork", "org"], source=_ACC)
    onk, ong, ork, org = k(inputs=[kv, gate, pc.buf_kv, pc.buf_gate, mx.array(rem, dtype=mx.int32)],
                           template=[("T", kv.dtype), ("G", gate.dtype), ("RATIO", r), ("HD", D), ("LR", Lr), ("ANYC", anyc)], grid=(D, r, B), threadgroup=(D, 1, 1),
                           output_shapes=[(B, r, D)] * 4, output_dtypes=[kv.dtype, gate.dtype, kv.dtype, gate.dtype])
    for i in range(B):
        pc._processed[i] += Lr
    pc.buf_kv, pc.buf_gate = onk, ong
    pc.remainder = [(x + Lr) % r for x in rem]
    if not anyc:
        return mx.zeros((B, 0, D), dtype=kv.dtype), mx.zeros((B, 0, D), dtype=gate.dtype), 0
    return ork, org, 0


ACC_ENABLED = os.environ.get("KYOJIN_GLM_FUSE_DSA_ACC", "1") == "1"


def acc_ok(pc, kv, gate) -> bool:
    return (ACC_ENABLED and kv.shape[1] <= pc.ratio and pc.buf_kv.dtype == kv.dtype and pc.buf_gate.dtype == gate.dtype and all(p == 0 for p in pc.left_padding)
            and kv.dtype in (mx.float16, mx.bfloat16) and gate.dtype in (mx.float16, mx.bfloat16, mx.float32) and int(kv.shape[2]) <= 1024)


def batch_why(self, x, padding_mask, cache, pool_cache):
    """None when the batch-cache indexer serves this call, else the reason (flag audit key; the guards of `batch_ok`, same order)."""
    from mlx_vlm.models.cache import BatchKVCache, BatchPoolingCache
    if self.training:
        return "training"
    if padding_mask is not None:
        return "padding_mask"
    if cache is None or pool_cache is None:
        return "no_cache"
    B, Lr = int(x.shape[0]), int(x.shape[1])
    if not 1 <= Lr <= MAX_ROWS:
        return f"rows={Lr}"
    if B > MAX_SESS:
        return f"sessions>{MAX_SESS}"
    if type(cache) is not BatchKVCache or type(pool_cache) is not BatchPoolingCache:
        return "cache=" + type(cache).__name__ + "/" + type(pool_cache).__name__
    if pool_cache.pooled is None or pool_cache.buf_kv is None:
        return "pool_empty"
    if pool_cache.batch_size != B:
        return "batch_size"
    if any(v != 0 for v in pool_cache.left_padding):
        return "pool_left_padding"
    if any(v < 2 ** 30 for v in pool_cache._lengths):
        return "pool_lengths"
    if any(cache._idx - v < 0 for v in pool_cache._processed):
        return "processed"
    if not (self.n_heads == 32 and self.head_dim == 128 and self.always_select_tail and self.index_kpool >= 2):
        return "indexer_dims"
    return None


def batch_ok(self, x, padding_mask, cache, pool_cache) -> bool:
    return batch_why(self, x, padding_mask, cache, pool_cache) is None


def indexer_batch(self, stock, x, q_resid, padding_mask=None, cache=None, pool_cache=None, offset=0):
    from . import glm_dsa as G
    why = "flag_off" if not (ENABLED and G.ENABLED) else batch_why(self, x, padding_mask, cache, pool_cache)
    if why is not None:
        STATS["ix_stock"] += 1
        FD.fb("fuse_dsa_batch", why)
        return stock(self, x, q_resid, padding_mask, cache, pool_cache, offset)
    B, Lr = int(x.shape[0]), int(x.shape[1])
    r = self.index_kpool
    sel_k = self.index_topk // r
    # the positions after this call, from host state only: session b holds processed[b] tokens, the physical columns [E - processed[b], E) of every cache
    proc = [p + Lr for p in pool_cache._processed]
    E = cache._idx + Lr
    lp = [E - p for p in proc]
    n_rows = [[(proc[b] - Lr + i + 1) // r for i in range(Lr)] for b in range(B)]
    if min(min(v) for v in n_rows) <= sel_k:
        STATS["ix_stock"] += 1
        FD.fb("fuse_dsa_batch", "few_pools")
        return stock(self, x, q_resid, padding_mask, cache, pool_cache, offset)
    if max(max(v) for v in n_rows) > MAX_POOLS:                          # select kernel bound: decided BEFORE _project_keys / any cache or pool write
        fallback("pools>32768", "fuse_dsa_batch")
        return stock(self, x, q_resid, padding_mask, cache, pool_cache, offset)
    k, gate = self._project_keys(x)
    if not keys_ok(k, pool_cache):
        fallback("pool_dtype_" + str(k.dtype).split(".")[-1], "fuse_dsa_batch")
        return stock(self, x, q_resid, padding_mask, cache, pool_cache, offset)
    cache.update_and_fetch(_arr(("onesb", B, Lr), lambda: mx.ones((B, 1, Lr, 1), dtype=mx.bool_)), _arr(("zerob", B, Lr), lambda: mx.zeros((B, 1, Lr, 0), dtype=mx.bool_)))
    g2 = gate.astype(k.dtype)
    ready_k, ready_gate, _ = accumulate_fast(pool_cache, k, g2) if acc_ok(pool_cache, k, g2) else pool_cache.accumulate_windows(k, g2, offset)
    if ready_k.shape[1] == r and ready_k.dtype in (mx.float16, mx.bfloat16) and ready_gate.dtype == ready_k.dtype and G.POOLC_ENABLED:
        new_pool = pool_compress_b(ready_k, ready_gate, self.index_kpool_compress_ape)
    else:
        new_pool = self._compress_pools(ready_k, ready_gate)
    pool_keys = pool_cache.update_and_fetch(new_pool)
    if (pool_keys.dtype not in (mx.float16, mx.bfloat16) or list(pool_cache._processed) != proc or cache._idx != E
            or any(n_rows[b][i] > pool_cache._pool_lengths[b] for b in range(B) for i in range(Lr))):
        raise RuntimeError(f"glm_fuse_dsa: unexpected batch pooling state: processed {pool_cache._processed} vs {proc}, idx {cache._idx} vs {E}, pools {pool_cache._pool_lengths} rows {n_rows}")
    P = int(pool_keys.shape[1])
    q, weights = self._project_queries(x, q_resid)
    q = q.astype(pool_keys.dtype).reshape(B * Lr, 32, 128)
    if IDX_SCALE_FUSE:
        # `weights` is already float32 (mlx_vlm's stock _project_queries casts + scales by n_heads**-0.5 itself);
        # the `.astype(mx.float32)` below was a no-op cast on an already-fp32 array -- value-identical, one fewer graph node.
        w = weights.reshape(B * Lr, 32) * self.softmax_scale
    else:
        w = weights.reshape(B * Lr, 32).astype(mx.float32) * self.softmax_scale
    flat = [(b, i) for b in range(B) for i in range(Lr)]
    n = mx.array([n_rows[b][i] for b, i in flat], dtype=mx.int32)
    nmax = max(max(v) for v in n_rows)
    meta = mx.array([v for b, i in flat for v in (lp[b] + r * n_rows[b][i], proc[b] - Lr + i + 1 - r * n_rows[b][i], lp[b])], dtype=mx.int32)
    sc = scores_rows(q, w, pool_keys, n, nmax, lrows=Lr)
    topk = select_expand_rows(sc, n, nmax, sel_k, r, meta)
    if topk is None:
        raise RuntimeError("glm_fuse_dsa: select kernel out of range")
    STATS["ix_batch"] += 1
    FD.fire("fuse_dsa_batch")
    return topk.reshape(B, Lr, -1)[:, :, :self.index_topk + r - 1]


def attend_batch_ok(self, q, topk, kv_cache) -> bool:
    from mlx_vlm.models.cache import BatchKVCache
    from . import latent_tq as LT, archs as AR, glm_dsa as G
    L = AR.lang(AR.GLM5)
    B, _, Lr, _ = q.shape
    return bool(ENABLED and G.plain_latent_attn(self) and 1 <= Lr <= MAX_ROWS and B <= MAX_SESS and not self.training and type(kv_cache) in (LT.LatentBatch, BatchKVCache) and kv_cache.keys is not None
                and kv_cache.keys.shape[1] == 1 and kv_cache.keys.shape[3] == 512 and kv_cache.keys.dtype in (mx.float16, mx.bfloat16) and topk is not None
                and topk.shape[0] == B and topk.shape[1] == Lr and topk.dtype == mx.int32 and Lr <= L.DECODE_BLOCK_SIZE)


def attend_batch(self, orig, q, latent, new_latent, topk, kv_cache, projected_cache, last_only=False):
    """Dense latent (LatentBatch) of B sessions x L rows: ONE gather launch, then, per session and row, the SDPA of the solo call (rows.py: a user's bits do not depend
    on the batch)."""
    from . import archs as AR
    L = AR.lang(AR.GLM5)
    B, _, Lr, _ = q.shape
    if not attend_batch_ok(self, q, topk, kv_cache):
        STATS["att_stock"] += 1
        FD.fb("fuse_dsa_att", "guard:" + type(kv_cache).__name__)
        return orig(self, q, latent, new_latent, topk, kv_cache, projected_cache, last_only)
    STATS["att_batch"] += 1
    FD.fire("fuse_dsa_att_batch")
    start = Lr - 1 if last_only else 0
    nq = Lr - start
    K = int(topk.shape[-1])
    if AR2.usable(self, q, topk, kv_cache, self.embed_q.weight, self.unembed_out.weight):      # V2-VERIFY (KYOJIN_DSA_ATTEND_ROWS)
        return kick(self, (AR2.attend_batch(self, q, topk, kv_cache, start, last_only), (topk[:, -1:] if last_only else topk)), Lr)
    if MLA.usable(self, q, topk, kv_cache, self.embed_q.weight, self.unembed_out.weight):      # G10d
        FD.fire("glm_mla")
        return kick(self, (MLA.attend_batch(self, q, topk, kv_cache, start, last_only), (topk[:, -1:] if last_only else topk)), Lr)
    rows, vm = dense_gather(kv_cache.keys, topk[:, start:].reshape(-1), nq * K, int(kv_cache._idx))
    outs = []
    for j in range(nq):
        i = start + j
        query = self.embed_q(q[:, :, i:i + 1])
        per = []
        for b in range(B):
            o = (b * nq + j) * K
            sel, valid = rows[o:o + K][None, None], vm[o:o + K][None, None]
            per.append(L.scaled_dot_product_attention(query[b:b + 1], sel, sel, cache=kv_cache, scale=self.scale, mask=valid[:, None]))
        outs.append(self.unembed_out(per[0] if B == 1 else mx.concatenate(per, axis=0)))
    out = outs[0] if nq == 1 else mx.concatenate(outs, axis=2)
    if last_only:
        topk = topk[:, -1:]
    return kick(self, (out.transpose(0, 2, 1, 3).reshape(B, nq, -1), topk), Lr)


# ------------------------------------------------------------------------------------------------------------------------------ install
def attend_dispatch(self, prev, q, latent, new_latent, topk, kv_cache, projected_cache, last_only=False):
    if type(kv_cache).__name__ in ("LatentBatch", "BatchKVCache"):
        return attend_batch(self, lambda s_, *a, **k: prev(s_, *a, **k), q, latent, new_latent, topk, kv_cache, projected_cache, last_only)
    return attend_rows(self, lambda s_, *a, **k: prev(s_, *a, **k), q, latent, new_latent, topk, kv_cache, projected_cache, last_only)


def late_bind() -> None:
    """rows._override resolves the wrapped method of its class at call time (once per process)."""
    from . import rows as R
    if getattr(R, "_kyojin_late_bind", False):
        return

    def _override(mod, method, fn):
        cls = type(mod)

        def wrapper(self, *a, **k):
            return fn(self, getattr(cls, method), *a, **k)

        wrapper._kyojin_site = fn
        mod.__class__ = type(cls.__name__, (cls,), {method: wrapper, "__module__": cls.__module__})

    R._override_stock = R._override
    R._override = _override
    R._kyojin_late_bind = True
    site = R.SITES["mla_attend_rows"]

    def mla_attend_rows(self, orig, q, latent, new_latent, topk, kv_cache, projected_cache, last_only=False):
        if attend_batch_ok(self, q, topk, kv_cache):
            return attend_batch(self, None, q, latent, new_latent, topk, kv_cache, projected_cache, last_only)       # per-user SDPA of the solo shape, like the site
        return site(self, orig, q, latent, new_latent, topk, kv_cache, projected_cache, last_only)

    mla_attend_rows._kyojin_site = site
    R.SITES["mla_attend_rows"] = mla_attend_rows


def install() -> bool:
    """Class level, before rows.enable (patch_model time): the indexer call for 2..MAX_ROWS rows, `_attend` for them, and the late binding of rows sites."""
    if not ENABLED:
        return False
    from . import archs as _AR, glm_dsa as G
    L = _AR.lang(_AR.GLM5)
    late_bind()
    ix = L.Glm5NextIndexer
    if not getattr(ix, "_kyojin_fuse_dsa", False):
        prev = ix.__call__

        def call(self, x, q_resid, padding_mask=None, cache=None, pool_cache=None, offset=0):
            if type(cache).__name__ == "BatchKVCache":
                return indexer_batch(self, prev, x, q_resid, padding_mask, cache, pool_cache, offset)
            return indexer_rows(self, prev, x, q_resid, padding_mask, cache, pool_cache, offset)

        ix.__call__ = call
        ix._kyojin_fuse_dsa = True
    at = L.Glm5NextAttention
    if not getattr(at, "_kyojin_fuse_dsa", False):
        prev_a = at._attend

        def _attend(self, q, latent, new_latent, topk, kv_cache, projected_cache, last_only=False):
            return attend_dispatch(self, getattr(at, "_kyojin_prev_attend"), q, latent, new_latent, topk, kv_cache, projected_cache, last_only)

        at._kyojin_prev_attend = prev_a
        at._attend = _attend
        at._kyojin_fuse_dsa = True
    return True
