# SPDX-License-Identifier: Apache-2.0
"""Decode "rest" seams for 1..8 rows (batched users or chained verify rows) on MLX quantized Qwen3.5 / Qwen3-Next packs.

decfuse (decfuse.py) cuts the launches of the compiled one-row step. The same forward at 2..8 rows (served multi-user steps, chained verify) ran the stock
mlx-lm body: 249 QuantizedMatmul + 81 RMSNorm + 72 Add + 32 sigmoid-mul + per-attention-layer norm / rope / split copies on the 4B (590 dispatches vs 366).
This module gives the rows path the same seams, per sub-layer launch:
  1. residual add + RMSNorm in one launch for both seams of a layer (the add of a sub-layer output is carried to the next norm: `pend`), decfuse.add_rms;
  2. same-input projections in one GEMV (q|k|v, gate|up; the gated-delta mixer through gdn_pf), decfuse.FusedQ;
  3. attention front end in ONE launch: split q / gate, q and k RMSNorm, partial rotary at the cache offset, v / k / q written head-major ([B, H, L, D]) for the
     cache append (stock: split + 2 rms_norm + 2 rope + 3 layout copies + gate copy).
Every piece keeps the arithmetic and roundings of the op it replaces (rms_norm lane layout, rope float math, rounding to the activation dtype), so the logits are bit-identical
to the stock forward (tools/decrows_ab.py checks 1..8 rows x batched / chained). Anything else (more rows, masks with padding, pipeline, other caches) runs the stock forward.
KYOJIN_DECROWS=1 on (default off); KYOJIN_DECROWS_QK=0 / _ADDNORM=0 / _PROJ=0 switch one class off."""
from __future__ import annotations

import math
import os

import mlx.core as mx
import mlx.nn as nn

from . import decfuse as DF
from .depchain import chain as _dep_chain      # test-only slope probe (KYOJIN_TEST_DEP_CHAIN), identity when 0

ENABLED = os.environ.get("KYOJIN_DECROWS", "0") == "1"                  # opt-in: bit-identical, wall-clock neutral on the Air (see tools/decrows_ab.py), -64..-96 dispatches per step
QK = os.environ.get("KYOJIN_DECROWS_QK", "1") != "0"
ADDNORM = os.environ.get("KYOJIN_DECROWS_ADDNORM", "1") != "0"
MAX_ROWS = 8
FOREIGN_ROWS = int(os.environ.get("KYOJIN_DECFUSE_ADDNORM_ROWS", "128"))      # rows of a foreign-pack round that take the add + norm seams (decode, verify, multi-user)
MLP_ROWS = int(os.environ.get("KYOJIN_DECROWS_MLP_ROWS", "5"))      # fused gate|up GEMV: neutral up to 5 rows, -6..-20 us per layer slower at 6..8 (tools/decrows_seam_bench.py)
STATS = {"rows_fast": 0, "rows_fallback": 0, "qk_fused": 0, "qk_stock": 0}

# ---------------------------------------------------------------- attention front end: split + q/k RMSNorm + rope + head-major layout in one launch

_QK_SRC = """
    uint slot = threadgroup_position_in_grid.x;
    uint row = threadgroup_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    uint sl = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    uint ntg = threads_per_threadgroup.x;
    uint b = row / L, l = row % L;
    threadgroup float part[32];
    threadgroup T nb[D];
    const device T* src;
    uint kind;                                        // 0 q, 1 k, 2 v
    uint head;
    if (slot < HQ) { kind = 0; head = slot; src = y + row * NT + head * 2 * D; }
    else if (slot < HQ + HK) { kind = 1; head = slot - HQ; src = y + row * NT + COLK + head * D; }
    else { kind = 2; head = slot - HQ - HK; src = y + row * NT + COLV + head * D; }
    if (kind == 2) {
        device T* dst = vo + ((b * HK + head) * L + l) * D;
        for (uint i = lid; i < D; i += ntg) dst[i] = src[i];
        return;
    }
    if (kind == 0) {
        device T* g = gate + row * HQ * D + head * D;
        for (uint i = lid; i < D; i += ntg) g[i] = src[D + i];
    }
    const device T* w = kind == 0 ? wq : wk;
    if (sg == 0) part[sl] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc = 0.0f;
    for (uint i0 = lid * 4; i0 < D; i0 += ntg * 4) {
        for (uint i = 0; i < 4; i++) {
            if (i0 + i < D) {
                float f = (float)src[i0 + i];
                acc += f * f;
            }
        }
    }
    acc = simd_sum(acc);
    if (sl == 0) part[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float t = simd_sum(part[sl]);
        if (sl == 0) part[0] = metal::precise::rsqrt(t / (float)D + prm[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = part[0];
    for (uint i0 = lid * 4; i0 < D; i0 += ntg * 4) {
        for (uint i = 0; i < 4; i++) {
            if (i0 + i < D) nb[i0 + i] = w[i0 + i] * static_cast<T>((float)src[i0 + i] * inv);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device T* dst = (kind == 0 ? qo : ko) + (((b * (kind == 0 ? HQ : HK) + head) * L + l) * D);
    float pos = prm[2] * (float)(l + (uint)ofs[0]);
    for (uint p = lid; p < DR / 2; p += ntg) {
        float d = (float)p / (float)(DR / 2);
        float theta = pos * metal::exp2(-d * prm[1]);
        float c = ROPE_COS(theta), s = ROPE_SIN(theta);
        float x1 = (float)nb[p], x2 = (float)nb[p + DR / 2];
        dst[p] = static_cast<T>(x1 * c - x2 * s);
        dst[p + DR / 2] = static_cast<T>(x1 * s + x2 * c);
    }
    for (uint i = DR + lid; i < D; i += ntg) dst[i] = nb[i];
"""
_k = {}
_f = {}


def _qk_kernel(variant):
    k = _k.get(variant)
    if k is None:
        fast, contract = variant
        hdr = "#pragma METAL fp contract(%s)\n" % ("on" if contract else "off")
        hdr += "#define ROPE_COS %s\n#define ROPE_SIN %s\n" % (("metal::fast::cos", "metal::fast::sin") if fast else ("metal::precise::cos", "metal::precise::sin"))
        k = _k[variant] = mx.fast.metal_kernel(name="kyojin_qk_prep", input_names=["y", "wq", "wk", "prm", "ofs"], output_names=["qo", "ko", "vo", "gate"],
                                               source=_QK_SRC, header=hdr)
    return k


# (fast trig, fp contract on): fp contract on equals mx.fast.rope + rms_norm bit for bit (tests/test_decrows.py); contract off differs on 20 of 2.7M elements, fast vs precise trig does not matter
VARIANT = (True, True)


def qk_prep(y, wq, wk, eps, HQ, HK, D, DR, rope_base, rope_scale, offset, B, L):
    """y [B, L, HQ * 2D + 2 * HK * D] (the fused q|k|v projection) -> q [B, HQ, L, D], k [B, HK, L, D], v [B, HK, L, D], gate [B, L, HQ * D]."""
    R = B * L
    NT = y.shape[-1]
    tg = min(1024, 32 * (-(-D // 128)))
    prm = _f.get((eps, rope_base, rope_scale))
    if prm is None:
        prm = _f[(eps, rope_base, rope_scale)] = mx.array([eps, math.log2(rope_base), rope_scale], dtype=mx.float32)
    outs = _qk_kernel(VARIANT)(inputs=[y, wq, wk, prm, mx.array([offset], dtype=mx.int32)],
                               template=[("T", y.dtype), ("D", D), ("DR", DR), ("HQ", HQ), ("HK", HK), ("NT", NT), ("COLK", HQ * 2 * D), ("COLV", HQ * 2 * D + HK * D), ("L", L)],
                               grid=(tg * (HQ + 2 * HK), R, 1), threadgroup=(tg, 1, 1),
                               output_shapes=[(B, HQ, L, D), (B, HK, L, D), (B, HK, L, D), (B, L, HQ * D)], output_dtypes=[y.dtype] * 4)
    return outs


def _attn_ok(a, x, cache):
    return (ENABLED and QK and DF.ENABLED and DF.PROJ and getattr(a, "_kr_fused", None) is not None and x.dtype in (mx.float16, mx.bfloat16) and cache is not None
            and type(a.rope) is nn.RoPE and not a.rope.traditional and getattr(a, "_kr_qk", False) and isinstance(getattr(cache, "offset", None), int)
            and a.q_norm.weight.dtype == x.dtype and a.k_norm.weight.dtype == x.dtype and x.shape[0] * x.shape[1] <= MAX_ROWS)


def patch_attention():
    from mlx_lm.models import qwen3_next as Q3N
    cls = Q3N.Qwen3NextAttention
    if getattr(cls, "_kyojin_decrows", False):
        return False
    stock = cls.__call__

    def call(self, x, mask=None, cache=None):
        if not _attn_ok(self, x, cache):
            STATS["qk_stock"] += 1
            return stock(self, x, mask, cache)
        STATS["qk_fused"] += 1
        B, L, _ = x.shape
        y = self._kr_fused(x)
        q, k, v, gate = qk_prep(y, self.q_norm.weight, self.k_norm.weight, self.q_norm.eps, self.num_attention_heads, self.num_key_value_heads, self.head_dim,
                                self.rope.dims, self.rope.base, self.rope.scale, cache.offset, B, L)
        k, v = cache.update_and_fetch(k, v)
        o = Q3N.scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        o = o.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(o * mx.sigmoid(gate))

    cls.__call__ = call
    cls._kyojin_decrows = True
    return True


def prepare(tm):
    """Mark the attention layers whose q / k / v are one FusedQ (decfuse.prepare ran) and whose shapes the kernel supports."""
    n = 0
    for layer in tm.layers:
        a = getattr(layer, "self_attn", None)
        f = getattr(a, "_kr_fused", None) if a is not None else None
        if f is None or tuple(s for _, s in f.spans) != (a.num_attention_heads * a.head_dim * 2, a.num_key_value_heads * a.head_dim, a.num_key_value_heads * a.head_dim):
            continue
        if a.head_dim % 4 or a.head_dim > 1024 or a.rope.dims % 2 or a.rope.dims > a.head_dim:
            continue
        object.__setattr__(a, "_kr_qk", True)
        n += 1
    return n


# ---------------------------------------------------------------- MoE router: softmax(precise) + top-k + renormalise in one launch (Qwen3-Next / Qwen3.5-MoE SparseMoeBlock)
# Stock: softmax, argpartition (a stable merge sort of the row, top k = last k, ascending), take_along_axis, sum, divide = 6 launches. Here one threadgroup per row: softmax with the lane layout
# of mx.softmax's single-row kernel (4 contiguous values per lane, simd_max / simd_sum, 1 / sum then multiply), rank of every expert in the stable ascending order (ties: lower index first),
# the k last ranks written in ascending order (so the weighted sum over experts adds in the stock order), scores / sum(scores) in the activation dtype.
_ROUTER_SRC = """
    uint row = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint sl = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    uint ntg = threads_per_threadgroup.x;
    threadgroup float part[32];
    threadgroup T pr[E];
    threadgroup T sc[K];
    const device T* g = lg + row * E;
    float ld[4];
    float m = -INFINITY;
    for (uint i = 0; i < 4; i++) {
        uint e = lid * 4 + i;
        ld[i] = e < E ? (float)g[e] : -INFINITY;
        m = max(m, ld[i]);
    }
    m = simd_max(m);
    if (sg == 0) part[sl] = -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sl == 0) part[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = simd_max(part[sl]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float nrm = 0.0f;
    for (uint i = 0; i < 4; i++) { ld[i] = SM_EXP(ld[i] - m); nrm += ld[i]; }
    nrm = simd_sum(nrm);
    if (sg == 0) part[sl] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sl == 0) part[sg] = nrm;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    nrm = simd_sum(part[sl]);
    float inv = 1.0f / nrm;
    for (uint i = 0; i < 4; i++) {
        uint e = lid * 4 + i;
        if (e < E) pr[e] = static_cast<T>(ld[i] * inv);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lid; e < E; e += ntg) {
        T pe = pr[e];
        uint rank = 0;
        for (uint j = 0; j < E; j++) {
            T pj = pr[j];
            rank += (pj < pe || (pj == pe && j < e)) ? 1u : 0u;
        }
        if (rank >= E - K) {
            uint pos = rank - (E - K);
            inds[row * K + pos] = e;
            sc[pos] = pe;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < K) {
        if (NORM) {
            T s = static_cast<T>(0.0f);
            for (uint i = 0; i < K; i++) s = static_cast<T>((float)s + (float)sc[i]);      // MLX sums a bf16 / fp16 row sequentially in T (rounded after every add)
            scores[row * K + lid] = static_cast<T>((float)sc[lid] / (float)s);
        } else {
            scores[row * K + lid] = sc[lid];
        }
    }
"""
_rk = {}
ROUTER_EXP = "metal::fast::exp"


def router_topk(logits, k, norm):
    """logits [..., E] (bf16 / fp16) -> (inds uint32 [..., k], scores [..., k]) as softmax(precise) + argpartition(-k)[-k:] + take_along_axis (+ renormalise)."""
    E = logits.shape[-1]
    lead = logits.shape[:-1]
    R = logits.size // E
    key = ROUTER_EXP
    kern = _rk.get(key)
    if kern is None:
        kern = _rk[key] = mx.fast.metal_kernel(name="kyojin_router_topk", input_names=["lg"], output_names=["inds", "scores"], source=_ROUTER_SRC,
                                               header="#define SM_EXP %s\n" % ROUTER_EXP)
    tg = min(1024, 32 * (-(-E // 128)))
    inds, scores = kern(inputs=[logits.reshape(R, E)], template=[("T", logits.dtype), ("E", E), ("K", k), ("NORM", bool(norm))], grid=(tg * R, 1, 1), threadgroup=(tg, 1, 1),
                        output_shapes=[(R, k), (R, k)], output_dtypes=[mx.uint32, logits.dtype])
    return inds.reshape(*lead, k), scores.reshape(*lead, k)


def router_ok(logits, k):
    E = logits.shape[-1]
    return (ENABLED and logits.dtype in (mx.float16, mx.bfloat16) and E <= 1024 and 1 <= k <= 32 and k < E and logits.size // E <= MAX_ROWS)

def patch_moe():
    """SparseMoeBlock (Qwen3-Next / Qwen3.5-MoE) for 1..8 rows: router softmax + top-k + renormalise in one launch (bit-identical, tests/test_decrows.py)."""
    from mlx_lm.models import qwen3_next as Q3N
    cls = Q3N.Qwen3NextSparseMoeBlock
    if getattr(cls, "_kyojin_decrows", False):
        return False
    stock = cls.__call__

    def call(self, x):
        if not (ENABLED and self.sharding_group is None and x.dtype in (mx.float16, mx.bfloat16) and x.size // x.shape[-1] <= MAX_ROWS and self.num_experts <= 1024
                and 1 <= self.top_k < self.num_experts):
            return stock(self, x)
        inds, scores = router_topk(self.gate(x), self.top_k, self.norm_topk_prob)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
        return y + shared_y

    cls.__call__ = call
    cls._kyojin_decrows = True
    return True


# ---------------------------------------------------------------- model forward: add + norm seams carried across sub-layers


def _body(tm, stock, inputs, cache=None, input_embeddings=None):
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    rows = inputs.shape[0] * inputs.shape[1] if inputs.ndim == 2 else 0
    ok = ((ENABLED or tm._kr_foreign) and ADDNORM and DF.ENABLED and DF.ADDNORM and input_embeddings is None and cache is not None and 1 <= rows <= tm._kr_rows
          and len(cache) == len(tm.layers) and getattr(tm, "pipeline_size", 1) == 1)
    if not ok:
        STATS["rows_fallback"] += 1
        return stock(inputs, cache, input_embeddings=input_embeddings)
    STATS["rows_fast"] += 1
    if rows == 1 and tm._kr_foreign:
        from . import foreign_regions
        got = foreign_regions.forward(tm, inputs, cache)
        if got is not None:
            return got
    x = tm.embed_tokens(inputs)
    fa_mask = create_attention_mask(x, cache[tm.fa_idx]) if tm.fa_idx is not None else None
    ssm_mask = create_ssm_mask(x, cache[tm.ssm_idx]) if tm.ssm_idx is not None else None
    pend = None
    for layer, c in zip(tm.layers, cache):
        x, h = DF.norm_of(x, pend, layer.input_layernorm)
        r = _dep_chain(layer.linear_attn(h, ssm_mask, c) if layer.is_linear else layer.self_attn(h, fa_mask, c))
        x, h = DF.add_rms(x, r, layer.post_attention_layernorm)
        pend = DF.mlp(layer.mlp, h, defer=True) if rows <= MLP_ROWS else layer.mlp(h)
    return DF.norm_of(x, pend, tm.norm)[1]


def enable_foreign(model, max_rows=FOREIGN_ROWS):
    """Install the add + norm forward (1..max_rows rows) on a hybrid Qwen3.5 text model whose linears are not MLX affine (foreign GGUF / NVFP4 packs):
    one launch per seam, bit-equal to the stock pair, for decode, verify and multi-user rounds. Returns True when installed."""
    if not (DF.ENABLED and DF.ADDNORM and ADDNORM):
        return False
    lm = getattr(model, "language_model", model)
    tm = getattr(lm, "model", None)
    if tm is None or not hasattr(tm, "pipeline_layers") or not hasattr(tm.layers[0], "is_linear") or getattr(tm, "_kr_stock", None) is not None or getattr(tm, "pipeline_size", 1) != 1:
        return False
    cls = type(tm)
    tm._kr_stock = lambda *a, **k: cls.__call__(tm, *a, **k)
    tm._kr_parent, tm._kr_foreign, tm._kr_rows = None, True, max_rows
    tm.__class__ = type("Kd" + cls.__name__, (cls,), {"__call__": _forward})
    return True


def _forward(self, inputs, cache=None, input_embeddings=None):
    # one row with the compiled step regions keeps its priority (their own fallback is the rows body)
    if self._kr_parent is not None and inputs.ndim == 2 and inputs.shape == (1, 1):
        return self._kr_parent(self, inputs, cache, input_embeddings=input_embeddings)
    return _body(self, self._kr_stock, inputs, cache, input_embeddings)


def enable(model):
    """Install the rows forward on the text model of `model` (after step_regions.enable). Returns True when installed."""
    if not (ENABLED and DF.ENABLED):
        return False
    lm = getattr(model, "language_model", model)
    tm = getattr(lm, "model", None)
    if tm is None or not hasattr(tm, "pipeline_layers") or not hasattr(tm.layers[0], "is_linear") or getattr(tm, "_kr_stock", None) is not None:
        return False
    if getattr(tm, "pipeline_size", 1) != 1:
        return False
    DF.prepare(tm)
    prepare(tm)
    patch_attention()
    patch_moe()
    cls = type(tm)
    tm._kr_foreign, tm._kr_rows = False, MAX_ROWS
    regions = getattr(tm, "_kr_plan", None) is not None
    tm._kr_stock = tm._kr_orig if regions else (lambda *a, **k: cls.__call__(tm, *a, **k))      # the unmodified forward (reference of the bit tests)
    tm._kr_parent = cls.__call__ if regions else None
    if regions:
        tm._kr_orig = lambda inputs, cache=None, input_embeddings=None: _body(tm, tm._kr_stock, inputs, cache, input_embeddings)      # fallback of the step regions
    tm.__class__ = type("Kd" + cls.__name__, (cls,), {"__call__": _forward})
    return True
