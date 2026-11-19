# SPDX-License-Identifier: Apache-2.0
"""Flag KYOJIN_FOREIGN_QSA_FUSE (default off): sparse attend over the QSA-selected blocks at decode, instead of mask build + dense masked SDPA.

A QSA full-attention layer past the indexer budget (qwen4_exp, 4K+ context) decodes as: indexer.build_mask(selection) (~25 small ops: scatter, arange, three
comparisons, where, put_along_axis ...) then a dense SDPA over the whole cache with that boolean mask. The stock code only uses its indexed kernel for prefill
(`allow_sparse_decode=False`: "the vector SDPA kernel remains faster for singleton decode"). Here ONE kernel attends the top-k blocks plus the tail block of the
query directly (online softmax, fp32), reading keys / values IN PLACE through their strides (no contiguous copy of the preallocated cache slice), no sort, no mask:
mask ops + SDPA (~25-30 launches) -> 1. Not bit-equal to the dense SDPA (different reduction order): gate by PPL / greedy tokens.

Scope (anything else runs the module's own `dispatch_qsa_attention`): S == 1, causal and no explicit mask (zero left padding, the `sparse_candidate` path),
bf16 / fp16 queries and cache on the GPU, head_dim % 32 == 0, any key / value strides (read through the kernel's own stride arguments). A module qualifies by structure (`indexer` with
block_topk / compress_ratio, and a language module exposing `dispatch_qsa_attention`), never by name. `ACTIVE` switches at run time (profile cell A/Bs in one process)."""
from __future__ import annotations

import os
import sys
from functools import lru_cache

import mlx.core as mx

ACTIVE = os.environ.get("KYOJIN_FOREIGN_QSA_FUSE", "0") not in ("0", "")
STATS = {"fast": 0, "stock": 0}

_SRC = r"""
    uint row_idx = threadgroup_position_in_grid.y;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int per_thread = D_SIZE / BD;
    typedef float U;
    thread U q[per_thread];
    thread U o[per_thread];
    threadgroup U outputs[BN * BD];
    threadgroup U max_scores[BN];
    threadgroup U sum_exp_scores[BN];

    int batch_idx = int(row_idx) / NUM_Q_HEADS;
    int q_head_idx = int(row_idx) - batch_idx * NUM_Q_HEADS;
    int kv_head_idx = q_head_idx / GQA_FACTOR;
    int query_end = int(query_ends[batch_idx]);

    const device T* qptr = queries + int(row_idx) * D_SIZE + int(simd_lid) * per_thread;
    device T* optr = out + int(row_idx) * D_SIZE + int(simd_gid) * per_thread;
    const device T* kbase = keys + batch_idx * keys_strides[0] + kv_head_idx * keys_strides[1] + int(simd_lid) * per_thread * keys_strides[3];
    const device T* vbase = values + batch_idx * values_strides[0] + kv_head_idx * values_strides[1] + int(simd_lid) * per_thread * values_strides[3];
    int ks3 = int(keys_strides[3]);
    int vs3 = int(values_strides[3]);
    int kstep = int(keys_strides[2]);
    int vstep = int(values_strides[2]);

    U s = U(scale[0]);
    for (int i = 0; i < per_thread; ++i) { q[i] = s * static_cast<U>(qptr[i]); o[i] = 0; }

    int blocks_offset = batch_idx * TOPK_BLOCKS;
    U max_score = -3.4028234663852886e38f;
    U sum_exp_score = 0;

    for (int selected_idx = int(simd_gid); selected_idx < SELECTED_LENGTH; selected_idx += BN) {
        int block_slot = selected_idx / BLOCK_SIZE;
        int block_offset = selected_idx - block_slot * BLOCK_SIZE;
        int block_idx = int(block_indices[blocks_offset + block_slot]);
        int key_pos = block_idx * BLOCK_SIZE + block_offset;
        bool valid = block_idx >= 0 && key_pos < query_end;
        U score = -3.4028234663852886e38f;
        if (valid) {
            const device T* kptr = kbase + key_pos * kstep;
            score = 0;
            for (int j = 0; j < per_thread; ++j) score += q[j] * static_cast<U>(kptr[j * ks3]);
            score = simd_sum(score);
        }
        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = valid ? fast::exp(score - new_max) : U(0);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        if (valid) {
            const device T* vptr = vbase + key_pos * vstep;
            for (int j = 0; j < per_thread; ++j) o[j] = o[j] * factor + exp_score * static_cast<U>(vptr[j * vs3]);
        } else {
            for (int j = 0; j < per_thread; ++j) o[j] *= factor;
        }
    }

    int tail_start = (query_end / BLOCK_SIZE) * BLOCK_SIZE;
    int tail_pos = tail_start + int(simd_gid);
    if (tail_pos < query_end) {
        const device T* kptr = kbase + tail_pos * kstep;
        U score = 0;
        for (int j = 0; j < per_thread; ++j) score += q[j] * static_cast<U>(kptr[j * ks3]);
        score = simd_sum(score);
        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        const device T* vptr = vbase + tail_pos * vstep;
        for (int j = 0; j < per_thread; ++j) o[j] = o[j] * factor + exp_score * static_cast<U>(vptr[j * vs3]);
    }

    if (simd_lid == 0) { max_scores[simd_gid] = max_score; sum_exp_scores[simd_gid] = sum_exp_score; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_score = max_scores[simd_lid];
    U new_max = simd_max(max_score);
    U factor = fast::exp(max_score - new_max);
    U total_sum = simd_sum(sum_exp_scores[simd_lid] * factor);
    for (int i = 0; i < per_thread; ++i) {
        outputs[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
        o[i] = total_sum == 0 ? U(0) : (o[i] / total_sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (simd_lid == 0) {
        for (int i = 0; i < per_thread; ++i) optr[i] = static_cast<T>(o[i]);
    }
"""


@lru_cache(maxsize=None)
def _kernel():
    return mx.fast.metal_kernel(name="kyojin_qsa_decode_attend", input_names=["queries", "keys", "values", "block_indices", "query_ends", "scale"], output_names=["out"],
                                header="#include <metal_simdgroup>\nusing namespace metal;\n", source=_SRC, ensure_row_contiguous=False)


@lru_cache(maxsize=64)
def _scale(v: float):
    return mx.array([v], dtype=mx.float32)


def decode_attend(queries, keys, values, block_indices, query_ends, scale, block_size):
    """queries (B, H, 1, D), keys / values (B, KH, T, D) (any batch / head stride), block_indices (B, 1, K) int32, query_ends (B, 1) -> (B, H, 1, D)."""
    B, H, _, D = queries.shape
    KH = keys.shape[1]
    K = block_indices.shape[-1]
    return _kernel()(
        inputs=[mx.contiguous(queries), keys, values, mx.contiguous(block_indices.astype(mx.int32)), mx.contiguous(query_ends.astype(mx.int32)), _scale(float(scale))],
        template=[("T", queries.dtype), ("D_SIZE", int(D)), ("NUM_Q_HEADS", int(H)), ("GQA_FACTOR", int(H // KH)), ("BLOCK_SIZE", int(block_size)), ("TOPK_BLOCKS", int(K)),
                  ("SELECTED_LENGTH", int(K * block_size))],
        grid=(1024, B * H, 1), threadgroup=(1024, 1, 1), output_shapes=[queries.shape], output_dtypes=[queries.dtype])[0]


def _eligible(queries, keys, values, block_indices, query_ends, block_size, causal, mask) -> bool:
    if not (ACTIVE and causal and mask is None and mx.default_device() == mx.gpu):
        return False
    if not all(isinstance(a, mx.array) for a in (queries, keys, values, block_indices, query_ends)):
        return False
    if queries.ndim != 4 or keys.ndim != 4 or queries.shape[2] != 1 or block_indices.ndim != 3 or block_indices.shape[1] != 1 or query_ends.ndim != 2:
        return False
    B, H, _, D = queries.shape
    KH = keys.shape[1]
    return (queries.dtype in (mx.bfloat16, mx.float16) and keys.dtype == queries.dtype and values.dtype == queries.dtype and values.shape == keys.shape
            and KH > 0 and H % KH == 0 and D % 32 == 0 and D == keys.shape[-1] and block_size > 0 and block_indices.shape[-1] >= 1
            and block_indices.shape[0] == B and keys.shape[0] == B and block_indices.shape[-1] * block_size >= 1)


def _wrap(stock):
    def dispatch(queries, keys, values, block_indices, query_ends, *, cache, scale, block_size, causal, mask, mask_factory, allow_sparse_decode=False):
        if _eligible(queries, keys, values, block_indices, query_ends, block_size, causal, mask):
            STATS["fast"] += 1
            return decode_attend(queries, keys, values, block_indices, query_ends, scale, block_size)
        STATS["stock"] += 1
        return stock(queries, keys, values, block_indices, query_ends, cache=cache, scale=scale, block_size=block_size, causal=causal, mask=mask, mask_factory=mask_factory,
                     allow_sparse_decode=allow_sparse_decode)
    dispatch._kyojin_qsa_fuse = True
    dispatch._stock = stock
    return dispatch


def qualifies(m) -> bool:
    ix = getattr(m, "indexer", None)
    return ix is not None and isinstance(getattr(ix, "block_topk", None), int) and isinstance(getattr(ix, "compress_ratio", None), int) and hasattr(ix, "build_mask")


def install(model, log=print) -> int:
    """Replace `dispatch_qsa_attention` once in the language module of every qualifying attention. Returns the number of attention modules armed (0 with the flag unset)."""
    if not ACTIVE:
        return 0
    n = 0
    done = set()
    for _, m in model.named_modules():
        if not qualifies(m):
            continue
        mod = sys.modules.get(type(m).__module__)
        fn = getattr(mod, "dispatch_qsa_attention", None)
        if fn is None:
            continue
        if mod not in done:
            if not getattr(fn, "_kyojin_qsa_fuse", False):
                mod.dispatch_qsa_attention = _wrap(fn)
            done.add(mod)
        n += 1
    log(f"[foreign] qsa_fuse: {n} QSA attention layers attend the selected blocks in one launch at decode")
    return n
