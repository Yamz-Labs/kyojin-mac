# SPDX-License-Identifier: Apache-2.0
"""DS4.1 attention decode cut (one token, one user): the ~35 small ops around the attention core become 2 kernels per layer.

Stock `DeepseekV41Attention._attention` for one decoded token of a compressed layer runs (reuse layer, launches): rope of q (1), kv rope + fp8 round trip
+ ring write + history gather + concatenate + window index math (~6), the sparse pooled attention (gather, 2 x cast, 4 x matmul, 2 x logsumexp, logaddexp,
split softmax, adds: 22) and the inverse rope (1). Here:

  kv_ring   kv (post kv_norm) -> rope tail -> fp8 round trip (blocks of 32, ue8m0) -> row `start_pos % window` of a NEW ring (a copy; caches are values:
            forks share buffers). Bit-identical to the stock rope + `fake_quant_fp8_ue8m0` + scatter (tests/test_fuse_attn.py).
  attend    q rope (tail), scores against the ring and the pooled rows picked by the indexer top-k, softmax with the sink in the denominator, P @ V, inverse rope.
            One threadgroup per head. The rope arithmetic is the stock one (cos / sin come from `mx.fast.rope` itself: one probe per token and rope
            module, shared by all layers); the dot products and the softmax are fp32 with a different summation order than the stock matmuls (same
            level of rounding: the stock composite is itself not order-defined), so the result is checked by error against an fp64 reference.

The ring after the write holds exactly the keys of positions start_pos - W + 1 .. start_pos (slot p % W), so the stock "history ++ new" concatenation is not built:
the kernel reads the ring directly and masks the slots that no position has reached yet (start_pos < W - 1).
Switch: module attribute ENABLED (env KYOJIN_FUSE_ATTN=0). Anything outside the covered case (batch > 1, S > 1, no pooled rows yet, head dim, dtype) runs the stock code."""
from __future__ import annotations

from . import archs as _A
import os

import mlx.core as mx
from .modeldesc import KERNEL as KG
from . import fired as FD
from . import ablate as AZ
from . import cdecpack as CD
import numpy as np

ENABLED = os.environ.get("KYOJIN_FUSE_ATTN", "1") != "0"
SPLIT_PV = os.environ.get("KYOJIN_ATTEND_SPLIT", "1") != "0"           # P @ V split over key groups (fp32 sum order differs from the sequential loop)
SU = int(os.environ.get("KYOJIN_ATTEND_SU", "1"))                  # keys per simdgroup pass in the score loop of the split kernel
MMA = os.environ.get("KYOJIN_ATTEND_MMA", "1") != "0"                 # multi-query simdgroup-matrix decode attention (dsa_attend.py), fp16
KMAX = 1024                                                   # largest top-k the kernel holds in threadgroup memory

_HEADER = r"""
inline float k_log1p(float x) {
  float xp1 = 1.0f + x;
  if (xp1 == metal::numeric_limits<float>::max()) { return metal::numeric_limits<float>::max(); }
  if (xp1 == 1.0f) { return x; }
  return x * (metal::log(xp1) / (xp1 - 1.0f));
}
inline float k_logaddexp(float x, float y) {
  const float mx_ = metal::max(x, y), mn_ = metal::min(x, y);
  return (mn_ == -metal::numeric_limits<float>::infinity() || mx_ == metal::numeric_limits<float>::infinity()) ? mx_ : (mx_ + k_log1p(metal::exp(mn_ - mx_)));
}
"""

_HEADER += r"""
inline float trunc10(float x) { return as_type<float>(as_type<uint>(x) & 0xFFFFE000u); }     // operand of an fp32 matmul on the tensor units: 10 mantissa bits, rounded toward zero
"""

_KV_RING = r"""
    // grid W threadgroups x 64 threads; threadgroup b = ring slot b, thread t = dims 8 t .. 8 t + 7 (a block of 32 dims = 4 consecutive threads).
    const uint b = threadgroup_position_in_grid.x;
    const uint t = thread_position_in_threadgroup.x;
    const uint slot = uint(pos[0]) % uint(W);
    const uint o = b * D + t * 8;
    if (b != slot) {
        *(device uint4*)(out + o) = *(const device uint4*)(ring + o);
        return;
    }
    float v[8];
    for (uint i = 0; i < 8; ++i) { v[i] = float(kv[t * 8 + i]); }
    if (t * 8 >= uint(D - RD)) {                                  // rope tail: 4 interleaved pairs per thread (mx.fast.rope traditional=True)
        const uint p0 = (t * 8 - uint(D - RD)) / 2;
        for (uint j = 0; j < 4; ++j) {
            const float c = rt[2 * (p0 + j)], s = rt[2 * (p0 + j) + 1];
            const float x1 = v[2 * j], x2 = v[2 * j + 1];
            v[2 * j] = float(static_cast<T>(x1 * c - x2 * s));
            v[2 * j + 1] = float(static_cast<T>(x1 * s + x2 * c));
        }
    }
    float amax = 0.0f;
    for (uint i = 0; i < 8; ++i) { amax = metal::max(amax, metal::abs(v[i])); }
    amax = metal::max(amax, simd_shuffle_xor(amax, 1u));
    amax = metal::max(amax, simd_shuffle_xor(amax, 2u));
    amax = metal::max(amax, 1e-4f);
    const uint bits = as_type<uint>(amax * (1.0f / 448.0f));
    const uint exponent = (bits >> 23) + ((bits & 0x7fffffu) != 0);
    const float scale = as_type<float>(exponent << 23);
    for (uint i = 0; i < 8; ++i) {
        const float vv = metal::clamp(v[i] / scale, -448.0f, 448.0f);
        const uint vbits = as_type<uint>(metal::abs(vv));
        const uint qexp = metal::max(int(vbits >> 23) - 3, 118);
        const float step = as_type<float>(qexp << 23);
        out[o + i] = static_cast<T>((metal::rint(vv / step) * step) * scale);
    }
"""

_ATTEND = r"""
    // one threadgroup (D / 2 threads) per head. Keys: the W ring slots, then the K pooled rows chosen by idx (-1 = none).
    const uint h = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    constexpr uint NT = D / 2;
    constexpr uint NSG = NT / 32;
    constexpr uint DPL = D / 32;                                   // dims per lane in the score phase
    const uint K = idx_shape[0];
    const uint N = pool_shape[0];
    const uint NK = uint(W) + K;
    const int posv = pos[0];
    const float inf = metal::numeric_limits<float>::infinity();
    threadgroup float sc[W + KMAX];
    threadgroup float wt[W + KMAX];
    threadgroup float red[NSG];

    float qf[DPL];
    {
        const device T* qh = q + ulong(h) * D + lane * DPL;
        for (uint i = 0; i < DPL; ++i) { qf[i] = float(qh[i]); }
        if (lane * DPL >= uint(D - RD)) {                          // q rope (mx.fast.rope traditional=True), rounded to T like the stock op
            const uint p0 = (lane * DPL - uint(D - RD)) / 2;
            for (uint j = 0; j < DPL / 2; ++j) {
                const float c = rt[2 * (p0 + j)], s = rt[2 * (p0 + j) + 1];
                const float x1 = qf[2 * j], x2 = qf[2 * j + 1];
                qf[2 * j] = float(static_cast<T>(x1 * c - x2 * s));
                qf[2 * j + 1] = float(static_cast<T>(x1 * s + x2 * c));
            }
        }
        for (uint i = 0; i < DPL; ++i) { qf[i] = qf[i] * SCALE; if (TRUNC) { qf[i] = trunc10(qf[i]); } }
    }
    // scores
    for (uint j = sg; j < NK; j += NSG) {
        const device T* row = ring;
        bool valid;
        if (j < uint(W)) {
            valid = (posv >= int(W) - 1) || (int(j) <= posv);
            row = ring + ulong(j) * D;
        } else {
            const int ii = idx[j - uint(W)];
            valid = (ii >= 0) && (uint(ii) < N);
            row = pool + ulong(valid ? ii : 0) * D;
        }
        float s = 0.0f;
        if (valid) {
            const device T* r = row + lane * DPL;
            float acc = 0.0f;
            for (uint i = 0; i < DPL; ++i) { acc = fma(qf[i], float(r[i]), acc); }
            s = simd_sum(acc);
        }
        if (lane == 0) { sc[j] = valid ? s : -inf; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // softmax with the sink in the denominator: w_j = exp(s_j - logaddexp(logsumexp(s), sink))
    float m = -inf;
    for (uint j = tid; j < NK; j += NT) { m = metal::max(m, sc[j]); }
    m = simd_max(m);
    if (lane == 0) { red[sg] = m; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = red[0];
    for (uint g = 1; g < NSG; ++g) { m = metal::max(m, red[g]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0f;
    for (uint j = tid; j < NK; j += NT) { sum += metal::exp(sc[j] - m); }
    sum = simd_sum(sum);
    if (lane == 0) { red[sg] = sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float S = 0.0f;
    for (uint g = 0; g < NSG; ++g) { S += red[g]; }
    const float nrm = k_logaddexp(m + metal::log(S), sink[h]);
    for (uint j = tid; j < NK; j += NT) { wt[j] = (sc[j] == -inf) ? 0.0f : metal::exp(sc[j] - nrm); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // P @ V: thread tid owns dims 2 tid, 2 tid + 1
    float a0 = 0.0f, a1 = 0.0f;
    for (uint j = 0; j < NK; ++j) {
        const float w = TRUNC ? trunc10(wt[j]) : wt[j];
        if (wt[j] != 0.0f) {
            const device T* row = (j < uint(W)) ? (ring + ulong(j) * D) : (pool + ulong(idx[j - uint(W)]) * D);
            a0 += w * float(row[2 * tid]);
            a1 += w * float(row[2 * tid + 1]);
        }
    }
    float x1 = float(static_cast<T>(a0)), x2 = float(static_cast<T>(a1));
    if (tid >= uint(D - RD) / 2) {                                 // inverse rope on the tail (the table holds cos / sin of the negated angles)
        const uint p = tid - uint(D - RD) / 2;
        const float c = it[2 * p], s = it[2 * p + 1];
        const float y1 = float(static_cast<T>(x1 * c - x2 * s));
        const float y2 = float(static_cast<T>(x1 * s + x2 * c));
        x1 = y1; x2 = y2;
    }
    out[ulong(h) * D + 2 * tid] = static_cast<T>(x1);
    out[ulong(h) * D + 2 * tid + 1] = static_cast<T>(x2);
"""

_ATTEND_SCORES_OLD = _ATTEND[_ATTEND.index("    // scores\n"):_ATTEND.index("    threadgroup_barrier(mem_flags::mem_threadgroup);\n    // softmax")]
_ATTEND_SU = r"""    // scores, SU keys per simdgroup pass (independent row loads in flight; same per-key arithmetic as the sequential loop)
    for (uint j0 = sg; j0 < NK; j0 += SU * NSG) {
        float xs[SU][DPL];
        bool vld[SU];
        for (uint u = 0; u < SU; ++u) {
            const uint j = j0 + u * NSG;
            const device T* row = ring;
            bool valid = false;
            if (j < NK) {
                if (j < uint(W)) {
                    valid = (posv >= int(W) - 1) || (int(j) <= posv);
                    row = ring + ulong(j) * D;
                } else {
                    const int ii = idx[j - uint(W)];
                    valid = (ii >= 0) && (uint(ii) < N);
                    row = pool + ulong(valid ? ii : 0) * D;
                }
            }
            vld[u] = valid;
            if (valid) {
                const device T* r = row + lane * DPL;
                for (uint i = 0; i < DPL; ++i) { xs[u][i] = float(r[i]); }
            }
        }
        for (uint u = 0; u < SU; ++u) {
            const uint j = j0 + u * NSG;
            float s = 0.0f;
            if (vld[u]) {
                float acc = 0.0f;
                for (uint i = 0; i < DPL; ++i) { acc = fma(qf[i], xs[u][i], acc); }
                s = simd_sum(acc);
            }
            if (j < NK && lane == 0) { sc[j] = vld[u] ? s : -inf; }
        }
    }
"""
_ATTEND_SPLIT = _ATTEND.replace(_ATTEND_SCORES_OLD, _ATTEND_SU).split("    // P @ V: thread tid owns dims 2 tid, 2 tid + 1")[0] + r"""    // P @ V split over the keys: NKG key groups of D / 8 threads, thread (g, d) owns dims 8 d .. 8 d + 7 of the keys j = g, g + NKG, ...; 16-byte row reads
    constexpr uint NKG = NT / (D / 8);
    const uint kg = tid / (D / 8), d8 = tid % (D / 8);
    float acc[8];
    for (uint i = 0; i < 8; ++i) { acc[i] = 0.0f; }
    for (uint j = kg; j < NK; j += NKG) {
        const float wj = wt[j];
        if (wj != 0.0f) {
            const float w = TRUNC ? trunc10(wj) : wj;
            const device T* row = (j < uint(W)) ? (ring + ulong(j) * D) : (pool + ulong(idx[j - uint(W)]) * D);
            const uint4 raw = *(const device uint4*)(row + d8 * 8);
            const thread T* rv = (const thread T*)&raw;
            for (uint i = 0; i < 8; ++i) { acc[i] += w * float(rv[i]); }
        }
    }
    threadgroup float part[NKG * D];
    for (uint i = 0; i < 8; ++i) { part[kg * D + d8 * 8 + i] = acc[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kg == 0) {
        float v[8];
        for (uint i = 0; i < 8; ++i) {
            float t = 0.0f;
            for (uint g = 0; g < NKG; ++g) { t += part[g * D + d8 * 8 + i]; }
            v[i] = float(static_cast<T>(t));
        }
        for (uint i = 0; i < 8; i += 2) {
            const uint dim = d8 * 8 + i;
            float x1 = v[i], x2 = v[i + 1];
            if (dim >= uint(D - RD)) {                             // inverse rope on the tail (the table holds cos / sin of the negated angles)
                const uint p = (dim - uint(D - RD)) / 2;
                const float c = it[2 * p], s = it[2 * p + 1];
                const float y1 = float(static_cast<T>(x1 * c - x2 * s));
                const float y2 = float(static_cast<T>(x1 * s + x2 * c));
                x1 = y1; x2 = y2;
            }
            out[ulong(h) * D + dim] = static_cast<T>(x1);
            out[ulong(h) * D + dim + 1] = static_cast<T>(x2);
        }
    }
"""

_k: dict = {}


def kv_ring(kv: mx.array, ring: mx.array, rt: mx.array, pos: mx.array, rd: int) -> mx.array:
    """kv [1, 1, D] (post kv_norm), ring [1, W, D], rt = interleaved cos / sin of the position [rd] fp32, pos int32 [1] -> new ring [1, W, D]."""
    _, W, D = ring.shape
    k = _k.get("kv")
    if k is None:
        k = _k["kv"] = mx.fast.metal_kernel(name="kyojin_attn_kv_ring", input_names=["kv", "ring", "rt", "pos"], output_names=["out"], source=_KV_RING)
    assert D % 64 == 0 and D // 8 == 64, "kv_ring: one threadgroup of 64 threads per ring row (head dim 512)"
    return k(inputs=[kv, ring, rt, pos], template=[("T", ring.dtype), ("W", W), ("D", D), ("RD", rd)], grid=(W * 64, 1, 1), threadgroup=(64, 1, 1),
             output_shapes=[(1, W, D)], output_dtypes=[ring.dtype])[0]


_trunc: list = []


def tensor_unit_trunc() -> bool:
    """True when this machine's fp32 matmul rounds its operands toward zero to 10 mantissa bits (M5 tensor units, MLX 0.32: measured error against the
    emulation 1e-7, against exact fp32 4e-4). The stock attention matmuls then have that systematic rounding and the kernel reproduces it, so it agrees
    with the stock path up to the accumulation order. Probed once with a random matmul."""
    if not _trunc:
        rng = np.random.default_rng(0)
        a = rng.standard_normal((16, 512)).astype(np.float32)
        b = np.array(mx.array(rng.standard_normal((512, 64)).astype(np.float32)).astype(mx.bfloat16).astype(mx.float32))
        got = np.array(mx.array(a) @ mx.array(b)).astype(np.float64)
        emu = (a.view(np.uint32) & np.uint32(0xFFFFE000)).view(np.float32).astype(np.float64) @ b.astype(np.float64)
        exact = a.astype(np.float64) @ b.astype(np.float64)
        n = np.linalg.norm(exact)
        _trunc.append(bool(np.linalg.norm(got - emu) / n < 5e-6 < np.linalg.norm(got - exact) / n))
    return _trunc[0]


def attend(q: mx.array, ring: mx.array, pool: mx.array, idx: mx.array, sink: mx.array, rt: mx.array, it: mx.array, pos: mx.array, heads: int, rd: int,
           scale: float) -> mx.array:
    """q [1, 1, H * D] (post wq_b, before rope), ring [1, W, D] (after the write), pool [1, N, D], idx int32 [K], sink fp32 [H], rt / it = rope tables of the
    position (forward / inverse) [rd] -> o [1, H, 1, D] in q.dtype (attention output after the inverse rope)."""
    _, W, D = ring.shape
    if MMA:
        from . import dsa_attend
        if hasattr(pool, "codes"):
            if dsa_attend.usable(q, ring, heads):
                return dsa_attend.attend_packed(q, ring, pool.codes, pool.scales, idx, sink, rt, it, pos, heads, rd, scale, tensor_unit_trunc())
            if CD.ENABLED:
                pool, idx = CD.compact(pool, idx)                       # cdecpack: the K selected rows only, no dense mirror of the layer
            else:
                pool = pool.dense()
        if dsa_attend.usable(q, ring, heads):
            return dsa_attend.attend(q, ring, pool, idx, sink, rt, it, pos, heads, rd, scale, tensor_unit_trunc())
    sc = float(np.float32(scale))
    key = ("attend", repr(sc), SPLIT_PV)
    k = _k.get(key)
    if k is None:
        k = _k[key] = mx.fast.metal_kernel(name="kyojin_attn_attend", input_names=["q", "ring", "pool", "idx", "sink", "rt", "it", "pos"], output_names=["out"],
                                           source=(_ATTEND_SPLIT if SPLIT_PV else _ATTEND).replace("SCALE", f"{sc!r}f"), header=_HEADER)
    return k(inputs=[q, ring, pool.reshape(-1, D), idx.reshape(-1), sink, rt, it, pos],
             template=[("T", q.dtype), ("W", W), ("D", D), ("RD", rd), ("KMAX", KMAX), ("TRUNC", tensor_unit_trunc()), ("SU", SU)], grid=(heads * (D // 2), 1, 1), threadgroup=(D // 2, 1, 1),
             output_shapes=[(1, heads, 1, D)], output_dtypes=[q.dtype])[0]


# --------------------------------------------------------------------------------------------------------------- n rows at once (chained verify)
# A reuse layer of a chained verify of n rows runs, per row, kv_ring (a new ring) + attend: 2 n launches. Here: kv_proc (the n new kv rows through rope tail +
# fp8 round trip, one launch), attend_rows (n x heads threadgroups; the keys of row j are the base ring with the slots of rows 0 .. j replaced by their
# processed kv, read through the index math instead of through n rings) and, deferred, kv_apply: the n rings of the rows in ONE launch, evaluated only when
# a later step reads the ring of the accepted row (the rings of rejected rows are never evaluated). Same arithmetic as kv_ring / attend per element (the
# index lists are padded with -1, which adds exp(-inf) = 0 to a sum and skips the P @ V term: bit-identical).
_KV_PROC = _KV_RING.split("    const uint slot = uint(pos[0]) % uint(W);", 1)[1].split("    float v[8];", 1)
assert len(_KV_PROC) == 2
_KV_PROC = r"""
    // grid n threadgroups x 64 threads; threadgroup b = row b (its own rope table), thread t = dims 8 t .. 8 t + 7.
    const uint b = threadgroup_position_in_grid.x;
    const uint t = thread_position_in_threadgroup.x;
    const device float* rt = rt_all + b * RD;
    const uint o = b * D + t * 8;
    float v[8];""" + _KV_PROC[1]
_KV_PROC = _KV_PROC.replace("kv[t * 8 + i]", "kv[o + i]")
assert "kv[o + i]" in _KV_PROC

_KV_APPLY = r"""
    // grid (n * W) threadgroups x 64 threads; threadgroup (j, b): slot b of the ring of row j = processed kv of row i if slot b is the slot of row i <= j, else the base ring.
    const uint tg = threadgroup_position_in_grid.x;
    const uint t = thread_position_in_threadgroup.x;
    const uint j = tg / uint(W);
    const uint b = tg % uint(W);
    const uint i = (b + uint(W) - uint(pos[0]) % uint(W)) % uint(W);
    const device T* src = (i <= j) ? (kvp + ulong(i) * D) : (ring + ulong(b) * D);
    *(device uint4*)(out + ulong(tg) * D + t * 8) = *(const device uint4*)(src + t * 8);
"""

_ATTEND_ROWS = _ATTEND
for old, new_ in (
    ("    const uint h = threadgroup_position_in_grid.x;\n",
     "    const uint tgi = threadgroup_position_in_grid.x;\n    const uint h = tgi % uint(H);\n    const uint r = tgi / uint(H);\n"),
    ("    const uint K = idx_shape[0];\n", "    const uint K = idx_shape[1];\n    const device int* idr = idx + ulong(r) * K;\n    const device float* rtr = rt + ulong(r) * RD;\n    const device float* itr = it + ulong(r) * RD;\n"),
    ("    const int posv = pos[0];\n", "    const int posv = pos[r];\n    const uint pr = uint(pos[0]) % uint(W);\n"),
    ("const device T* qh = q + ulong(h) * D + lane * DPL;", "const device T* qh = q + (ulong(r) * H + h) * D + lane * DPL;"),
    ("const float c = rt[2 * (p0 + j)], s = rt[2 * (p0 + j) + 1];", "const float c = rtr[2 * (p0 + j)], s = rtr[2 * (p0 + j) + 1];"),
    ("            row = ring + ulong(j) * D;\n",
     "            const uint si = (j + uint(W) - pr) % uint(W);\n            row = (si <= r) ? (kvp + ulong(si) * D) : (ring + ulong(j) * D);\n"),
    ("const int ii = idx[j - uint(W)];", "const int ii = idr[j - uint(W)];"),
    ("            const device T* row = (j < uint(W)) ? (ring + ulong(j) * D) : (pool + ulong(idx[j - uint(W)]) * D);",
     "            const uint si = (j + uint(W) - pr) % uint(W);\n"
     "            const device T* row = (j < uint(W)) ? ((si <= r) ? (kvp + ulong(si) * D) : (ring + ulong(j) * D)) : (pool + ulong(idr[j - uint(W)]) * D);"),
    ("const float c = it[2 * p], s = it[2 * p + 1];", "const float c = itr[2 * p], s = itr[2 * p + 1];"),
    ("    out[ulong(h) * D + 2 * tid] = static_cast<T>(x1);\n    out[ulong(h) * D + 2 * tid + 1] = static_cast<T>(x2);",
     "    out[(ulong(r) * H + h) * D + 2 * tid] = static_cast<T>(x1);\n    out[(ulong(r) * H + h) * D + 2 * tid + 1] = static_cast<T>(x2);"),
):
    assert _ATTEND_ROWS.count(old) == 1, old
    _ATTEND_ROWS = _ATTEND_ROWS.replace(old, new_)
assert "rt[" not in _ATTEND_ROWS and " it[" not in _ATTEND_ROWS and "idx[" not in _ATTEND_ROWS

# --------------------------------------------------------------------------------------------------------------- n rows of a pass, several sessions
# The same kernel for the rows of a served pass (kyojin_mlx.dsv4_core.RowsCache: counts[s] consecutive positions of session s, laid out session after
# session). Row r reads the ring and the pool of sess[r] (never a copy of them: `ring` [S, W, D], `pool` [S, PN, D]) and the kv rows its own session's
# earlier rows wrote, kvp[first[r] .. r]; a slot of the base ring that no row of this pass wrote is read from ring[sess[r]]. `fst[r]` = the first row
# of r's session in the pass, so `i = fst[r] + (slot - pos[r]) mod W` is the row that wrote slot j and `i <= r` is the pass's own write.
_ATTEND_ROWS_M = _ATTEND_ROWS
for old, new_ in (
    ("    const device float* itr = it + ulong(r) * RD;\n",
     "    const ulong ridx = ulong(r) * K, rrop = ulong(r) * RD;\n"          # indexed, not pointer arithmetic: MLX may bind a small array as constant
     "    const uint PN = uint(POOLN);\n    const uint st0 = uint(fst[r]);\n"
     "    const device T* ringr = ring + ulong(sess[r]) * uint(W) * D;\n    const device T* poolr = pool + ulong(sess[r]) * PN * D;\n"),
    ("    const device int* idr = idx + ulong(r) * K;\n    const device float* rtr = rt + ulong(r) * RD;\n", ""),
    ("    const uint pr = uint(pos[0]) % uint(W);\n", "    const uint pr = uint(pos[st0]) % uint(W);\n"),
    ("const float c = rtr[2 * (p0 + j)], s = rtr[2 * (p0 + j) + 1];", "const float c = rt[rrop + 2 * (p0 + j)], s = rt[rrop + 2 * (p0 + j) + 1];"),
    ("const float c = itr[2 * p], s = itr[2 * p + 1];", "const float c = it[rrop + 2 * p], s = it[rrop + 2 * p + 1];"),
    ("const int ii = idr[j - uint(W)];", "const int ii = idx[ridx + (j - uint(W))];"),
    ("            const uint si = (j + uint(W) - pr) % uint(W);\n            row = (si <= r) ? (kvp + ulong(si) * D) : (ring + ulong(j) * D);\n",
     "            const uint i = st0 + (j + uint(W) - pr) % uint(W);\n            row = (i <= r) ? (kvp + ulong(i) * D) : (ringr + ulong(j) * D);\n"),
    ("            valid = (ii >= 0) && (uint(ii) < N);\n            row = pool + ulong(valid ? ii : 0) * D;\n",
     "            valid = (ii >= 0) && (uint(ii) < PN);\n            row = poolr + ulong(valid ? ii : 0) * D;\n"),
    ("            const uint si = (j + uint(W) - pr) % uint(W);\n"
     "            const device T* row = (j < uint(W)) ? ((si <= r) ? (kvp + ulong(si) * D) : (ring + ulong(j) * D)) : (pool + ulong(idr[j - uint(W)]) * D);",
     "            const uint i = st0 + (j + uint(W) - pr) % uint(W);\n"
     "            const device T* row = (j < uint(W)) ? ((i <= r) ? (kvp + ulong(i) * D) : (ringr + ulong(j) * D)) : (poolr + ulong(idx[ridx + (j - uint(W))]) * D);"),
):
    assert _ATTEND_ROWS_M.count(old) == 1, old
    _ATTEND_ROWS_M = _ATTEND_ROWS_M.replace(old, new_)
assert "ring + ulong(j)" not in _ATTEND_ROWS_M and "kvp + ulong(si)" not in _ATTEND_ROWS_M
assert "idr[" not in _ATTEND_ROWS_M and "rtr[" not in _ATTEND_ROWS_M and "itr[" not in _ATTEND_ROWS_M

# The rings of the rows of a pass over the base ring of each row's session: same index math as _KV_APPLY with the session offset and the session's
# first row of the pass. Row j = the ring of sess[j] after the writes of the rows first[j] .. j (the ring of a row is what its session keeps when the
# pass commits that row; a rejected row's ring is never written into the state).
_KV_APPLY_M = r"""
    // grid (n * W) threadgroups x 64 threads; threadgroup (j, b): slot b of the ring of row j's session.
    const uint tg = threadgroup_position_in_grid.x;
    const uint t = thread_position_in_threadgroup.x;
    const uint j = tg / uint(W);
    const uint b = tg % uint(W);
    const uint ss = sess[j];
    const int i = int(fst[j]) + int((b + uint(W) - uint(pos[fst[j]]) % uint(W)) % uint(W));
    const device T* src = (i <= int(j)) ? (kvp + ulong(i) * D) : (ring + (ulong(ss) * uint(W) + b) * D);
    *(device uint4*)(out + ulong(tg) * D + t * 8) = *(const device uint4*)(src + t * 8);
"""


def kv_proc(kv: mx.array, rt_all: mx.array, rd: int) -> mx.array:
    """kv [n, D] (post kv_norm), rt_all [n, rd] rope tables of the n positions -> processed kv rows [n, D] (rope tail + fp8 round trip, as kv_ring writes them)."""
    n, D = kv.shape
    k = _k.get("kvp")
    if k is None:
        k = _k["kvp"] = mx.fast.metal_kernel(name="kyojin_attn_kv_proc", input_names=["kv", "rt_all"], output_names=["out"], source=_KV_PROC)
    assert D // 8 == 64
    return k(inputs=[kv, rt_all], template=[("T", kv.dtype), ("D", D), ("RD", rd)], grid=(n * 64, 1, 1), threadgroup=(64, 1, 1),
             output_shapes=[(n, D)], output_dtypes=[kv.dtype])[0]


def kv_apply(ring: mx.array, kvp: mx.array, pos: mx.array) -> mx.array:
    """ring [1, W, D] (before the n rows), kvp [n, D] (kv_proc), pos int32 [n] (positions) -> rings [n, W, D]: row j = the ring after the writes of rows 0 .. j."""
    _, W, D = ring.shape
    n = kvp.shape[0]
    k = _k.get("kva")
    if k is None:
        k = _k["kva"] = mx.fast.metal_kernel(name="kyojin_attn_kv_apply", input_names=["ring", "kvp", "pos"], output_names=["out"], source=_KV_APPLY)
    return k(inputs=[ring, kvp, pos], template=[("T", ring.dtype), ("W", W), ("D", D)], grid=(n * W * 64, 1, 1), threadgroup=(64, 1, 1),
             output_shapes=[(n, W, D)], output_dtypes=[ring.dtype])[0]


def attend_rows(q: mx.array, ring: mx.array, kvp: mx.array, pool: mx.array, idx: mx.array, sink: mx.array, rt_all: mx.array, it_all: mx.array, pos: mx.array,
                heads: int, rd: int, scale: float) -> mx.array:
    """q [n, H * D] (post wq_b, before rope), ring [1, W, D] (before the n rows), kvp [n, D], pool [1, N, D] (the longest of the rows; each row reads its own
    top-k), idx int32 [n, K] (-1 = none), rope tables [n, rd], pos int32 [n] -> o [n, H, D] in q.dtype (attention output after the inverse rope)."""
    _, W, D = ring.shape
    n = q.shape[0]
    sc = float(np.float32(scale))
    key = ("attend_rows", repr(sc))
    k = _k.get(key)
    if k is None:
        k = _k[key] = mx.fast.metal_kernel(name="kyojin_attn_attend_rows", input_names=["q", "ring", "kvp", "pool", "idx", "sink", "rt", "it", "pos"],
                                           output_names=["out"], source=_ATTEND_ROWS.replace("SCALE", f"{sc!r}f"), header=_HEADER)
    return k(inputs=[q, ring, kvp, pool.reshape(-1, D), idx, sink, rt_all, it_all, pos],
             template=[("T", q.dtype), ("W", W), ("D", D), ("RD", rd), ("KMAX", KMAX), ("TRUNC", tensor_unit_trunc()), ("H", heads)],
             grid=(n * heads * (D // 2), 1, 1), threadgroup=(D // 2, 1, 1), output_shapes=[(n, heads, D)], output_dtypes=[q.dtype])[0]


def kv_apply_m(ring: mx.array, kvp: mx.array, sess: mx.array, fst: mx.array, pos: mx.array) -> mx.array:
    """ring [S, W, D] (the committed rings of the sessions, before the pass), kvp [n, D], sess / fst / pos int32 [n] -> rings [n, W, D]: row j = the
    ring of sess[j] after the writes of rows first[j] .. j (sessions of one pass in one tensor; the state's own buffers are never written)."""
    S, W, D = ring.shape
    n = kvp.shape[0]
    k = _k.get("kvam")
    if k is None:
        k = _k["kvam"] = mx.fast.metal_kernel(name="kyojin_attn_kv_apply_m", input_names=["ring", "kvp", "sess", "fst", "pos"], output_names=["out"],
                                              source=_KV_APPLY_M)
    assert D // 8 == 64
    return k(inputs=[ring, kvp, sess, fst, pos], template=[("T", ring.dtype), ("W", W), ("D", D)], grid=(n * W * 64, 1, 1), threadgroup=(64, 1, 1),
             output_shapes=[(n, W, D)], output_dtypes=[ring.dtype])[0]


def attend_rows_m(q: mx.array, ring: mx.array, kvp: mx.array, pool: mx.array, idx: mx.array, sink: mx.array, rt_all: mx.array, it_all: mx.array,
                  pos: mx.array, sess: mx.array, fst: mx.array, heads: int, rd: int, scale: float) -> mx.array:
    """As attend_rows for the rows of a pass over S sessions: ring [S, W, D], pool [S, PN, D], sess / fst / pos int32 [n] (fst = the first row of the
    row's session in the pass) -> o [n, H, D]. Same per-row arithmetic as attend_rows; the ring and the pool of a session are read, not copied."""
    _, W, D = ring.shape
    n = q.shape[0]
    PN = pool.shape[1]
    sc = float(np.float32(scale))
    key = ("attend_rows_m", repr(sc))
    k = _k.get(key)
    if k is None:
        k = _k[key] = mx.fast.metal_kernel(name="kyojin_attn_attend_rows_m",
                                           input_names=["q", "ring", "kvp", "pool", "idx", "sink", "rt", "it", "pos", "sess", "fst"],
                                           output_names=["out"], source=_ATTEND_ROWS_M.replace("SCALE", f"{sc!r}f"), header=_HEADER)
    return k(inputs=[q, ring, kvp, pool.reshape(-1, D), idx, sink, rt_all, it_all, pos, sess, fst],
             template=[("T", q.dtype), ("W", W), ("D", D), ("RD", rd), ("KMAX", KMAX), ("TRUNC", tensor_unit_trunc()), ("H", heads), ("POOLN", PN)],
             grid=(n * heads * (D // 2), 1, 1), threadgroup=(D // 2, 1, 1), output_shapes=[(n, heads, D)], output_dtypes=[q.dtype])[0]


# --------------------------------------------------------------------------------------------------------------- per token shared inputs
_tab: dict = {}
_pos: dict = {}
_KEEP = 64                                                   # entries kept per table: a chained verify asks for several positions per layer, a solo token for one


def rope_tables(rope, start_pos: int) -> tuple[mx.array, mx.array]:
    """(forward, inverse) interleaved cos / sin of `start_pos` for the rope tail of a DeepseekV4RoPE, taken from `mx.fast.rope` itself on an
    fp32 probe [1, 0, 1, 0, ...] (y1 = 1 * cos - 0 * sin = cos, y2 = 1 * sin + 0 * cos = sin: exactly the factors the stock op uses). Built once per
    position for all layers that share the same frequencies (a chained verify of n rows asks for n positions in every layer: a one-entry table would
    rebuild them, 2 rope launches + the numpy probe, for every row of every layer)."""
    fkey = rope.__dict__.get("_kfa_key")
    if fkey is None:
        fr = np.array(rope._freqs).astype(np.float32)
        fkey = rope.__dict__["_kfa_key"] = (rope.dims, rope.freq_scale, fr.tobytes())
    key = (fkey, start_pos)
    e = _tab.get(key)
    if e is not None:
        return e
    d = rope.dims
    probe = mx.array(np.tile(np.array([1.0, 0.0], np.float32), d // 2)).reshape(1, 1, 1, d)
    fwd = mx.fast.rope(probe, d, traditional=True, base=None, scale=1.0, offset=start_pos, freqs=rope._get_freqs(d, False)).reshape(d)
    inv = mx.fast.rope(probe, d, traditional=True, base=None, scale=1.0, offset=start_pos, freqs=rope._get_freqs(d, True)).reshape(d)
    if len(_tab) >= _KEEP:
        _tab.pop(next(iter(_tab)))
    _tab[key] = (fwd, inv)
    return fwd, inv


def pos_array(start_pos: int) -> mx.array:
    e = _pos.get(start_pos)
    if e is None:
        if len(_pos) >= _KEEP:
            _pos.pop(next(iter(_pos)))
        e = _pos[start_pos] = mx.array([start_pos], dtype=mx.int32)
    return e


_rtab: dict = {}
_prng: dict = {}
_probe: dict = {}


def rope_range(rope, p0: int, n: int) -> tuple[mx.array, mx.array]:
    """(forward, inverse) tables [n, dims] of the n consecutive positions p0 .. p0 + n - 1: one `mx.fast.rope` per table on a probe with the n positions along
    the sequence axis (the same cos / sin the one-position probe of `rope_tables` gives)."""
    rope_tables(rope, p0)                                                   # primes the frequency key
    fkey = rope.__dict__["_kfa_key"]
    key = (fkey, p0, n)
    e = _rtab.get(key)
    if e is not None:
        return e
    d = rope.dims
    pr = _probe.get((d, n))
    if pr is None:
        pr = _probe[(d, n)] = mx.array(np.tile(np.array([1.0, 0.0], np.float32), (n, d // 2))).reshape(1, 1, n, d)
    fwd = mx.fast.rope(pr, d, traditional=True, base=None, scale=1.0, offset=p0, freqs=rope._get_freqs(d, False)).reshape(n, d)
    inv = mx.fast.rope(pr, d, traditional=True, base=None, scale=1.0, offset=p0, freqs=rope._get_freqs(d, True)).reshape(n, d)
    if len(_rtab) >= _KEEP:
        _rtab.pop(next(iter(_rtab)))
    _rtab[key] = (fwd, inv)
    return fwd, inv


def pos_range(p0: int, n: int) -> mx.array:
    e = _prng.get((p0, n))
    if e is None:
        if len(_prng) >= _KEEP:
            _prng.pop(next(iter(_prng)))
        e = _prng[(p0, n)] = mx.array(list(range(p0, p0 + n)), dtype=mx.int32)
    return e


# --------------------------------------------------------------------------------------------------------------- wiring
def decode_ok(self, cache, batch: int, seq: int, dtype) -> bool:
    """True when the fused core applies to this attention call: one user, one token, a ring buffer of the cache, bf16 / fp16, a compressing layer
    that already holds a pooled row."""
    ratio = self.compress_ratio
    ring = cache.window[self.layer_idx]
    return bool(ENABLED and ratio and batch == 1 and seq == 1 and ring is not None and ring.shape == (1, self.window_size, self.head_dim)
                and dtype in (mx.float16, mx.bfloat16) and ring.dtype == dtype and self.head_dim == KG.HEAD_DIM and (cache.offset + 1) // ratio > 0)


def core(self, L, cache, x, qr, q, kv):
    """Attention after the q / kv projections of one token: the compressor and indexer of the layer (stock modules), the ring write and the attention
    kernel. Returns (o, None) with o [1, G, 1, (H / G) * D] to feed wo_a, or (None, out) with the stock tail's output (after wo_b) in the rare cases."""
    if AZ.ACTIVE and AZ.on("attn"):                                           # ablate: zeros of the core's output, ring / pool / compressor untouched
        AZ.hit("attn")
        return AZ.zeros_like_shape((1, self.o_groups, 1, (self.n_heads // self.o_groups) * self.head_dim), x.dtype), None
    start_pos = cache.offset
    comp = getattr(cache, "compress", None)
    lazy = PACKED_ATTEND and MMA and hasattr(comp, "lazy") and type(comp).BLOCK == 16
    if lazy:
        comp.lazy = True                                                      # a long fp4-packed pool layer comes back as a PackedView: no unpack of the whole pool per layer and step
    try:
        pool, idxs = self._compress_part(x, qr, start_pos, cache)
    finally:
        if lazy:
            comp.lazy = False
    if pool.shape[1] == 0 or idxs.shape[-1] > KMAX:                           # nothing pooled yet / top-k beyond the kernel: stock tail
        return None, _stock_tail(self, L, cache, x, qr, q, start_pos, pool, idxs)
    rd = self.rope.dims
    rt, it = rope_tables(self.rope, start_pos)
    pos = pos_array(start_pos)
    new_ring = kv_ring(kv, cache.window[self.layer_idx], rt, pos, rd)
    cache.window[self.layer_idx] = new_ring
    o = attend(q, new_ring, pool, idxs, self.attn_sink, rt, it, pos, self.n_heads, rd, self.scale)
    return o.reshape(1, self.o_groups, 1, -1), None                          # heads of a group are contiguous


PACKED_ATTEND = os.environ.get("KYOJIN_PACKED_ATTEND", "1") != "0"        # decode attention reads the fp4-packed pool directly (fp4pool.PackedView), no per-layer unpack
ROWS = os.environ.get("KYOJIN_FUSE_ROWS_CORE", "1") != "0"


def is_reuse(self) -> bool:
    """A compressing layer that owns neither the compressor nor the indexer: its core reads the pooled rows and the top-k of an earlier layer."""
    return bool(self.compress_ratio and not self.is_kv_source and not self.is_index_source)


def core_rows(self, L, chain, q_all, kv_all):
    """The cores of the n rows of a chained verify for one reuse layer in 3 launches (kv_proc, attend_rows, and the deferred kv_apply of the n rings), or None
    when the case is outside what the kernels cover (the caller then runs `core` row by row). q_all [n, 1, H * D], kv_all [n, 1, D] (the region A outputs).
    Returns o_all [n, G, 1, (H / G) * D]. Sets the ring of every row (lazy slices of one [n, W, D] array)."""
    rows = chain.rows
    n = len(rows)
    if not (ROWS and ENABLED and chain.link and len(chain.starts) == 1 and n >= 2 and is_reuse(self)):     # one segment: the rows chain through each other
        return None
    l, ratio = self.layer_idx, self.compress_ratio
    r0 = rows[0]
    ring = r0.window[l]
    p0 = r0.offset
    D, W = self.head_dim, self.window_size
    if (ring is None or ring.shape != (1, W, D) or D != 512 or n > W or q_all.dtype not in (mx.float16, mx.bfloat16) or ring.dtype != q_all.dtype
            or (p0 + 1) // ratio <= 0 or any(r.offset != p0 + j for j, r in enumerate(rows))):
        return None
    last = rows[-1]
    cl = (p0 + n) // ratio
    if last.compress_kv is None or last.compress_kv.shape[1] < cl:
        return None
    kvpool = last.compress_kv[:, :cl]
    packed = hasattr(kvpool, "codes")
    if packed and not CD.ENABLED:
        kvpool, packed = kvpool.dense(), False
    ids = []
    for r in rows:
        t = r.topk_idxs
        if t is None or t.ndim != 3 or t.shape[0] != 1 or t.shape[1] != 1 or t.shape[2] == 0 or t.shape[2] > KMAX:
            return None
        ids.append(t)
    key = tuple(id(t) for t in ids)
    memo = chain.__dict__.get("_idx_stack")
    if memo is None or memo[0] != key:
        kmax = max(t.shape[2] for t in ids)                                                   # short contexts: top-k = number of keys, which grows along the rows
        parts = [t if t.shape[2] == kmax else mx.pad(t, [(0, 0), (0, 0), (0, kmax - t.shape[2])], constant_values=-1) for t in ids]
        memo = chain._idx_stack = (key, ids, mx.concatenate(parts, axis=0).reshape(n, -1))      # keeps `ids` alive: the ids stay unique while memoised
    idx_all = memo[2]
    if packed:
        kvpool, idx_all = CD.compact(kvpool, idx_all)                                         # cdecpack: gather the rows the n top-k lists name, remap to them
    rd = self.rope.dims
    rt_all, it_all = rope_range(self.rope, p0, n)
    pos = pos_range(p0, n)
    kvp = kv_proc(kv_all.reshape(n, D), rt_all, rd)
    o = attend_rows(q_all.reshape(n, -1), ring, kvp, kvpool, idx_all, self.attn_sink, rt_all, it_all, pos, self.n_heads, rd, self.scale)
    rings = kv_apply(ring, kvp, pos)
    for j, r in enumerate(rows):
        r.window[l] = rings[j:j + 1]
    return o.reshape(n, self.o_groups, 1, -1)


def project_out(self, L, o):
    """wo_a (grouped, fake quant inside) and wo_b on the attention output o [1, G, 1, (H / G) * D]."""
    o = self.wo_a(o)
    o = o.transpose(0, 2, 1, 3).flatten(-2)
    return self.wo_b(L.fake_quant_fp8_ue8m0(o))


def _impl_attention(orig):
    """Replacement of `DeepseekV41Attention._attention` (see module doc); same module calls in the same order as the stock method."""
    import importlib
    L = _A.lang()

    def _attention(self, cache, x):
        if not (x.ndim == 3 and decode_ok(self, cache, x.shape[0], x.shape[1], x.dtype)):
            return orig(self, cache, x)
        fq = L.fake_quant_fp8_ue8m0
        qr = self.q_norm(self.wq_a(fq(x)))
        q = self.wq_b(fq(qr))                                                     # [1, 1, H * D]
        kv = self.kv_norm(self.wkv(fq(x)))                                        # [1, 1, D]
        o, out = core(self, L, cache, x, qr, q, kv)
        return out if o is None else project_out(self, L, o)

    return _attention


# --------------------------------------------------------------------------------------------------------------- no host sync in the compressor rope
_hint: list = [None]
ROPE_POS_HINT = os.environ.get("KYOJIN_ROPE_POS_HINT", "0") != "0"   # lever rope_pos_hint: the same hint on the verify rows (s > 1), not only solo decode
HINT_ROWS_MAX = 64                                                    # rows the hint covers; a longer chunk is a prompt pass and keeps the stock `.item()`


def _impl_compress_part(orig):
    """`_compress_part` with the position of the closing compression group known on the host: the stock `_apply_rope_at_positions` reads the largest position
    back with `.item()` (a wait for the GPU on every closing step of every compressing layer), here it comes from the python ints that built the positions.
    `rope_pos_hint` extends the hint to a verify pass (`s > 1` rows, whose latent positions are `(start_pos // ratio + i) * ratio` for i < n_latent <= s)."""
    def _compress_part(self, x, qr, start_pos, cache):
        s = x.shape[1]
        if ENABLED and s == 1 and self.compress_ratio:
            _hint[0] = (start_pos // self.compress_ratio) * self.compress_ratio
            try:
                return orig(self, x, qr, start_pos, cache)
            finally:
                _hint[0] = None
        if ROPE_POS_HINT and self.compress_ratio and 1 < s <= HINT_ROWS_MAX:
            r = self.compress_ratio
            _hint[0] = (start_pos // r + s - 1) * r                   # upper bound of the last latent position: the table grows past it, exact
            try:
                return orig(self, x, qr, start_pos, cache)
            finally:
                _hint[0] = None
        if s > 1 and self.compress_ratio:
            FD.hit("rope_item_sync")                                  # the stock `.item()` runs on this pass: one full-queue drain per closing group per layer
        return orig(self, x, qr, start_pos, cache)
    return _compress_part


def _impl_rope_positions(orig):
    import importlib
    L = _A.lang()

    def _apply_rope_at_positions(x, positions, rope_dim, theta, yarn):
        p = _hint[0]
        if p is None:
            return orig(x, positions, rope_dim, theta, yarn)
        cos, sin = L._index_cos_sin(p + 1, rope_dim, theta, yarn)               # the table the stock function builds from max(positions) + 1; the rows an index
        shape = (1,) * (x.ndim - positions.ndim - 1) + positions.shape + (cos.shape[-1],)   # selects are the same, a longer table is the same rows
        return L._apply_index_rotary(x, cos[positions].reshape(shape), sin[positions].reshape(shape), rope_dim)
    return _apply_rope_at_positions


def _stock_tail(self, L, cache, x, qr, q, start_pos, pool, idxs):
    """The stock method after `_compress_part` (rare cases), with the projections already built."""
    pool = pool.dense() if hasattr(pool, "codes") else pool
    batch, seqlen = 1, 1
    q = q.reshape(batch, seqlen, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
    q = self.rope(q, start_pos)
    window_kv, window_mask = self._window_part(x, start_pos, cache)
    out = None
    if pool.shape[1]:
        out = self._sparse_attention(q, window_kv, pool, idxs, window_mask)
    if out is None:
        kv = window_kv[:, None].astype(q.dtype)
        out = mx.fast.scaled_dot_product_attention(q, kv, kv, scale=self.scale, mask=window_mask, sinks=self.attn_sink.astype(q.dtype))
    out = self.rope(out, start_pos, inverse=True)
    out = out.reshape(batch, self.o_groups, -1, seqlen, self.head_dim)
    out = out.transpose(0, 1, 3, 2, 4).flatten(-2)
    out = self.wo_a(out)
    out = out.transpose(0, 2, 1, 3).flatten(-2)
    return self.wo_b(L.fake_quant_fp8_ue8m0(out))
