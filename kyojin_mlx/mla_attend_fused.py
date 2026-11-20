# SPDX-License-Identifier: Apache-2.0
"""Decode attention core of an absorbed-MLA layer on the fixed-state KV buffer, in two launches (flag KYOJIN_MLA_ATTEND_FUSE, default off).

The stock core (foreign_skeleton_mla.MlaAttention.attend) is ~15 small launches per layer: the latent scores, the rope scores, their sum, the causal / window mask
(built from the offset array), the `keep` select, the sink concat, the precise softmax, the slice, P @ latent. Each is a few microseconds of dispatch for a few KB
of data (Hy4 plain step: ~5 ms of 24.8 ms). Here, after the k_b absorb (qa = q_nope @ k_b^T, stock) and before the v_b absorb (stock):

  part   one simdgroup per (key split, head, row): its lanes own 18 contiguous dims of the 576-wide key row [latent 512 | k_pe 64]; scores s = (qa . lat + q_pe . k_pe) * scale
         (fp32 accumulate, one simd_sum per key), online softmax, the same loaded row feeds P @ latent. Keys visible to row r: slots k <= off + r (and > off + r - window on a
         windowed layer), and keep[r, k] when a kpool layer gives a mask. Only those keys are read (the stock core reads the whole buffer and masks it).
  merge  one simdgroup per (head, row): combine the splits, the sink joins the denominator only, normalise, round to the model dtype.

Same maths as the stock core; differences are the rounding points only (scores and P stay fp32 here, the stock path rounds the two score matmuls and P to fp16).
Needs a fp16, bf16 or fp32 buffer (Hy4 runs its hidden stream, so this buffer, in fp32) with qa / q_pe of the same dtype, rows <= 16, kv_lora + rope = 576 (any kv_lora % 2 == 0 and rope % 2 == 0 with (kv_lora + rope) / 32 contiguous dims per lane
whole: here DPL = (kv + rope) / 32 must be an even integer). Anything else runs the stock core."""
from __future__ import annotations

import os

import mlx.core as mx

NSPLIT = 8
SHARE = os.environ.get("KYOJIN_MLA_ATTEND_SHARE", "0") not in ("0", "")      # n9 v3: heads of a threadgroup share each loaded key tile; n9-b: SLOWER than the per-head kernel (p5 4000 ctx 29.5 vs 34.7 tok/s, no key prefetch / few resident groups), so off (0 = n7's one simdgroup per head)
_k: dict = {}
DT = {mx.float16: "f16", mx.bfloat16: "bf16", mx.float32: "f32"}

_PART = r"""
    const uint split = threadgroup_position_in_grid.x;
    const uint head = threadgroup_position_in_grid.y;
    const uint row = threadgroup_position_in_grid.z;
    const uint lane = thread_index_in_simdgroup;
    const int qpos = off[0] + int(row);
    const int hi = qpos;                                                     // inclusive
    const int lo = WIN > 0 ? max(0, qpos - WIN + 1) : 0;
    const int n = hi - lo + 1;
    const int cs = (n + NSPLIT - 1) / NSPLIT;
    const int k0 = lo + int(split) * cs;
    const int k1 = min(lo + (int(split) + 1) * cs, hi + 1);
    const float sc = par[0];
    float q[DPL];
    const uint d0 = lane * DPL;
    for (uint i = 0; i < DPL; ++i) {
        const uint d = d0 + i;
        q[i] = (d < uint(KV) ? float(qa[(ulong(head) * ROWS + row) * KV + d]) : float(qp[(ulong(head) * ROWS + row) * RD + (d - KV)])) * sc;
    }
    float m = -INFINITY, l = 0.0f;
    float acc[DPL];
    for (uint i = 0; i < DPL; ++i) acc[i] = 0.0f;
    for (int k = k0; k < k1; ++k) {
        if (HASKEEP && !keep[ulong(row) * C + k]) continue;
        const device T* kr = buf + ulong(k) * DSTR + d0;
        float v[DPL];
        float s = 0.0f;
        for (uint i = 0; i < DPL; ++i) { v[i] = float(kr[i]); s += q[i] * v[i]; }
        s = simd_sum(s);
        const float mn = max(m, s);
        const float a = exp(m - mn);
        const float p = exp(s - mn);
        l = l * a + p;
        for (uint i = 0; i < DPL; ++i) acc[i] = acc[i] * a + p * v[i];
        m = mn;
    }
    const ulong base = ((ulong(head) * ROWS + row) * NSPLIT + split) * (2 + KV);
    if (lane == 0) { pt[base] = m; pt[base + 1] = l; }
    for (uint i = 0; i < DPL; ++i) { const uint d = d0 + i; if (d < uint(KV)) pt[base + 2 + d] = acc[i]; }
"""

_PART3 = r"""
    const uint split = threadgroup_position_in_grid.x;
    const uint hg = threadgroup_position_in_grid.y;
    const uint row = threadgroup_position_in_grid.z;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = tid % 32u, sg = tid / 32u;
    const uint head = hg * uint(HG) + sg;
    const int qpos = off[0] + int(row);
    const int hi = qpos;
    const int lo = WIN > 0 ? max(0, qpos - WIN + 1) : 0;
    const int n = hi - lo + 1;
    const int cs = (n + NSPLIT - 1) / NSPLIT;
    const int k0 = lo + int(split) * cs;
    const int k1 = min(lo + (int(split) + 1) * cs, hi + 1);
    const float sc = par[0];
    float q[DPL];
    const uint d0 = lane * DPL;
    for (uint i = 0; i < DPL; ++i) {
        const uint d = d0 + i;
        q[i] = (d < uint(KV) ? float(qa[(ulong(head) * ROWS + row) * KV + d]) : float(qp[(ulong(head) * ROWS + row) * RD + (d - KV)])) * sc;
    }
    float m = -INFINITY, l = 0.0f;
    float acc[DPL];
    for (uint i = 0; i < DPL; ++i) acc[i] = 0.0f;
    threadgroup T tile[KT * DD];
    const uint NTH = 32u * uint(HG);
    for (int kb = k0; kb < k1; kb += KT) {
        const int nk = min(int(KT), k1 - kb);
        for (int kk = 0; kk < nk; ++kk) {                                      // one key row of the tile at a time: every thread of the group, coalesced
            if (HASKEEP && !keep[ulong(row) * C + kb + kk]) continue;
            const device T* src = buf + ulong(kb + kk) * DSTR;
            for (uint d = tid; d < uint(DD); d += NTH) tile[kk * DD + d] = src[d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < nk; ++kk) {
            if (HASKEEP && !keep[ulong(row) * C + kb + kk]) continue;
            const threadgroup T* kr = tile + kk * DD + d0;
            float v[DPL];
            float s = 0.0f;
            for (uint i = 0; i < DPL; ++i) { v[i] = float(kr[i]); s += q[i] * v[i]; }
            s = simd_sum(s);
            const float mn = max(m, s);
            const float a = exp(m - mn);
            const float p = exp(s - mn);
            l = l * a + p;
            for (uint i = 0; i < DPL; ++i) acc[i] = acc[i] * a + p * v[i];
            m = mn;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const ulong base = ((ulong(head) * ROWS + row) * NSPLIT + split) * (2 + KV);
    if (lane == 0) { pt[base] = m; pt[base + 1] = l; }
    for (uint i = 0; i < DPL; ++i) { const uint d = d0 + i; if (d < uint(KV)) pt[base + 2 + d] = acc[i]; }
"""


_MERGE = r"""
    const uint head = threadgroup_position_in_grid.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    const ulong b0 = (ulong(head) * ROWS + row) * NSPLIT * (2 + KV);
    float M = -INFINITY;
    for (uint s = 0; s < NSPLIT; ++s) M = max(M, pt[b0 + ulong(s) * (2 + KV)]);
    if (HASSINK) M = max(M, sinks[head]);
    float l = HASSINK ? exp(sinks[head] - M) : 0.0f;
    float w[NSPLIT];
    for (uint s = 0; s < NSPLIT; ++s) {
        const float ms = pt[b0 + ulong(s) * (2 + KV)];
        w[s] = ms == -INFINITY ? 0.0f : exp(ms - M);
        l += w[s] * pt[b0 + ulong(s) * (2 + KV) + 1];
    }
    const float inv = 1.0f / l;
    for (uint d = lane; d < uint(KV); d += 32) {
        float o = 0.0f;
        for (uint s = 0; s < NSPLIT; ++s) o += w[s] * pt[b0 + ulong(s) * (2 + KV) + 2 + d];
        out[(ulong(head) * ROWS + row) * KV + d] = static_cast<OT>(o * inv);
    }
"""


def _const(name, make):
    """a placeholder input of the kernel as a CONSTANT array (mx.zeros inside a traced layer is a Full launch per layer and call; a python-built array is not)"""
    v = _k.get(("const", name))
    if v is None:
        v = _k[("const", name)] = make()
    return v


def eligible(qa: mx.array, q_pe: mx.array, buf: mx.array, kv: int, rd: int, keep, why: dict | None = None) -> bool:
    """qa [1, nh, L, kv], q_pe [1, nh, L, rd], buf [1, 1, C, D] (fixed-state buffer), keep None or bool [1, 1, L, C]. `why`: a counter dict that gets the reason of a refusal."""
    def no(r):
        if why is not None:
            why["attend_fuse_no_" + r] = why.get("attend_fuse_no_" + r, 0) + 1
        return False
    if mx.default_device() != mx.gpu:
        return no("device")
    if qa.dtype not in DT or q_pe.dtype != qa.dtype or buf.dtype != qa.dtype:
        return no(f"dtype_{str(qa.dtype)[-8:]}_{str(q_pe.dtype)[-8:]}_{str(buf.dtype)[-8:]}")
    if qa.ndim != 4 or buf.ndim != 4:
        return no("ndim")
    B, nh, L, _ = qa.shape
    if B != 1 or not 1 <= L <= 16 or qa.shape[-1] != kv or q_pe.shape[-1] != rd or (kv + rd) % 64 != 0 or buf.shape[-1] < kv + rd:
        return no("shape")
    if keep is not None and not (keep.shape == (1, 1, L, buf.shape[2]) and keep.dtype == mx.bool_):
        return no("keep")
    return True


def attend(qa: mx.array, q_pe: mx.array, buf: mx.array, off: mx.array, keep, sinks, scale: float, window: int, kv: int, rd: int) -> mx.array:
    """-> o [1, nh, L, kv] fp16 (P @ latent per head; the v_b absorb follows). off: int32 [1], the write offset BEFORE the new rows (row r sits at slot off + r)."""
    _, nh, L, _ = qa.shape
    C, D = buf.shape[2], buf.shape[3]
    hs, hk = sinks is not None, keep is not None
    hg = next((g for g in (8, 4, 2) if nh % g == 0), 1) if SHARE else 1                  # heads per threadgroup: they share every key tile (the key rows are the same for all heads: MLA)
    kt = (16 if qa.dtype != mx.float32 else 8) if hg > 1 else 0
    key = (nh, L, kv, rd, D, C, window, hs, hk, qa.dtype, hg)
    ks = _k.get(key)
    if ks is None:
        part = mx.fast.metal_kernel(name=f"kyojin_mla_part{'3' if hg > 1 else ''}_{nh}_{L}_{D}_{window}_{int(hk)}_{DT[qa.dtype]}_{hg}", input_names=["qa", "qp", "buf", "off", "keep", "par"], output_names=["pt"], source=_PART3 if hg > 1 else _PART)
        merge = mx.fast.metal_kernel(name=f"kyojin_mla_merge_{nh}_{L}_{int(hs)}_{DT[qa.dtype]}", input_names=["pt", "sinks"], output_names=["out"], source=_MERGE)
        ks = _k[key] = (part, merge)
    part, merge = ks
    par = _k.get(("par", scale))
    if par is None:
        par = _k[("par", scale)] = mx.array([scale], dtype=mx.float32)
    dpl = (kv + rd) // 32
    tpl = [("NSPLIT", NSPLIT), ("ROWS", L), ("KV", kv), ("RD", rd), ("DPL", dpl), ("DSTR", D), ("C", C), ("WIN", int(window)), ("HASKEEP", hk), ("T", qa.dtype)]
    if hg > 1:
        tpl += [("HG", hg), ("KT", kt), ("DD", kv + rd)]
    pt = part(inputs=[qa, q_pe, buf, off.astype(mx.int32).reshape(-1), keep if hk else _const("nokeep", lambda: mx.array([False])), par],
              template=tpl, grid=(NSPLIT * 32 * hg, nh // hg, L), threadgroup=(32 * hg, 1, 1), output_shapes=[(nh * L * NSPLIT * (2 + kv),)], output_dtypes=[mx.float32])[0]
    out = merge(inputs=[pt, sinks.astype(mx.float32) if hs else _const("nosink", lambda: mx.array([0.0], dtype=mx.float32))],
                template=[("NSPLIT", NSPLIT), ("ROWS", L), ("KV", kv), ("HASSINK", hs), ("OT", qa.dtype)],
                grid=(nh * 32, L, 1), threadgroup=(32, 1, 1), output_shapes=[(1, nh, L, kv)], output_dtypes=[qa.dtype])[0]
    return out


ON = os.environ.get("KYOJIN_MLA_ATTEND_FUSE", "0") not in ("0", "")      # flag mla_attend_fuse (flags.py rewrites it at runtime)
