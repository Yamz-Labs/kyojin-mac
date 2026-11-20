# SPDX-License-Identifier: Apache-2.0
"""Flag mla_decode_vec (KYOJIN_HY4_ATTN_VEC, default 0): the decode attention core of an absorbed-MLA layer (1 .. 8 rows) as ONE launch, the flash-attention-vec shape of llama.cpp
(ggml-metal flash_attn_ext_vec, nwg = 1: no cross-threadgroup split, no reduce dispatch).

The stock core after the k_b absorb is ~15-25 small launches per layer (scores matmul, rope-score matmul, add, mask, keep, sink concat, softmax, slice, strided copies, P @ latent). Here, between the two
absorb products (qa = q_nope @ k_b^T before, o @ v_b^T after), ONE kernel:

  grid     one threadgroup (4 simdgroups = 128 threads) per (head, query row). The latent is ONE kv head shared by every head and row: keys are the 576-wide cache rows [latent 512 | k_pe 64]; V = the first
           512 columns of the same row (a view of K, nothing is copied).
  keys     the row's cell list is dealt to 8 lane groups (4 simdgroups x 2 groups of 16 lanes), one key per group per step. A lane owns 9 float4 of the key row (strided by 16, coalesced across the group)
           and 8 float4 of the output; the 16-lane dot product is a 4-step xor shuffle. Online softmax (m, l, acc) per group; the two groups of a simdgroup merge by shuffle, the four simdgroups in
           threadgroup memory, together with the attention sink (it joins the denominator only). One launch writes o [heads, rows, 512].
  cells    kind 0 window: the visible slots (off + r - window, off + r], read straight from the buffer with the device offset (no dynamic slice, no mask build); window 0 = every slot <= off + r.
           kind 1 keep: slots <= off + r with keep[r, slot] (the kpool_fuse mask; only kept keys are loaded).
           kind 2 ids: the key list ids [rows, W] (-1 = none) of the gather path (kpool picks + tail, past FIXED_GATHER_DEC slots).
  dtypes   the buffer (K / V) and the queries are read in their own dtype (fp32 or fp16; fp32 scores and accumulation). The fp16 cache of llama.cpp is the attn_fp16 arm; its buffer rounding is a
           separate, measured quality cost (+3.1 % PPL at 4096 in n9-a), so the kernel does not require it.
Same maths as the stock core; the differences are rounding points only (scores and P stay fp32). Metal only; anything outside the contract runs the stock core (`why` counts the reason)."""
from __future__ import annotations

import mlx.core as mx

MODE = 0                                          # bound by flags.set (attrs): 0 off, 1 windowed (SWA) layers, 2 + kpool layers (keep mask and gather ids)
MAX_ROWS = 8
MIN_ROWS = 2                                      # bound by flags.set (attrs): flag mla_decode_vec_min_rows, fewer rows than this (plain decode, one row) run the stock core
NSG = 4                                           # simdgroups per threadgroup
KV, RD = 512, 64
_k: dict = {}
DT = {mx.float16: "f16", mx.bfloat16: "bf16", mx.float32: "f32"}

_SRC = r"""
    constexpr uint NL = 16;                                    // lanes per key
    constexpr uint NVL = KV / 4 / NL;                          // float4 of the latent part per lane (8); one more float4 per lane covers the rope part
    const uint h = threadgroup_position_in_grid.x;
    const uint r = threadgroup_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32u;
    const uint lane = tid % 32u;
    const uint g = lane / NL;
    const uint jl = lane % NL;

    const float sc = par[0];
    float4 q4[NVL + 1];
    {
        const device QT* pa = qa + (ulong(h) * ROWS + r) * KV;
        const device QT* pr = qp + (ulong(h) * ROWS + r) * RD;
        for (uint i = 0; i < NVL; ++i) {
            const uint e = 4 * (jl + NL * i);
            q4[i] = float4(float(pa[e]), float(pa[e + 1]), float(pa[e + 2]), float(pa[e + 3])) * sc;
        }
        const uint e = 4 * jl;
        q4[NVL] = float4(float(pr[e]), float(pr[e + 1]), float(pr[e + 2]), float(pr[e + 3])) * sc;
    }

    const int hi = min(int(off[0]) + int(r), int(C) - 1);
    const int lo = (WIN > 0) ? max(0, hi - int(WIN) + 1) : 0;
    const int n = (KIND == 2) ? int(W) : (hi - lo + 1);

    float m = -INFINITY, l = 0.0f;
    float4 acc[NVL];
    for (uint i = 0; i < NVL; ++i) acc[i] = float4(0.0f);

    for (int j0 = 0; j0 < n; j0 += 2 * int(NSG)) {
        const int j = j0 + int(sg) * 2 + int(g);
        int slot = -1;
        if (j < n) {
            if (KIND == 2) {
                slot = ids[ulong(r) * W + j];
                if (slot > hi) slot = -1;
            } else {
                slot = lo + j;
                if (KIND == 1 && !keep[ulong(r) * C + slot]) slot = -1;
            }
        }
        float4 kv[NVL];
        float part = 0.0f;
        for (uint i = 0; i < NVL; ++i) kv[i] = float4(0.0f);
        if (slot >= 0) {
            const device vec<T, 4>* kp = (const device vec<T, 4>*)(buf + ulong(slot) * DSTR);
            for (uint i = 0; i < NVL; ++i) {
                kv[i] = float4(kp[jl + NL * i]);
                part += dot(kv[i], q4[i]);
            }
            part += dot(float4(kp[jl + NL * NVL]), q4[NVL]);
        }
        part += simd_shuffle_xor(part, ushort(8));
        part += simd_shuffle_xor(part, ushort(4));
        part += simd_shuffle_xor(part, ushort(2));
        part += simd_shuffle_xor(part, ushort(1));
        const float s = (slot >= 0) ? part : -INFINITY;
        const float mn = max(m, s);
        const float a = (mn == -INFINITY) ? 1.0f : exp(m - mn);
        const float p = (s == -INFINITY) ? 0.0f : exp(s - mn);
        l = l * a + p;
        for (uint i = 0; i < NVL; ++i) acc[i] = acc[i] * a + p * kv[i];
        m = mn;
    }

    // the two lane groups of this simdgroup
    {
        const float m2 = simd_shuffle_xor(m, ushort(16));
        const float l2 = simd_shuffle_xor(l, ushort(16));
        const float M = max(m, m2);
        const float wa = (m == -INFINITY) ? 0.0f : exp(m - M);
        const float wb = (m2 == -INFINITY) ? 0.0f : exp(m2 - M);
        l = l * wa + l2 * wb;
        for (uint i = 0; i < NVL; ++i) acc[i] = acc[i] * wa + simd_shuffle_xor(acc[i], ushort(16)) * wb;
        m = M;
    }

    threadgroup float4 ta[NSG][KV / 4];
    threadgroup float tm[NSG];
    threadgroup float tl[NSG];
    if (g == 0) {
        for (uint i = 0; i < NVL; ++i) ta[sg][jl + NL * i] = acc[i];
        if (jl == 0) { tm[sg] = m; tl[sg] = l; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float Mx = -INFINITY;
    for (uint s = 0; s < NSG; ++s) Mx = max(Mx, tm[s]);
    float den = 0.0f;
    if (HASSINK) { Mx = max(Mx, sinks[h]); den = exp(sinks[h] - Mx); }
    float4 o = float4(0.0f);
    for (uint s = 0; s < NSG; ++s) {
        const float ws = (tm[s] == -INFINITY) ? 0.0f : exp(tm[s] - Mx);
        den += ws * tl[s];
        o += ws * ta[s][tid];
    }
    o *= (den > 0.0f) ? (1.0f / den) : 0.0f;
    device OT* po = out + (ulong(h) * ROWS + r) * KV + 4 * tid;
    po[0] = static_cast<OT>(o.x); po[1] = static_cast<OT>(o.y); po[2] = static_cast<OT>(o.z); po[3] = static_cast<OT>(o.w);
"""


def _const(name, make):
    """a placeholder input as a CONSTANT array (mx.zeros inside a traced layer would be a launch per layer and call)"""
    v = _k.get(("const", name))
    if v is None:
        v = _k[("const", name)] = make()
    return v


def eligible(qa, q_pe, buf, kind: int, extra=None, why: dict | None = None) -> bool:
    """qa [1, nh, L, 512], q_pe [1, nh, L, 64], buf [1, 1, C, D] (fixed-state buffer, D % 4 == 0). kind 1: extra = keep bool [1, 1, L, C]; kind 2: extra = ids int32 [1, L, W]."""
    def no(r):
        if why is not None:
            why["attn_vec_no_" + r] = why.get("attn_vec_no_" + r, 0) + 1
        return False
    if mx.default_device() != mx.gpu:
        return no("device")
    if qa.dtype not in (mx.float16, mx.float32) or q_pe.dtype not in (mx.float16, mx.float32) or buf.dtype not in (mx.float16, mx.float32):
        return no("dtype")
    if qa.ndim != 4 or buf.ndim != 4 or q_pe.ndim != 4:
        return no("ndim")
    B, nh, L, _ = qa.shape
    if B != 1 or not 1 <= L <= MAX_ROWS or qa.shape[-1] != KV or q_pe.shape[-1] != RD or buf.shape[-1] < KV + RD or buf.shape[-1] % 4 or buf.shape[1] != 1:
        return no("shape")
    if kind == 1 and not (extra is not None and extra.shape == (1, 1, L, buf.shape[2]) and extra.dtype == mx.bool_):
        return no("keep")
    if kind == 2 and not (extra is not None and extra.ndim == 3 and extra.shape[:2] == (1, L) and extra.dtype == mx.int32):
        return no("ids")
    return True


def attend(qa, q_pe, buf, off, kind: int, extra, sinks, scale: float, window: int):
    """-> o [1, nh, L, 512] in qa's dtype (P @ latent per head; the v_b absorb follows). off: int32 [1], the write offset BEFORE the new rows (row r sits at slot off + r)."""
    _, nh, L, _ = qa.shape
    C, D = buf.shape[2], buf.shape[3]
    hs = sinks is not None
    if q_pe.dtype != qa.dtype:
        q_pe = q_pe.astype(qa.dtype)
    W = int(extra.shape[-1]) if kind == 2 else 0
    key = (kind, hs, qa.dtype, buf.dtype)
    kern = _k.get(key)
    if kern is None:
        kern = _k[key] = mx.fast.metal_kernel(name=f"kyojin_mla_vec_{kind}_{int(hs)}_{DT[qa.dtype]}_{DT[buf.dtype]}", input_names=["qa", "qp", "buf", "off", "keep", "ids", "sinks", "par"],
                                              output_names=["out"], source=_SRC)
    par = _k.get(("par", scale))
    if par is None:
        par = _k[("par", scale)] = mx.array([scale], dtype=mx.float32)
    keep = extra if kind == 1 else _const("nokeep", lambda: mx.array([False]))
    ids = extra if kind == 2 else _const("noids", lambda: mx.array([0], dtype=mx.int32))
    sk = sinks.astype(mx.float32) if hs else _const("nosink", lambda: mx.array([0.0], dtype=mx.float32))
    tpl = [("ROWS", L), ("KV", KV), ("RD", RD), ("C", C), ("DSTR", D), ("WIN", int(window)), ("KIND", kind), ("W", W), ("HASSINK", hs), ("NSG", NSG), ("T", buf.dtype), ("QT", qa.dtype), ("OT", qa.dtype)]
    return kern(inputs=[qa, q_pe, buf, off.astype(mx.int32).reshape(-1), keep, ids, sk, par], template=tpl, grid=(32 * NSG * nh, L, 1), threadgroup=(32 * NSG, 1, 1),
                output_shapes=[(1, nh, L, KV)], output_dtypes=[qa.dtype])[0]


def reference(qa, q_pe, buf, off, kind: int, extra, sinks, scale: float, window: int):
    """The same cell lists and maths in plain mx ops (fp32): the unit-test oracle of the kernel and of the stock chain. -> [1, nh, L, 512] fp32.
    Runs on the CPU stream whatever the default device: the GPU fp32 matmul of this Mac (applegpu_g17d, mlx 0.29) is not a full-fp32 product (the f32 kernel case differed by 6e-4 from it with ONE visible key,
    where the kernel is a plain fp32 dot + exp), so an oracle on the GPU would cap the fp32 contract at the matmul's own error."""
    with mx.stream(mx.cpu):
        return _reference(qa, q_pe, buf, off, kind, extra, sinks, scale, window)


def _reference(qa, q_pe, buf, off, kind: int, extra, sinks, scale: float, window: int):
    _, nh, L, _ = qa.shape
    C = buf.shape[2]
    b = buf[0, 0].astype(mx.float32)
    lat, kpe = b[:, :KV], b[:, KV:KV + RD]
    q = qa[0].astype(mx.float32)                                           # [nh, L, KV]
    qp = q_pe[0].astype(mx.float32)
    sc = (q @ lat.T + qp @ kpe.T) * scale                                  # [nh, L, C]
    pos = off.reshape(-1)[0] + mx.arange(L)
    slot = mx.arange(C)
    vis = slot[None, :] <= pos[:, None]
    if window:
        vis = vis & (pos[:, None] - slot[None, :] < window)
    if kind == 1:
        vis = vis & extra[0, 0]
    if kind == 2:
        ids = extra[0]                                                     # [L, W]
        okid = (ids >= 0) & (ids <= pos[:, None])
        hit = mx.any((ids[:, :, None] == slot[None, None, :]) & okid[:, :, None], axis=1)
        vis = vis & hit
    sc = mx.where(vis[None], sc, -mx.inf)
    if sinks is not None:
        sc = mx.concatenate([sc, mx.broadcast_to(sinks.astype(mx.float32)[:, None, None], (nh, L, 1))], axis=-1)
    p = mx.softmax(sc, axis=-1, precise=True)[..., :C]
    return (p @ lat)[None]
