# SPDX-License-Identifier: Apache-2.0
"""One-dispatch decode step of a TurboQuant full-attention layer (GQA, one row, batch 1).

The stock chain around the cache costs ~28 dispatches per layer: q / k norm, two ropes, the K / V rotation matmul, the casts, two quantize launches, four pool
slice updates, the ring concat + slice for K and V, the sigmoid and the gate multiply. This module runs all of it in ONE launch (`step`), then the flash-decode
kernel and a merge that also applies the output gate:

    kernel "tq_step" (one 32-lane threadgroup per job):
        q jobs     rms norm, rope  -> the decode query (model dtype)
        k jobs     rms norm, rope, rotation H diag(s) / sqrt(D), quantize, K words / scale written INTO the pool, rotated row at the end of the new ring
        v jobs     rotation, quantize, V words / scale written INTO the pool, rotated row at the end of the new ring
        copy jobs  the previous ring rows (shifted by one once the ring is full) into the new ring arrays
    merge: the partials of the decode kernel + inverse rotation + sigmoid(gate) multiply (the gate is read from the q projection).

The pool arrays are written in place (the kernel writes through its const input pointers; the rows it touches, position `pos`, are never read by anything else
of the step). The new ring arrays are real outputs, so the decode kernel that reads them runs after the writes, and a snapshot that holds a ring array never sees
a later write. The cache owns its pool buffers (`_own`): a pool that came from a state restore or a fork is copied once before the first in-place write.
Works for any head count / GQA ratio, head dim 64 / 128 / 256, any partial rotary width, any K / V bits that have a codebook. KYOJIN_TQ_STEP=0 turns it off."""
from __future__ import annotations

import os

import mlx.core as mx

from . import tq as TQ

ENABLED = os.environ.get("KYOJIN_TQ_STEP", "1") != "0"
CH = 8                                                           # ring rows per copy job

_HEADER = """
// rms norm (model dtype rounding like the stock kernel), then non-traditional rope over the first RD dims. o[g] = value of dim g * 32 + lane.
template <int D, int RD, typename T>
inline void tq_norm_rope(const device T* src, const device T* w, float eps, float l2b, int pos, uint lane, threadgroup float* buf, thread float* o) {
    constexpr int G = D / 32;
    float ss = 0.0f;
    for (int g = 0; g < G; g++) { float x = (float)src[g * 32 + lane]; o[g] = x; ss += x * x; }
    float r = metal::precise::rsqrt(simd_sum(ss) / (float)D + eps);
    for (int g = 0; g < G; g++) {
        T t = (T)(o[g] * r);
        float y = (float)(T)((float)w[g * 32 + lane] * (float)t);
        buf[g * 32 + lane] = y; o[g] = y;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (RD > 0) {
        for (int g = 0; g < G; g++) {
            int i = g * 32 + (int)lane;
            if (i < RD) {
                int j = i < RD / 2 ? i : i - RD / 2;
                float th = (float)pos * metal::exp2(-(float)j * l2b);
                float c = metal::precise::cos(th), s = metal::precise::sin(th);
                float a = buf[j], b = buf[j + RD / 2];
                o[g] = i < RD / 2 ? a * c - b * s : b * c + a * s;
            }
            o[g] = (float)(T)o[g];
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

// y = H diag(s) x / sqrt(D), in place on the lane-owned values
template <int D>
inline void tq_rot(thread float* o, const device float* sgn, uint lane, threadgroup float* buf) {
    constexpr int G = D / 32;
    for (int g = 0; g < G; g++) buf[g * 32 + lane] = o[g] * sgn[g * 32 + lane];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int len = 1; len < D; len <<= 1) {
        for (int k = (int)lane; k < D / 2; k += 32) {
            int a = (k / len) * 2 * len + (k % len);
            float x0 = buf[a], x1 = buf[a + len];
            buf[a] = x0 + x1; buf[a + len] = x0 - x1;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rs = metal::precise::rsqrt((float)D);
    for (int g = 0; g < G; g++) o[g] = buf[g * 32 + lane] * rs;
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

// the quantizer of tq_write for one row held lane-wise; words go to dst[w * STR]; returns the row scale
template <int D, int BITS, int STR>
inline half tq_put(thread float* v, const device float* mid, const device float* cb, uint lane, device uint* dst) {
    constexpr int G = D / 32;
    constexpr int NM = (1 << BITS) - 1;
    float ss = 0.0f;
    for (int g = 0; g < G; g++) ss += v[g] * v[g];
    float nrm = max(sqrt(simd_sum(ss)), 1e-12f);
    uint code[G];
    float cn = 0.0f;
    for (int g = 0; g < G; g++) {
        float t = metal::precise::divide(v[g], nrm);
        int lo = 0, hi = NM;
        while (lo < hi) { int m = (lo + hi) >> 1; if (mid[m] < t) lo = m + 1; else hi = m; }
        code[g] = (uint)lo;
        float c = cb[lo];
        cn += c * c;
    }
    cn = max(sqrt(simd_sum(cn)), 1e-12f);
    int rem = BITS, wo = 0;
    for (int W = 4; W >= 1; W >>= 1) {
        if (!(BITS & W)) continue;
        rem -= W;
        int vpw = 32 / W;
        for (int g = 0; g < G; g++) {
            uint field = (code[g] >> rem) & ((1u << W) - 1u);
            uint word = 0;
            for (int j = 0; j < vpw; j++) word |= simd_shuffle(field, (lane * vpw + j) & 31) << (j * W);
            if ((int)lane < W) dst[(g * BITS + wo + (int)lane) * STR] = word;
        }
        wo += W;
    }
    return (half)metal::precise::divide(nrm, cn);
}
"""

_SRC = """
    uint gi = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    constexpr int G = D / 32;
    constexpr int NWK = D / 32 * KB, NWV = D / 32 * VB;
    threadgroup float buf[D];
    const int pos = ia[0], nro = ia[1], nrn = ia[2], cap = ia[3], nch = ia[4];
    const float l2b = fpar[2];
    float o[G];
    if (gi < (uint)HQ) {
        tq_norm_rope<D, RD, T>(qp + gi * QS, qw, fpar[0], l2b, pos, lane, buf, o);
        for (int g = 0; g < G; g++) qout[gi * D + g * 32 + lane] = (T)o[g];
    } else if (gi < (uint)(HQ + HK)) {
        int hk = (int)gi - HQ;
        tq_norm_rope<D, RD, T>(kp + hk * D, kw_, fpar[1], l2b, pos, lane, buf, o);
        tq_rot<D>(o, sgn, lane, buf);
        for (int g = 0; g < G; g++) rko[((size_t)hk * nrn + nrn - 1) * D + g * 32 + lane] = (T)o[g];
        device uint* dst = const_cast<device uint*>(kwd) + (((size_t)hk * (cap / 32) + (pos >> 5)) * NWK) * 32 + (pos & 31);
        half sc = tq_put<D, KB, 32>(o, midk, cbk, lane, dst);
        if (lane == 0) const_cast<device half*>(ksd)[(size_t)hk * cap + pos] = sc;
    } else if (gi < (uint)(HQ + 2 * HK)) {
        int hk = (int)gi - HQ - HK;
        for (int g = 0; g < G; g++) o[g] = (float)vp[hk * D + g * 32 + lane];
        tq_rot<D>(o, sgn, lane, buf);
        for (int g = 0; g < G; g++) rvo[((size_t)hk * nrn + nrn - 1) * D + g * 32 + lane] = (T)o[g];
        device uint* dst = const_cast<device uint*>(vwd) + ((size_t)hk * cap + pos) * NWV;
        half sc = tq_put<D, VB, 1>(o, midv, cbv, lane, dst);
        if (lane == 0) const_cast<device half*>(vsd)[(size_t)hk * cap + pos] = sc;
    } else {
        int c = (int)gi - HQ - 2 * HK;
        int which = c / (HK * nch), rem = c % (HK * nch);
        int hk = rem / nch, r0 = (rem % nch) * CHR;
        int sh = nro + 1 - nrn;
        const device T* src = which == 0 ? rk : rv;
        device T* dstp = which == 0 ? rko : rvo;
        for (int r = r0; r < min(r0 + CHR, nrn - 1); r++)
            for (int g = 0; g < G; g++) dstp[((size_t)hk * nrn + r) * D + g * 32 + lane] = src[((size_t)hk * nro + r + sh) * D + g * 32 + lane];
    }
"""

# merge of the decode partials + inverse rotation + sigmoid(gate): _MRG_SRC with the gate multiply on the last line (gate = the q projection, [Hq, 2 D], gate at +D)
_MRGG_SRC = TQ._MRG_SRC.replace(
    "if (tid < (uint)D) out[h * D + tid] = (T)(xs[tid] * sgn[tid] * scl[0]);",
    "if (tid < (uint)D) { T o_ = (T)(xs[tid] * sgn[tid] * scl[0]); T g_ = (T)(1.0f / (1.0f + metal::exp(-(float)qg[h * 2 * D + D + tid]))); out[h * D + tid] = (T)((float)o_ * (float)g_); }")
assert _MRGG_SRC != TQ._MRG_SRC

_k = {}
_KI = {}


def _step_kernel():
    if "step" not in _k:
        _k["step"] = mx.fast.metal_kernel(
            "tq_step", ["qp", "kp", "vp", "qw", "kw_", "kwd", "ksd", "vwd", "vsd", "rk", "rv", "cbk", "midk", "cbv", "midv", "sgn", "ia", "fpar"], ["qout", "rko", "rvo"],
            _SRC, header=_HEADER, ensure_row_contiguous=False)
    return _k["step"]


def merge_gated_kernel():
    if "mrgg" not in _k:
        _k["mrgg"] = mx.fast.metal_kernel("tq_merge_gate", ["po", "pml", "po2", "pml2", "sgn", "scl", "qg"], ["out"], _MRGG_SRC)
    return _k["mrgg"]


def rope_ok(rope) -> bool:
    """The stock plain rope (nn.RoPE, not traditional, scale 1): anything else (yarn, mrope, ...) keeps the stock chain."""
    return type(rope).__name__ == "RoPE" and not getattr(rope, "traditional", True) and float(getattr(rope, "scale", 0)) == 1.0 and hasattr(rope, "base")


def _fpar(mod):
    key = id(mod)
    v = _k.get(("fpar", key))
    if v is None or v[0] is not mod:
        import math
        rd = int(mod.rope.dims)
        l2b = math.log2(float(mod.rope.base)) / (rd / 2) if rd > 0 else 0.0
        v = (mod, mx.array([float(mod.q_norm.eps), float(mod.k_norm.eps), l2b], dtype=mx.float32))
        _k[("fpar", key)] = v
    return v[1]


def _own(c):
    """The in-place writes need pool buffers nobody else holds: a restored or forked pool is copied (once) first."""
    cur = (c.kq[0], c.kq[1], c.vq[0], c.vq[1])
    if getattr(c, "_own_ids", None) is not None and all(a is b for a, b in zip(cur, c._own_ids)):
        return
    kq = tuple(mx.contiguous(x * 1 if x.dtype != mx.uint32 else x | mx.array(0, dtype=mx.uint32)) for x in c.kq)
    vq = tuple(mx.contiguous(x * 1 if x.dtype != mx.uint32 else x | mx.array(0, dtype=mx.uint32)) for x in c.vq)
    mx.eval(*kq, *vq)
    c.kq, c.vq = kq, vq
    c._own_ids = (c.kq[0], c.kq[1], c.vq[0], c.vq[1])


def mark_owned(c):
    c._own_ids = (c.kq[0], c.kq[1], c.vq[0], c.vq[1])


def _heads(mod):
    return (getattr(mod, "num_attention_heads", None) or getattr(mod, "n_heads", None), getattr(mod, "num_key_value_heads", None) or getattr(mod, "n_kv_heads", None))


def ok(c, mod, qp, gated=True) -> bool:
    """True when `step` takes this cache / layer / row (no side effect): the multi-user caller checks every user before writing any cache."""
    if not ENABLED or type(c) is not TQ.TQKVCache or not c.fused or c.kq is None or c.rk is None or c.sk is None or c.window <= 0:
        return False
    if not rope_ok(mod.rope):
        return False
    Hq, Hk = _heads(mod)
    D = qp.shape[-1] // ((2 if gated else 1) * Hq)
    return not (D not in (64, 128, 256) or c.offset < c.sink or c.rk.dtype != qp.dtype or c.rk.shape[1] != Hk or c.rk.shape[2] < 1 or c.rk.shape[0] != 1)


def step(c, mod, qp, kp, vp, scale, gated=True):
    """qp [1, 1, Hq * 2 D] (queries and gates per head), kp / vp [1, 1, Hk * D] projections of ONE row -> the gated attention output [1, 1, Hq * D], or None when
    the cache / layer is not in the fused class (the caller then runs the stock chain). Appends the row to the cache like update_and_fetch."""
    if not ok(c, mod, qp, gated):
        return None
    Hq, Hk = _heads(mod)
    D = qp.shape[-1] // ((2 if gated else 1) * Hq)
    prev = c.offset
    if prev + 1 > c._cap():
        c._alloc(1, Hk, D, D, qp.dtype, 1)
    _own(c)
    cap = c._cap()
    nro = c.rk.shape[2]
    nrn = min(nro + 1, c.window + c.margin)
    nch = max((nrn - 1 + CH - 1) // CH, 1)
    key = (prev, nro, nrn, cap, nch)
    ia = _KI.get(key)
    if ia is None:
        if len(_KI) > 8:
            _KI.clear()
        ia = _KI[key] = mx.array(list(key), dtype=mx.int32)
    cbk, midk = TQ._cb_mx(D, c.kbits)
    cbv, midv = TQ._cb_mx(D, c.vbits)
    dt = qp.dtype
    cg = mx.contiguous
    rd = int(mod.rope.dims)
    qout, rko, rvo = _step_kernel()(
        inputs=[cg(qp), cg(kp), cg(vp), mod.q_norm.weight, mod.k_norm.weight, c.kq[0], c.kq[1], c.vq[0], c.vq[1], cg(c.rk), cg(c.rv), cbk, midk, cbv, midv,
                TQ._sgn_mx(D), ia, _fpar(mod)],
        template=[("D", D), ("QS", (2 if gated else 1) * D), ("KB", c.kbits), ("VB", c.vbits), ("HQ", Hq), ("HK", Hk), ("RD", rd), ("CHR", CH), ("T", dt)],
        grid=(32 * (Hq + 2 * Hk + 2 * Hk * nch), 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(Hq * D,), (1, Hk, nrn, D), (1, Hk, nrn, D)], output_dtypes=[dt, dt, dt])
    c.offset = prev + 1
    c._prev = (c.rk, c.rv, prev)
    c._chunk = None
    c.rk, c.rv = rko, rvo
    if gated:
        return c._decode(qout.reshape(1, Hq, 1, D), scale, gate=qp)
    return c._decode(qout.reshape(1, Hq, 1, D), scale).reshape(1, 1, Hq * D)     # ungated attention (q_proj = Hq * D): plain merge


_REQ = ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm", "rope", "scale")
_patched = set()


def install(model) -> int:
    """Route the one-row decode call of every gated-q GQA attention class of `model` (q_proj = [query | gate] per head, q / k norms, plain rope) through `step`
    when its cache is a TurboQuant cache (alone or the only user of a RowsKV). Idempotent; returns the number of classes patched."""
    if model is None or not ENABLED:
        return 0
    n = 0
    for _, m in model.named_modules():
        cls = type(m)
        if cls in _patched or not all(hasattr(m, a) for a in _REQ) or None in _heads(m):
            continue
        cls.__call__ = _wrap_call(cls.__call__)
        _patched.add(cls)
        n += 1
    return n


def _wrap_call(orig):
    """The patched __call__ of one attention class. `orig` is closed over, not a default argument: an attention that takes a 4th positional argument (mlx-vlm
    lfm3 passes position_ids) would otherwise overwrite it."""
    def call(self, x, mask=None, cache=None, *args, **kw):
        if cache is not None and x.shape[1] == 1 and not kw and not args:
            cs = cache.caches if hasattr(cache, "caches") else [cache]                  # RowsKV: one private cache per user (row b of the batch)
            if len(cs) == x.shape[0] and all(type(c) is TQ.TQKVCache for c in cs):
                qp, kp = self.q_proj(x), self.k_proj(x)                                  # lazy: unused when the stock chain runs below
                Hq, Hk = _heads(self)
                D = kp.shape[-1] // Hk
                gated = qp.shape[-1] == 2 * Hq * D
                if qp.shape[-1] in (Hq * D, 2 * Hq * D) and all(ok(c, self, qp[b:b + 1], gated) for b, c in enumerate(cs)):    # gated q (Qwen3.5 / 3.6) or plain q (Qwen3)
                    vp = self.v_proj(x)
                    outs = [step(c, self, qp[b:b + 1], kp[b:b + 1], vp[b:b + 1], self.scale, gated=gated) for b, c in enumerate(cs)]
                    return self.o_proj(outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0))
        return orig(self, x, mask, cache, *args, **kw)

    return call
