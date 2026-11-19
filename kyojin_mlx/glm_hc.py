# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 mHC glue at decode, two launches per branch (flag KYOJIN_GLM_HC_FUSE, G3b).

Stock decode (L = 1) of one hyper-connection branch is: astype fp32, rms_norm of the 4 streams, the [24, 4H] mix GEMV (rowdot), the sinkhorn + collapse
kernel, the input RMSNorm of the branch, [branch], then the compiled expand (post * out + comb^T @ residual: a multiply, a matmul, an add, two casts).
About ten launches per branch, ~20 per layer, ~900 per token, almost all of them tiny.

Here a branch is
    B  (`_B`)  one threadgroup per token: reduces the partial sums handed over by the previous kernel, mixes = dots * rsqrt(mean(x^2) + eps),
               pre / post / comb (sigmoid, softmax, 20 sinkhorn iterations in simdgroup 0), collapse of the 4 streams with pre, RMSNorm of the
               collapsed row with the branch's input norm weight -> the branch input, plus post / comb for the expand;
    [branch]
    A  (`_A`)  the expand (the new 4 streams in the model dtype) FUSED with the partial sums the NEXT branch's B needs: for every (stream, 64-wide chunk) a
               threadgroup produces x_new, rounds it to the model dtype, and reduces sum(x^2) and the 24 dot products with the next branch's mix weight.
So the glue is 2 launches per branch (the first branch of a pass: a partial-sums-only A, no expand; the last branch: expand only).

Exactness: sinkhorn / sigmoid / softmax are the stock kernel's arithmetic (fp32, metal::fast::exp). The mix is dot(x, fn) * rsqrt(mean(x^2) + eps) instead
of dot(x * rsqrt, fn): same value up to fp32 rounding (1e-7 relative). Every token is reduced independently in a fixed order, so a row has the same bits
alone or in a batch of rows (row invariance).

Hand-over between layers: A's partials ride in a one-slot side channel keyed by the identity of the output array and of the next branch's mix weight;
a miss (the array came from elsewhere) recomputes the partials with the partials-only A. Nothing but identity is trusted."""
from __future__ import annotations

import functools
import os

import mlx.core as mx

from . import fired as FD

ENABLED = os.environ.get("KYOJIN_GLM_HC_FUSE", "0") not in ("0", "")
MAX_TOKENS = int(os.environ.get("KYOJIN_GLM_HC_MAX_TOKENS", "16"))     # B * L above this = prompt rows: the stock path (prefill mHC is G5b's)
CW = 64                                                                 # hidden positions per A threadgroup (4 streams x 64 = 256 threads)
NPART = 25                                                              # sum(x^2) + 24 dot products
STATS = {"fused": 0, "stock": 0, "miss": 0}

_HEADER = ""

_SRC_A = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const uint c = threadgroup_position_in_grid.x, tok = threadgroup_position_in_grid.y;
    constexpr uint K = 4 * H, NC = H / 64;
    const uint s = tid >> 6, h = c * 64 + (tid & 63);
    float v;
    EXPAND_BODY
    threadgroup float red[8 * 25];
    float acc[25];
    NEXT_BODY
"""

_EXPAND = r"""
    const device float* pcp = pc + tok * 20;
    float y = pcp[s] * float(o[tok * H + h]);
    for (uint j = 0; j < 4; ++j) y += pcp[4 + j * 4 + s] * float(x[(tok * 4 + j) * H + h]);
    const T yt = static_cast<T>(y);
    xn[(tok * 4 + s) * H + h] = yt;
    v = float(yt);
"""
_NOEXPAND = r"""
    v = float(x[(tok * 4 + s) * H + h]);
"""
_NEXT = r"""
    acc[0] = v * v;
    for (uint j = 0; j < 24; ++j) acc[1 + j] = v * fn[j * K + s * H + h];
    for (uint i = 0; i < 25; ++i) {
        const float r = simd_sum(acc[i]);
        if (lane == 0) red[sg * 25 + i] = r;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 25) {
        float t = 0.0f;
        for (uint g = 0; g < 8; ++g) t += red[g * 25 + tid];
        part[(tok * NC + c) * 25 + tid] = t;
    }
"""


@functools.lru_cache(maxsize=None)
def _kernel_a(expand: bool, nxt: bool):
    src = _SRC_A.replace("EXPAND_BODY", _EXPAND if expand else _NOEXPAND).replace("NEXT_BODY", _NEXT if nxt else "")
    ins = (["o", "pc"] if expand else []) + ["x"] + (["fn"] if nxt else [])
    outs = (["xn"] if expand else []) + (["part"] if nxt else [])
    if not nxt:
        src = src.replace("threadgroup float red[8 * 25];", "").replace("float acc[25];", "")
    return mx.fast.metal_kernel(name=f"kyojin_glm_hc_a{int(expand)}{int(nxt)}", input_names=ins, output_names=outs, source=src)


_SRC_B = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const uint tok = threadgroup_position_in_grid.x;
    constexpr uint K = 4 * H, NC = H / 64, NV = H / 1024;
    constexpr float EPS = EPS_INT * 1e-9f;
    constexpr float NEPS = NEPS_INT * 1e-9f;
    threadgroup float red[8 * 25];
    threadgroup float tot[25];
    threadgroup float mixv[24];
    threadgroup float pre_s[4];
    threadgroup float wsum[8];

    // 1. the partial sums of the previous kernel, fixed order
    if (tid < 200) {
        const uint j = tid % 25, g = tid / 25;
        float a = 0.0f;
        for (uint cc = g; cc < NC; cc += 8) a += part[(tok * NC + cc) * 25 + j];
        red[g * 25 + j] = a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 25) {
        float t = 0.0f;
        for (uint g = 0; g < 8; ++g) t += red[g * 25 + tid];
        tot[tid] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rs = metal::precise::rsqrt(tot[0] / float(K) + MEPS);
    if (tid < 24) mixv[tid] = tot[1 + tid] * rs;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. pre / post / comb (the stock sinkhorn kernel's arithmetic), simdgroup 0
    if (sg == 0) {
        const float pre_scale = scale[0], post_scale = scale[1], comb_scale = scale[2];
        const float active = (lane < 4u) ? 1.0f : 0.0f;
        const uint llane = metal::min(lane, 3u);
        float pre_z = mixv[llane] * pre_scale + base[llane];
        float post_z = mixv[4 + llane] * post_scale + base[4 + llane];
        float pre_v = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + EPS;
        float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));
        if (lane < 4u) {
            pre_s[lane] = pre_v;
            pc[tok * 20 + lane] = post_v;
        }
        float4 v4 = (float4(mixv[8 + llane * 4], mixv[9 + llane * 4], mixv[10 + llane * 4], mixv[11 + llane * 4]) * comb_scale
                     + *(const device float4*)(base + 8 + llane * 4)) * active;
        float row_max = metal::max(metal::max(v4.x, v4.y), metal::max(v4.z, v4.w));
        float4 e = metal::fast::exp(v4 - row_max) * active;
        float4 r = e * (1.0f / (e.x + e.y + e.z + e.w + EPS)) + EPS * active;
        float4 col_inv = 1.0f / (float4(simd_sum(r.x), simd_sum(r.y), simd_sum(r.z), simd_sum(r.w)) + EPS);
        r *= col_inv;
        for (int it = 1; it < ITERS; ++it) {
            r *= (1.0f / (r.x + r.y + r.z + r.w + EPS)) * active;
            col_inv = 1.0f / (float4(simd_sum(r.x), simd_sum(r.y), simd_sum(r.z), simd_sum(r.w)) + EPS);
            r *= col_inv;
        }
        if (lane < 4u) *(device float4*)(pc + tok * 20 + 4 + lane * 4) = r;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. collapse of the 4 streams with pre (rounded to the model dtype, as the stock kernel's output), sum of squares of the rounded row
    const float p0 = pre_s[0], p1 = pre_s[1], p2 = pre_s[2], p3 = pre_s[3];
    float4 col[NV];
    float ss = 0.0f;
    for (uint i = 0; i < NV; ++i) {
        const uint h4 = tid + i * 256;
        const device vec<T, 4>* xp = (const device vec<T, 4>*)(x + tok * 4 * H) + h4;
        float4 a = p0 * float4(xp[0]);
        a = metal::fma(float4(p1), float4(xp[H / 4]), a);
        a = metal::fma(float4(p2), float4(xp[2 * (H / 4)]), a);
        a = metal::fma(float4(p3), float4(xp[3 * (H / 4)]), a);
        a = float4(vec<T, 4>(a));
        col[i] = a;
        ss += a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
    }
    ss = simd_sum(ss);
    if (lane == 0) wsum[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = 0.0f;
    for (uint g = 0; g < 8; ++g) t += wsum[g];
    const float nrm = metal::precise::rsqrt(t / float(H) + NEPS);

    // 4. the branch's input RMSNorm (mlx rms_norm: w * T(x * nrm))
    for (uint i = 0; i < NV; ++i) {
        const uint h4 = tid + i * 256;
        const vec<T, 4> q = vec<T, 4>(col[i] * nrm);
        const vec<T, 4> w4 = ((const device vec<T, 4>*)nw)[h4];
        ((device vec<T, 4>*)xo)[tok * (H / 4) + h4] = w4 * q;
    }
"""


@functools.lru_cache(maxsize=None)
def _kernel_b():
    src = _SRC_B.replace("MEPS", "NEPS")
    return mx.fast.metal_kernel(name="kyojin_glm_hc_b", input_names=["x", "part", "scale", "base", "nw"], output_names=["xo", "pc"], source=src)


# ------------------------------------------------------------------------------------------------------------------------------ launches
def why_shape(x, hc):
    """None when the kernels serve x / hc, else the reason (flag audit key; the guards of `ok_shape`, same order)."""
    if x.ndim != 4 or x.shape[2] != 4:
        return "x_shape"
    if x.dtype not in (mx.float16, mx.bfloat16):
        return "x_dtype=" + str(x.dtype).split(".")[-1]
    H = x.shape[3]
    if x.shape[0] * x.shape[1] > MAX_TOKENS:
        return f"tokens>{MAX_TOKENS}"
    if H % 1024 != 0:
        return "hidden%1024"
    if hc.fn.dtype not in (mx.float32, mx.float16):
        return "fn_dtype=" + str(hc.fn.dtype).split(".")[-1]
    if hc.fn.shape != (24, 4 * H):
        return "fn_shape"
    if hc.base.dtype != mx.float32 or hc.scale.dtype != mx.float32:
        return "base_scale_dtype"
    if hc.hc_mult != 4:
        return "hc_mult"
    return None


def ok_shape(x, hc) -> bool:
    """x [B, L, 4, H] in fp16 / bf16, B * L <= MAX_TOKENS, hc an mHC with a 4-stream [24, 4H] mix weight (fp32, or fp16 as the served loader stores it: the kernel reads either exactly)."""
    return why_shape(x, hc) is None


def partials(x, fn):
    """The partial sums of x (a stream state, no expand) against mix weight fn: [T, H/64, 25]."""
    B, L, _, H = x.shape
    T = B * L
    return _kernel_a(False, True)(inputs=[x, fn], template=[("T", x.dtype), ("H", H)], grid=((H // CW) * 256, T, 1), threadgroup=(256, 1, 1),
                                   output_shapes=[(T, H // CW, NPART)], output_dtypes=[mx.float32])[0]


_NW: dict = {}


def _norm_w(w, dt):
    """The branch norm weight in the activation dtype. A weight stored in another dtype was cast on every call (one launch per layer and token); with KYOJIN_GLM_FUSE_DSA (G10c)
    the cast is made once per (weight, dtype) outside any trace and kept (the weights are constants of the served model)."""
    if w.dtype == dt:
        return w
    from . import glm_fuse_dsa
    if not glm_fuse_dsa.ENABLED:
        return w.astype(dt)
    e = _NW.get((id(w), dt))
    if e is None or e[0] is not w:
        c = w.astype(dt)
        try:
            mx.eval(c)
        except Exception:                                      # a tracer (mx.compile region): no cache
            return c
        e = _NW[(id(w), dt)] = (w, c)
    return e[1]


def pre(x, part, hc, norm_w, norm_eps):
    """B: (branch input [B, L, H] in x dtype, post/comb [T, 20] fp32)."""
    B, L, _, H = x.shape
    T = B * L
    xo, pc = _kernel_b()(inputs=[x, part, hc.scale, hc.base, _norm_w(norm_w, x.dtype)],
                         template=[("T", x.dtype), ("H", H), ("ITERS", hc.sinkhorn_iters), ("EPS_INT", round(hc.hc_eps / 1e-9)), ("NEPS_INT", round(norm_eps / 1e-9))],
                         grid=(T * 256, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(B, L, H), (T, 20)], output_dtypes=[x.dtype, mx.float32])
    return xo, pc


def expand(out, x, pc, fn_next):
    """A: the new streams [B, L, 4, H] (and the partial sums against fn_next, or None)."""
    B, L, _, H = x.shape
    T = B * L
    nxt = fn_next is not None
    k = _kernel_a(True, nxt)
    shapes = [x.shape] + ([(T, H // CW, NPART)] if nxt else [])
    dts = [x.dtype] + ([mx.float32] if nxt else [])
    r = k(inputs=[out, pc, x] + ([fn_next] if nxt else []), template=[("T", x.dtype), ("H", H)], grid=((H // CW) * 256, T, 1), threadgroup=(256, 1, 1),
          output_shapes=shapes, output_dtypes=dts)
    return (r[0], r[1]) if nxt else (r[0], None)


# ------------------------------------------------------------------------------------------------------------------------------ layer call
class _Slot:
    x = None
    part = None
    fn = None


SLOT = _Slot()


def _take(x, hc):
    s = SLOT
    if s.x is x and s.fn is hc.fn:
        part = s.part
        s.x = s.part = s.fn = None
        return part
    STATS["miss"] += 1
    return partials(x, hc.fn)


def _layer_call(self, x, mask=None, cache=None, prev_topk_indices=None):
    a, f = self.attn_hc, self.ffn_hc
    if not (ok_shape(x, a) and ok_shape(x, f)):                 # (ok_shape stays the decision: glm_compile and its tests swap it)
        STATS["stock"] += 1
        FD.fb("hc_fuse", why_shape(x, a) or why_shape(x, f) or "?")
        return self._stock_call(x, mask, cache, prev_topk_indices)
    from . import glm_hc_defer as DF
    if DF.ENABLED:                                              # G10d: one launch per branch on the dependent chain
        return DF.layer_call(self, x, mask, cache, prev_topk_indices)
    STATS["fused"] += 1
    FD.fire("hc_fuse")
    ne = self.input_layernorm.eps
    part = _take(x, a)
    xo, pc = pre(x, part, a, self.input_layernorm.weight, ne)
    if self.is_linear:
        o = self.self_attn(xo, mask, cache)
        topk = prev_topk_indices
    else:
        o, topk = self.self_attn(xo, mask, cache, prev_topk_indices)
    x, part = expand(o, x, pc, f.fn)
    xo, pc = pre(x, part, f, self.post_attention_layernorm.weight, self.post_attention_layernorm.eps)
    o = self.mlp(xo)
    nxt = self.__dict__.get("_g3b_next")
    x2, part = expand(o, x, pc, None if nxt is None else nxt.fn)
    if part is not None:
        SLOT.x, SLOT.part, SLOT.fn = x2, part, nxt.fn
    return x2, topk


def install(model, log=print) -> int:
    """Swap the call of every trunk decoder layer (language_model.model.layers). Returns the number of layers."""
    if not ENABLED:
        return 0
    layers = model.language_model.model.layers
    n = 0
    for i, ly in enumerate(layers):
        cls = ly.__class__
        if getattr(cls, "_g3b_hc", False) or not hasattr(ly, "attn_hc"):
            continue
        ly.__dict__["_g3b_next"] = layers[i + 1].attn_hc if i + 1 < len(layers) else None
        ly.__class__ = type("G3b" + cls.__name__, (cls,), {"__call__": _layer_call, "_stock_call": cls.__call__, "_g3b_hc": True})
        n += 1
    if n:
        log(f"[glm_hc] {n} decoder layers: mHC glue in 2 launches per branch (<= {MAX_TOKENS} tokens per call)")
    return n
