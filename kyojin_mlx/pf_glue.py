# SPDX-License-Identifier: Apache-2.0
"""Prompt-row glue of the DS4.1 attention block, one pass each (fp16 activations, bit-identical to the stock op chains).

The stock block moves the [T, H, D] activations through separate copy passes between the GEMMs (T = 4096: 537 MB per pass, 3.5 ms at the memory floor):
  q_rope      wq_b out [T, H * D] -> rope tail -> q [H, T, D]     stock: rope reading a transposed view (strided, general rope path)
  o_rope_inv  attention out [H, T, D] -> inverse rope tail -> wo_a input [G, T, (H / G) * D]     stock: inverse rope, then a transpose copy
  fq_t        wo_a out [G, T, r] -> fp8 round trip -> wo_b input [T, G * r]     stock: transpose copy, then the round trip
cos / sin are the factors of `mx.fast.rope` itself (fuse_attn.rope_range); the rotation is rounded to the activation dtype like the stock op."""
from __future__ import annotations

from . import archs as _A
import functools
import os

import mlx.core as mx
from .modeldesc import KERNEL as KG

from . import ablit as AB, fuse_attn as FA, wcache as WC

NH = int(os.environ.get("KYOJIN_PF_GLUE_NH", "2"))                  # heads per threadgroup
ENABLED = os.environ.get("KYOJIN_PF_GLUE", "1") != "0"
DEC_ROWS = os.environ.get("KYOJIN_PF_GLUE_DEC", "1") != "0"                     # the MoE combine also serves decode rows (R, 1, D); 0 = prompts only
STATS = {"block": 0, "block_fallback": 0}                           # prompt blocks that took the fused glue / the stock call (served /v1/stats "engage")
def _fq_on() -> bool:
    """False when the fp8 fake quant is switched off (mlx-vlm fakequant.DISABLE, the accuracy gate's --no-fq): the fused kernels then pass the values through."""
    return _A.fq_on()


_HEADER = ""                                              # default contraction: the stock rope kernel fuses the products (fma) like this

_ROPE_ROW = r"""
    // threadgroup = 64 threads = one row of D = 512 dims, thread t = dims 8 t .. 8 t + 7; rope tail = the last RD dims: interleaved pairs (mx.fast.rope traditional=True)
    float v[8];
    {
        const uint4 u = *(const device uint4*)(SRC);
        const thread T* e = (const thread T*)&u;
        for (uint i = 0; i < 8; ++i) { v[i] = float(e[i]); }
    }
    if (t * 8 >= uint(D - RD)) {
        const uint p0 = (t * 8 - uint(D - RD)) / 2;
        for (uint j = 0; j < 4; ++j) {
            const float c = rt[ulong(srow) * RD + 2 * (p0 + j)], s = rt[ulong(srow) * RD + 2 * (p0 + j) + 1];
            const float x1 = v[2 * j], x2 = v[2 * j + 1];
            v[2 * j] = float(static_cast<T>(x1 * c - x2 * s));
            v[2 * j + 1] = float(static_cast<T>(x1 * s + x2 * c));
        }
    }
    uint4 w;
    thread T* o = (thread T*)&w;
    for (uint i = 0; i < 8; ++i) { o[i] = static_cast<T>(v[i]); }
    *(device uint4*)(DST) = w;
"""
_Q_ROPE = r"""
    const uint t = thread_position_in_threadgroup.x;
    const uint h = threadgroup_position_in_grid.x * NH + thread_position_in_threadgroup.y, srow = threadgroup_position_in_grid.y;
""" + _ROPE_ROW.replace("SRC", "q + (ulong(srow) * H + h) * D + t * 8").replace("DST", "out + (ulong(h) * S + srow) * D + t * 8")
_O_ROPE = r"""
    const uint t = thread_position_in_threadgroup.x;
    const uint h = threadgroup_position_in_grid.x * NH + thread_position_in_threadgroup.y, srow = threadgroup_position_in_grid.y;
    const uint hg = H / G;
""" + _ROPE_ROW.replace("SRC", "ao + (ulong(h) * S + srow) * D + t * 8").replace("DST", "out + ((ulong(h / hg) * S + srow) * hg + (h % hg)) * D + t * 8")
_FQ_T = r"""
    // thread = 8 consecutive columns of a row of the [G, S, R] source, written to row s, columns g * R + c of [S, G * R]; 4 threads = one block of 32 (fake_quant_fp8_ue8m0)
    const uint c8 = thread_position_in_grid.x * 8;
    const uint s = thread_position_in_grid.y, g = thread_position_in_grid.z;
    float v[8];
    {
        const uint4 u = *(const device uint4*)(x + (ulong(g) * S + s) * R + c8);
        const thread T* e = (const thread T*)&u;
        for (uint i = 0; i < 8; ++i) { v[i] = float(e[i]); }
    }
    if (!FQ) {
        uint4 w;
        thread T* o = (thread T*)&w;
        for (uint i = 0; i < 8; ++i) { o[i] = static_cast<T>(v[i]); }
        *(device uint4*)(out + ulong(s) * G * R + ulong(g) * R + c8) = w;
        return;
    }
    float m = 0.0f;
    for (uint i = 0; i < 8; ++i) { m = metal::max(m, metal::abs(v[i])); }
    m = metal::max(m, simd_shuffle_xor(m, 1u));
    m = metal::max(m, simd_shuffle_xor(m, 2u));
    const float amax = metal::max(m, 1e-4f);
    const uint bits = as_type<uint>(amax * (1.0f / 448.0f));
    const uint exponent = (bits >> 23) + ((bits & 0x7fffff) != 0);
    const float scale = as_type<float>(exponent << 23);
    uint4 w;
    thread T* o = (thread T*)&w;
    for (uint i = 0; i < 8; ++i) {
        const float q = metal::clamp(v[i] / scale, -448.0f, 448.0f);
        const uint vbits = as_type<uint>(metal::abs(q));
        const uint qexp = metal::max(int(vbits >> 23) - 3, 118);
        const float step = as_type<float>(qexp << 23);
        o[i] = T((metal::rint(q / step) * step) * scale);
    }
    *(device uint4*)(out + ulong(s) * G * R + ulong(g) * R + c8) = w;
"""


@functools.lru_cache(maxsize=None)
def _k(name: str):
    src, ins = {"q": (_Q_ROPE, ["q", "rt"]), "o": (_O_ROPE, ["ao", "rt"]), "fq": (_FQ_T, ["x"])}[name]
    return mx.fast.metal_kernel(name=f"kyojin_pf_{name}", input_names=ins, output_names=["out"], source=src, header=_HEADER)


def supported(x: mx.array, D: int, RD: int) -> bool:
    return x.dtype in (mx.float16, mx.bfloat16) and D == KG.HEAD_DIM and RD % 8 == 0 and RD <= D


def q_rope(qb: mx.array, rope, start: int, H: int, D: int) -> mx.array:
    """qb [1, S, H * D] (wq_b out) -> [1, H, S, D] = rope(qb.reshape(1, S, H, D).transpose(0, 2, 1, 3), start)."""
    S = qb.shape[1]
    rt, _ = FA.rope_range(rope, start, S)
    return _k("q")(inputs=[qb, rt], template=[("T", qb.dtype), ("H", H), ("S", S), ("D", D), ("RD", rope.dims), ("NH", NH)], grid=(64 * H // NH, S * NH, 1), threadgroup=(64, NH, 1),
                   output_shapes=[(1, H, S, D)], output_dtypes=[qb.dtype])[0]


def o_rope_inv(o: mx.array, rope, start: int, G: int) -> mx.array:
    """o [1, H, S, D] (attention out) -> [1, G, S, (H / G) * D] = rope(o, start, inverse=True).reshape(1, G, H / G, S, D).transpose(0, 1, 3, 2, 4).flatten(-2)."""
    _, H, S, D = o.shape
    _, it = FA.rope_range(rope, start, S)
    return _k("o")(inputs=[o, it], template=[("T", o.dtype), ("H", H), ("S", S), ("D", D), ("RD", rope.dims), ("G", G), ("NH", NH)], grid=(64 * H // NH, S * NH, 1), threadgroup=(64, NH, 1),
                   output_shapes=[(1, G, S, H // G * D)], output_dtypes=[o.dtype])[0]


def fq_t(x: mx.array) -> mx.array:
    """x [1, G, S, R] (wo_a out) -> [1, S, G * R] = fake_quant_fp8_ue8m0(x.transpose(0, 2, 1, 3).flatten(-2))."""
    _, G, S, R = x.shape
    return _k("fq")(inputs=[x], template=[("T", x.dtype), ("G", G), ("S", S), ("R", R), ("FQ", _fq_on())], grid=(R // 8, S, G), threadgroup=(32, 1, 1),
                    output_shapes=[(1, S, G * R)], output_dtypes=[x.dtype])[0]


# ---------------------------------------------------------------------------------------------------------- mHC expand and FFN combine (prompt rows)
_EXPAND = r"""
    // thread = 8 consecutive dims d of row b: y[b, j, d] = T(post[b, j] * x[b, d] + sum_a comb[b, a, j] * res[b, a, d]), the sum sequential from 0 in fp32 (products rounded first,
    // contraction off: the arithmetic of fuse.mhc_expand without its fp32 copy of y, which prompts never read)
    const uint d8 = thread_position_in_grid.x * 8;
    const uint b = thread_position_in_grid.y;
    if (d8 >= D) { return; }
    float res[4][8], xv[8];
    for (uint a = 0; a < 4; ++a) {
        const uint4 u = *(const device uint4*)(r + (ulong(b) * 4 + a) * D + d8);
        const thread T* e = (const thread T*)&u;
        for (uint i = 0; i < 8; ++i) { res[a][i] = float(e[i]); }
    }
    {
        const uint4 u = *(const device uint4*)(x + ulong(b) * D + d8);
        const thread T* e = (const thread T*)&u;
        for (uint i = 0; i < 8; ++i) { xv[i] = float(e[i]); }
    }
    for (uint j = 0; j < 4; ++j) {
        const float pj = post[b * 4 + j];
        float cj[4];
        for (uint a = 0; a < 4; ++a) { cj[a] = comb[(ulong(b) * 4 + a) * 4 + j]; }
        uint4 w;
        thread T* o = (thread T*)&w;
        for (uint i = 0; i < 8; ++i) {
            float mixed = 0.0f;
            for (uint a = 0; a < 4; ++a) { mixed += cj[a] * res[a][i]; }
            const float v = pj * xv[i] + mixed;
            o[i] = static_cast<T>(is_same<T, half>::value ? clamp(v, -65504.0f, 65504.0f) : v);   // saturate at the fp16 range: see fuse.sat
        }
        *(device uint4*)(y + (ulong(b) * 4 + j) * D + d8) = w;
    }
"""
_COMBINE = r"""
    // out[i] = T(y[i] + float(sh[i])): the routed fp32 sum plus the shared expert, one pass
    const ulong i = ulong(thread_position_in_grid.x) * 8;
    const device float4* yp = (const device float4*)(y + i);
    const float4 y0 = yp[0], y1 = yp[1];
    const uint4 u = *(const device uint4*)(sh + i);
    const thread T* e = (const thread T*)&u;
    uint4 w;
    thread T* o = (thread T*)&w;
    const float yy[8] = {y0.x, y0.y, y0.z, y0.w, y1.x, y1.y, y1.z, y1.w};
    for (uint k = 0; k < 8; ++k) { o[k] = static_cast<T>(yy[k] + float(e[k])); }
    *(device uint4*)(out + i) = w;
"""


@functools.lru_cache(maxsize=None)
def _k2(name: str):
    src, ins, outs, hdr = {"expand": (_EXPAND, ["x", "r", "post", "comb"], ["y"], "#pragma METAL fp contract(off)\n"),
                           "combine": (_COMBINE, ["y", "sh"], ["out"], "#pragma METAL fp contract(off)\n")}[name]
    return mx.fast.metal_kernel(name=f"kyojin_pf_{name}", input_names=ins, output_names=outs, source=src, header=hdr)


def expand(x: mx.array, residual: mx.array, post: mx.array, comb: mx.array) -> mx.array:
    """x [1, S, D] fp16/bf16, residual [1, S, 4, D], post [1, S, 4] fp32, comb [1, S, 4, 4] fp32 -> [1, S, 4, D]; equals language.hc_expand."""
    _, S, D = x.shape
    return _k2("expand")(inputs=[x, residual, post, comb], template=[("T", x.dtype), ("D", D)], grid=(D // 8, S, 1), threadgroup=(128 if D % 1024 == 0 else 32, 1, 1),
                         output_shapes=[(1, S, 4, D)], output_dtypes=[x.dtype])[0]


def combine(y: mx.array, shared: mx.array) -> mx.array:
    """y fp32, shared fp16/bf16 (same number of elements, a multiple of 8) -> (y + shared.astype(fp32)).astype(shared.dtype), shape of shared."""
    return _k2("combine")(inputs=[y, shared], template=[("T", shared.dtype)], grid=(shared.size // 8, 1, 1), threadgroup=(256, 1, 1),
                          output_shapes=[shared.shape], output_dtypes=[shared.dtype])[0]


SHARED_MLP = os.environ.get("KYOJIN_SHARED_MLP_FUSE", "1") != "0"      # shared expert: swiglu and its cast to the stream dtype in one compiled launch
_swiglu_cast = {}


def _swiglu_h(limit: float, dt):
    """The stock compiled `_swiglu` (fp32 inside) with the cast to the stream dtype inside the same compiled region: one launch fewer, the same rounding."""
    key = (limit, dt)
    f = _swiglu_cast.get(key)
    if f is None:
        import mlx.nn as nn

        @mx.compile
        def f(gate, up):
            gate, up = gate.astype(mx.float32), up.astype(mx.float32)
            if limit > 0:
                gate = mx.minimum(gate, limit)
                up = mx.clip(up, -limit, limit)
            return (nn.silu(gate) * up).astype(dt)
        _swiglu_cast[key] = f
    return f


def _impl_shared_mlp(orig):
    """Replacement of `DeepseekV41MLP.__call__` (the shared expert, every layer, every row count): `activated.astype(dtype)` rides in the swiglu launch."""
    L = _A.lang()

    def __call__(self, x):
        if not SHARED_MLP or x.dtype not in (mx.float16, mx.bfloat16):
            return orig(self, x)
        dtype = x.dtype
        x = L.fake_quant_fp8_ue8m0(x)
        activated = _swiglu_h(self.swiglu_limit, dtype)(self.gate_proj(x), self.up_proj(x))
        return self.down_proj(L.fake_quant_fp8_ue8m0(activated))
    return __call__


def _impl_moe(orig):
    """Replacement of `DeepseekV41MoE.__call__` for prompts: the routed sum (fp32) and the shared expert are added and rounded in one pass (stock: sum copy, cast, add, cast)."""
    import importlib
    L = _A.lang()

    def __call__(self, x, image_mask=None):
        rows = x.ndim == 3 and x.shape[1] == 1 and DEC_ROWS                      # decode rows (R, 1, D): the same sum + add + casts, one pass
        if not ENABLED or x.ndim != 3 or not (rows or (x.shape[0] == 1 and x.shape[1] >= 64)) or x.dtype not in (mx.float16, mx.bfloat16) or x.shape[-1] % 8:
            return orig(self, x, image_mask)
        inds, scores = self.gate(x, image_mask)
        y = self.switch_mlp(L.fake_quant_fp8_ue8m0(x), inds, scores)
        if y.dtype != mx.float32 or y.ndim != 4 or y.shape[-2] != 1:
            return (y.astype(mx.float32).sum(-2) + self.shared_experts(x).astype(mx.float32)).astype(x.dtype)
        return combine(y, self.shared_experts(x))
    return __call__


# ---------------------------------------------------------------------------------------------------------- hc_pre + RMSNorm + fp8 round trip (prompt rows)
_PRENORM = r"""
    // One threadgroup of 1024 threads per row b (D = 5120). Thread lid owns dims 4 lid .. 4 lid + 3 and 4096 + 4 lid .. (lid < 256): the element split, the per-thread
    // accumulation order and the two-level simd_sum of mlx's `rms_looped` kernel, so the row scale is the stock one bit for bit.
    //   x_n[d] = w[d] * T(h[d] * rsqrt(mean(h^2) + eps)),  h[d] = T(sum_j pre[b, j] * x[b, j, d]) (sequential from 0, products rounded first: the stock hc_pre)
    //   xq     = fake_quant_fp8_ue8m0(x_n) (blocks of 32 = 8 threads)
    const uint lid = thread_position_in_threadgroup.x;
    const uint b = threadgroup_position_in_grid.y;
    const uint sg = simdgroup_index_in_threadgroup, ln = thread_index_in_simdgroup;
    float v[8];
    float acc = 0.0f;
    for (uint r = 0; r < 2; ++r) {
        const uint off = r * 4096 + lid * 4;
        if (off + 4 <= D) {
            for (uint i = 0; i < 4; ++i) {
                float a = 0.0f;
                for (uint j = 0; j < 4; ++j) { a += pre[b * 4 + j] * float(x[(ulong(b) * 4 + j) * D + off + i]); }
                const float h = float(static_cast<T>(is_same<T, half>::value ? clamp(a, -65504.0f, 65504.0f) : a));
                v[r * 4 + i] = h;
                acc += h * h;
            }
        }
    }
    threadgroup float ls[32];
    threadgroup float inv_s[1];
    acc = simd_sum(acc);
    if (sg == 0) { ls[ln] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ln == 0) { ls[sg] = acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) { acc = simd_sum(ls[ln]); if (ln == 0) { inv_s[0] = metal::precise::rsqrt(acc / float(D) + eps[0]); } }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = inv_s[0];
    for (uint r = 0; r < 2; ++r) {
        const uint off = r * 4096 + lid * 4;
        if (off + 4 <= D) {
            float n[4];
            float m = 0.0f;
            for (uint i = 0; i < 4; ++i) {
                const T t = w[off + i] * static_cast<T>(v[r * 4 + i] * inv);
                out[ulong(b) * D + off + i] = t;
                n[i] = float(t);
                m = metal::max(m, metal::abs(n[i]));
            }
            if (!FQ) {
                for (uint i = 0; i < 4; ++i) { outq[ulong(b) * D + off + i] = T(n[i]); }
                continue;
            }
            m = metal::max(m, simd_shuffle_xor(m, 1u));
            m = metal::max(m, simd_shuffle_xor(m, 2u));
            m = metal::max(m, simd_shuffle_xor(m, 4u));
            const float amax = metal::max(m, 1e-4f);
            const uint bits = as_type<uint>(amax * (1.0f / 448.0f));
            const uint exponent = (bits >> 23) + ((bits & 0x7fffff) != 0);
            const float scale = as_type<float>(exponent << 23);
            for (uint i = 0; i < 4; ++i) {
                const float q = metal::clamp(n[i] / scale, -448.0f, 448.0f);
                const uint vbits = as_type<uint>(metal::abs(q));
                const uint qexp = metal::max(int(vbits >> 23) - 3, 118);
                const float step = as_type<float>(qexp << 23);
                outq[ulong(b) * D + off + i] = T((metal::rint(q / step) * step) * scale);
            }
        }
    }
"""


@functools.lru_cache(maxsize=None)
def _k3():
    return mx.fast.metal_kernel(name="kyojin_pf_prenorm", input_names=["x", "pre", "w", "eps"], output_names=["out", "outq"], source=_PRENORM, header="#pragma METAL fp contract(off)\n")


def pre_norm_fq(h: mx.array, pre: mx.array, w: mx.array, eps: float) -> tuple[mx.array, mx.array]:
    """h [1, S, 4, D] fp16 (D = 5120), pre [1, S, 4] fp32, w [D] (the norm weight, activation dtype) -> (x_n, fq(x_n)) [1, S, D] with
    x_n = nn.RMSNorm(w)(hc_pre(h, pre)) and fq = fake_quant_fp8_ue8m0."""
    _, S, _, D = h.shape
    return tuple(_k3()(inputs=[h.reshape(S, 4, D), pre.reshape(S, 4), w, _eps(eps)], template=[("T", h.dtype), ("D", D), ("FQ", _fq_on())], grid=(1024, S, 1), threadgroup=(1024, 1, 1),
                       output_shapes=[(1, S, D)] * 2, output_dtypes=[h.dtype] * 2))


_eps_cache: dict = {}


def _eps(e: float) -> mx.array:
    a = _eps_cache.get(e)
    if a is None:
        a = _eps_cache[e] = mx.array([e], dtype=mx.float32)
    return a


# ---------------------------------------------------------------------------------------------------------- fp8 round trip memo and the prompt block
_ring: list = []          # (x, fq(x)) of the last calls, newest last; both kept alive so that `is` tests stay valid


def seed(x: mx.array, y: mx.array) -> None:
    from . import fqfuse as FQF
    _ring.append((x, y))
    del _ring[:-4]
    if y is not x and _fq_on():                      # fakequant.DISABLE: y is no fp8 round trip of x, a rot must not fold one in
        FQF.note(x, y)


def memoize_fq() -> bool:
    """DS4.1: `fake_quant_fp8_ue8m0(x)` once per input array (same output array for the same input array: attention wq_a / wkv, routed / shared experts read one x), and
    the outputs of `pre_norm_fq` are looked up here instead of being recomputed. Compatible with fused_group.memoize_fake_quant (which then leaves it alone)."""
    import importlib
    L = _A.lang()
    f = getattr(L, "fake_quant_fp8_ue8m0", None)
    if f is None:
        return False
    if not getattr(f, "_s5_memo", False):
        def g(x, *args, **kw):
            if args or kw or not ENABLED:
                return f(x, *args, **kw)
            for xi, yi in _ring:
                if xi is x:
                    return yi
            y = f(x)
            seed(x, y)
            return y
        g._s5_memo = True
        g.__wrapped__ = f
        L.fake_quant_fp8_ue8m0 = g
    return True


def _impl_block(orig):
    """Replacement of `DeepseekV41Block.__call__` for prompts (batch 1, >= 64 rows, hidden 5120): hc_pre + RMSNorm + fp8 round trip are one kernel (`pre_norm_fq`) per sublayer;
    the module calls are the stock ones otherwise (same order)."""
    import importlib
    L = _A.lang()
    cache_cls = _A.cls("cache")                       # resolved once at install, the per-call test stays one isinstance

    def __call__(self, h, pre_mix, image_mask, cache):
        if (not ENABLED or not isinstance(cache, cache_cls) or h.ndim != 4 or h.shape[0] != 1 or h.shape[1] < 64 or h.shape[2] != KG.HC or h.shape[3] != KG.PF_NORM_HIDDEN or h.dtype != mx.float16 or pre_mix.dtype != mx.float32
                or self.attn_norm.weight.dtype != h.dtype or self.ffn_norm.weight.dtype != h.dtype):
            STATS["block_fallback"] += h.ndim == 4 and h.shape[1] >= 64
            return orig(self, h, pre_mix, image_mask, cache)
        STATS["block"] += 1
        residual = h
        attn_pre, attn_post, attn_comb = self.hc_mixes(h, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x, xq = pre_norm_fq(h, pre_mix, self.attn_norm.weight, self.norm_eps)
        seed(x, xq)
        x = AB.attn_out(self, self.attn(x, cache))
        x = L.hc_expand(x, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        y, yq = pre_norm_fq(x, attn_pre, self.ffn_norm.weight, self.norm_eps)
        seed(y, yq)
        x = AB.ffn_out(self, self.ffn(y, image_mask))
        x = L.hc_expand(x, residual, ffn_post, ffn_comb)
        return x, ffn_pre
    return __call__


# ---------------------------------------------------------------------------------------------------------- wo_a on prompt rows: inverse rope + fp8 round trip + rot in one kernel
_FQ8 = r"""
    if (FQ) {
        float am = metal::max(metal::max(metal::abs(XV.x), metal::abs(XV.y)), metal::max(metal::abs(XV.z), metal::abs(XV.w)));
        am = metal::max(am, simd_shuffle_xor(am, 1));
        am = metal::max(am, simd_shuffle_xor(am, 2));
        am = metal::max(am, simd_shuffle_xor(am, 4));
        am = metal::max(am, 1e-4f);
        const uint amb = as_type<uint>(am * (1.0f / 448.0f));
        const float fq_scale = as_type<float>(((amb >> 23) + ((amb & 0x7fffff) != 0 ? 1u : 0u)) << 23);
FQ8_ELEMS
    }
"""
_FQ8_ELEM = r"""        { const float q = metal::clamp(XV.C / fq_scale, -448.0f, 448.0f);
          const uint qb = as_type<uint>(metal::abs(q));
          const float step = as_type<float>(uint(metal::max(int(qb >> 23) - 3, 118)) << 23);
          XV.C = float(half((metal::rint(q / step) * step) * fq_scale)); }
"""


def _fq8(var: str) -> str:
    return _FQ8.replace("FQ8_ELEMS", "".join(_FQ8_ELEM.replace(".C", f".{c}") for c in "xyzw")).replace("XV", var)


_ROT_ATT = r"""
    // xh[g, s] = H128(fq(rope_inv(o[g * HG + hl, s, :])) * suh_g) / sqrt(128) for the 8 heads hl of group g, row s: one simdgroup per 128-block (4 lanes-worth of 4 dims), the
    // attention output read in place ([H, S, D]); the stock chain is inverse rope -> reshape / transpose -> fp8 round trip -> rot (kernels._ROT_V1) per group.
    const uint lane = thread_index_in_simdgroup;
    const uint blk = threadgroup_position_in_grid.x * 4 + simdgroup_index_in_threadgroup;
    const uint s = threadgroup_position_in_grid.y, g = threadgroup_position_in_grid.z;
    const uint S = params[0], tc = params[1];
    if (blk >= IN / 128) return;
    const uint hl = blk / (D / 128), d = (blk % (D / 128)) * 128 + lane * 4;
    const uint c = s / tc, ls = s - c * tc;                                  // launch group of the row and its row in the group
    const uint rows = (c + 1) * tc <= S ? tc : S - c * tc;
    const device half* oc = (c == 0 ? o0 : c == 1 ? o1 : c == 2 ? o2 : c == 3 ? o3 : c == 4 ? o4 : c == 5 ? o5 : c == 6 ? o6 : o7);
    float4 xv = float4(*((const device half4*)(oc + ((ulong(g * HG + hl)) * rows + ls) * D + d)));
    if (d >= D - RD) {
        const uint p0 = (d - (D - RD)) / 2;
        const float c0 = it[ulong(s) * RD + 2 * p0], s0 = it[ulong(s) * RD + 2 * p0 + 1], c1 = it[ulong(s) * RD + 2 * p0 + 2], s1 = it[ulong(s) * RD + 2 * p0 + 3];
        const float a = xv.x, b = xv.y, c = xv.z, e = xv.w;
        xv = float4(float(half(a * c0 - b * s0)), float(half(a * s0 + b * c0)), float(half(c * c1 - e * s1)), float(half(c * s1 + e * c1)));
    }
FQ8_XV
    const float4 sv = float4(*((const device half4*)(suh + g * IN + blk * 128 + lane * 4)));
    float v0 = xv.x * sv.x, v1 = xv.y * sv.y, v2 = xv.z * sv.z, v3 = xv.w * sv.w;
BUTTERFLY
    *((device half4*)(xh + (ulong(g) * S + s) * IN + blk * 128 + lane * 4)) = half4(float4(v0, v1, v2, v3) * RSQRT);
"""
_FIN_ATT = r"""
    // out[s, g * R + c] = fq(H128(inner_g[s]) / sqrt(128) * svh_g): the group axis folded into the columns, the fp8 round trip of the wo_b input in the same pass
    const uint lane = thread_index_in_simdgroup;
    const uint blk = threadgroup_position_in_grid.x * 4 + simdgroup_index_in_threadgroup;
    const uint s = threadgroup_position_in_grid.y, g = threadgroup_position_in_grid.z;
    if (blk >= R / 128) return;
    const device half* part = PICK;
    float4 acc = float4(0.0f);
    acc += float4(*((const device half4*)(part + ulong(s) * R + blk * 128 + lane * 4)));
    float v0 = acc.x, v1 = acc.y, v2 = acc.z, v3 = acc.w;
BUTTERFLY
    const float4 sv = float4(*((const device half4*)(svh + g * R + blk * 128 + lane * 4)));
    const float4 o4 = float4(v0, v1, v2, v3) * RSQRT * sv;
    float4 xv = float4(half4(o4));
FQ8_XV
    *((device half4*)(out + ulong(s) * G * R + g * R + blk * 128 + lane * 4)) = half4(xv);
"""


@functools.lru_cache(maxsize=None)
def _k4(name: str):
    from . import metal_src as S
    if name == "rot":
        src = _ROT_ATT.replace("FQ8_XV", _fq8("xv")).replace("BUTTERFLY", S._BUTTERFLY).replace("RSQRT", S.RSQRT_HAD)
        return mx.fast.metal_kernel(header=S.HEADER, name="kyojin_pf_rot_att", input_names=[f"o{i}" for i in range(8)] + ["it", "suh", "params"], output_names=["xh"], source=src)
    pick = "(g == 0 ? p0 : g == 1 ? p1 : g == 2 ? p2 : g == 3 ? p3 : g == 4 ? p4 : g == 5 ? p5 : g == 6 ? p6 : p7)"
    src = _FIN_ATT.replace("PICK", pick).replace("FQ8_XV", _fq8("xv")).replace("BUTTERFLY", S._BUTTERFLY).replace("RSQRT", S.RSQRT_HAD)
    return mx.fast.metal_kernel(header=S.HEADER, name="kyojin_pf_fin_att", input_names=[f"p{i}" for i in range(8)] + ["svh"], output_names=["out"], source=src)


def _o_shape(o):
    """(1, H, S, D) of the attention output, an array [1, H, S, D] or the list of launch-group outputs [H, rows, D]."""
    if isinstance(o, list):
        return (1, o[0].shape[0], sum(c.shape[1] for c in o), o[0].shape[2])
    return o.shape


def unwrap(m):
    """The linear under a passive rows.Hoist / rows.Sink (the spec path wraps the attention linears; a prompt chunk never arms them), else `m`."""
    from . import rows as RW
    if type(m) is RW.Hoist and m.st.pre is None or type(m) is RW.Sink and m.st.grab is None:
        return m.inner
    return m


def wo_a_ok(ml, o) -> bool:
    """`ml` an Exl3MultiLinear with 8 groups, o [1, H, S, D] fp16 attention output (D = 512): rows above the fused-dense crossover (the decode + matmul path)."""
    from . import exl3_linear as EL
    from . import patch as P
    shp = _o_shape(o)
    return (ENABLED and type(ml) is P.Exl3MultiLinear and getattr(ml, "fq_in", False) and ml.num_groups == KG.O_GROUPS and (o[0] if isinstance(o, list) else o).dtype == mx.float16 and len(shp) == 4 and shp[0] == 1 and shp[-1] == KG.HEAD_DIM
            and shp[1] % KG.O_GROUPS == 0 and shp[1] // KG.O_GROUPS * KG.HEAD_DIM == ml.in_features and ml.in_features % 128 == 0 and ml.out_features % 128 == 0 and shp[2] > EL.DENSE_FUSED_MAX
            and not ml.groups[0]._fused_ok(shp[2]))


def wo_a_prompt(ml, o: mx.array, rope, start: int) -> mx.array:
    """fq(wo_a(fq(rope_inv(o) regrouped)) regrouped) for the prompt rows of `o` [1, H, S, D]: [1, S, G * R], the wo_b input. Bit-identical to the stock chain
    (inverse rope, regroup, fp8 round trip, per-group dense EXL3 linear, stack, regroup, fp8 round trip)."""
    from . import kernels as KN
    _, H, S, D = _o_shape(o)
    G, IN, R = ml.num_groups, ml.in_features, ml.out_features
    parts = o if isinstance(o, list) else [o[0]]
    tc = parts[0].shape[1]
    assert len(parts) <= 8 and all(c.shape[1] == tc for c in parts[:-1])
    parts = parts + [parts[0]] * (8 - len(parts))
    st = ml.__dict__.get("_pf_stack")
    if st is None:
        st = ml.__dict__["_pf_stack"] = (mx.stack([g._suh.astype(mx.float16) for g in ml.groups]), mx.stack([g._svh.astype(mx.float16) for g in ml.groups]))
    suh, svh = st
    _, it = FA.rope_range(rope, start, S)
    params = mx.array([S, tc], dtype=mx.uint32)
    xh = _k4("rot")(inputs=parts + [it, suh, params], template=[("IN", IN), ("D", D), ("RD", rope.dims), ("HG", H // G), ("FQ", _fq_on())], grid=(-(-(IN // 128) // 4) * 128, S, G), threadgroup=(128, 1, 1),
                    output_shapes=[(G, S, IN)], output_dtypes=[mx.float16])[0]
    inner = []
    for g, lin in enumerate(ml.groups):
        wr = WC.decode_wr(lin)
        inner.append(xh[g] @ wr.T)
    return _k4("fin")(inputs=inner + [svh], template=[("R", R), ("G", G), ("FQ", _fq_on())], grid=(-(-(R // 128) // 4) * 128, S, G), threadgroup=(128, 1, 1),
                      output_shapes=[(1, S, G * R)], output_dtypes=[mx.float16])[0]


# ---------------------------------------------------------------------------------------------------------- wq_b on prompt rows: finalize with the rope tail, written as [H, S, D]
_FIN_Q = r"""
    // q[h, s, :] = rope(H128(inner[s, 128-blocks of head h]) / sqrt(128) * svh): the finalize of the dense EXL3 linear (kernels._FINAL_V1, one k-block, fp16 partials) with the rope tail
    // of the head applied to its fp16 result and the output laid out [H, S, D] (the stock chain: finalize, reshape / transpose, rope)
    const uint lane = thread_index_in_simdgroup;
    const uint blk = threadgroup_position_in_grid.x * 4 + simdgroup_index_in_threadgroup;
    const uint s = threadgroup_position_in_grid.y;
    const uint N = H * D;
    if (blk * 128 >= N) return;
    float4 acc = float4(0.0f);
    acc += float4(*((const device half4*)(part + ulong(s) * N + blk * 128 + lane * 4)));
    float v0 = acc.x, v1 = acc.y, v2 = acc.z, v3 = acc.w;
BUTTERFLY
    const float4 sv = float4(*((const device half4*)(svh + blk * 128 + lane * 4)));
    const float4 o4 = float4(v0, v1, v2, v3) * RSQRT * sv;
    float4 xv = float4(half4(o4));
    const uint h = blk / (D / 128), d = (blk % (D / 128)) * 128 + lane * 4;
    if (d >= D - RD) {
        const uint p0 = (d - (D - RD)) / 2;
        const float c0 = rt[ulong(s) * RD + 2 * p0], s0 = rt[ulong(s) * RD + 2 * p0 + 1], c1 = rt[ulong(s) * RD + 2 * p0 + 2], s1 = rt[ulong(s) * RD + 2 * p0 + 3];
        const float a = xv.x, b = xv.y, c = xv.z, e = xv.w;
        xv = float4(float(half(a * c0 - b * s0)), float(half(a * s0 + b * c0)), float(half(c * c1 - e * s1)), float(half(c * s1 + e * c1)));
    }
    *((device half4*)(q + (ulong(h) * S + s) * D + d)) = half4(xv);
"""


@functools.lru_cache(maxsize=None)
def _k5():
    from . import metal_src as S
    src = _FIN_Q.replace("BUTTERFLY", S._BUTTERFLY).replace("RSQRT", S.RSQRT_HAD)
    return mx.fast.metal_kernel(header=S.HEADER, name="kyojin_pf_fin_q", input_names=["part", "svh", "rt"], output_names=["q"], source=src)


def wq_b_ok(lin, x: mx.array, H: int, D: int) -> bool:
    from . import exl3_linear as EL
    return (ENABLED and type(lin) is EL.Exl3Linear and x.dtype == mx.float16 and x.ndim == 3 and x.shape[0] == 1 and D == KG.HEAD_DIM and lin.out_features == H * D and lin.in_features % 128 == 0
            and lin._bias is None and lin.out_dtype == mx.float16 and x.shape[1] > EL.DENSE_FUSED_MAX and not lin._fused_ok(x.shape[1]))


def wq_b_rope(lin, x: mx.array, rope, start: int, H: int, D: int) -> mx.array:
    """x [1, S, in] (fp8-rounded q latent) -> [1, H, S, D] = rope(lin(x).reshape(1, S, H, D).transpose(0, 2, 1, 3), start), bit for bit."""
    from . import kernels as KN
    from . import fqfuse as FQF
    S = x.shape[1]
    src = FQF.source(x)                                     # x = fq(src): the round trip inside the rot
    xh = KN.rot((x if src is None else src).reshape(S, -1), lin._suh, mx.float16, fq8=src is not None)
    wr = WC.decode_wr(lin)
    inner = xh @ wr.T
    rt, _ = FA.rope_range(rope, start, S)
    return _k5()(inputs=[inner, lin._svh if lin._svh.dtype == mx.float16 else lin._svh.astype(mx.float16), rt], template=[("H", H), ("D", D), ("RD", rope.dims), ("S", S)],
                 grid=(-(-(H * D // 128) // 4) * 128, S, 1), threadgroup=(128, 1, 1), output_shapes=[(1, H, S, D)], output_dtypes=[mx.float16])[0]
