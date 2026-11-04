# SPDX-License-Identifier: Apache-2.0
"""Launch cuts for the decode step: small ops that the stock model spreads over many kernels become one kernel each, with the arithmetic of the
stock sequence (same formulas, same summation order, `fp contract(off)`), so the result is bit-identical to the unfused path. tests/test_fuse.py
compares every kernel with the stock op sequence on many random inputs (ties and extremes included).

Why: a decode token of a 40-layer model is ~5000 launches; the CPU needs ~5 us per launch (python build + encode), so on a wide GPU (M5 Ultra) the
step is CPU bound, not bandwidth bound. Every launch removed is ~5 us of the token time.

Each cut has a switch (module attribute, env KYOJIN_FUSE=0 turns all off) to A/B inside one process. Every kernel is row independent: a row gets
the same bits whatever the number of rows (multi-user decode and speculative verify rely on it).

  router_select   sqrt(softplus(logits)) + correction bias, top-k by (value desc, index asc), weights normalised and scaled: 14 launches -> 2
  mhc_expand      hyper-connection expand (post * x + comb^T residual) for one token: 5 launches -> 1, also writes the fp32 copy the next mix reads
  mhc_pre         hyper-connection collapse (sum of pre-weighted copies): 3 launches -> 1
"""
from __future__ import annotations

import os

import mlx.core as mx

ENABLED = os.environ.get("KYOJIN_FUSE", "1") != "0"
ROUTER = True
MHC = True
MIXPRE = os.environ.get("KYOJIN_FUSE_MIXPRE", "1") != "0"

_HEADER = "#pragma METAL fp contract(off)\n"

# log1p of mlx/backend/metal/kernels/utils.h (Goldberg), so that softplus matches mx.logaddexp(x, 0) bit for bit
_LOG1P = r"""
inline float k_log1p(float x) {
  float xp1 = 1.0f + x;
  if (xp1 == metal::numeric_limits<float>::max()) { return metal::numeric_limits<float>::max(); }
  if (xp1 == 1.0f) { return x; }
  return x * (metal::log(xp1) / (xp1 - 1.0f));
}
"""

_ROUTER = r"""
    // one threadgroup (128 threads) per row: scores, selection key, K rounds of (min key, min index), normalised weights.
    const uint r = threadgroup_position_in_grid.x;
    const uint t = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    threadgroup float sval[E];
    threadgroup float skey[E];
    threadgroup float rk[4];
    threadgroup uint ri[4];
    threadgroup uint chosen[K];
    const float inf = metal::numeric_limits<float>::infinity();
    for (uint e = t; e < E; e += 128) {
        const float x = lg[ulong(r) * E + e];
        const float maxv = metal::max(x, 0.0f);
        const float minv = metal::min(x, 0.0f);
        float sp = (minv == -inf || maxv == inf) ? maxv : (maxv + k_log1p(metal::exp(minv - maxv)));
        if (metal::isnan(x)) { sp = metal::numeric_limits<float>::quiet_NaN(); }
        const float s = metal::precise::sqrt(sp);
        sval[e] = s;
        const float b = BIAS_EXPR;
        skey[e] = -(s + b);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < K; ++j) {
        float bk = inf;
        uint bi = 0xffffffffu;
        for (uint e = t; e < E; e += 128) {
            const float k = skey[e];
            if (k < bk || (k == bk && e < bi)) { bk = k; bi = e; }
        }
        const float mk = simd_min(bk);
        const uint mi = simd_min(bk == mk ? bi : 0xffffffffu);
        if (lane == 0) { rk[sg] = mk; ri[sg] = mi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0) {
            float fk = rk[0];
            uint fi = ri[0];
            for (uint g = 1; g < 4; ++g) {
                if (rk[g] < fk || (rk[g] == fk && ri[g] < fi)) { fk = rk[g]; fi = ri[g]; }
            }
            chosen[j] = fi;
            skey[fi] = inf;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0) {
        float w[K];
        float sum = 0.0f;
        for (uint j = 0; j < K; ++j) { w[j] = sval[chosen[j]]; sum += w[j]; }
        const float den = sum + par[1];
        for (uint j = 0; j < K; ++j) {
            ids[ulong(r) * K + j] = int(chosen[j]);
            const float wn = NORM ? (w[j] / den) : w[j];
            wts[ulong(r) * K + j] = wn * par[0];
        }
    }
"""
_k_router: dict = {}
_par: dict = {}


def router_select(logits: mx.array, bias: mx.array, top_k: int, norm: bool, scale: float, bias_vl: mx.array | None = None,
                  image_mask: mx.array | None = None) -> tuple[mx.array, mx.array]:
    """logits fp32 [R, E] (router matmul), bias fp32 [E] -> (ids int32 [R, K], weights fp32 [R, K]); the stock sequence
    scores = sqrt(softplus(logits)); ids = argpartition(-(scores + bias), K - 1)[:K]; w = take(scores, ids); w = w / (w.sum() + 1e-20); w * scale.
    image_mask (bool, R values) with bias_vl selects the bias per row like mx.where(image_mask[..., None], bias_vl, bias)."""
    masked = image_mask is not None
    from . import alucut as AC
    r1 = AC.router1()                                                              # KYOJIN_ALU_ROUTER1: one barrier per selection round (exact)
    kern = _k_router.get((masked, r1))
    if kern is None:
        kern = _k_router[(masked, r1)] = mx.fast.metal_kernel(
            name=f"kyojin_router_select{int(masked)}" + ("_r1" if r1 else ""), input_names=["lg", "bias"] + (["bias_vl", "msk"] if masked else []) + ["par"],
            output_names=["ids", "wts"], header=_HEADER + _LOG1P,
            source=(AC.router_source(_ROUTER) if r1 else _ROUTER).replace("BIAS_EXPR", "(msk[r] ? bias_vl[e] : bias[e])" if masked else "bias[e]"))
    R, E = logits.shape
    p = _par.get(scale)
    if p is None:
        p = _par[scale] = mx.array([scale, 1e-20], dtype=mx.float32)
    ins = [logits, bias] + ([bias_vl, image_mask.reshape(R)] if masked else []) + [p]
    return kern(inputs=ins, template=[("E", E), ("K", top_k), ("NORM", bool(norm))], grid=(128 * R, 1, 1), threadgroup=(128, 1, 1),
                output_shapes=[(R, top_k), (R, top_k)], output_dtypes=[mx.int32, mx.float32])


# --------------------------------------------------------------------------------------------------------------- hyper-connections (mHC)
_EXPAND = r"""
    // y[b, j, d] = T(post[b, j] * x[b, d] + sum_a comb[b, a, j] * res[b, a, d]); the sum is sequential from 0 in fp32 (products rounded first),
    // the last multiply-add is an fma (the stock compiled kernel contracts it); y32 is the exact fp32 copy of the rounded y.
    const uint d = thread_position_in_grid.x;
    const uint b = thread_position_in_grid.y;
    if (d >= D) { return; }
    float res[4];
    for (uint a = 0; a < 4; ++a) { res[a] = float(r[(ulong(b) * 4 + a) * D + d]); }
    const float xv = float(x[ulong(b) * D + d]);
    for (uint j = 0; j < 4; ++j) {
        float mixed = 0.0f;
        for (uint a = 0; a < 4; ++a) { mixed += comb[(ulong(b) * 4 + a) * 4 + j] * res[a]; }
        const float v = FMA_LAST ? fma(post[b * 4 + j], xv, mixed) : (post[b * 4 + j] * xv + mixed);
        const T o = static_cast<T>(is_same<T, half>::value ? clamp(v, -65504.0f, 65504.0f) : v);   // saturate at the fp16 range: see fuse.sat
        y[(ulong(b) * 4 + j) * D + d] = o;
        y32[(ulong(b) * 4 + j) * D + d] = float(o);
    }
"""
_PRE = r"""
    // y[b, d] = T(sum_j pre[b, j] * x[b, j, d]) sequential from 0 in fp32.
    const uint d = thread_position_in_grid.x;
    const uint b = thread_position_in_grid.y;
    if (d >= D) { return; }
    float acc = 0.0f;
    for (uint j = 0; j < 4; ++j) { acc += pre[b * 4 + j] * float(x[(ulong(b) * 4 + j) * D + d]); }
    y[ulong(b) * D + d] = static_cast<T>(is_same<T, half>::value ? clamp(acc, -65504.0f, 65504.0f) : acc);
"""
_MIXPRE = r"""
    // Two independent jobs per row in one launch (grid x: threadgroups 0 .. NDT - 1, then one more; grid y: the row b):
    //  hc_pre   y[d] = T(sum_j pre[j] * x[j, d]) sequential from 0 in fp32 (products rounded first: contraction off for this scope, like _PRE)
    //  sinkhorn the mlx-vlm kernel hc_sinkhorn_split (pre / post / comb of the mix just computed), verbatim for one row, default contraction like that kernel
    const uint tg = threadgroup_position_in_grid.x;
    const uint b = threadgroup_position_in_grid.y;
    if (tg < NDT) {
        const uint d = tg * 128 + thread_position_in_threadgroup.x;
        if (d >= D) { return; }
        float acc = 0.0f;
        {
#pragma METAL fp contract(off)
            for (uint j = 0; j < 4; ++j) { acc += pre[b * 4 + j] * float(x[(ulong(b) * 4 + j) * D + d]); }
        }
        y[ulong(b) * D + d] = static_cast<T>(is_same<T, half>::value ? clamp(acc, -65504.0f, 65504.0f) : acc);
        return;
    }
    if (simdgroup_index_in_threadgroup != 0) { return; }
    uint lane = thread_position_in_threadgroup.x;
    constexpr int HC = 4;
    constexpr int MIX = (2 + HC) * HC;
    constexpr int BASE_OFF = 2 * HC;
    constexpr float EPS = EPS_INT * 1e-9;
    const device float* mix      = (const device float*)mixes + b * MIX;
    device float*       pre_out  = (device float*)pre_o + b * HC;
    device float*       post_out = (device float*)post + b * HC;
    device float*       comb_out = (device float*)comb + b * HC * HC;
    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    const float active = (lane < (uint)HC) ? 1.0f : 0.0f;
    const uint  llane  = metal::min(lane, (uint)(HC - 1));
    float pre_z  = mix[llane]      * pre_scale  + base[llane];
    float post_z = mix[HC + llane] * post_scale + base[HC + llane];
    float pre_v  = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + EPS;
    float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));
    if (lane < (uint)HC) {
        pre_out[lane]  = pre_v;
        post_out[lane] = post_v;
    }
    float4 v = (*(const device float4*)(mix  + BASE_OFF + llane * HC)
                    * comb_scale
              + *(const device float4*)(base + BASE_OFF + llane * HC))
             * active;
    float row_max = metal::max(metal::max(v.x, v.y), metal::max(v.z, v.w));
    float4 e = metal::fast::exp(v - row_max) * active;
    float4 r = e * (1.0f / (e.x + e.y + e.z + e.w + EPS)) + EPS * active;
    float4 col_inv = 1.0f / (float4(simd_sum(r.x), simd_sum(r.y), simd_sum(r.z), simd_sum(r.w)) + EPS);
    r *= col_inv;
    for (int iter = 1; iter < ITERS; ++iter) {
        r *= (1.0f / (r.x + r.y + r.z + r.w + EPS)) * active;
        col_inv = 1.0f / (float4(simd_sum(r.x), simd_sum(r.y), simd_sum(r.z), simd_sum(r.w)) + EPS);
        r *= col_inv;
    }
    if (lane < (uint)HC) {
        *(device float4*)(comb_out + lane * HC) = r;
    }
"""
_k_mhc: dict = {}
FMA_LAST = False


def mhc_expand(x: mx.array, residual: mx.array, post: mx.array, comb: mx.array) -> tuple[mx.array, mx.array]:
    """x [B, 1, D] (sublayer output), residual [B, 1, 4, D], post [B, 1, 4] fp32, comb [B, 1, 4, 4] fp32 -> (y [B, 1, 4, D] in x.dtype, y32 = y as fp32).
    Equals language.hc_expand(x, residual, post, comb)."""
    B, S, D = x.shape
    assert S == 1 and residual.shape == (B, 1, 4, D) and post.dtype == mx.float32 and comb.dtype == mx.float32
    from . import alucut as AC
    if AC.noy32():                                                                 # KYOJIN_ALU_NOY32: the fp32 copy is not stored (y32 is None)
        k = _k_mhc.get("expand_n")
        if k is None:
            k = _k_mhc["expand_n"] = mx.fast.metal_kernel(name="kyojin_mhc_expand_n", input_names=["x", "r", "post", "comb"], output_names=["y"],
                                                          source=AC.expand_source_no_y32(_EXPAND), header=_HEADER)
        y = k(inputs=[x, residual, post, comb], template=[("T", x.dtype), ("D", D), ("FMA_LAST", FMA_LAST)], grid=(((D + 127) // 128) * 128, B, 1),
              threadgroup=(128, 1, 1), output_shapes=[(B, 1, 4, D)], output_dtypes=[x.dtype])[0]
        return y, None
    k = _k_mhc.get("expand")
    if k is None:
        k = _k_mhc["expand"] = mx.fast.metal_kernel(name="kyojin_mhc_expand", input_names=["x", "r", "post", "comb"], output_names=["y", "y32"],
                                                    source=_EXPAND, header=_HEADER)
    y, y32 = k(inputs=[x, residual, post, comb], template=[("T", x.dtype), ("D", D), ("FMA_LAST", FMA_LAST)], grid=(((D + 127) // 128) * 128, B, 1),
               threadgroup=(128, 1, 1), output_shapes=[(B, 1, 4, D), (B, 1, 4, D)], output_dtypes=[x.dtype, mx.float32])
    return y, y32


def mhc_pre(x: mx.array, pre: mx.array) -> mx.array:
    """x [B, 1, 4, D], pre [B, 1, 4] fp32 -> [B, 1, D] in x.dtype. Equals the stock hc_pre."""
    B, S, H, D = x.shape
    assert S == 1 and H == 4 and pre.dtype == mx.float32
    k = _k_mhc.get("pre")
    if k is None:
        k = _k_mhc["pre"] = mx.fast.metal_kernel(name="kyojin_mhc_pre", input_names=["x", "pre"], output_names=["y"], source=_PRE, header=_HEADER)
    return k(inputs=[x, pre], template=[("T", x.dtype), ("D", D)], grid=(((D + 127) // 128) * 128, B, 1), threadgroup=(128, 1, 1),
             output_shapes=[(B, 1, D)], output_dtypes=[x.dtype])[0]


def mhc_mix_pre(mixes: mx.array, scale: mx.array, base: mx.array, x: mx.array, pre: mx.array, iters: int, eps: float):
    """One launch for `hc_split_sinkhorn(mixes, scale, base, 4, iters, eps)` (-> pre', post, comb) and `hc_pre(x, pre)` (-> y): the two are independent.
    mixes [B, 1, 24] fp32, scale [3], base [24] fp32, x [B, 1, 4, D], pre [B, 1, 4] fp32 -> (y [B, 1, D] in x.dtype, pre' [B, 1, 4], post [B, 1, 4], comb [B, 1, 4, 4]).
    Bit-identical to the two separate launches (the sinkhorn text is the mlx-vlm kernel's, the pre sum is `mhc_pre`'s); a row does not depend on B (one grid row each)."""
    B, S, H, D = x.shape
    assert S == 1 and H == 4 and pre.dtype == mx.float32 and mixes.shape == (B, 1, 24) and pre.shape == (B, 1, 4)
    k = _k_mhc.get("mixpre")
    if k is None:
        k = _k_mhc["mixpre"] = mx.fast.metal_kernel(name="kyojin_mhc_mix_pre", input_names=["mixes", "scale", "base", "x", "pre"], output_names=["y", "pre_o", "post", "comb"],
                                                    source=_MIXPRE)
    ndt = (D + 127) // 128
    from . import gluefuse as _GF                                                    # stays a no-op unless KYOJIN_GLUE_FUSE=1 (read per call)
    y, pre_o, post, comb = k(inputs=[_GF.f32_arg(mixes), _GF.f32_arg(scale), _GF.f32_arg(base), x, pre],
                             template=[("T", x.dtype), ("D", D), ("NDT", ndt), ("ITERS", iters), ("EPS_INT", round(eps / 1e-9))],
                             grid=((ndt + 1) * 128, B, 1), threadgroup=(128, 1, 1),
                             output_shapes=[(B, 1, D), (B, 1, 4), (B, 1, 4), (B, 1, 4, 4)], output_dtypes=[x.dtype, mx.float32, mx.float32, mx.float32])
    return y, pre_o, post, comb


# --------------------------------------------------------------------------------------------------------------- wiring
MIN_PROMPT_ROWS = 64
_last = (None, None)          # (array, its fp32 copy) written by the last mhc_expand: the next mHC mix reads the copy instead of casting


def take_f32(x: mx.array) -> mx.array | None:
    """fp32 copy of `x` when `x` is the array the last mhc_expand returned, else None."""
    from . import alucut as AC
    return _last[1] if _last[0] is x and not AC.noy32() else None


def region_f32(h: mx.array) -> mx.array | None:
    """What a compiled region returns as the fp32 copy of its output: take_f32, or a scalar placeholder under KYOJIN_ALU_NOY32 (a compiled region cannot
    return None; take_f32 never hands the placeholder out, the next block runs without a copy)."""
    t = take_f32(h)
    if t is None:
        from . import alucut as AC
        if AC.noy32():
            return mx.zeros((), mx.float32)
    return t


def sat(y: mx.array) -> mx.array:
    """Saturate a stream array at the fp16 range. The model is bf16 (range 1e38), the engine runs it in fp16 (limit 65504): the last-layer streams come within
    x4 of the limit on ordinary text and the BOS input overflows from layer 22, and one inf is NaN at the next norm (the row then samples token 0 for ever).
    The expand and collapse kernels do the same inline (the collapse weights are sigmoids, their sum goes above 1); bit-identical below the limit."""
    return mx.clip(y, -65504, 65504) if y.dtype == mx.float16 else y


def _impl_mhc_expand(orig):
    from . import pf_glue as PG
    def expand(x, residual, post, comb):
        global _last
        if ENABLED and MHC and x.ndim == 3 and x.shape[1] == 1 and residual.ndim == 4 and residual.shape[2] == 4 and post.dtype == mx.float32 \
                and comb.dtype == mx.float32 and x.dtype in (mx.float16, mx.bfloat16):
            y, y32 = mhc_expand(x, residual, post, comb)
            _last = (y, y32) if y32 is not None else (None, None)
            return y
        if ENABLED and MHC and x.ndim == 3 and x.shape[0] == 1 and x.shape[1] > 1 and residual.shape == (1, x.shape[1], 4, x.shape[2]) and post.dtype == mx.float32 \
                and comb.dtype == mx.float32 and x.dtype in (mx.float16, mx.bfloat16):
            S, D = x.shape[1], x.shape[2]                                      # prompt rows: one grid row per token, the arithmetic of the 1-row case
            if PG.ENABLED and D % 8 == 0 and S >= MIN_PROMPT_ROWS:
                _last = (None, None)
                return PG.expand(x, residual, post, comb)
            y, _ = mhc_expand(x.reshape(S, 1, D), residual.reshape(S, 1, 4, D), post.reshape(S, 1, 4), comb.reshape(S, 1, 4, 4))
            _last = (None, None)
            return y.reshape(1, S, 4, D)
        return sat(orig(x, residual, post, comb))
    expand.saturates = True                                                    # every branch returns a stream already clamped at the fp16 range
    return expand


def _impl_mhc_pre(orig):
    def pre(x, pre_mix):
        if ENABLED and MHC and x.ndim == 4 and x.shape[1] == 1 and x.shape[2] == 4 and pre_mix.dtype == mx.float32 and x.dtype in (mx.float16, mx.bfloat16):
            return mhc_pre(x, pre_mix)
        if ENABLED and MHC and x.ndim == 4 and x.shape[0] == 1 and x.shape[1] > 1 and x.shape[2] == 4 and pre_mix.dtype == mx.float32 and x.dtype in (mx.float16, mx.bfloat16):
            S, D = x.shape[1], x.shape[3]
            return mhc_pre(x.reshape(S, 1, 4, D), pre_mix.reshape(S, 1, 4)).reshape(1, S, D)
        return sat(orig(x, pre_mix))
    pre.saturates = True                                                       # every branch returns a collapse already clamped at the fp16 range
    return pre


IMPLS = {"mhc_expand": _impl_mhc_expand, "mhc_pre": _impl_mhc_pre}
from . import fuse_attn as _fa                                                 # noqa: E402  (attention decode cut, see fuse_attn)
IMPLS["attn_decode"] = _fa._impl_attention
IMPLS["indexer"] = lambda orig: __import__(f"{__package__}.fuse_index", fromlist=["_impl_indexer"])._impl_indexer(orig)
IMPLS["compress_part"] = _fa._impl_compress_part
IMPLS["rope_positions"] = _fa._impl_rope_positions
IMPLS["sparse_prefill"] = lambda orig: __import__(f"{__package__}.attn_prefill", fromlist=["_impl_sparse"])._impl_sparse(orig)
IMPLS["mix_prompt"] = lambda orig: __import__(f"{__package__}.mhc_prefill", fromlist=["_impl_mix"])._impl_mix(orig)
IMPLS["moe_prompt"] = lambda orig: __import__(f"{__package__}.pf_glue", fromlist=["_impl_moe"])._impl_moe(orig)
IMPLS["shared_mlp"] = lambda orig: __import__(f"{__package__}.pf_glue", fromlist=["_impl_shared_mlp"])._impl_shared_mlp(orig)
IMPLS["block_prompt"] = lambda orig: __import__(f"{__package__}.pf_glue", fromlist=["_impl_block"])._impl_block(orig)
IMPLS["local_prefill"] = lambda orig: __import__(f"{__package__}.attn_prefill", fromlist=["_impl_attention_local"])._impl_attention_local(orig)
IMPLS["block_decode"] = lambda orig: __import__(f"{__package__}.fuse_layer", fromlist=["_impl_block"])._impl_block(orig)     # compiled decode regions of the block (fuse_layer imports this module)
PREFILL_IMPLS = {"sparse_prefill", "local_prefill", "mix_prompt", "mhc_expand", "mhc_pre", "moe_prompt", "shared_mlp", "block_prompt"}              # replacements that serve prompts: installed by patch_model itself
_installed: set = set()


def install(spec, only=None) -> int:
    """Replace the functions listed in `spec.fuse` (process wide, once). Returns the number of replacements."""
    import importlib
    if spec.fuse is None or not ENABLED:
        return 0
    if os.environ.get("KYOJIN_DSA_INDEX_GLM", "0") == "1" and spec.index_patch == "glm":
        from .glm_index import install as _glm_install
        _glm_install()
    n = 0
    for path, attr, impl in spec.fuse.funcs:
        if only is not None and impl not in only:
            continue
        key = (path, attr, impl)
        if key in _installed:
            continue
        modpath, _, cls = path.partition(":")
        owner = importlib.import_module(modpath)
        if cls:
            owner = getattr(owner, cls)
        orig = owner.__dict__[attr] if cls else getattr(owner, attr)
        static = isinstance(orig, staticmethod)
        fn = IMPLS[impl](orig.__func__ if static else orig)
        setattr(owner, attr, staticmethod(fn) if static else fn)
        _installed.add(key)
        n += 1
    return n
