# SPDX-License-Identifier: Apache-2.0
"""G2c: fused decode MoE block of GLM-5.3 (glm5_next Glm5NextMoE: sigmoid router with e_score_correction_bias, top-k, shared expert).

The stock block at decode rows is ~30 dispatches per layer: x.astype(f32), the router matmul, sigmoid, + bias, argpartition (a sort kernel), take_along_axis,
sum, + 1e-20, divide, * scale, the index casts, the routed-expert GEMVs, a gate-sized multiply, a sum over the slots and the shared-expert add. Here:

    logits   kyojin_rowdot2 (rowinv.rowdot, the very kernel the stock override calls; fp16 x read directly: no cast launch)
    select   one launch   sigmoid, + bias, top-K by (value desc, index asc), normalise, * routed_scaling_factor -> uint32 indices + f32 weights
    experts  G2a pair gather (gate | up), glu_rows, down gather   (ForeignSwitchGLU's own kernels, indices stay on the device)
    combine  one launch   sum_k fp16(y_k * fp16(w_k)) in f32, rounded to fp16, + shared expert (fp16 add)

The products and the fp16 roundings of the stock `SwitchGLU._combine` are kept; the logits are bit-identical (same kernel); the sigmoid / sum orders differ in f32 rounding only.
Anything else (n_group > 1 with dropped groups, sharded blocks, > GATHER_MAX slots, non-fp16 streams, experts outside the G2a format) runs the stock block.
Flag: KYOJIN_GLM_MOE (read per call).
"""
from __future__ import annotations

import os

import mlx.core as mx

from . import fired as FD
from . import glm_moe_e as GE
from . import glm_moe_rows as R
from . import kernels as KN


def on() -> bool:
    return os.environ.get("KYOJIN_GLM_MOE", "0") not in ("0", "")


T_MAX = 16


def rows_on() -> bool:
    """Flag KYOJIN_GLM_MOE_ROWS (read per call, default 1; only read when KYOJIN_GLM_MOE is on): glm_moe_rows kernels (glu in the pair launch, distinct-expert columns)."""
    return os.environ.get("KYOJIN_GLM_MOE_ROWS", "1") not in ("0", "")

# select_par: BP-MOEPIPE (branch glm-speed-moepipe, tested there against `select`): one threadgroup of E threads per token
_SELECT_PAR = r"""
    const uint t = threadgroup_position_in_grid.x;
    const uint i = thread_position_in_threadgroup.x;
    threadgroup float chs[NT];
    threadgroup float tops[K];
    threadgroup uint topi[K];
    float s = 0.0f, c = -INFINITY;
    if (i < uint(E)) {
      const float z = lg[t * uint(E) + i];
      const float y = 1.0f / (1.0f + exp(-abs(z)));
      s = z < 0.0f ? 1.0f - y : y;                                                   // mx.sigmoid, the expression of the walking kernel
      c = s + bias[i];
      if (!(c == c)) c = -INFINITY;                                                  // a NaN never wins the walking kernel's `>`
    }
    chs[i] = c;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i < uint(E)) {
      uint rank = 0u;                                                                // candidates ahead of me: larger value, or an equal one at a lower index
      for (uint j = 0; j < uint(E); ++j) { const float o = chs[j]; rank += (o > c || (o == c && j < i)) ? 1u : 0u; }
      if (rank < uint(K)) { tops[rank] = s; topi[rank] = i; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i < uint(K)) {
      float tot = 0.0f;
      for (uint k = 0; k < uint(K); ++k) tot += tops[k];                             // slot order = the walking kernel's f32 sum order
      const float w = NORM ? tops[i] / (tot + 1e-20f) : tops[i];
      inds[t * uint(K) + i] = topi[i];
      wts[t * uint(K) + i] = w * SCALE;
    }
"""

_SELP: dict = {}


def select_par(lg, bias, T, E, K, norm, scale):
    """`select` with one threadgroup of ceil(E / 32) * 32 threads per token (E <= 1024, K <= 32): every candidate counts the ones ahead of it, the K winners write their slot. Same outputs bit for bit
    (lowest index wins ties, same sigmoid expression, same f32 sum order); a NaN logit is skipped like the walking kernel's `>`."""
    NT = -(-E // 32) * 32
    key = (E, K, bool(norm), float(scale))
    k = _SELP.get(key)
    if k is None:
        src = _SELECT_PAR.replace("SCALE", repr(float(scale)) + "f")
        k = _SELP[key] = mx.fast.metal_kernel(name=f"kyojin_glm_selpar_{E}_{K}_{_scale_tag(scale)}", input_names=["lg", "bias"], output_names=["inds", "wts"], header="", source=src)
    return k(inputs=[lg, bias], template=[("E", E), ("K", K), ("NT", NT), ("NORM", bool(norm) and K > 1)], grid=(T * NT, 1, 1), threadgroup=(NT, 1, 1),
             output_shapes=[(T * K,), (T * K,)], output_dtypes=[mx.uint32, mx.float32])


_SELECT_HEAD = r"""
    const uint t = threadgroup_position_in_grid.x;
    const uint lane = thread_index_in_simdgroup;
    device const float* L = lg + t * uint(E);
"""
_SELECT_BODY = r"""    float sc[EL], ch[EL];
    for (uint j = 0; j < EL; ++j) {
      const uint i = lane + 32u * j;
      if (i < uint(E)) {
        const float z = L[i];
        const float y = 1.0f / (1.0f + exp(-abs(z)));
        const float s = z < 0.0f ? 1.0f - y : y;                       // mx.sigmoid
        sc[j] = s; ch[j] = s + bias[i];
      } else { sc[j] = 0.0f; ch[j] = -INFINITY; }
    }
    float top_s[K]; uint top_i[K];
    for (uint k = 0; k < uint(K); ++k) {
      float bv = -INFINITY; uint bi = 0u;                                           // a row of all-NaN logits never passes `>`: keep index 0 valid instead of 0xFFFFFFFF (out-of-range gather)
      for (uint j = 0; j < EL; ++j) if (ch[j] > bv) { bv = ch[j]; bi = lane + 32u * j; }
      const float gv = simd_max(bv);
      const uint gi = simd_min(bv == gv ? bi : 0xFFFFFFFFu);           // lowest index among equal values
      float s = 0.0f;
      for (uint j = 0; j < EL; ++j) if (lane + 32u * j == gi) { s = sc[j]; ch[j] = -INFINITY; }
      top_s[k] = simd_sum(s); top_i[k] = gi;
    }
    float tot = 0.0f;
    for (uint k = 0; k < uint(K); ++k) tot += top_s[k];
    if (lane == 0u) {
      for (uint k = 0; k < uint(K); ++k) {
        const float w = NORM ? top_s[k] / (tot + 1e-20f) : top_s[k];
        inds[t * uint(K) + k] = top_i[k];
        wts[t * uint(K) + k] = w * SCALE;
      }
    }
"""
_SELECT = _SELECT_HEAD + _SELECT_BODY

_COMBINE = r"""
    const uint i = thread_position_in_grid.x;
    const uint h4 = i % (uint(H) / 4u), t = i / (uint(H) / 4u);
    float4 acc = 0.0f;
    for (uint k = 0; k < uint(K); ++k) {
      const float w = float(half(wts[t * uint(K) + k]));
      const float4 yv = float4(((device const half4*)y)[(t * uint(K) + k) * (uint(H) / 4u) + h4]);
      acc += float4(half4(yv * w));                                    // routed * weights.astype(fp16): the product is rounded to fp16
    }
    float4 r = float4(half4(acc));                                     // sum over the slots: f32 accumulate, one fp16 rounding
    if (HAS_S) r += float4(((device const half4*)sh)[t * (uint(H) / 4u) + h4]);
    ((device half4*)o)[t * (uint(H) / 4u) + h4] = half4(r);
"""

_SEL: dict = {}


def _scale_tag(scale: float) -> str:
    """Name-safe rendering of a baked SCALE (mlx caches compiled metal kernels by name): 2.5 -> '2p5', 1e-06 -> '1em06'."""
    return repr(float(scale)).replace(".", "p").replace("-", "m").replace("+", "")


def selpar_on() -> bool:
    """Flag KYOJIN_GLM_MOE_E_SELPAR (read per call, default 0, only read in the E branch): the router select as one threadgroup of E threads (`select_par`)."""
    return os.environ.get("KYOJIN_GLM_MOE_E_SELPAR", "0") not in ("0", "")


_SEL_ONE = "        const float z = L[i];\n        const float y = 1.0f / (1.0f + exp(-abs(z)));\n        const float s = z < 0.0f ? 1.0f - y : y;                       // mx.sigmoid\n"
_SEL_ROW = "device const float* L = lg + t * uint(E);"


def select_source(routers: int = 1, scale_vec: bool = False) -> str:
    """The select kernel body. routers 1 and a scalar scale: _SELECT unchanged. routers R > 1: the logits are [T, R * E] (router r = columns r * E ..) and the expert score is the
    sum over the routers of sigmoid(logit), added in router order in f32. scale_vec: the weight of the expert in slot k is w * sv[expert] (a per-expert factor) instead of w * SCALE."""
    src = _SELECT
    if routers > 1:
        assert src.count(_SEL_ONE) == 1 and src.count(_SEL_ROW) == 1
        src = src.replace(_SEL_ROW, f"device const float* L = lg + t * uint(E) * {int(routers)}u;")
        src = src.replace(_SEL_ONE, "        float s = 0.0f;\n        for (uint r = 0; r < " + f"{int(routers)}u" + "; ++r) {\n          const float z = L[r * uint(E) + i];\n          const float y = 1.0f / (1.0f + exp(-abs(z)));\n"
                          "          s += z < 0.0f ? 1.0f - y : y;                                   // mx.sigmoid, routers added in order\n        }\n")
    if scale_vec:
        assert src.count("wts[t * uint(K) + k] = w * SCALE;") == 1
        src = src.replace("wts[t * uint(K) + k] = w * SCALE;", "wts[t * uint(K) + k] = w * sv[top_i[k]];")
    return src


def select(lg, bias, T, E, K, norm, scale, routers=1):
    """lg f32 [T, routers * E] router logits -> (inds uint32 [T * K], weights f32 [T * K]). scale: a float, or an f32 array [E] (the weight of expert e is multiplied by scale[e])."""
    vec = hasattr(scale, "shape")
    key = (E, K, bool(norm), "vec" if vec else float(scale), routers)
    k = _SEL.get(key)
    if k is None:
        src = select_source(routers, vec).replace("SCALE", "1.0f" if vec else repr(float(scale)) + "f")
        k = _SEL[key] = mx.fast.metal_kernel(name=f"kyojin_glm_select_{E}_{K}_{'v' if vec else _scale_tag(scale)}" + (f"_r{routers}" if routers > 1 else ""),
                                             input_names=["lg", "bias"] + (["sv"] if vec else []), output_names=["inds", "wts"], header="", source=src)
    return k(inputs=[lg, bias] + ([scale] if vec else []), template=[("E", E), ("K", K), ("EL", -(-E // 32)), ("NORM", bool(norm) and K > 1)], grid=(T * 32, 1, 1), threadgroup=(32, 1, 1),
             output_shapes=[(T * K,), (T * K,)], output_dtypes=[mx.uint32, mx.float32])


_COMB_W16 = "const float w = float(half(wts[t * uint(K) + k]));"
_COMB_P16 = "acc += float4(half4(yv * w));                                    // routed * weights.astype(fp16): the product is rounded to fp16"


def combine_source(wround: bool = True) -> str:
    """wround True: _COMBINE unchanged (fp16-rounded weight and product, the stock GLM block). False: f32 weight and product, f32 sum, one fp16 rounding (y.astype(f32) * w).sum(-2).astype(f16)."""
    if wround:
        return _COMBINE
    assert _COMBINE.count(_COMB_W16) == 1 and _COMBINE.count(_COMB_P16) == 1
    return _COMBINE.replace(_COMB_W16, "const float w = wts[t * uint(K) + k];").replace(_COMB_P16, "acc += yv * w;")


def combine(y, wts, sh, T, K, H, wround=True):
    """y fp16 [T * K, H] (slot order of select), wts f32 [T * K], sh fp16 [T, H] or None -> fp16 [T, H]. wround False: f32 products (see combine_source)."""
    k = KN._kern(("glm_combine",) + (() if wround else ("f32w",)), name="kyojin_glm_combine" + ("" if wround else "_f32w"), input_names=["y", "wts", "sh"], output_names=["o"], header="", source=combine_source(wround))
    return k(inputs=[y, wts, sh if sh is not None else y], template=[("H", H), ("K", K), ("HAS_S", sh is not None)], grid=(T * H // 4, 1, 1), threadgroup=(256, 1, 1),
             output_shapes=[(T, H)], output_dtypes=[mx.float16])[0]


class Route:
    """Generic description of a routed MoE block, in numbers and callables (no model attribute names). A block that is not shaped like the GLM one exposes `moe_route()` returning one.

    weight   [routers * E, Hr] router matrix (fp32 / bf16 / fp16), the routers stacked on the rows: logits of router r = rows r * E .. (r + 1) * E
    routers  how many routers: the expert score is the sum of sigmoid(logit) over the routers
    bias     f32 [E] added to the score for the top-k choice only (the weights use the plain score)
    top_k, norm (weights renormalised over the chosen slots), scale (float, or f32 [E]: per-expert factor on the weight)
    groups   group-limited routing: only 1 (or n_group == topk_group, set groups=1) is fused
    pre      None: the experts read the block input (width Hr). Else callable fp16 [T, Hr] -> fp16 [T, width] (a latent projection in front of the experts)
    post     None: the combined routed output is the block output (the shared expert is added by the combine kernel). Else callable fp16 [T, width] -> fp16 [T, H]
             (norm and up projection after the combine); the shared expert is added to its result
    width    width of the expert input and of the combined routed output (the experts' k and the down projection's n); None = Hr
    wround   True: the stock GLM combine (fp16 weight and product); False: f32 weight and product, one fp16 rounding at the end
    shared   callable (block input) -> shared-expert output [.., H] or None
    """

    def __init__(self, weight, top_k, bias, norm=True, scale=1.0, routers=1, groups=1, pre=None, post=None, width=None, wround=True, shared=None):
        self.weight, self.top_k, self.bias, self.norm, self.scale, self.routers, self.groups = weight, top_k, bias, norm, scale, routers, groups
        self.pre, self.post, self.width, self.wround, self.shared = pre, post, width, wround, shared

    @property
    def E(self):
        return self.weight.shape[0] // self.routers

    @property
    def plain(self):
        """The router / select / combine of the GLM block (one router, scalar scale, fp16 combine, no latent): the single-launch variants (rowsel, select_par, prompt chunk) apply."""
        return self.routers == 1 and not hasattr(self.scale, "shape") and self.pre is None and self.post is None and self.wround


def route_of(blk):
    """The Route of a block: its own `moe_route()` (built once, then kept), else the GLM attributes (gate.weight / e_score_correction_bias / top_k / n_group ...), else None."""
    r = blk.__dict__.get("_moe_route")
    if r is None:
        fn = getattr(blk, "moe_route", None)
        if callable(fn):
            r = blk.__dict__["_moe_route"] = fn()
    if r is not None:
        return r
    g = getattr(blk, "gate", None)
    if g is None or not hasattr(g, "e_score_correction_bias") or not hasattr(g, "n_group"):
        return None
    return Route(g.weight, g.top_k, g.e_score_correction_bias, g.norm_topk_prob, g.routed_scaling_factor, groups=1 if g.n_group <= 1 or g.n_group == g.topk_group else g.n_group,
                 shared=getattr(blk, "shared_experts", None))


def _why(blk, x):
    """None when the fused block serves this call, else the reason it does not (flag audit key; the guards of the former `_eligible`, same order)."""
    from . import foreign_linear as FL
    sm = blk.switch_mlp
    if getattr(blk, "sharding_group", None) is not None:
        return "sharded"
    if not isinstance(sm, FL.ForeignSwitchGLU):
        return "experts=" + type(sm).__name__
    if x.dtype != mx.float16:
        return "x_dtype=" + str(x.dtype).split(".")[-1]
    if x.shape[-1] % 256 != 0:
        return "hidden%256"
    r = route_of(blk)
    if r is None:
        return "no_route"
    T = 1
    for d in x.shape[:-1]:
        T *= d
    K, E = r.top_k, r.E
    if T > T_MAX:
        return f"rows>{T_MAX}"                                          # prompt chunk: the stock block (ForeignSwitchGLU prefill / kqgmm)
    if T * K > FL.GATHER_MAX:
        return "slots>GATHER_MAX"
    if E > 1024 or K > 16 or r.weight.shape[1] != x.shape[-1] or r.weight.shape[0] % r.routers != 0:
        return "router_dims"
    if r.weight.dtype not in (mx.float32, mx.bfloat16, mx.float16):
        return "router_dtype=" + str(r.weight.dtype).split(".")[-1]
    if r.groups > 1:                                                    # dropped groups: stock block (GLM-5.3 has n_group = 1)
        return "groups"
    if not sm.g.g2a_ok() or not sm.d.g2a_ok():
        return "g2a_" + ("off" if not FL.g2a_on() else "pack")
    xw = x.shape[-1] if r.width is None else r.width                    # width of the expert input and of the combined routed output
    if not (sm.g.pair_ok(sm.u) and sm.g.n % 4 == 0 and sm.d.k == sm.g.n and sm.d.n == xw and sm.g.k == xw and xw % 4 == 0):
        return "pair_dims"
    return None


def _eligible(blk, x) -> bool:
    return _why(blk, x) is None


def pf_on() -> bool:
    """Flag KYOJIN_GLM_PF_MOE (read per call, default 0): prompt chunks (> T_MAX rows) run the router select and the combine as one launch each (see `_pf_call`)."""
    return os.environ.get("KYOJIN_GLM_PF_MOE", "0") not in ("0", "")


def _pf_why(blk, x):
    """None when the prompt-chunk MoE (fused select + combine around the stock ForeignSwitchGLU) serves this call, else the reason (flag audit key)."""
    from . import foreign_linear as FL
    if not pf_on():
        return "flag_off"
    if getattr(blk, "sharding_group", None) is not None:
        return "sharded"
    if not isinstance(blk.switch_mlp, FL.ForeignSwitchGLU):
        return "experts=" + type(blk.switch_mlp).__name__
    if callable(getattr(blk, "moe_route", None)):
        return "route"                                                # a block with its own route description: decode only
    if x.dtype not in (mx.float16, mx.float32):                   # mHC collapse hands prompt chunks fp32 (T > DECODE_BLOCK_SIZE); tiled_linear / shared_experts /
        return "x_dtype=" + str(x.dtype).split(".")[-1]           # switch_mlp already cast x internally, so fp32 is fine here (unlike the decode fused path's rowdot)
    H = x.shape[-1]
    T = 1
    for d in x.shape[:-1]:
        T *= d
    if T <= T_MAX:
        return f"rows<={T_MAX}"
    g = blk.gate
    K, E = g.top_k, g.weight.shape[0]
    if H % 4 != 0 or E > 1024 or K > 16 or g.weight.shape[1] != H:
        return "router_dims"
    if g.n_group > 1 and g.n_group != g.topk_group:
        return "groups"
    if g.weight.dtype not in (mx.float32, mx.bfloat16, mx.float16):
        return "router_dtype=" + str(g.weight.dtype).split(".")[-1]
    return None


PF_SELPAR = os.environ.get("KYOJIN_MOE_ROUTER_SELECT_PAR", "0") not in ("0", "")      # prompt chunks: `select_par` (one threadgroup per token, same outputs bit for bit) instead of the walking one-simdgroup kernel


def _pf_call(self, x, residual=None):
    """Prompt chunk: stock router GEMM (`tiled_linear`, so glm_pf_untile applies), then ONE launch for sigmoid + bias + top-k + normalise (`select`), the stock expert block, and ONE launch
    for sum_k fp16(y_k * fp16(w_k)) + shared (`combine`; the stock `_combine` is a multiply pass that writes [T, K, H] and a sum pass that reads it back: 2 x 0.8 GB at 8192 rows)."""
    from . import archs as _A
    from . import foreign_linear as FL
    g = self.gate
    lead, H = x.shape[:-1], x.shape[-1]
    K, E = g.top_k, g.weight.shape[0]
    x2 = x.reshape(-1, H)
    T = x2.shape[0]
    L = _A.lang(_A.GLM5)
    lg = L.tiled_linear(lambda v: v @ g.weight.T, x.astype(mx.float32)).reshape(T, E)
    sel = select_par if PF_SELPAR and E <= 1024 and K <= 32 else select
    inds, wts = sel(lg, g.e_score_correction_bias, T, E, K, g.norm_topk_prob, g.routed_scaling_factor)
    from . import hc_pf_fuse as HF
    xh = HF.twin_of(x)                                           # the fused mHC norm already wrote x.astype(float16): the shared expert and the routed experts read that instead of casting twice
    xe = x if xh is None else xh
    sh = self.shared_experts(xe).reshape(T, H) if getattr(self, "shared_experts", None) is not None else None
    y = self.switch_mlp(xe, inds.reshape(*lead, K)).reshape(T * K, H)
    if y.dtype != mx.float16 or (sh is not None and sh.dtype != mx.float16):       # the combine kernel reads y and sh as half4: a fp32 stream would be reinterpreted, not cast
        FD.fb("glm_pf_moe", "combine_dtype=" + ("y" if y.dtype != mx.float16 else "sh"))
        return self._stock_call(x) if residual is None else self._stock_call(x, residual=residual)       # mlx-lm MoE classes take no `residual`
    out = combine(y, wts, sh, T, K, H).reshape(*lead, H)
    if residual is not None:
        from mlx_vlm.models.deepseek_v4.hyper_connection import hc_expand
        out = hc_expand(out, *residual)
    return out


def _call(self, x, *args, residual=None, **kwargs):
    why = "flag_off" if not on() else ("args" if (args or kwargs) else _why(self, x))
    if why is not None:
        FD.fb("glm_moe", why)
        if not (args or kwargs) and pf_on():
            pw = _pf_why(self, x)
            if pw is None:
                FD.fire("glm_pf_moe")
                return _pf_call(self, x, residual)
            FD.fb("glm_pf_moe", pw)
        return self._stock_call(x, *args, **kwargs) if residual is None else self._stock_call(x, *args, residual=residual, **kwargs)
    FD.fire("glm_moe")
    from . import foreign_linear as FL
    from .rowinv import rowdot
    r, sm = route_of(self), self.switch_mlp
    lead, H = x.shape[:-1], x.shape[-1]
    K, E = r.top_k, r.E
    x2 = x.reshape(-1, H)
    T = x2.shape[0]
    from . import glm_moe_rs as RS
    if r.plain and GE.on() and RS.on() and x2.dtype in (mx.float16, mx.bfloat16, mx.float32) and RS.eligible(T, E, K, H):
        FD.fire("glm_moe_rs")                      # router rowdot + select in one launch (BP-MOE2, flag KYOJIN_GLM_MOE_E_RS): same bits
        _, inds, wts = RS.rowsel(x2, r.weight, r.bias, T, E, K, r.norm, r.scale)
    else:
        lg = rowdot(x2, r.weight, out_dtype=mx.float32)
        sel = select_par if (r.plain and GE.on() and selpar_on() and E <= 1024 and K <= 32) else select
        inds, wts = sel(lg, r.bias, T, E, K, r.norm, r.scale) if r.plain else select(lg, r.bias, T, E, K, r.norm, r.scale, r.routers)
    if r.pre is not None:
        FD.fire("glm_moe_latent")
        x2 = r.pre(x2).reshape(T, -1)                   # latent projection in front of the experts: the experts and the combine run at its width
    W = x2.shape[-1]
    S = T * K
    if rows_on() and R.ok(sm.g, sm.u, sm.d, T, K) and FL.G2A_NBR == 1:
        FD.fire("glm_moe_rows")
        if GE.on() and GE.ok(sm.g, sm.d, T, K):
            FD.fire("glm_moe_e")                         # flag KYOJIN_GLM_MOE_E: same sums bit for bit, oMLX launch structure (small threadgroups, in-kernel dedupe, no group launch)
            rps, nsg = GE.cfg(T)
            drps, dnsg, dxv, diu = GE.down_cfg(T)
            ju = GE.rows_unroll(T)
            y = GE.down("g2m", sm.d, GE.pair_glu("g2m", sm.g, sm.u, x2, inds, T, K, sm.limit, rps, nsg, 0, 0, 0, ju, sm.act_kind), inds, T, K, drps, dnsg, 0, dxv, diu, ju)
        else:
            tab = R.group(inds, T, K) if T > 1 else None                # one threadgroup column per distinct expert: its weights are read once for all the rows that picked it
            y = R.down(sm.d, R.pair_glu(sm.g, sm.u, x2, inds, tab, T, K, sm.limit, sm.act_kind), inds, tab, T, K)
    else:
        if FD.AUDIT:
            FD.fb("glm_moe_rows", "rows_off" if not rows_on() else ("nbr" if FL.G2A_NBR != 1 else "R.ok"))
        tok, ar = FL._slot_rows(S, K)
        h = FL.glu_rows(sm.g.gather_pair_raw(sm.u, x2, tok, inds), S, sm.g.n, sm.limit, sm.act_kind)
        y = sm.d.gather(h, ar, inds)
    sh = r.shared(x).reshape(T, H) if r.shared is not None else None
    if r.post is None:
        out = combine(y, wts, sh, T, K, W, r.wround).reshape(*lead, H)
    else:
        FD.fire("glm_moe_post")
        out = r.post(combine(y, wts, None, T, K, W, r.wround)).reshape(T, H)          # e.g. norm + up projection of a latent block, then the shared expert
        if sh is not None:
            out = out + sh
        out = out.reshape(*lead, H)
    if residual is not None:
        from mlx_vlm.models.deepseek_v4.hyper_connection import hc_expand
        out = hc_expand(out, *residual)
    return out


def install(model, log=print) -> int:
    """Swap the call of every GLM MoE block (gate with e_score_correction_bias, ForeignSwitchGLU experts). Returns the number of blocks."""
    from . import foreign_linear as FL
    n = 0
    for _, mod in model.named_modules():
        g = getattr(mod, "gate", None)
        if (isinstance(getattr(mod, "switch_mlp", None), FL.ForeignSwitchGLU) and ((hasattr(g, "e_score_correction_bias") and hasattr(g, "n_group")) or callable(getattr(mod, "moe_route", None)))
                and not getattr(mod.__class__, "_glm_moe", False)):
            cls = mod.__class__
            mod.__class__ = type("GlmFused" + cls.__name__, (cls,), {"__call__": _call, "_stock_call": cls.__call__, "_glm_moe": True})
            n += 1
    if n:
        log(f"[foreign] glm_moe: {n} GLM MoE blocks (router + experts + combine in fused launches, flag KYOJIN_GLM_MOE={'on' if on() else 'off'})")
    return n
