# SPDX-License-Identifier: Apache-2.0
"""Generic, spec-driven MoE feed-forward block for the GGUF arch kit. Every choice is a MoeSpec field; no architecture name appears in this file.

The maths is `llm_graph_context::build_moe_ffn` + the shared-expert / dense / latent wrappers of llama.cpp (src/llama-graph.cpp:2252-2667,
src/models/qwen2moe.cpp:122-160, kimi-k3.cpp:546-600, arctic.cpp:115-142), written in plain MLX ops (any device, any dtype, no kernel).

How the block DESCRIBES itself to the engine (so Metal runs it through the existing fused routed-expert kernels on stored-format weights):

  kind "foreign_moe"  softmax / softmax_weight gating, no router bias, no scale, no groups, no latent. The block exposes the attributes
                      foreign_moe.install reads (gate.weight f32 [E, H], top_k, num_experts, norm_topk_prob, shared_expert, shared_expert_gate,
                      switch_mlp). An ungated always-on shared expert is described as gate weight 0 (sigmoid(0) = 0.5) and a shared output times 2 (exact).
  kind "glm_route"    sigmoid gating (+ optional selection bias, normalise, scale, optional latent pre / post, shared expert with or without gate).
                      The block exposes `moe_route()` returning glm_moe.Route; glm_moe.install picks it up.
  kind "stock"        anything the kernels cannot describe (see `describe()["why"]`): the plain-MLX ops below run (correct but slow).
The class of a block is chosen at construction from the spec, so only the attributes of its kind exist (the two installers never both match).
`switch_mlp` (tensors["switch_mlp"]) is the engine's ForeignSwitchGLU (stored-format experts) when the loader has one; otherwise the block builds a dense
expert stack from tensors gate_exps / up_exps / down_exps.

Entry point for the decoder skeleton: ffn_factory(layer_index, spec, tensors) -> module, __call__(x) -> same shape as x.
"""
from __future__ import annotations

import dataclasses
import math
import os
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from . import rowinv as _RI
from .rowinv import skinny_mm, route_norm

F16_MIN_NORM = 6.103515625e-5            # llama.cpp: weights_sum clamp when no eps is given (llama-graph.cpp:2410)


@dataclasses.dataclass
class MoeSpec:
    # sizes
    n_expert: int = 0
    top_k: int = 0
    # router
    gating: str = "softmax"              # softmax | sigmoid | softmax_weight | none | sqrt_softplus
    router_bias: bool = False            # gate_inp_b: added to the logits before the gating function
    select_bias: bool = False            # exp_probs_b: added to the scores for the top-k choice only
    norm_weights: bool = False           # renormalise the selected weights
    norm_eps: float = 0.0                # > 0: sum + eps; 0: max(sum, 6.1e-5) (llama.cpp)
    weight_scale: float = 1.0
    temperature: float = 1.0             # logits / temperature (no llama.cpp arch uses it; spec field only)
    n_group: int = 1                     # group-limited routing: top `group_used` groups of E / n_group experts, group score = sum of its 2 best
    group_used: int = 1
    # experts
    act: str = "silu"                    # silu | gelu | relu | relu2 (non gated) | situ | swiglu_oai
    gated: bool = True
    clamp: float = 0.0                   # 0 = none
    clamp_mode: str = "post_act"         # post_act: min(act(gate), c), clip(up) (llama.cpp generic) | pre_act: min(gate, c) then act (swiglu_clamp archs)
    situ_beta: float = 0.0
    situ_linear_beta: float = 0.0
    # shared experts (one FFN of width n_shared * ff, always on)
    shared: bool = False
    shared_gate: bool = False            # sigmoid(gate_inp_shexp x) * shared
    shared_act: str = "silu"
    # parallel dense FFN added to the MoE output
    parallel_dense: bool = False
    # latent projection around the experts (router and shared read the full-width input)
    latent: bool = False
    latent_norm: bool = False
    norm_eps_rms: float = 1e-6           # eps of the latent rms norm
    # norms around the expert block (muse-spark.cpp:111-112 pre, llama-graph.cpp:2622-2625 post)
    pre_exp_norm: bool = False           # expert input (after the latent down projection) = weight * rms_norm(x); the router still reads the un-normed input
    pre_exp_eps: float = 1e-6
    post_exp_norm: bool = False          # each expert output y = rms_norm(y * weight), before the router weights multiply it
    post_exp_eps: float = 1e-5
    down_scale: bool = False             # a per-expert scale [n_expert] on the down projection output of each chosen expert (llama-graph.cpp:1783-1789, build_lora_mm_id w_s); per layer
    shared_clamp: float = 0.0            # swiglu clamp of the shared expert (llama-graph.cpp:2066-2080; same form as the experts' post_act clamp); 0 = none
    # interleaving with dense layers
    n_dense_leading: int = 0
    moe_every: int = 1                   # layer i is MoE when i >= n_dense_leading and (i - n_dense_leading) % moe_every == 0
    dense_layers: Optional[tuple] = None  # explicit set of dense layers (overrides the two fields above)

    @staticmethod
    def from_any(s) -> "MoeSpec":
        if isinstance(s, MoeSpec):
            return s
        get = (lambda k: s.get(k)) if isinstance(s, dict) else (lambda k: getattr(s, k, None))
        kw = {}
        for f in dataclasses.fields(MoeSpec):
            v = get(f.name)
            if v is not None:
                kw[f.name] = v
        return MoeSpec(**kw)

    def is_moe_layer(self, i: int) -> bool:
        if self.dense_layers is not None:
            return i not in self.dense_layers
        return i >= self.n_dense_leading and (i - self.n_dense_leading) % max(1, self.moe_every) == 0


_GATINGS = ("softmax", "sigmoid", "softmax_weight", "none", "sqrt_softplus")
_ACTS = ("silu", "gelu", "relu", "relu2", "situ", "swiglu_oai")


_ALIAS = {"ffn_gate_inp": "gate_inp", "ffn_gate_inp_b": "gate_inp_b", "ffn_gate_exps": "gate_exps", "ffn_up_exps": "up_exps", "ffn_down_exps": "down_exps",
          "ffn_gate_shexp": "gate_shexp", "ffn_up_shexp": "up_shexp", "ffn_down_shexp": "down_shexp", "ffn_gate_inp_shexp": "gate_inp_shexp",
          "ffn_routed_down": "routed_down", "ffn_routed_up": "routed_up", "ffn_routed_norm": "routed_norm",
          "ffn_pre_exp_norm": "pre_exp_norm", "ffn_post_exp_norm": "post_exp_norm"}


def _norm_names(tensors: dict) -> dict:
    """Accepts `blk.N.<name>.weight` / `<name>.weight` / `<name>`; returns <name> -> value."""
    out = {}
    for k, v in tensors.items():
        n = k
        if n.startswith("blk."):
            n = n.split(".", 2)[2]
        if n.endswith(".weight"):
            n = n[:-len(".weight")]
        elif n.endswith(".bias"):
            n = n[:-len(".bias")]
            n = n if n.endswith("_b") else n + "_b"
        out[_ALIAS.get(n, n)] = v
    return out


def _lin(w):
    """w: [out, in] array (x @ w.T) or a callable (a ForeignLinear / nn.Linear)."""
    if callable(w):
        return w
    return lambda x: x @ w.astype(x.dtype).T


def apply_act(kind: str, gate, up, spec: Optional[MoeSpec] = None, clamp: float = 0.0, mode: str = "post_act"):
    """h = f(gate, up) for a gated FFN; for a non-gated one `gate` is None and the activation is applied to `up`. f32 in, f32 out."""
    s = spec or MoeSpec()
    if gate is None:
        if kind == "silu":
            return up * mx.sigmoid(up)
        if kind == "gelu":
            return nn.gelu_approx(up)
        if kind == "relu":
            return mx.maximum(up, 0)
        if kind == "relu2":
            r = mx.maximum(up, 0)
            return r * r
        raise ValueError(f"act {kind!r} needs a gate")
    if kind == "silu":
        if clamp > 1e-6:
            if mode == "pre_act":
                g = mx.minimum(gate, clamp)
                return g * mx.sigmoid(g) * mx.clip(up, -clamp, clamp)
            return mx.minimum(gate * mx.sigmoid(gate), clamp) * mx.clip(up, -clamp, clamp)
        return gate * mx.sigmoid(gate) * up
    if kind == "gelu":
        return nn.gelu_approx(gate) * up
    if kind == "relu":
        return mx.maximum(gate, 0) * up
    if kind == "situ":
        b, lb = s.situ_beta, s.situ_linear_beta
        a = b * mx.tanh(gate / b) * mx.sigmoid(gate)
        if lb > 0:
            up = lb * mx.tanh(up / lb)
        return a * up
    if kind == "swiglu_oai":                                    # ggml_swiglu_oai: alpha 1.702, limit 7 (llama-graph.cpp:2554)
        g = mx.minimum(gate, 7.0)
        u = mx.clip(up, -7.0, 7.0)
        return g * mx.sigmoid(1.702 * g) * (u + 1.0)
    raise ValueError(f"unknown act {kind!r}")


class DenseFFN(nn.Module):
    """Dense FFN (llama.cpp build_ffn, parallel gate): down(act(gate x, up x)). Used for shared experts, dense layers and the parallel dense branch."""

    def __init__(self, gate, up, down, act: str = "silu", spec: Optional[MoeSpec] = None, clamp: float = 0.0):
        super().__init__()
        self.clamp = clamp
        self._g = None if gate is None else _lin(gate)
        self._u, self._d = _lin(up), _lin(down)
        self.act, self.spec = act, spec or MoeSpec()

    def __call__(self, x):
        u = self._u(x)
        if self._g is None:
            h = apply_act(self.act, None, u.astype(mx.float32), self.spec)
        else:
            h = apply_act(self.act, self._g(x).astype(mx.float32), u.astype(mx.float32), self.spec, self.clamp)
        return self._d(h.astype(x.dtype))


class _Gate(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight


GATEUP_PAIR = os.environ.get("KYOJIN_EXPERT_GATEUP_PAIR", "0") not in ("0", "")      # flag expert_gateup_pair (flags.set updates it)

_RED_HDR = "#pragma clang fp contract(off)\n"
_RED_SRC = """
    const uint j = thread_position_in_grid.x;
    const uint t = thread_position_in_grid.y;
    float acc = 0.0f;
    for (uint k = 0; k < K; k++) {
        const float p = float(y[(t * K + k) * W + j]) * w[t * K + k];
        acc += p;
    }
    out[t * W + j] = static_cast<OT>(acc);
"""


def moe_reduce_ok(y, w) -> bool:
    """The weighted slot sum can run as one kernel: GPU stream, y [T, K, W] half / bfloat16 / float32, w [T, K] float32."""
    return (mx.default_device() == mx.gpu and y.ndim == 3 and w.ndim == 2 and w.dtype == mx.float32 and y.dtype in (mx.float16, mx.bfloat16, mx.float32)
            and tuple(w.shape) == tuple(y.shape[:2]))


def moe_reduce(y, w, out_dtype):
    """out [T, W] = sum_k float32(y[t, k]) * w[t, k] (products and the running sum in float32, slot order, no fma contraction), written as out_dtype: GenericMoE's
    (y.astype(f32) * w[..., None]).sum(-2).astype(out_dtype) in one launch (the same arithmetic; the stock reduction's association over the K terms is the only difference)."""
    from . import kernels as KN
    T, K, W = y.shape
    kern = KN._kern(("moe_reduce",), name="kyojin_moe_reduce", input_names=["y", "w"], output_names=["out"], header=_RED_HDR, source=_RED_SRC)
    return kern(inputs=[y, w], template=[("K", K), ("W", W), ("OT", out_dtype)], grid=(W, T, 1), threadgroup=(min(W, 256), 1, 1), output_shapes=[(T, W)], output_dtypes=[out_dtype])[0]


class _Stack:
    """Dense expert stack, the stock-ops counterpart of ForeignSwitchGLU: __call__(x [.., H], inds [.., K]) -> [.., K, H] (un-weighted)."""

    def __init__(self, gate_w, up_w, down_w, spec: MoeSpec):
        self.g, self.u, self.d, self.spec = gate_w, up_w, down_w, spec

    def __call__(self, x, inds):
        s = self.spec
        f = mx.float32                                              # gather_mm on the CPU device only takes f32
        xe = mx.expand_dims(x.astype(f), (-2, -3))
        u = mx.gather_mm(xe, self.u.astype(f).swapaxes(-1, -2), rhs_indices=inds)
        g = mx.gather_mm(xe, self.g.astype(f).swapaxes(-1, -2), rhs_indices=inds) if self.g is not None else None
        h = apply_act(s.act, g, u, s, s.clamp, s.clamp_mode)
        y = mx.gather_mm(h, self.d.astype(f).swapaxes(-1, -2), rhs_indices=inds)
        return y.squeeze(-2).astype(x.dtype)


def describe(spec: MoeSpec, fmts=None) -> dict:
    """What the engine's fused kernels can take. -> {"kind": "foreign_moe" | "glm_route" | "stock", "why": [reasons the block is not (fully) fused]}.
    `fmts` = the stored weight formats of the expert projections (ForeignSwitchGLU .g / .u / .d `.fmt`), None = unknown (the spec alone decides). With formats given, a variant the
    kernels cannot read in that format is reported as "stock" (plain path) with the reason, never as fused: foreign_moe needs stored experts (not "dense"); glm_route needs nvfp4 experts
    (glm_moe._why: the decode path reads them through the g2a gather, `g2a_ok`; any other format falls back with reason g2a_pack)."""
    d = _describe_spec(spec)
    if fmts is None or d["kind"] == "stock":
        return d
    f = sorted({str(x) for x in fmts})
    if d["kind"] == "foreign_moe" and "dense" in f:
        return {"kind": "stock", "why": [f"experts not stored (fmt {f}): foreign_moe fuses stored-format experts only"]}
    if d["kind"] == "glm_route" and f != ["nvfp4"]:
        return {"kind": "stock", "why": [f"experts stored as {f}: glm_moe's fused decode reads nvfp4 experts only (g2a gather; glm_moe fallback g2a_pack), so this variant runs the plain path"]}
    return d


def _describe_spec(spec: MoeSpec) -> dict:
    why = []
    s = spec
    if s.gating not in _GATINGS:
        why.append(f"gating {s.gating!r}")
    if s.act not in _ACTS:
        why.append(f"act {s.act!r}")
    if s.router_bias:
        why.append("router logit bias (kernels compute logits as one row-dot, no bias term)")
    if s.temperature != 1.0:
        why.append("router temperature (no kernel field)")
    if s.n_group > 1 and s.n_group != s.group_used:
        why.append("group-limited routing with dropped groups (Route.groups > 1 is not fused)")
    if not s.gated:
        why.append("non-gated experts (the pair kernels read gate and up)")
    if s.act not in ("silu", "situ"):
        why.append(f"act {s.act!r} has no kernel kind (ForeignSwitchGLU.act_kind: silu, situ)")
    if s.pre_exp_norm or s.post_exp_norm:
        why.append("norms around the expert block (no pre / post hook in the fused kernels)")
    if s.down_scale:
        why.append("per-expert down scale (no field in the fused kernels)")
    if s.shared_clamp > 1e-6:
        why.append("shared expert clamp (the fused shared expert has no clamp)")
    if s.clamp > 1e-6 and s.clamp_mode != "pre_act":
        why.append("post-activation clamp (ForeignSwitchGLU.limit is the pre-activation form)")
    if why:
        return {"kind": "stock", "why": why}
    if s.gating in ("softmax", "softmax_weight"):
        w2 = []
        if s.select_bias:
            w2.append("selection bias with softmax (the softmax select kernel has no bias input)")
        if s.weight_scale not in (0.0, 1.0):
            w2.append("weight scale (softmax select kernel has no scale)")
        if s.latent:
            w2.append("latent projection (foreign_moe has no pre / post hook)")
        if s.shared and s.shared_act != "silu":
            w2.append("shared act")
        return {"kind": "foreign_moe", "why": []} if not w2 else {"kind": "stock", "why": w2}
    if s.gating == "sigmoid":
        return {"kind": "glm_route", "why": []}
    return {"kind": "stock", "why": [f"gating {s.gating!r} has no select kernel (softmax and sigmoid only)"]}


ROUTE_RS = os.environ.get("KYOJIN_MOE_ROUTE_RS", "0") not in ("0", "")      # flag moe_route_rs (flags.py rewrites it at runtime)
_ZB: dict = {}


def _zeros_bias(E: int):
    z = _ZB.get(E)
    if z is None:
        z = _ZB[E] = mx.zeros((E,), dtype=mx.float32)
        mx.eval(z)
    return z


class GenericMoE(nn.Module):
    """Router + experts + optional shared experts (+ optional latent projection), all from the spec. Plain-MLX forward (the kind-specific subclasses add the
    description attributes the engine's installers read; their forward is this one)."""

    def __init__(self, spec: MoeSpec, tensors: dict, layer: int = 0):
        super().__init__()
        t = _norm_names(tensors)
        s = self.spec_ = spec
        self.layer = layer
        E, K = s.n_expert, s.top_k
        if not (0 < K <= E):
            raise ValueError(f"top_k {K} / n_expert {E}")
        gw = t["gate_inp"]
        self.router_w = gw.astype(mx.float32)
        self.router_b = t["gate_inp_b"].astype(mx.float32) if s.router_bias else None
        self.sel_b = t["exp_probs_b"].astype(mx.float32) if s.select_bias else None
        if s.router_bias and self.router_b is None or s.select_bias and self.sel_b is None:
            raise ValueError("spec asks for a bias tensor the loader did not give")
        self.switch_mlp = t.get("switch_mlp")
        if self.switch_mlp is None:
            self.switch_mlp = _Stack(t.get("gate_exps") if s.gated else None, t["up_exps"], t["down_exps"], s)
        self.latent_down = _lin(t["routed_down"]) if s.latent and "routed_down" in t else None
        self.latent_up = _lin(t["routed_up"]) if s.latent and "routed_up" in t else None
        self.latent_norm_w = t.get("routed_norm") if s.latent_norm else None
        self.pre_exp_w = t["pre_exp_norm"].astype(mx.float32) if s.pre_exp_norm else None
        self.post_exp_w = t["post_exp_norm"].astype(mx.float32) if s.post_exp_norm else None
        self.down_s = t["down_exps_s"].astype(mx.float32) if s.down_scale else None
        self.shared_ffn = None
        self.shared_gate_w = None
        if s.shared:
            self.shared_ffn = DenseFFN(t.get("gate_shexp"), t["up_shexp"], t["down_shexp"], s.shared_act, s, s.shared_clamp)
            if s.shared_gate:
                self.shared_gate_w = t["gate_inp_shexp"].reshape(-1)
        self._describe_extra(t)

    def _describe_extra(self, t):
        pass

    # ---- router (llama-graph.cpp:2284-2422) ----
    def _rs_ok(self, x2) -> bool:
        """The fused router (glm_moe_rs.rowsel) computes this block's router: sigmoid gating, one group, no logit bias / temperature, 1e-20 or max() norm epsilon, <= 8 rows, Metal."""
        from . import glm_moe_rs as _RS
        s = self.spec_
        return bool(s.gating == "sigmoid" and s.n_group <= 1 and not s.router_bias and s.temperature == 1.0 and s.norm_eps <= 1e-12 and s.weight_scale != 0.0
                    and x2.ndim == 2 and x2.dtype in (mx.float16, mx.bfloat16, mx.float32) and self.router_w.dtype in (mx.float16, mx.bfloat16, mx.float32)
                    and mx.default_device() == mx.gpu and _RS.eligible(x2.shape[0], s.n_expert, s.top_k, x2.shape[1]))

    def route(self, x2):
        """x2 [T, H] -> (inds [T, K] int, weights [T, K] f32)."""
        s = self.spec_
        E, K = s.n_expert, s.top_k
        if ROUTE_RS and self._rs_ok(x2):                                             # flag moe_route_rs: router GEMV + sigmoid + bias + top-k + normalise + scale in ONE launch
            from . import glm_moe_rs as _RS
            T = x2.shape[0]
            bias = self.sel_b if self.sel_b is not None else _zeros_bias(E)
            _, inds, w = _RS.rowsel(x2, self.router_w, bias, T, E, K, s.norm_weights, s.weight_scale if s.weight_scale not in (0.0, 1.0) else 1.0)
            return inds.reshape(T, K), w.reshape(T, K)
        lg = skinny_mm(x2.astype(mx.float32), self.router_w)                         # flag skinny_rows: few rows = one rowdot launch
        if self.router_b is not None:
            lg = lg + self.router_b
        if s.temperature != 1.0:
            lg = lg / s.temperature
        if s.gating == "softmax":
            probs = mx.softmax(lg, axis=-1, precise=True)
        elif s.gating == "sigmoid":
            probs = mx.sigmoid(lg)
        elif s.gating in ("softmax_weight", "none"):
            probs = lg
        elif s.gating == "sqrt_softplus":
            probs = mx.sqrt(nn.softplus(lg))
        else:
            raise ValueError(f"unknown gating {s.gating!r}")
        sel = probs + self.sel_b if self.sel_b is not None else probs
        if s.n_group > 1:                                                           # top `group_used` groups (llama-graph.cpp:2346)
            T, G = sel.shape[0], s.n_group
            grp = sel.reshape(T, G, E // G)
            top2 = mx.sort(grp, axis=-1)[..., -2:]
            gscore = top2.sum(-1)
            keep = mx.argsort(-gscore, axis=-1)[:, :s.group_used]
            mask = mx.zeros((T, G), dtype=mx.bool_)
            mask = mx.put_along_axis(mask, keep, mx.ones(keep.shape, dtype=mx.bool_), axis=-1)
            sel = mx.where(mask[..., None], grp, -mx.inf).reshape(T, E)
        inds = mx.argsort(-sel, axis=-1)[:, :K]                                      # ties: lowest index first
        w = mx.take_along_axis(probs, inds, axis=-1)
        if s.gating == "softmax_weight":
            w = mx.softmax(w, axis=-1, precise=True)
        if s.norm_weights and _RI.ROUTE_NORM_FUSE and w.dtype == mx.float32 and 1 < w.shape[0] <= 8 and mx.default_device() == mx.gpu:   # flag route_norm_fuse: one launch for sum / eps|max / divide / scale
            return inds, route_norm(w, s.norm_eps, F16_MIN_NORM, s.weight_scale)
        if s.norm_weights:
            tot = w.sum(-1, keepdims=True)
            tot = tot + s.norm_eps if s.norm_eps > 0 else mx.maximum(tot, F16_MIN_NORM)
            w = w / tot
        if s.weight_scale not in (0.0, 1.0):
            w = w * s.weight_scale
        return inds, w

    def _norm_latent(self, y):
        if self.latent_norm_w is None:
            return y
        y32 = y.astype(mx.float32)
        r = y32 * mx.rsqrt((y32 * y32).mean(-1, keepdims=True) + self.spec_.norm_eps_rms)
        return (r * self.latent_norm_w.astype(mx.float32)).astype(y.dtype)

    def shared_out(self, x):
        sh = self.shared_ffn(x)
        if self.shared_gate_w is not None:
            g = mx.sigmoid((x.astype(mx.float32) @ self.shared_gate_w.astype(mx.float32))[..., None])
            sh = (sh.astype(mx.float32) * g).astype(sh.dtype)
        return sh

    def __call__(self, x):
        s = self.spec_
        lead, H = x.shape[:-1], x.shape[-1]
        x2 = x.reshape(-1, H)
        inds, w = self.route(x2)
        xe = self.latent_down(x2) if self.latent_down is not None else x2
        if self.pre_exp_w is not None:                                               # the router above read x2, not this
            xe = mx.fast.rms_norm(xe, self.pre_exp_w.astype(xe.dtype), s.pre_exp_eps)
        y = self.switch_mlp(xe, inds)                                                # [T, K, W]
        if self.down_s is not None:                                                  # before the post-expert norm and the router weights
            y = y * self.down_s[inds][..., None].astype(y.dtype)
        if self.post_exp_w is not None:
            y = mx.fast.rms_norm(y * self.post_exp_w.astype(y.dtype), mx.ones((y.shape[-1],), dtype=y.dtype), s.post_exp_eps)
        if GATEUP_PAIR and self.down_s is None and self.post_exp_w is None and moe_reduce_ok(y, w):    # flag expert_gateup_pair
            out = moe_reduce(y, w, x.dtype)
        else:
            out = (y.astype(mx.float32) * w[..., None]).sum(-2)
        out = self._norm_latent(out.astype(x.dtype))
        if self.latent_up is not None:
            out = self.latent_up(out)
        out = out.astype(x.dtype)
        if self.shared_ffn is not None:
            out = out + self.shared_out(x2)
        return out.reshape(*lead, H)


class _ForeignMoeKind(GenericMoE):
    """Described to foreign_moe.install (softmax kernels): the attributes it reads."""

    def _describe_extra(self, t):
        s = self.spec_
        self.gate = _Gate(self.router_w)
        self.top_k = s.top_k
        self.num_experts = s.n_expert
        self.norm_topk_prob = bool(s.norm_weights or s.gating == "softmax_weight")
        if self.shared_ffn is not None:
            if self.shared_gate_w is not None:
                self.shared_expert = self.shared_ffn
                self.shared_expert_gate = _Gate(self.shared_gate_w.reshape(1, -1).astype(mx.float32))
            else:                                                                    # always-on: sigmoid(0) * (2 * shared) = shared, exactly
                ffn = self.shared_ffn
                self.shared_expert = lambda x: ffn(x) * 2
                self.shared_expert_gate = _Gate(mx.zeros((1, self.router_w.shape[1]), dtype=mx.float32))


class _GlmRouteKind(GenericMoE):
    """Described to glm_moe.install (sigmoid select kernel + combine): `moe_route()` returns a Route."""

    def moe_route(self):
        from .glm_moe import Route
        s = self.spec_
        bias = self.sel_b if self.sel_b is not None else mx.zeros((s.n_expert,), dtype=mx.float32)
        pre = post = None
        width = None
        if s.latent:
            pre, width = self.latent_down, self.latent_down(mx.zeros((1, self.router_w.shape[1]), dtype=mx.float16)).shape[-1]
            up = self.latent_up

            def post(y):
                y = self._norm_latent(y)
                return up(y) if up is not None else y
        sh = (lambda x: self.shared_out(x.reshape(-1, x.shape[-1]))) if self.shared_ffn is not None else None
        return Route(self.router_w, s.top_k, bias, norm=s.norm_weights, scale=s.weight_scale if s.weight_scale != 0.0 else 1.0, routers=1, groups=1,
                     pre=pre, post=post, width=width, wround=False, shared=sh)


_KINDS = {"foreign_moe": _ForeignMoeKind, "glm_route": _GlmRouteKind, "stock": GenericMoE}


def expert_fmts(tensors: dict):
    """Stored formats of the engine's ForeignSwitchGLU in tensors["switch_mlp"] (None when the experts are not one: the spec alone decides)."""
    sm = tensors.get("switch_mlp")
    ps = [getattr(sm, a, None) for a in ("g", "u", "d")]
    return tuple(getattr(p, "fmt", None) or "dense" for p in ps) if all(p is not None for p in ps) else None


def moe_block(layer: int, spec: MoeSpec, tensors: dict) -> GenericMoE:
    return _KINDS[describe(spec, expert_fmts(tensors))["kind"]](spec, tensors, layer)


class ParallelDenseMoE(nn.Module):
    """out = moe(x_moe) + dense(x): a dense FFN in parallel with the MoE (arctic.cpp:115-142: the two branches read different norms of the stream, so
    __call__ takes an optional second input for the MoE branch)."""

    def __init__(self, moe: GenericMoE, dense: DenseFFN):
        super().__init__()
        self.moe, self.dense = moe, dense

    def __call__(self, x, x_moe=None):
        return self.moe(x if x_moe is None else x_moe) + self.dense(x)


def ffn_factory(layer_index: int, spec, tensors: dict):
    """Feed-forward block of one layer. spec: MoeSpec or a dict / object with its field names. tensors: name -> array, in HF orientation ([out, in]; experts
    [E, out, in]), with or without the `blk.N.` prefix and `.weight` suffix:
      gate_inp [E, H], gate_inp_b, exp_probs_b, gate_exps / up_exps / down_exps (or switch_mlp = ForeignSwitchGLU), gate_shexp / up_shexp / down_shexp,
      gate_inp_shexp [1, H], ffn_gate / ffn_up / ffn_down (dense layer or parallel dense), routed_down / routed_up / routed_norm (latent).
    Linear tensors may also be callables (ForeignLinear). -> a module x -> same shape (ParallelDenseMoE also takes x_moe)."""
    s = MoeSpec.from_any(spec)
    t = _norm_names(tensors)
    if not s.is_moe_layer(layer_index):
        return DenseFFN(t.get("ffn_gate"), t["ffn_up"], t["ffn_down"], s.act if s.act != "situ" else "silu", s)
    moe = moe_block(layer_index, s, t)
    if s.parallel_dense:
        return ParallelDenseMoE(moe, DenseFFN(t.get("ffn_gate"), t["ffn_up"], t["ffn_down"], "silu", s))
    return moe
