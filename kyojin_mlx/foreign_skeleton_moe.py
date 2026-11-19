# SPDX-License-Identifier: Apache-2.0
"""Wiring of the generic MoE block (foreign_moe_generic) into the decoder skeleton (foreign_skeleton) through the skeleton's `ffn_factory` hook.

Nothing here names an architecture. A file is MoE when its metadata has `<arch>.expert_count` > 0 and a block has the router tensor `ffn_gate_inp.weight`.
The MoeSpec is built from:
  * the file: expert_count, expert_used_count, expert_weights_norm / _scale, expert_gating_func (0 = not set), expert_group_count / _used_count are
    refused (not placed), leading_dense_block_count (a layer without a router tensor is dense whatever the count says), and the tensor list
    (ffn_gate_exps = gated experts, ffn_*_shexp = shared experts, ffn_gate_inp_shexp = sigmoid-gated shared expert, exp_probs_b = selection bias,
    ffn_gate_inp.bias = router logit bias);
  * the architecture's skeleton row: `act` (as for the dense FFN) and an optional `moe` dict of MoeSpec fields for what the file does not carry
    (e.g. norm_weights=True, gating="softmax"). The metadata wins over the row; the default gating is softmax (llama.cpp build_moe_ffn callers).
The generic block takes arrays; the skeleton loads weights into parameters by name, so each MoE layer is a `MoeFFN` module holding the tensors as
parameters (`mlp.<name>.weight`) and building the block on its first call.
"""
from __future__ import annotations

import dataclasses

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import foreign_gguf as GF
from . import foreign_moe_generic as G

# metadata keys (after `<arch>.`) this wiring places; every other expert-ish key makes the skeleton refuse
MOE_KEYS = {"expert_count", "expert_used_count", "expert_shared_count", "expert_feed_forward_length", "expert_shared_feed_forward_length",
            "expert_weights_scale", "expert_weights_norm", "expert_gating_func", "leading_dense_block_count",
            # group-limited routing, latent projection, per-layer swiglu clamps (arrays or scalars)
            "expert_group_count", "expert_group_used_count", "moe_latent_size", "swiglu_clamp_exp", "swiglu_clamp_shexp",
            # stored by the converter, not read by the graph of the archs that carry it (step35.cpp decides MoE by the router tensor, as detect() does)
            "moe_every_n_layers"}

# GGUF block tensor kind -> attribute of MoeFFN (the module path is `mlp.<attr>`; param_name appends .weight / .bias)
NAMES = {"ffn_gate_inp": "gate_inp", "ffn_gate_exps": "gate_exps", "ffn_up_exps": "up_exps", "ffn_down_exps": "down_exps",
         "ffn_gate_shexp": "gate_shexp", "ffn_up_shexp": "up_shexp", "ffn_down_shexp": "down_shexp", "ffn_gate_inp_shexp": "gate_inp_shexp",
         "exp_probs_b": "exp_probs_b", "ffn_latent_down": "routed_down", "ffn_latent_up": "routed_up",
         "ffn_pre_exp_norm": "pre_exp_norm", "ffn_post_exp_norm": "post_exp_norm"}
_GATING = {1: "softmax", 2: "sigmoid", 3: "softmax_weight", 4: "sqrt_softplus"}

# block N6, latent-MoE alias: llama.cpp's FFN_ROUTED_* tensors and the key expert_latent_length are the same op as FFN_LATENT_* / moe_latent_size
# (a down projection to the latent width before the routed experts, an up projection after them; the router and the shared expert see the full width)
ALIASES = {"ffn_routed_down": "ffn_latent_down", "ffn_routed_up": "ffn_latent_up"}
KEY_ALIASES = {"expert_latent_length": "moe_latent_size"}
NAMES.update({a: NAMES[k] for a, k in ALIASES.items()})
MOE_KEYS |= set(KEY_ALIASES)
# predicate kind of N6: a weight-only norm of the latent width (Kimi-K3 writes blk.N.ffn_routed_norm [latent] on every routed layer, Hy4-flash
# none). Present on a layer = the norm form fires there; absent = the plain alias. Header-mapped only (op not read, not in NAMES): mapped, not run.
ROUTED_NORM = "ffn_routed_norm"


def routed_norm_fires(layer_tensors) -> bool:
    return ROUTED_NORM + ".weight" in layer_tensors


def _canon(kind: str) -> str:
    """`ffn_routed_down.weight` -> `ffn_latent_down.weight` (N6); any other kind unchanged."""
    base, _, tail = kind.partition(".")
    return ALIASES.get(base, base) + ("." + tail if tail else "")


class _W(nn.Module):
    def __init__(self, ne, bias_ne=None, scale_ne=None):
        super().__init__()
        if ne is not None:
            self.weight = mx.zeros(tuple(int(x) for x in reversed(ne)))
        if scale_ne is not None:
            self.scale = mx.ones(tuple(int(x) for x in reversed(scale_ne)))
        if bias_ne is not None:
            self.bias = mx.zeros(tuple(int(x) for x in reversed(bias_ne)))


class _Box:
    pass


class MoeFFN(nn.Module):
    """One MoE layer: tensors as parameters (loaded by name), the generic block built on first use."""

    def __init__(self, il: int, spec: G.MoeSpec, tensors: dict):
        super().__init__()
        self.il, self.moe_spec = il, spec
        self._box = _Box()
        for kind, attr in NAMES.items():
            if kind + ".weight" in tensors or kind + ".bias" in tensors:
                setattr(self, attr, _W(tensors.get(kind + ".weight"), tensors.get(kind + ".bias"), tensors.get(kind + ".scale")))
        self._attrs = tuple(a for a in NAMES.values() if a in self)

    def set_stored(self, attr: str, value):
        """Serve the tensor `attr` (a NAMES value, or `switch_mlp` for the three expert stacks) by `value` instead of a parameter: a callable linear
        (ForeignLinear) or the expert switch (ForeignSwitchGLU / a row-decoding switch). The parameters it replaces leave the module (never loaded)."""
        st = self._box.__dict__.setdefault("stored", {})
        st[attr] = value
        for a in (("gate_exps", "up_exps", "down_exps") if attr == "switch_mlp" else (attr,)):
            if a in self:
                self.pop(a)
        self._box.blk = None

    def block(self):
        if getattr(self._box, "blk", None) is None:
            t = {}
            for a in self._attrs:
                if a not in self:                                      # served by set_stored
                    continue
                if "weight" in self[a]:
                    t[a] = self[a]["weight"]
                    if "scale" in self[a]:
                        t[a + "_s"] = self[a]["scale"]
                    if "bias" in self[a]:
                        t[a + "_b"] = self[a]["bias"]
                else:                                                  # exp_probs_b: the file stores it as a bias
                    t[a] = self[a]["bias"]
            t.update(self._box.__dict__.get("stored", {}))
            self._box.blk = G.ffn_factory(self.il, self.moe_spec, t)
        return self._box.blk

    def __call__(self, x):
        return self.block()(x)


class Detected:
    def __init__(self, spec, factory, names):
        self.spec, self.factory, self.names = spec, factory, names


def _per_layer(kv, key, nl, default, what):
    """A key that is a scalar or a per-layer array -> list of nl values (an array may be longer than nl: the extra entries belong to the MTP layers)."""
    from .foreign_skeleton import SkeletonError
    v = kv.get(key, default)
    if isinstance(v, (list, tuple, np.ndarray)):
        if len(v) < nl:
            raise SkeletonError(f"{key}: {len(v)} entries for {nl} layers")
        return list(v)[:nl]
    return [v] * nl


def moe_layers_idx(lt):
    return [i for i, t in enumerate(lt) if "ffn_gate_inp.weight" in t]


def detect(arch: str, kv: dict, tensors: dict, row: dict | None):
    """-> Detected (MoeSpec, ffn_factory, extra_names) when the file is a MoE, None when it is dense. Raises SkeletonError for a MoE it cannot place."""
    from .foreign_skeleton import SkeletonError
    p = arch + "."
    n_exp = int(GF.kv_counts(kv.get(p + "expert_count"))[1] or 0)      # scalar or per-layer array (largest value); the per-layer counts come from the router tensors below
    routers = {k for k in tensors if k.startswith("blk.") and k.endswith(".ffn_gate_inp.weight")}
    if n_exp <= 0 or not routers:
        if n_exp > 0 or routers:
            raise SkeletonError(f"{arch}: expert_count {n_exp} and {len(routers)} router tensors disagree")
        return None
    row = row or {}
    nl = int(kv[p + "block_count"]) - int(kv.get(p + "nextn_predict_layers", 0) or 0)      # MTP layers are not decoded
    lt = [{} for _ in range(nl)]
    for name, ne in tensors.items():
        parts = name.split(".", 2)
        if parts[0] == "blk" and int(parts[1]) < nl:
            lt[int(parts[1])][_canon(parts[2])] = ne
    dense = tuple(i for i, t in enumerate(lt) if "ffn_gate_inp.weight" not in t)
    moe_layers = [t for t in lt if "ffn_gate_inp.weight" in t]
    for t in moe_layers:
        for need in ("ffn_up_exps.weight", "ffn_down_exps.weight"):
            if need not in t:
                raise SkeletonError(f"{arch}: a MoE layer without {need}")
    t0 = moe_layers[0]
    shared = "ffn_up_shexp.weight" in t0
    latent = "ffn_latent_down.weight" in t0
    top_k = _per_layer(kv, p + "expert_used_count", nl, None, "expert_used_count")
    n_group, group_used = int(kv.get(p + "expert_group_count", 1) or 1), int(kv.get(p + "expert_group_used_count", 1) or 1)
    if n_group > 1 and (n_exp % n_group or not 0 < group_used <= n_group):
        raise SkeletonError(f"{arch}: expert_group_count {n_group} / used {group_used} do not fit {n_exp} experts")
    clamp_exp = [float(x or 0.0) for x in _per_layer(kv, p + "swiglu_clamp_exp", nl, 0.0, "swiglu_clamp_exp")]
    clamp_sh = [float(x or 0.0) for x in _per_layer(kv, p + "swiglu_clamp_shexp", nl, 0.0, "swiglu_clamp_shexp")]
    rms_eps = float(kv.get(p + "attention.layer_norm_rms_epsilon", 1e-6))
    cnt = GF.kv_counts(kv.get(p + "expert_count"))[2]                  # per-layer array form (list / tuple / numpy array), None for a scalar
    for i in moe_layers_idx(lt):                                       # the metadata count of a layer must match its router rows (a scalar is the largest count)
        rows = int(lt[i]["ffn_gate_inp.weight"][1])
        if cnt is not None and len(cnt) <= i:
            raise SkeletonError(f"{arch}: expert_count has {len(cnt)} entries, layer {i} has a router")
        if (cnt is not None and cnt[i] != rows) or (cnt is None and rows > n_exp):
            raise SkeletonError(f"{arch}: layer {i}: expert_count {cnt[i] if cnt is not None else n_exp} but the router has {rows} rows")
    fields = dict(
        n_expert=n_exp, top_k=int(top_k[moe_layers_idx(lt)[0]]), n_group=n_group, group_used=group_used, latent=latent, clamp=clamp_exp[moe_layers_idx(lt)[0]],
         gating=_GATING.get(int(kv.get(p + "expert_gating_func", 0) or 0), "softmax"),
        router_bias="ffn_gate_inp.bias" in t0, select_bias="exp_probs_b.bias" in t0, gated="ffn_gate_exps.weight" in t0,
        norm_weights=bool(kv.get(p + "expert_weights_norm", False)), weight_scale=float(kv.get(p + "expert_weights_scale", 1.0) or 1.0),
        pre_exp_norm="ffn_pre_exp_norm.weight" in t0, post_exp_norm="ffn_post_exp_norm.weight" in t0, pre_exp_eps=rms_eps,
        post_exp_eps=float(kv.get(p + "post_norm_epsilon", 0.0) or row.get("post_norm_eps", 0.0) or rms_eps),
        act=row.get("act", "silu"), shared=shared, shared_gate="ffn_gate_inp_shexp.weight" in t0, shared_act=row.get("act", "silu"), dense_layers=dense)
    if "expert_gating_func" in kv and not int(kv[p + "expert_gating_func"] or 0):
        fields.pop("gating")
    for k, v in (row.get("moe") or {}).items():                      # row: only what the file does not say
        if k in ("n_expert", "top_k", "dense_layers"):
            raise SkeletonError(f"{arch}: row key moe.{k} would override the file")
        if k == "gating" and int(kv.get(p + "expert_gating_func", 0) or 0):
            continue
        if k in ("norm_weights", "weight_scale") and (p + "expert_weights_" + ("norm" if k == "norm_weights" else "scale")) in kv:
            continue
        fields[k] = v
    for t in moe_layers:                                              # the layers must agree with the spec built from the first one
        for kind, flag in (("ffn_up_shexp.weight", shared), ("ffn_latent_down.weight", latent), ("ffn_latent_up.weight", latent), ("exp_probs_b.bias", fields["select_bias"]), ("ffn_gate_inp.bias", fields["router_bias"]),
                           ("ffn_gate_exps.weight", fields["gated"]), ("ffn_pre_exp_norm.weight", fields["pre_exp_norm"]), ("ffn_post_exp_norm.weight", fields["post_exp_norm"]), ("ffn_gate_inp_shexp.weight", fields["shared_gate"])):
            if (kind in t) != flag:
                raise SkeletonError(f"{arch}: MoE layers differ on {kind}")
    lk = next((k for k in ("moe_latent_size", *KEY_ALIASES) if p + k in kv), None)
    if latent and lk and int(kv[p + lk]) != int(t0["ffn_latent_down.weight"][1]):
        raise SkeletonError(f"{arch}: {lk} {kv[p + lk]} differs from ffn_latent_down {t0['ffn_latent_down.weight']}")
    spec = G.MoeSpec(**fields)
    # one spec per MoE layer: the per-layer top-k, expert count (router rows) and clamp of the arrays; every other field is shared
    per = {i: dataclasses.replace(spec, top_k=int(top_k[i]), n_expert=int(lt[i]["ffn_gate_inp.weight"][1]), clamp=clamp_exp[i], shared_clamp=clamp_sh[i], down_scale="ffn_down_exps.scale" in lt[i]) for i in range(nl)
           if "ffn_gate_inp.weight" in lt[i]}
    for i, sp_i in per.items():
        if not 0 < sp_i.top_k <= sp_i.n_expert or (n_group > 1 and sp_i.n_expert % n_group):
            raise SkeletonError(f"{arch}: layer {i}: top_k {sp_i.top_k} with {sp_i.n_expert} experts / {n_group} groups")

    def factory(il, sp, tensors_):
        if "ffn_gate_inp.weight" not in tensors_:                    # leading / interleaved dense layer: the skeleton's own FFN
            from .foreign_skeleton import dense_ffn_factory
            return dense_ffn_factory(il, sp, tensors_)
        return MoeFFN(il, per.get(il, spec), tensors_)
    return Detected(spec, factory, {k: "mlp." + v for k, v in NAMES.items()})
