# SPDX-License-Identifier: Apache-2.0
"""Config-driven decoder skeleton: a decoder-only transformer whose every choice is a field of `ModelArgs`, built from the GGUF itself
(tensor list + metadata) and one row of foreign_skeleton_rows.ROWS. No architecture name appears in this file.

Used only for a GGUF architecture that has no model class (foreign_archkit_cpu / foreign_archkit_serve call it when the class lookup fails);
`load_cpu(..., force)` can also take any architecture that has a row, for checks.

Serving contract (the one mlx-lm classes meet, which serve/backends.py relies on): `Model(args)`, `model(tokens[B, L], cache=None) -> logits`,
`model.layers` (list, one KV cache each: mlx-lm KVCache / BatchKVCache through `update_and_fetch`, `offset` may be an int or a per-row array),
`model.model.embed_tokens`, `sanitize`. Linears are plain `nn.Linear` named like the HF llama modules, so foreign_linear swaps them for the stored-format
ForeignLinear exactly as for a stock class (same objects, no second matmul path)."""
from __future__ import annotations

import dataclasses
import math
import os
import re
import sys
import types
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import headq as HQ
from .foreign_skeleton_rows import ROWS
from . import ablate as _ABL


# KYOJIN_SPLIT_EVAL="1,10,20": a pass of KYOJIN_SPLIT_ROWS_MIN..KYOJIN_SPLIT_ROWS rows (default 2..8: an MTP verify) calls mx.async_eval(h) after those layers, so the GPU runs the
# first layers while the host builds the rest (n10 measure: with one eval at the end the GPU idles for the whole host build, 2.6 ms at 1 row, 7.2 ms at 6 rows, then waits no more).
# Read at every call (arms of one load flip it through the environment); ROWS_MIN=1 also covers the plain one-row step.
_SPLIT_CACHE: dict = {}
SPLIT_ON = False                  # flag split_eval: layers 1 and 8 submitted early on every pass of 1..8 rows (n10: plain +8.6 % ids equal, verify T=6 -6.1 ms); explicit KYOJIN_SPLIT_* env wins


def _split_cfg():
    e = os.environ
    key = (e.get("KYOJIN_SPLIT_EVAL", ""), e.get("KYOJIN_SPLIT_ROWS", "8"), e.get("KYOJIN_SPLIT_ROWS_MIN", "2"), SPLIT_ON)
    v = _SPLIT_CACHE.get(key)
    if v is None:
        at, hi, lo, on = key
        if on and not at:
            at, lo = "1,8", "1"
        v = _SPLIT_CACHE[key] = (frozenset(int(x) for x in at.split(",") if x.strip()), int(hi), int(lo))
    return v


class SkeletonError(ValueError):
    pass


ACTS = {
    "silu": nn.silu,
    "gelu": nn.gelu_approx,                                    # ggml_gelu is the tanh form
    "gelu_erf": nn.gelu,
    "gelu_quick": lambda x: x * mx.sigmoid(1.702 * x),
    "relu": nn.relu,
    "relu2": lambda x: mx.square(nn.relu(x)),
}


def _swiglu_oai(gate, up, alpha=1.702, limit=7.0):
    """ggml_swiglu_oai (ggml-cpu/ops.cpp:3388-3452; constants llama-graph.cpp:2145-2148): min(g, limit) * sigmoid(alpha * min(g, limit)) * (clip(u, -limit, limit) + 1).
    Plain MLX ops: there is no fused kernel for it (foreign_moe_generic.describe() says so for the experts)."""
    g = mx.minimum(gate, limit)
    return g * mx.sigmoid(alpha * g) * (mx.clip(up, -limit, limit) + 1.0)


# two-input gated activations (gate, up) -> hidden; they need a gated FFN
GLU_ACTS = {"swiglu_oai": _swiglu_oai}

# GGUF tensor kind (after `blk.N.`, without .weight / .bias) -> module path under `model.layers.N.`
BLK_NAMES = {
    "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm", "post_attention_norm": "post_self_attn_layernorm",
    "post_ffw_norm": "post_mlp_layernorm", "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
    "attn_qkv": "self_attn.qkv_proj", "attn_output": "self_attn.o_proj", "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm",
    "attn_gate": "self_attn.gate_proj", "attn_sinks": "self_attn.sinks",
    "indexer.q_proj": "self_attn.idx_q_proj", "indexer.k_proj": "self_attn.idx_k_proj", "indexer.q_norm": "self_attn.idx_q_norm", "indexer.k_norm": "self_attn.idx_k_norm",
    "attn_resid_alpha": "attn_resid_alpha", "attn_resid_beta": "attn_resid_beta", "ffn_resid_alpha": "ffn_resid_alpha", "ffn_resid_beta": "ffn_resid_beta",
    "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj",
}
TOP_NAMES = {
    "token_embd.weight": "model.embed_tokens.weight", "position_embd.weight": "model.embed_positions.weight",
    "token_embd_norm.weight": "model.embed_norm.weight", "token_embd_norm.bias": "model.embed_norm.bias",
    "output_norm.weight": "model.norm.weight", "output_norm.bias": "model.norm.bias", "output.weight": "lm_head.weight",
    "rope_freqs.weight": "model.rope_factors",
}

# blocks N1 (MLA), N2 (hyper-connection streams), N4 (Engram, Hy4 form): GGUF block kind -> module path under model.layers.N. (N3 kpool: foreign_skeleton_kpool.KINDS).
# The kinds in N_RAW are plain arrays: the file's `.weight` is the array itself (no Linear / Norm around it).
N_NAMES = {
    "attn_q_a": "self_attn.q_a_proj", "attn_q_a_norm": "self_attn.q_a_norm", "attn_q_b": "self_attn.q_b_proj", "attn_kv_a_mqa": "self_attn.kv_a_proj",
    "attn_kv_a_norm": "self_attn.kv_a_norm", "attn_k_b": "self_attn.k_b", "attn_v_b": "self_attn.v_b",
    **{f"hc_{s}_{p}": f"hc_{s}.{p}" for s in ("attn", "ffn") for p in ("fn", "base", "scale")},
    **{k: "engram." + k[len("engram_"):] for k in ("engram_kv", "engram_kv_b", "engram_gate", "engram_hc_scale", "engram_lookup_norm", "engram_pre_gamma")},
}
N_TOP = {f"output_hc_{p}.weight": f"model.hc_head.{p}" for p in ("fn", "base", "scale")}
N_RAW = ({"attn_k_b", "attn_v_b", "indexer_compressor_ape", "indexer_compressor_norm"} | {k for k in N_NAMES if k.startswith(("hc_", "engram_"))})
ENGRAM_TABLE = "per_layer_token_embd.weight"                    # the ONE Engram table: read by row (Model.engram_source), never a parameter

# metadata keys (after `<arch>.`) the skeleton places; any other key makes the build refuse instead of ignoring it
_KNOWN = {
    "context_length", "embedding_length", "block_count", "feed_forward_length", "vocab_size", "attention.head_count", "attention.head_count_kv",
    "attention.key_length", "attention.value_length", "attention.layer_norm_epsilon", "attention.layer_norm_rms_epsilon", "attention.clamp_kqv",
    "attention.scale", "rope.freq_base", "rope.dimension_count", "rope.scaling.type", "rope.scaling.factor", "rope.scaling.alpha",
    "rope.scaling.original_context_length", "rope.scaling.finetuned", "rope.dimension_sections", "use_parallel_residual", "embedding_scale",
    "logit_scale", "residual_scale",
    # attention mask kinds (group 1): llama.cpp load_arch_hparams of the sliding-window archs
    "final_logit_softcapping", "post_norm_epsilon", "nextn_predict_layers",
    "attention.sliding_window", "attention.sliding_window_pattern", "rope.freq_base_swa", "attention.temperature_scale", "attention.temperature_length",
    "attention.value_scale",
    "attention.indexer.head_count", "attention.indexer.key_length", "attention.indexer.top_k", "attention.indexer.block_size", "attention.indexer.local_blocks",
    "attention.indexer.types",
}


@dataclass
class ModelArgs:
    model_type: str = "kyojin_skeleton"
    hidden_size: int = 0
    num_hidden_layers: int = 0
    vocab_size: int = 0
    num_attention_heads: list = field(default_factory=list)       # per layer
    num_key_value_heads: list = field(default_factory=list)       # per layer
    head_dim: int = 0
    intermediate_size: list = field(default_factory=list)         # per layer
    norm_kind: str = "rms"
    norm_eps: float = 1e-5
    norm_pos: str = "pre"
    parallel_residual: bool = False
    final_norm: bool = True
    act: str = "silu"
    rope_kind: str = "neox"
    rope_dims: int = 0
    rope_base: float = 10000.0
    rope_scale: float = 1.0
    rope_factors: bool = False                                    # a rope_freqs.weight tensor divides the angles
    pos_learned: bool = False
    max_position: int = 0
    qk_norm_after_rope: bool = False
    clamp_qkv: float = 0.0
    embed_scale: float = 1.0
    logit_scale: float = 1.0
    residual_scale: float = 1.0
    attn_scale: float = 0.0                                       # 0: 1 / sqrt(head_dim)
    tie_word_embeddings: bool = True
    layer_tensors: list = field(default_factory=list)             # per layer: {"kind.weight" / "kind.bias": ne}
    top_tensors: dict = field(default_factory=dict)
    # attention mask kinds. swa_layers[il] True = the layer attends to the last `swa_window` positions only (llama.cpp LLAMA_SWA_TYPE_STANDARD: a key at
    # distance q - k >= window is masked, so a query sees `swa_window` positions including itself). Empty = all layers global.
    swa_window: int = 0
    swa_layers: list = field(default_factory=list)
    rope_layers: list = field(default_factory=list)               # per layer: False = no positional encoding (NoPE); empty = rope everywhere
    rope_base_swa: float = 0.0                                    # rope base of the windowed layers; 0 = same as rope_base
    logit_softcap: float = 0.0                                    # final logits: tanh(logits * logit_scale / c) * c (after the logit scale: muse-glimmer.cpp:173-179)
    post_norm_eps: float = 0.0                                    # eps of the sandwich post norms; 0 = norm_eps
    residual_kind: str = "plain"                                  # plain: x + branch | depth: branch / sqrt(i) + x * sqrt((i - 1) / i), i = 2 (l + 1) [+ 1 for the ffn] |
                                                                  # vector: x * alpha + branch * beta (learned [d] vectors per sub-layer)
    embed_rms: bool = False                                       # weightless rms norm of the embedding (row `embed_norm_plain`)
    ffn_post_mul_first: bool = False                              # ffn post norm = rms_norm(weight * x) instead of weight * rms_norm(x) (row `ffn_post_norm_mul_first`)
    attn_out_norm: bool = False                                   # weightless rms norm of each head's attention output, before the gate (row `attn_out_norm`)
    rope_factors_swa: bool = True                                 # False: the windowed layers do not use the rope_freqs.weight factors
    attn_temp_scale: float = 0.0                                  # NoPE layers: q *= (1 + (pos + 1) / attn_temp_length) ** attn_temp_scale (0 = off)
    attn_temp_length: int = 2048
    head_dim_v: int = 0                                           # value head dim when it differs from head_dim (attention.value_length); 0 = head_dim
    indexer_heads: int = 0                                        # block-sparse attention indexer (minimax-m3.cpp:339-532): 0 = none; the layers that have indexer.* tensors use it
    indexer_dim: int = 0
    indexer_top_k: int = 0
    indexer_block: int = 0
    indexer_local: int = 0
    attn_value_scale: float = 0.0                                 # attention.value_scale: the attention block output (after o_proj) is multiplied by it (mimo2.cpp:163); 0 / 1 = off
    # N1 MLA (layers with attn_kv_a_mqa + attn_k_b + attn_v_b): absorbed attention, scale (nope + rope) ** -0.5; 0 = no MLA layer
    mla_q_lora: int = 0
    mla_kv_lora: int = 0
    mla_nope: int = 0
    mla_v: int = 0
    # N3 kpool selection behind N1 (layers where foreign_skeleton_kpool.fires)
    kpool: int = 0
    kpool_select_tail: bool = False
    # N2 hyper-connection streams: 0 = plain residual
    hc_count: int = 0
    hc_eps: float = 0.0
    hc_magnitude: float = 1.0
    hc_iters: int = 0
    hc_post_eps: bool = False                                     # identity form: post = magnitude * sigmoid(..) + eps (llama.cpp build_hc_pre, hy-v4-flash.cpp:575-580)
    # N4 Engram (Hy4 form): layer ids that carry one; eps of its norms (FLT_EPSILON)
    engram_layers: list = field(default_factory=list)
    engram_eps: float = 0.0
    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def enabled() -> bool:
    """Flag KYOJIN_SKELETON (flags.py arch_skeleton, default 0, read at load): the skeleton is used by the arch kit loaders for an architecture that has skeleton rows."""
    return os.environ.get("KYOJIN_SKELETON", "0") not in ("0", "")


def has_rows(arch: str) -> bool:
    """A row of an unproven architecture (foreign_unproven.UNPROVEN_ARCHS) exists only while KYOJIN_KIT_UNPROVEN_ROWS is on."""
    from . import foreign_unproven as U
    return arch in ROWS and (arch not in U.UNPROVEN_ARCHS or U.on(arch))


def build_spec(arch: str, kv: dict, tensors: dict, row: dict | None = None, allow_moe: bool = False) -> ModelArgs:
    """GGUF metadata + tensor shapes (name -> ne) + the row of the architecture -> ModelArgs. Raises SkeletonError on anything it cannot place."""
    row = (ROWS.get(arch) if has_rows(arch) else None) if row is None else row
    if row is None:
        raise SkeletonError(f"{arch}: no skeleton row")
    p = arch + "."
    from . import foreign_skeleton_blocks as SB
    from . import foreign_skeleton_kpool as KP
    placed = _KNOWN | SB.MLA_KEYS | SB.HC_KEYS | KP.KEYS | SB.ENGRAM_KEYS          # the N1 / N2 / N3 / N4 keys: read below when their tensors are there
    extra = sorted(k[len(p):] for k in kv if k.startswith(p) and k[len(p):] not in placed)
    if allow_moe:
        from .foreign_skeleton_moe import MOE_KEYS
        extra = [k for k in extra if k not in MOE_KEYS]
    if extra:
        raise SkeletonError(f"{arch}: metadata keys the skeleton does not place: {extra}")
    if kv.get(p + "rope.scaling.type", "none") not in ("none", "linear"):
        raise SkeletonError(f"{arch}: rope scaling type {kv[p + 'rope.scaling.type']!r} is not in the skeleton")

    def get(key, default=None):
        return kv.get(p + key, default)

    nextn = int(get("nextn_predict_layers", 0) or 0)                # the last `nextn` blocks are multi-token-prediction layers (llama_hparams::n_layer): not decoded
    n = int(get("block_count")) - nextn
    d = int(get("embedding_length"))

    def per_layer(v, what):
        if isinstance(v, (list, tuple, np.ndarray)):
            v = [int(x) for x in v][:n + nextn]
            if len(v) < n:
                raise SkeletonError(f"{arch}: {what} has {len(v)} entries for {n} layers")
            return v[:n]
        return [int(v)] * n
    nh = per_layer(get("attention.head_count"), "head_count")
    nkv = per_layer(get("attention.head_count_kv", nh), "head_count_kv")
    hd = int(get("attention.key_length", d // nh[0]))
    hv = int(get("attention.value_length", hd))
    vs = float(get("attention.value_scale", 0.0) or 0.0)
    norm = row.get("norm", "rms")
    eps = get("attention.layer_norm_epsilon" if norm == "layer" else "attention.layer_norm_rms_epsilon")
    if eps is None:
        raise SkeletonError(f"{arch}: no {'layer_norm' if norm == 'layer' else 'layer_norm_rms'}_epsilon in the file")
    rope = row.get("rope", "neox")
    base = float(get("rope.freq_base", 10000.0))
    alpha = float(get("rope.scaling.alpha", 0.0))
    nrot = int(get("rope.dimension_count", hd))
    if alpha > 0:                                                  # NTK-aware: base * alpha^(dim / (dim - 2)) (llama.cpp load_arch_hparams)
        base *= alpha ** (hd / (hd - 2))
    scale = 1.0 / float(get("rope.scaling.factor", 1.0)) if get("rope.scaling.type", "none") == "linear" else 1.0
    lt = [{} for _ in range(n)]
    top = {}
    for name, ne in tensors.items():
        m = re.match(r"blk\.(\d+)\.(.+)$", name)
        if m:
            if int(m.group(1)) < n:
                lt[int(m.group(1))][m.group(2)] = list(ne)
        else:
            top[name] = list(ne)
    kinds = {k.rsplit(".", 1)[0] for t in lt for k in t}
    bad = sorted(kinds - set(BLK_NAMES) - set(N_NAMES) - set(KP.KINDS))
    if bad and not allow_moe:
        raise SkeletonError(f"{arch}: block tensors the skeleton has no module for: {bad}")
    if ENGRAM_TABLE in top:
        top.pop(ENGRAM_TABLE)
    gated = [("ffn_gate.weight" in t) for t in lt if "ffn_gate_inp.weight" not in t] or [False]     # MoE layers have their own experts
    if len(set(gated)) != 1:
        raise SkeletonError(f"{arch}: gated and non-gated FFN layers mixed")
    ff = get("feed_forward_length")
    if ff is None:                                                 # not in the metadata: read it off the up projection (ne = [in, out])
        ff = [t["ffn_up.weight"][1] if "ffn_up.weight" in t else 0 for t in lt] if all("ffn_up.weight" in t or "ffn_gate_inp.weight" in t for t in lt) else None
    if ff is None:
        raise SkeletonError(f"{arch}: feed_forward_length unknown")
    par = row.get("parallel")
    if par is None:
        par = bool(get("use_parallel_residual", False))
    ls = float(get("logit_scale", 1.0))
    if row.get("logit_div") and ls:
        ls = 1.0 / ls
    pos = "position_embd.weight" in top
    swa_w, swa_layers, rope_layers = _swa_fields(arch, get, row, n, pos)
    nb = _n_block_fields(arch, get, lt, top, nh, nrot, eps)
    return ModelArgs(
        hidden_size=d, num_hidden_layers=n, vocab_size=int(top["token_embd.weight"][1]), num_attention_heads=nh, num_key_value_heads=nkv, head_dim=hd,
        intermediate_size=per_layer(ff, "feed_forward_length"), norm_kind=norm, norm_eps=float(eps), norm_pos=row.get("norm_pos", "pre"),
        parallel_residual=par, act=row.get("act", "silu"), rope_kind="none" if pos else rope, rope_dims=nrot, rope_base=base, rope_scale=scale,
        rope_factors="rope_freqs.weight" in top, pos_learned=pos, max_position=int(top["position_embd.weight"][1]) if pos else 0,
        qk_norm_after_rope=bool(row.get("qk_norm_after_rope", False)), clamp_qkv=float(get("attention.clamp_kqv", 0.0) or 0.0),
        embed_scale=float(get("embedding_scale", 1.0)), logit_scale=ls, residual_scale=float(get("residual_scale", 1.0)),
        attn_scale=float(get("attention.scale", 0.0) or 0.0), tie_word_embeddings="output.weight" not in top, layer_tensors=lt, top_tensors=top,
        swa_window=swa_w, swa_layers=swa_layers, rope_layers=rope_layers, rope_base_swa=float(get("rope.freq_base_swa", 0.0) or 0.0),
        rope_factors_swa=bool(row.get("rope_factors_swa", True)), attn_out_norm=bool(row.get("attn_out_norm", False)),
        logit_softcap=float(get("final_logit_softcapping", 0.0) or 0.0), post_norm_eps=float(get("post_norm_epsilon", 0.0) or row.get("post_norm_eps", 0.0)),
        residual_kind=_residual_kind(arch, row, lt), embed_rms=bool(row.get("embed_norm_plain", False)) and "token_embd_norm.weight" not in top,
        ffn_post_mul_first=bool(row.get("ffn_post_norm_mul_first", False)),
        indexer_heads=int(get("attention.indexer.head_count", 0)), indexer_dim=int(get("attention.indexer.key_length", 0)), indexer_top_k=int(get("attention.indexer.top_k", 0)),
        indexer_block=int(get("attention.indexer.block_size", 0)), indexer_local=int(get("attention.indexer.local_blocks", 0)),
        head_dim_v=hv if hv != hd else 0, attn_value_scale=vs if vs != 1.0 else 0.0,
        attn_temp_scale=float(get("attention.temperature_scale", 0.0) or 0.0), attn_temp_length=int(get("attention.temperature_length", 2048) or 2048), **nb)


def _n_block_fields(arch, get, lt, top, nh, nrot, eps):
    """ModelArgs fields of N1 / N2 / N3 / N4, each read when its tensors are there (and then its keys must be: SkeletonError otherwise)."""
    from . import foreign_skeleton_mla as MLA, foreign_skeleton_kpool as KP, foreign_skeleton_hc as HC
    out = {}

    def need(key):
        v = get(key)
        if v is None:
            raise SkeletonError(f"{arch}: tensors of a block are there but the key {key} is not")
        return v
    mla = [il for il, t in enumerate(lt) if MLA.fires(t)]
    if mla:
        q, kvl, kl, vl = (int(need("attention." + k)) for k in ("q_lora_rank", "kv_lora_rank", "key_length_mla", "value_length_mla"))
        nope = kl - nrot
        for il in mla:
            t = lt[il]
            want = {"attn_q_a.weight": [None, q], "attn_q_b.weight": [q, nh[il] * kl], "attn_kv_a_mqa.weight": [None, kvl + nrot],
                    "attn_k_b.weight": [nope, kvl, nh[il]], "attn_v_b.weight": [kvl, vl, nh[il]]}
            for k, w in want.items():
                if k not in t or any(a is not None and a != b for a, b in zip(w, t[k])) or len(t[k]) != len(w):
                    raise SkeletonError(f"layer {il}: {k} {t.get(k)} does not fit the MLA keys (want {w})")
        out.update(mla_q_lora=q, mla_kv_lora=kvl, mla_nope=nope, mla_v=vl)
    if any(KP.fires(t) for t in lt):
        if not mla or any(KP.fires(t) and not MLA.fires(t) for t in lt):
            raise SkeletonError(f"{arch}: kpool indexer tensors on a layer without MLA (N3 reads N1's q_lora residual)")
        out.update(kpool=int(need("attention.indexer.kpool")), kpool_select_tail=bool(get("attention.indexer.kpool_select_tail", False)))
    hcl = [("hc_attn_fn.weight" in t) for t in lt]
    if any(hcl):
        if not all(hcl) or any(k not in top for k in N_TOP):
            raise SkeletonError(f"{arch}: hyper-connection tensors on some layers only, or no output_hc_* head")
        sub = {k[len(arch) + 1:]: v for k, v in ((f"{arch}.{s}", get(s)) for s in HC.KEYS) if v is not None}
        form = HC.form_of(sub)
        out.update(hc_count=int(need("hyper_connection.count")), hc_eps=float(need("hyper_connection.epsilon")),
                   hc_magnitude=float(get("hyper_connection.magnitude", 1.0)), hc_iters=int(get("hyper_connection.sinkhorn_iterations", 0) or 0),
                   hc_post_eps=form == "identity")
    eng = [il for il, t in enumerate(lt) if "engram_kv.weight" in t]
    if eng:
        miss = [(il, k) for il in eng for k in ("engram_kv_b", "engram_lookup_norm", "engram_pre_gamma", "engram_hc_scale", "engram_gate") if k + ".weight" not in lt[il]]
        if miss:
            raise SkeletonError(f"{arch}: Engram layers without the Hy4-form tensors {miss[:4]} (the DS4.1 form is served by its own class)")
        lids = [int(x) for x in need("ple.layers")]
        if lids != eng:
            raise SkeletonError(f"{arch}: ple.layers {lids} != the layers with engram tensors {eng}")
        if not out.get("hc_count"):
            raise SkeletonError(f"{arch}: Engram without hyper-connection streams")
        out.update(engram_layers=eng, engram_eps=float(np.finfo(np.float32).eps))
    return out


def _residual_kind(arch, row, lt):
    kind = row.get("residual", "plain")
    if kind not in ("plain", "depth", "vector"):
        raise SkeletonError(f"{arch}: row residual {kind!r}")
    if kind == "vector":
        for il, t in enumerate(lt):
            for k in ("attn_resid_alpha", "attn_resid_beta", "ffn_resid_alpha", "ffn_resid_beta"):
                if k + ".weight" not in t:
                    raise SkeletonError(f"{arch}: residual vector form without blk.{il}.{k}.weight")
    return kind


def _swa_fields(arch, get, row, n, pos):
    """-> (window, swa_layers, rope_layers). The pattern is the per-layer array of the file, or an integer period (`swa_period` in the file, else the row's
    `swa_period`): layer il is windowed iff il % period < period - 1 (llama_hparams::set_swa_pattern, llama-hparams.cpp:8-22, dense_first = false).
    A row's `rope_on` says which layers carry rope: "all" (default), "swa" (rope_pattern = is_swa: the global layers are NoPE), "global"."""
    w = int(get("attention.sliding_window", 0) or 0)
    if not w:
        return 0, [], []
    pat = get("attention.sliding_window_pattern")
    if pat is None:
        pat = row.get("swa_period")
    if pat is None:
        raise SkeletonError(f"{arch}: sliding_window {w} without a layer pattern (file key or row swa_period)")
    if isinstance(pat, (list, tuple, np.ndarray)):
        if len(pat) < n:
            raise SkeletonError(f"{arch}: sliding_window_pattern has {len(pat)} entries for {n} layers")
        swa = [bool(x) for x in list(pat)[:n]]                     # entries past block_count belong to the MTP layers
    else:
        per = int(pat)
        swa = [True] * n if per == 0 else [(il % per) < per - 1 for il in range(n)]
    on = row.get("rope_on", "all")
    if on not in ("all", "swa", "global"):
        raise SkeletonError(f"{arch}: row rope_on {on!r}")
    rope = [] if on == "all" else [s if on == "swa" else not s for s in swa]
    if rope and pos:
        raise SkeletonError(f"{arch}: rope_on with learned positions")
    return w, swa, rope


class Norm(nn.Module):
    """RMS or LayerNorm (with mean); weight / bias exist only when the file has them (a weightless LayerNorm has no parameters)."""

    def __init__(self, dim, kind, eps, weight=True, bias=False, mul_first=False):
        super().__init__()
        self.kind, self.eps, self.mul_first = kind, eps, mul_first
        if weight:
            self.weight = mx.ones((dim,))
        if bias:
            self.bias = mx.zeros((dim,))

    def __call__(self, x):
        w = getattr(self, "weight", None)
        if self.mul_first and w is not None:                       # muse-spark.cpp:190-192: the weight scales the input, then the norm
            return mx.fast.rms_norm(x * w, mx.ones((x.shape[-1],), dtype=x.dtype), self.eps)
        if self.kind == "rms":
            if w is None:
                w = mx.ones((x.shape[-1],), dtype=x.dtype)
            return mx.fast.rms_norm(x, w, self.eps)
        return mx.fast.layer_norm(x, w, getattr(self, "bias", None), self.eps)


def _norm(sp, dim, tens, kind, eps=None, mul_first=False):
    return Norm(dim, sp.norm_kind, eps or sp.norm_eps, weight=f"{kind}.weight" in tens, bias=f"{kind}.bias" in tens, mul_first=mul_first)


class Vec(nn.Module):
    """A learned per-channel vector (`weight`, ones until loaded): the alpha / beta of the vector residual form."""

    def __init__(self, dim):
        super().__init__()
        self.weight = mx.ones((dim,))


def _lin(tens, kind, i, o):
    return nn.Linear(i, o, bias=f"{kind}.bias" in tens)


class Attention(nn.Module):
    def __init__(self, sp: ModelArgs, il: int):
        super().__init__()
        t, d, hd = sp.layer_tensors[il], sp.hidden_size, sp.head_dim
        self.nh, self.nkv, self.hd, self.sp = sp.num_attention_heads[il], sp.num_key_value_heads[il], hd, sp
        self.scale = sp.attn_scale or hd ** -0.5
        self.hv = hv = sp.head_dim_v or hd                         # value head dim (llama.cpp n_embd_head_v)
        if "attn_qkv.weight" in t:
            self.qkv_proj = _lin(t, "attn_qkv", d, self.nh * hd + self.nkv * (hd + hv))
        else:
            self.q_proj, self.k_proj, self.v_proj = (_lin(t, "attn_q", d, self.nh * hd), _lin(t, "attn_k", d, self.nkv * hd), _lin(t, "attn_v", d, self.nkv * hv))
        self.o_proj = _lin(t, "attn_output", self.nh * hv, d)
        self.idx = "indexer.q_proj.weight" in t
        if self.idx:
            Hd, dx = sp.indexer_heads, sp.indexer_dim
            if not (Hd and dx and sp.indexer_top_k and sp.indexer_block) or Hd != self.nkv or dx != hd:
                raise SkeletonError(f"layer {il}: indexer tensors need attention.indexer.* keys with one indexer head per kv head ({self.nkv}) and key length = head dim ({hd}); "
                                    f"file has heads {Hd}, length {dx}, top_k {sp.indexer_top_k}, block {sp.indexer_block}")
            self.idx_q_proj, self.idx_k_proj = nn.Linear(d, Hd * dx, bias=False), nn.Linear(d, dx, bias=False)
            self.idx_q_norm, self.idx_k_norm = Norm(dx, "rms", sp.norm_eps), Norm(dx, "rms", sp.norm_eps)
        self.sinks = None
        if "attn_sinks.weight" in t:                               # one extra logit per head in the softmax denominator, no value (llama-graph.cpp:3120-3130)
            if list(t["attn_sinks.weight"]) != [self.nh]:
                raise SkeletonError(f"layer {il}: attn_sinks must have one value per head ({self.nh}), has {t['attn_sinks.weight']}")
            self.sinks = mx.zeros((self.nh,))
        self.qk_norm = None                                        # "head": one norm over head_dim; "full": over the projection width (applied before the head split)
        if "attn_q_norm.weight" in t:
            w = t["attn_q_norm.weight"][0]
            self.qk_norm = "head" if w == hd else "full"
            self.q_norm = Norm(w, sp.norm_kind, sp.norm_eps)
            self.k_norm = Norm(t["attn_k_norm.weight"][0], sp.norm_kind, sp.norm_eps)
        self.gate = None                                           # attention output gate: sigmoid(W x [+ bias per head]) * attention output, before o_proj
        if "attn_gate.weight" in t:
            gout = t["attn_gate.weight"][1]
            if gout not in (self.nh, self.nh * hv):
                raise SkeletonError(f"layer {il}: attn_gate has {gout} outputs for {self.nh} heads of {hv}")
            self.gate = "head" if gout == self.nh else "element"
            self.gate_proj = nn.Linear(d, gout, bias=False)
            if "attn_gate.bias" in t:
                if list(t["attn_gate.bias"]) != [self.nh]:
                    raise SkeletonError(f"layer {il}: attn_gate.bias must have one value per head")
                self.gate_bias = mx.zeros((self.nh,))
        self.dims = sp.rope_dims
        self.swa = bool(sp.swa_layers[il]) if sp.swa_layers else False
        self.rope_on = bool(sp.rope_layers[il]) if sp.rope_layers else True
        self.base = sp.rope_base_swa if (self.swa and sp.rope_base_swa) else sp.rope_base

    def _rope(self, x, offset, freqs):
        sp = self.sp
        if sp.rope_kind == "none" or not self.rope_on:
            return x
        return mx.fast.rope(x, self.dims, traditional=sp.rope_kind == "norm", base=None if freqs is not None else self.base, scale=sp.rope_scale,
                            offset=offset, freqs=freqs)

    def _temp(self, q, off):
        """NoPE layers' query scale (llama-graph.cpp:318-323, attn_temp_pow): (1 + (pos + 1) / length) ** scale with the ABSOLUTE position (cache offset + index)."""
        sp = self.sp
        pos = mx.arange(q.shape[2], dtype=mx.float32)
        pos = pos + off if not isinstance(off, mx.array) or off.ndim == 0 else pos[None, :] + off[:, None].astype(mx.float32)
        f = mx.exp(sp.attn_temp_scale * mx.log1p((pos + 1.0) / sp.attn_temp_length))
        f = f[None, None, :, None] if f.ndim == 1 else f[:, None, :, None]
        return (q.astype(mx.float32) * f).astype(q.dtype)

    def __call__(self, x, mask, cache, freqs):
        from mlx_lm.models.base import scaled_dot_product_attention
        B, L, _ = x.shape
        sp, hd, hv = self.sp, self.hd, self.hv
        if hasattr(self, "qkv_proj"):
            q, k, v = mx.split(self.qkv_proj(x), [self.nh * hd, (self.nh + self.nkv) * hd], axis=-1)
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if sp.clamp_qkv:
            q, k, v = (mx.clip(a, -sp.clamp_qkv, sp.clamp_qkv) for a in (q, k, v))
        if self.qk_norm == "full":
            q, k = self.q_norm(q), self.k_norm(k)
        q = q.reshape(B, L, self.nh, hd).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.nkv, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.nkv, hv).transpose(0, 2, 1, 3)
        if self.qk_norm == "head" and not sp.qk_norm_after_rope:
            q, k = self.q_norm(q), self.k_norm(k)
        off = cache.offset if cache is not None else 0
        q, k = self._rope(q, off, freqs), self._rope(k, off, freqs)
        if sp.attn_temp_scale and not self.rope_on:
            q = self._temp(q, off)
        if self.qk_norm == "head" and sp.qk_norm_after_rope:
            q, k = self.q_norm(q), self.k_norm(k)
        keep = None
        if self.idx:                                                  # the indexer keys ride in the cache as one extra kv head (value: zeros), so any mlx-lm cache holds them
            Hd, dx = sp.indexer_heads, sp.indexer_dim
            iq = self.idx_q_norm(self.idx_q_proj(x).reshape(B, L, Hd, dx)).transpose(0, 2, 1, 3)
            ik = self.idx_k_norm(self.idx_k_proj(x).reshape(B, L, 1, dx)).transpose(0, 2, 1, 3)
            iq, ik = self._rope(iq, off, freqs), self._rope(ik, off, freqs)
            k, v = mx.concatenate([k, ik], axis=1), mx.concatenate([v, mx.zeros((B, 1, L, hv), dtype=v.dtype)], axis=1)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        if self.idx:
            ik, k, v = k[:, self.nkv], k[:, :self.nkv], v[:, :self.nkv]
            keep = self._block_keep(iq, ik, off)
        if self.sinks is not None or keep is not None:
            o = self._attend_plain(q, k, v, mask, keep)
        else:
            o = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        o = o.transpose(0, 2, 1, 3)                                   # [B, L, heads, hv]
        if sp.attn_out_norm:                                          # muse-spark.cpp:129-133
            o = mx.fast.rms_norm(o, mx.ones((hv,), dtype=o.dtype), sp.norm_eps)
        if self.gate:                                                 # beam.cpp:174-183 (per element), step35.cpp:253-269 (per head)
            gt = self.gate_proj(x)
            if self.gate == "element":
                gt = gt.reshape(B, L, self.nh, hv)
                if "gate_bias" in self:
                    gt = gt + self.gate_bias[None, None, :, None]
            else:
                gt = gt[..., None]
                if "gate_bias" in self:
                    gt = gt + self.gate_bias[None, None, :, None]
            o = o * mx.sigmoid(gt)
        out = self.o_proj(o.reshape(B, L, -1))
        return out * sp.attn_value_scale if sp.attn_value_scale else out    # mimo2.cpp:163: the scale is applied to build_attn's output (after wo)

    def _block_keep(self, iq, ik, off):
        """Block-sparse selection (minimax-m3.cpp:339-532): per kv group, score each past position by iq . ik (unscaled, f32, causal), take the max per block of
        `indexer_block` positions, force the `indexer_local` newest blocks (+1e30), keep the `indexer_top_k` best blocks. -> bool [B, n_head, L, S]: key visible."""
        from .foreign_skeleton_kpool import row_pad
        sp = self.sp
        B, Hd, L, _ = iq.shape
        S, blk = ik.shape[1], sp.indexer_block
        host_off = off if isinstance(off, int) else None             # a python offset: the padding is known on the host, no device read
        pad, off = row_pad(off, L, S, B)                             # per-row offsets: row b's slot s holds position s - pad[b] (right-aligned batch cache)
        moved = B > 1 or (S - L - host_off != 0 if host_off is not None else True)   # array offsets: the reorder runs on the device (a no-op gather when pad is 0, bit-equal), never a host sync
        if moved:
            ik = mx.take_along_axis(ik, mx.clip(mx.arange(S)[None] + pad[:, None], 0, S - 1)[..., None], axis=1)   # position order
        sc = iq.astype(mx.float32) @ ik.astype(mx.float32)[:, None].swapaxes(-1, -2)         # [B, Hd, L, S] over positions
        qpos = (off[:, None] + mx.arange(L)[None])[:, None, :, None]                         # [B, 1, L, 1]
        sc = mx.where(mx.arange(S) <= qpos, sc, -mx.inf)
        nblk = -(-S // blk)
        sc = mx.pad(sc, [(0, 0), (0, 0), (0, 0), (0, nblk * blk - S)], constant_values=-mx.inf)
        bs = sc.reshape(B, Hd, L, nblk, blk).max(-1)
        qb, b = qpos // blk, mx.arange(nblk)
        bs = bs + mx.where((b <= qb) & (b > qb - sp.indexer_local), 1e30, 0.0)
        top = mx.argsort(-bs, axis=-1)[..., :min(sp.indexer_top_k, nblk)]
        sel = mx.put_along_axis(mx.zeros(bs.shape, dtype=mx.bool_), top, mx.ones(top.shape, dtype=mx.bool_), axis=-1)
        keep = mx.take(sel, mx.arange(S) // blk, axis=-1)                                      # [B, Hd, L, S] over positions
        if moved:                                                                             # back to slots; padding slots hidden
            kpos = mx.arange(S)[None] - pad[:, None]
            idx = mx.broadcast_to(mx.clip(kpos, 0, S - 1)[:, None, None, :], keep.shape)
            keep = mx.logical_and(mx.take_along_axis(keep, idx, axis=-1), (kpos >= 0)[:, None, None, :])
        return mx.repeat(keep, self.nh // Hd, axis=1)

    def _attend_plain(self, q, k, v, mask, keep=None):
        """Plain-op attention: the mask, an optional bool `keep` [B, nh, L, S] (block-sparse), and optional sinks: softmax over [scores, sink] (the sink is an unscaled
        logit with a zero value), the sink column dropped (llama-graph.cpp:3120-3130 / ggml_soft_max_add_sinks). q [B, nh, L, hd], k [B, nkv, S, hd], v [B, nkv, S, hv]."""
        from mlx_lm.models.base import create_causal_mask
        B, nh, L, _ = q.shape
        S = k.shape[2]
        if self.nh != self.nkv:
            k, v = mx.repeat(k, self.nh // self.nkv, axis=1), mx.repeat(v, self.nh // self.nkv, axis=1)
        sc = (q @ k.swapaxes(-1, -2)).astype(mx.float32) * self.scale
        if isinstance(mask, str):
            mask = create_causal_mask(L, S - L) if mask == "causal" else None
        if mask is not None:
            sc = sc + (mx.where(mask, 0.0, -mx.inf) if mask.dtype == mx.bool_ else mask.astype(mx.float32))
        if keep is not None:
            sc = mx.where(keep, sc, -mx.inf)
        if self.sinks is not None:
            sc = mx.concatenate([sc, mx.broadcast_to(self.sinks.astype(mx.float32)[None, :, None, None], (B, nh, L, 1))], axis=-1)
        p = mx.softmax(sc, axis=-1, precise=True)[..., :S]
        return p.astype(v.dtype) @ v


class DenseFFN(nn.Module):
    """The default FFN: gated (act(gate x) * up x) or not (act(up x)), then down."""

    def __init__(self, sp: ModelArgs, il: int, tensors: dict):
        super().__init__()
        d, f = sp.hidden_size, sp.intermediate_size[il]
        self.gated = "ffn_gate.weight" in tensors
        if self.gated:
            self.gate_proj = _lin(tensors, "ffn_gate", d, f)
        self.up_proj = _lin(tensors, "ffn_up", d, f)
        self.down_proj = _lin(tensors, "ffn_down", f, d)
        if sp.act in GLU_ACTS:
            if not self.gated:
                raise SkeletonError(f"act {sp.act!r} needs a gated FFN (layer {il} has no ffn_gate)")
            self.act = None
            self.glu = GLU_ACTS[sp.act]
        else:
            self.act, self.glu = ACTS[sp.act], None

    def __call__(self, x):
        if self.glu is not None:
            return self.down_proj(self.glu(self.gate_proj(x), self.up_proj(x)))
        h = self.act(self.gate_proj(x)) * self.up_proj(x) if self.gated else self.act(self.up_proj(x))
        return self.down_proj(h)


def dense_ffn_factory(il, spec, tensors):
    """ffn_factory(layer_index, spec, tensors) -> module x -> y. `tensors` = this layer's {"kind.weight": ne}. A MoE block plugs in by replacing this."""
    return DenseFFN(spec, il, tensors)


def _mla_class():
    from .foreign_skeleton_mla import MlaAttention

    class SkeletonMla(MlaAttention):
        """N1 (+ N3 behind it) with the skeleton's attention call contract (x, mask, cache, freqs) and its `swa` flag."""

        def __call__(self, x, mask, cache, freqs=None):
            return MlaAttention.__call__(self, x, mask=mask, cache=cache)
    return SkeletonMla


def make_attention(sp: ModelArgs, il: int):
    """The layer's attention: N1 MlaAttention when foreign_skeleton_mla.fires on its tensors (N3 behind it when foreign_skeleton_kpool.fires), else Attention."""
    from . import foreign_skeleton_mla as MLA, foreign_skeleton_kpool as KP
    t = sp.layer_tensors[il]
    if not MLA.fires(t):
        return Attention(sp, il)
    nh = sp.num_attention_heads[il]
    swa = bool(sp.swa_layers[il]) if sp.swa_layers else False
    base = sp.rope_base_swa if (swa and sp.rope_base_swa) else sp.rope_base
    if "attn_gate.weight" in t and t["attn_gate.weight"][1] != nh * sp.mla_v:
        raise SkeletonError(f"layer {il}: MLA takes the element gate only ({nh * sp.mla_v} outputs), attn_gate has {t['attn_gate.weight'][1]}")
    if sp.rope_kind not in ("norm", "none"):
        raise SkeletonError(f"layer {il}: MLA ropes consecutive pairs (rope 'norm'), the row says {sp.rope_kind!r}")
    kp = None
    if KP.fires(t):
        kp = dict(H=sp.indexer_heads, dx=sp.indexer_dim, kp=sp.kpool, top_k=sp.indexer_top_k, select_tail=sp.kpool_select_tail, n_rot=sp.rope_dims, base=base)
    m = _mla_class()(sp.hidden_size, nh, sp.mla_q_lora, sp.mla_kv_lora, sp.mla_nope, sp.rope_dims if sp.rope_kind == "norm" else 0, sp.mla_v,
                     eps=sp.norm_eps, base=base, gate="attn_gate.weight" in t, sinks="attn_sinks.weight" in t, kpool=kp)
    m.swa = swa
    m.window = sp.swa_window if swa else 0
    return m


class HcParams(nn.Module):
    """One sub-layer's hyper-connection weights (fn [rows, hc * D], base [rows], scale [2 or 3]) as plain arrays, loaded by name."""

    def __init__(self, ne_fn, ne_base, ne_scale):
        super().__init__()
        self.fn, self.base, self.scale = (mx.zeros(tuple(int(x) for x in reversed(ne))) for ne in (ne_fn, ne_base, ne_scale))

    def stream(self, sp: ModelArgs):
        from .foreign_skeleton_hc import HcStream
        return HcStream(self.fn, self.base, self.scale, sp.hc_count, sp.hc_eps, sp.hc_magnitude, sp.norm_eps, sp.hc_iters, sp.hc_post_eps)


class EngramParams(nn.Module):
    """N4 Engram of one layer (Hy4 form), the projections as plain arrays; the table rows come from Model.engram_source."""

    def __init__(self, t):
        super().__init__()
        for kind in ("engram_kv", "engram_kv_b", "engram_gate", "engram_hc_scale", "engram_lookup_norm", "engram_pre_gamma"):
            setattr(self, kind[len("engram_"):], mx.zeros(tuple(int(x) for x in reversed(t[kind + ".weight"]))))

    _KINDS = ("gate", "kv", "kv_b", "lookup_norm", "pre_gamma", "hc_scale")

    def _dev(self):
        """The projections as f32 mx arrays, converted once; rebuilt when any parameter object was replaced (load_weights, update)."""
        src = tuple(getattr(self, k) for k in self._KINDS)
        c = self.__dict__.get("_devc")
        if c is None or any(a is not b for a, b in zip(c[0], src)):
            w = {k: a.astype(mx.float32) for k, a in zip(self._KINDS, src)}
            mx.eval(list(w.values()))
            c = self.__dict__["_devc"] = (src, w)
        return c[1]

    def __call__(self, h, rows, valid, eps):
        """h [1, L, hc, D] stream, rows [L, n_cols, head_dim], valid [L] -> the stream after the Engram write (engram_gguf.engram_forward)."""
        from .engram_gguf import engram_forward
        if h.shape[0] != 1:
            raise SkeletonError("the skeleton's Engram source is one sequence (batch 1)")
        from . import engram_gguf as EG
        if EG.DEVICE:                                                      # KYOJIN_ENGRAM_DEVICE: f32 device copies, made once per parameter object
            w = self._dev()
            _, out = engram_forward(h[0], rows, w["gate"], w["kv"], eps, act_fp8=False, kv_b=w["kv_b"],
                                    lookup_norm=w["lookup_norm"].reshape(rows.shape[1], rows.shape[2]), pre_gamma=w["pre_gamma"],
                                    hc_scale=w["hc_scale"], valid=valid)
            return out[None]
        f32 = lambda a: np.array(a.astype(mx.float32))                     # noqa: E731
        _, out = engram_forward(h[0], rows, f32(self.gate), f32(self.kv), eps, act_fp8=False, kv_b=f32(self.kv_b),
                                lookup_norm=f32(self.lookup_norm).reshape(rows.shape[1], rows.shape[2]), pre_gamma=f32(self.pre_gamma),
                                hc_scale=f32(self.hc_scale), valid=valid)
        return out[None]


from . import ablit as _AB  # noqa: E402  (decensor hook; unarmed blocks never call it)


class Block(nn.Module):
    def __init__(self, sp: ModelArgs, il: int, ffn_factory):
        super().__init__()
        t, d = sp.layer_tensors[il], sp.hidden_size
        self.sp, self.il = sp, il
        if sp.residual_kind != "plain" and (sp.parallel_residual or sp.norm_pos == "post" or sp.residual_scale != 1.0):
            raise SkeletonError("residual forms other than plain need the sequential pre / sandwich block without residual_scale")
        if sp.residual_kind == "vector":
            self.attn_resid_alpha, self.attn_resid_beta, self.ffn_resid_alpha, self.ffn_resid_beta = (Vec(d) for _ in range(4))
        self.self_attn = make_attention(sp, il)
        self.mlp = ffn_factory(il, sp, t)
        if sp.hc_count:
            if sp.parallel_residual or sp.norm_pos == "post" or sp.residual_kind != "plain" or sp.residual_scale != 1.0:
                raise SkeletonError("hyper-connection streams take the sequential pre / sandwich block with the plain residual")
            self.hc_attn = HcParams(*(t[f"hc_attn_{p}.weight"] for p in ("fn", "base", "scale")))
            self.hc_ffn = HcParams(*(t[f"hc_ffn_{p}.weight"] for p in ("fn", "base", "scale")))
        if il in sp.engram_layers:
            self.engram = EngramParams(t)
        self.input_layernorm = _norm(sp, d, t, "attn_norm")
        self.shared_norm = sp.parallel_residual and "ffn_norm.weight" not in t and "attn_norm.weight" in t
        if not self.shared_norm:
            self.post_attention_layernorm = _norm(sp, d, t, "ffn_norm")
        if sp.norm_pos == "sandwich":
            self.post_self_attn_layernorm = _norm(sp, d, t, "post_attention_norm", sp.post_norm_eps)
            self.post_mlp_layernorm = _norm(sp, d, t, "post_ffw_norm", sp.post_norm_eps, sp.ffn_post_mul_first)

    def __call__(self, x, mask, cache, freqs, engram_in=None):
        sp, r = self.sp, self.sp.residual_scale
        if sp.hc_count:
            return self._hc_call(x, mask, cache, freqs, engram_in)
        if sp.norm_pos == "post":                                  # norm after the residual add
            h = self.input_layernorm(x + r * self.self_attn(x, mask, cache, freqs))
            return self.post_attention_layernorm(h + r * self.mlp(h))
        a = self.self_attn(self.input_layernorm(x), mask, cache, freqs)
        if sp.norm_pos == "sandwich":
            a = self.post_self_attn_layernorm(a)
        if sp.parallel_residual:                                   # one residual: x + attn(n1 x) + ffn(n2 x)
            f = self.mlp(self.input_layernorm(x) if self.shared_norm else self.post_attention_layernorm(x))
            return x + r * a + r * f
        h = self._add(x, a, "attn", r)
        f = self.mlp(self.post_attention_layernorm(h))
        if sp.norm_pos == "sandwich":
            f = self.post_mlp_layernorm(f)
        return self._add(h, f, "ffn", r)

    def _hc_call(self, h, mask, cache, freqs, engram_in):
        """N2 (+ N4): h [B, L, hc, D]. Engram writes into the streams first, then each sub-layer reads its pre-mix and writes back through its post
        (llama.cpp hy-v4-flash.cpp:1221-1257): attention = post_attention_norm(attn(attn_norm(x))), ffn = post_ffw_norm(ffn(ffn_norm(x)))."""
        sp = self.sp
        if "engram" in self and engram_in is not False:               # False: the caller already wrote the Engram into the streams
            if engram_in is None:
                raise SkeletonError(f"layer {self.il}: an Engram layer needs Model.engram_source (the table rows)")
            h = self.engram(h, *engram_in, sp.engram_eps)
        sand = sp.norm_pos == "sandwich"
        sa = self.hc_attn.stream(sp)
        ab = self.__dict__.get("_ablit")                           # decensor hook (ablit.attach_skeleton): armed block = the unfused path, the edit sits after the post norms
        if ab is None and sa.fusable(h):                                          # KYOJIN_HC_ID_FUSE: the identity-form seams fused, norms folded (foreign_skeleton_hc.fused)
            h = sa.fused(h, lambda x: self.self_attn(x, mask, cache, freqs), self.input_layernorm, self.post_self_attn_layernorm if sand else None)
            return self.hc_ffn.stream(sp).fused(h, self.mlp, self.post_attention_layernorm, self.post_mlp_layernorm if sand else None)

        def attn(x):
            a = self.self_attn(self.input_layernorm(x), mask, cache, freqs)
            a = self.post_self_attn_layernorm(a) if sand else a
            return _AB.attn_out(self, a) if ab is not None else a

        def ffn(x):
            f = self.mlp(self.post_attention_layernorm(x))
            f = self.post_mlp_layernorm(f) if sand else f
            return _AB.ffn_out(self, f) if ab is not None else f
        h = self.hc_attn.stream(sp)(h, attn)
        return self.hc_ffn.stream(sp)(h, ffn)

    def _add(self, x, branch, kind, r):
        """The residual join of one sub-layer (beam.cpp:109-113 / muse-spark.cpp:153-156 / plain)."""
        sp = self.sp
        if sp.residual_kind == "depth":
            i = 2 * (self.il + 1) + (kind == "ffn")
            return branch * (1.0 / math.sqrt(i)) + x * math.sqrt((i - 1) / i)
        if sp.residual_kind == "vector":
            al, be = getattr(self, kind + "_resid_alpha").weight, getattr(self, kind + "_resid_beta").weight
            return x * al + branch * be
        return x + r * branch


class Inner(nn.Module):
    def __init__(self, sp: ModelArgs, ffn_factory):
        super().__init__()
        d = sp.hidden_size
        self.embed_tokens = nn.Embedding(sp.vocab_size, d)
        if sp.pos_learned:
            self.embed_positions = nn.Embedding(sp.max_position, d)
        if "token_embd_norm.weight" in sp.top_tensors or "token_embd_norm.bias" in sp.top_tensors:
            self.embed_norm = _norm(sp, d, sp.top_tensors, "token_embd_norm")
        self.layers = [Block(sp, i, ffn_factory) for i in range(sp.num_hidden_layers)]
        if sp.hc_count:
            t = sp.top_tensors
            self.hc_head = HcParams(*(t[f"output_hc_{p}.weight"] for p in ("fn", "base", "scale")))
        if sp.final_norm:
            self.norm = _norm(sp, d, sp.top_tensors, "output_norm")
        if sp.rope_factors:
            self.rope_factors = mx.ones((sp.rope_dims // 2,))


def _host_offset(c) -> int:
    """The cache position as a python int WITHOUT a device read: python offsets as they are; array offsets (batch caches) from the cache's own host-side write
    index `_idx` (one row: position == write index). Only a cache that holds neither falls back to one device read."""
    off = c.offset if c is not None else 0
    if not isinstance(off, mx.array):
        return int(off)
    idx = getattr(c, "_idx", None)
    if isinstance(idx, int) and off.size == 1:                       # one row has no left padding: position == the cache's host-side write index
        return idx
    return int(off.reshape(-1)[0].item())


class Model(nn.Module):
    def __init__(self, args: ModelArgs, ffn_factory=None):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Inner(args, ffn_factory or dense_ffn_factory)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        if args.hc_count and _AB.spec_path():                       # KYOJIN_ABLIT_SPEC: Hy4 decensor hook on the sub-layer outputs (off = untouched)
            _AB.attach_skeleton(self)

    @property
    def layers(self):
        return self.model.layers

    def _freqs(self, swa=False):
        sp = self.args
        if not sp.rope_factors or (swa and not sp.rope_factors_swa):
            return None
        base = sp.rope_base_swa if (swa and sp.rope_base_swa) else sp.rope_base
        return self.model.rope_factors * base ** (mx.arange(0, sp.rope_dims, 2, dtype=mx.float32) / sp.rope_dims)

    def _window_mask(self, L, c):
        """Mask of the windowed layers: visible iff 0 <= q - k < window (llama_hparams::is_masked_swa, llama-hparams.h:557). The cache keeps every key, the
        window is only in the mask."""
        from mlx_lm.models.base import create_causal_mask
        w = self.args.swa_window
        if c is not None and hasattr(c, "make_mask"):
            return c.make_mask(L, return_array=True, window_size=w)
        return create_causal_mask(L, 0 if c is None else c.offset, window_size=w)

    def __call__(self, inputs, cache=None, input_embeddings=None, hidden_sink=None):
        from mlx_lm.models.base import create_attention_mask
        sp, m = self.args, self.model
        if _ABL.ACTIVE:                                              # KYOJIN_ABLATE probe (timing only): structural stand-ins, see ablate_struct.py
            from . import ablate_struct
            ablate_struct.prepare(self, inputs)
        h = m.embed_tokens(inputs) if input_embeddings is None else input_embeddings
        if sp.embed_scale != 1.0:
            h = h * sp.embed_scale
        cache = cache if cache is not None else [None] * len(m.layers)
        if sp.pos_learned:
            off = cache[0].offset if cache[0] is not None else 0
            pos = mx.arange(inputs.shape[1])
            pos = pos + off if not isinstance(off, mx.array) or off.ndim == 0 else pos[None, :] + off[:, None]
            h = h + m.embed_positions(pos)
        if hasattr(m, "embed_norm"):
            h = m.embed_norm(h)
        elif sp.embed_rms:
            h = mx.fast.rms_norm(h, mx.ones((h.shape[-1],), dtype=h.dtype), sp.norm_eps)
        _noise = os.environ.get("KYOJIN_EMBED_NOISE", "")
        if _noise:                                                  # null arm of a quality gate (Hy4 is chaotic at 1e-6): "rel[,seed]" = the c120 N_EMB arm, a relative uniform(-1, 1) perturbation of the embedding output
            rel, _, seed = _noise.partition(",")
            h = (h.astype(mx.float32) * (1.0 + float(rel) * mx.random.uniform(-1.0, 1.0, h.shape, key=mx.random.key(int(seed or 1))))).astype(h.dtype)
        from . import attn_fp16 as AF
        if AF.ON or AF.pf_on():
            AF.prepare(self)                                        # flags attn_fp16 / KYOJIN_HY4_PF_FP16: fp16 absorb weights, once, eager
        fixed = cache[0] is not None and getattr(cache[0], "fixed_kv", False)
        if fixed:                                                   # fixed-state caches: the buffers grow here (eager, host), the masks are built in the attention from the offset array
            for c in cache:
                c.ensure(c._n + inputs.shape[1], AF.kv_dtype(h.dtype))
        mask = None if fixed else create_attention_mask(h, cache[0])
        wmask = self._window_mask(h.shape[1], cache[0]) if sp.swa_window and not fixed else None
        freqs, freqs_w = self._freqs(), (self._freqs(True) if sp.swa_window else None)
        eng = {}
        if sp.engram_layers:
            src = getattr(self, "engram_source", None)
            if src is None:
                raise SkeletonError("Engram layers need model.engram_source = callable(token ids [L], offset) -> {layer: (rows [L, n_cols, hd], valid [L])}")
            if inputs.shape[0] != 1:
                raise SkeletonError("the skeleton's Engram source is one sequence (batch 1)")
            eng = src(np.array(inputs[0]), _host_offset(cache[0]))
        if sp.hc_count:
            from .foreign_skeleton_hc import embed
            h = embed(h, sp.hc_count)
        _sa, _hi, _lo = _split_cfg()
        split_at = _sa if _lo <= inputs.shape[1] <= _hi else ()
        for il, (layer, c) in enumerate(zip(m.layers, cache)):
            sw = layer.self_attn.swa
            if sp.hc_count:
                if getattr(eng, "pending", None) is not None and eng.pending(il):    # prompt chunk, this layer's table rows still being read: the layers built so far run on the GPU meanwhile
                    mx.async_eval(h)
                eg = eng.get(il)
                if eg is not None and "engram" in layer:            # the Engram write is host numpy (table rows): it runs here, eagerly, so the layer itself is a pure array program (step_compile)
                    h, eg = layer.engram(h, *eg, sp.engram_eps), False
                h = layer(h, wmask if sw else mask, c, freqs_w if sw else freqs, eg)
            else:
                h = layer(h, wmask if sw else mask, c, freqs_w if sw else freqs)
            if il in split_at:                                  # KYOJIN_SPLIT_EVAL: submit the layers built so far, so the GPU runs them while the host builds the rest
                mx.async_eval(h)
        if fixed:
            for c in cache:
                c._n += inputs.shape[1]                             # host mirror of the device write offset (the layers only touch cache.cache)
        if sp.hc_count:
            from .foreign_skeleton_hc import head
            hh = m.hc_head
            h = head(h, hh.fn, hh.base, hh.scale, sp.hc_eps, sp.norm_eps)
        if sp.final_norm:
            h = m.norm(h)
        if hidden_sink is not None:                                   # the state the output head reads (an MTP draft head takes it as its `h` input)
            hidden_sink.append(h)
        from . import headlast
        if headlast.active() and h.shape[1] > headlast.LAST_ROW_MIN:       # a caller that reads the last row only (serve prefill): [B, 1, vocab], not [B, rows, vocab]
            h = h[:, -1:]
        out = m.embed_tokens.as_linear(h) if sp.tie_word_embeddings else (HQ.project(self.lm_head, h) if (HQ.BITS or HQ.FMM8) else None)
        if out is None:
            out = self.lm_head(h)
        out = out * sp.logit_scale if sp.logit_scale != 1.0 else out
        if sp.logit_softcap:
            c = sp.logit_softcap
            out = mx.tanh(out / c) * c
        return out

    def sanitize(self, weights):
        return weights


def param_name(gguf_name: str, extra: dict | None = None, kpool_layers=None) -> str | None:
    """GGUF tensor name -> parameter path of `Model` (None: not a skeleton tensor). `extra` adds block kinds of a plugged-in FFN (kind -> module path).
    `kpool_layers`: layer ids where N3 fires (there `indexer.k_norm` is the kpool index key norm, not the block-sparse indexer's)."""
    from .foreign_skeleton_kpool import KINDS as KPK
    if gguf_name in TOP_NAMES:
        return TOP_NAMES[gguf_name]
    if gguf_name in N_TOP:
        return N_TOP[gguf_name]
    m = re.match(r"blk\.(\d+)\.(.+)\.(weight|bias|scale)$", gguf_name)
    if m and m.group(2) == "attn_sinks" and m.group(3) == "weight":
        return f"model.layers.{m.group(1)}.self_attn.sinks"
    if m and m.group(2) == "attn_gate" and m.group(3) == "bias":      # a per-head bias vector, not the bias of the projection
        return f"model.layers.{m.group(1)}.self_attn.gate_bias"
    if m:
        il, kind, suf = m.groups()
        kp = kind in KPK and (kind not in BLK_NAMES or (kpool_layers is not None and int(il) in kpool_layers))
        path = KPK[kind] if kp else ({**BLK_NAMES, **N_NAMES, **(extra or {})}.get(kind))
        if path:
            if kind in N_RAW and suf == "weight":
                return f"model.layers.{il}.{path}"
            return f"model.layers.{il}.{path}.{suf}"
    return None


def install_module():
    """Make `mlx_lm.models.kyojin_skeleton` importable (mlx_lm.utils._get_classes imports the model class by model_type)."""
    mod = types.ModuleType("mlx_lm.models.kyojin_skeleton")
    mod.Model, mod.ModelArgs = Model, ModelArgs
    sys.modules["mlx_lm.models.kyojin_skeleton"] = mod
    return mod


def load_cpu(path, dtype=mx.float32, ffn_factory=None, extra_names=None, row=None, allow_moe=False):
    """GGUF -> (model, report) on the dense CPU tier: every tensor dequantised to `dtype`, strict load_weights (a parameter without a tensor, or a tensor without
    a parameter, raises)."""
    from . import foreign_gguf as GF
    g = GF.GgufFile(GF.find_gguf(path))
    keep_open = False
    try:
        arch = g.kv["general.architecture"]
        ne = {k: ti.ne for k, ti in g.tensors.items()}
        if ffn_factory is None:                                    # the file says MoE (expert count + router tensors): the generic MoE block through the hook
            from . import foreign_skeleton_moe as SM
            r = ROWS.get(arch) if row is None else row
            moe = SM.detect(arch, g.kv, ne, r) if r is not None else None
            if moe is not None:
                ffn_factory, extra_names, allow_moe = moe.factory, {**(extra_names or {}), **moe.names}, True
        sp = build_spec(arch, g.kv, ne, row=row, allow_moe=allow_moe or ffn_factory is not None)
        model = Model(sp, ffn_factory)
        from .foreign_skeleton_kpool import fires as kp_fires
        kpl = {il for il, t in enumerate(sp.layer_tensors) if kp_fires(t)}
        w, skipped = [], []
        nextn = int(g.kv.get(arch + ".nextn_predict_layers", 0) or 0)
        for name in g.tensors:
            mm = re.match(r"blk\.(\d+)\.", name)
            if nextn and mm and int(mm.group(1)) >= sp.num_hidden_layers:     # MTP blocks sit after the decoded layers: not a skeleton tensor, not an error
                continue
            if name == ENGRAM_TABLE and sp.engram_layers:
                continue
            pn = param_name(name, extra_names, kpl)
            if pn is None:
                skipped.append(name)
                continue
            w.append((pn, mx.array(g.tensor_f32(name).astype(np.float32)).astype(dtype)))
        if skipped:
            raise SkeletonError(f"{arch}: tensors with no skeleton parameter: {skipped[:8]}")
        model.load_weights(w, strict=True)
        mx.eval(model.parameters())
        if sp.engram_layers:                                       # the table stays in the file, read by row: the file stays open with the model
            model.engram_source, keep_open = engram_source_of(g), True
        return model, dict(arch=arch, spec=sp, n_tensors=len(w), model_type=sp.model_type, config="skeleton rows")
    finally:
        if not keep_open:
            g.close()


def engram_source_of(g):
    """GgufFile with an Engram table -> callable(token ids [L], offset) -> {layer: (rows [L, n_cols, hd] f32, valid [L])}: engram_gguf's row hash
    (the history restarts when offset is 0) and its row reader."""
    from . import engram_gguf as EG
    spec = EG.EngramGgufSpec.from_gguf(g)
    rm = EG.GgufRowMap(spec)
    rd = EG.RowReader(g, spec)

    def src(ids, offset):
        if offset == 0:
            rm.reset()
        if len(rm.history) != offset:
            raise SkeletonError(f"Engram history holds {len(rm.history)} positions, the cache {offset}")
        dec = EG.DEC_LOG and EG.PHASE[0] == "dec"                  # KYOJIN_HY4_ENGRAM_DEC_LOG=1: the decode-phase counters (every read path)
        if dec:
            import time
            t0 = time.perf_counter()
        rows, valid = rm.rows_valid(ids)
        if dec and rows.shape[0] >= EG.LAZY_MIN:
            dec = False                                            # a prompt chunk that reached us outside EG.phase("pf")
        if dec:
            uq = np.unique(rows)
            EG.dec_add(calls=1, pos=rows.shape[0], rows=rows.size, unique=uq.size)
        if not EG.PREAD:                                           # KYOJIN_HY4_ENGRAM_PREAD=0: the memory-map gather (serial page-ins on a cold row)
            if dec:
                cold = rd.cold_count(uq)                           # before the gather: the pages it faults in are cached afterwards
                t1 = time.perf_counter()
            out = {lid: (EG.read_rows(g, spec, rows[:, i]), valid) for i, lid in enumerate(spec.layer_ids)}
            if dec:
                EG.dec_add(cold=cold, cold_sync=cold, miss=uq.size, sync_read_s=time.perf_counter() - t1, map_calls=1)
        else:
            out = rd.layers(rows, valid, offset)
        if dec:
            EG.dec_add(src_s=time.perf_counter() - t0)
            EG._DEC_N[0] += 1
            if EG._DEC_N[0] % EG.DEC_EVERY == 0:
                EG.dec_flush("run")
        return out

    def truncate(n):
        """Speculative rollback: the history keeps its first n positions (the rows of rejected drafts leave it)."""
        rm.history = rm.history[:n]

    def hint(ids, a, b):
        """The prompt positions [a, b) are the next prefill chunk: their rows are read in the background (content-addressed pieces: a wrong hint only misses)."""
        if not (EG.PREAD and EG.AHEAD):
            return False
        a, b = int(a), min(int(b), int(a) + EG.AHEAD_MAX, len(ids))
        if b - a < EG.LAZY_MIN:
            return False
        lo = max(0, a - (spec.ngram_size - 1))
        prev, _ = EG.compressed_ids(spec, ids[lo:a])
        cids, _ = EG.compressed_ids(spec, ids[a:b])
        rd.hint(EG.hash_rows(spec, prev, cids, start=a), a)
        return True
    src.truncate, src.history_len = truncate, lambda: len(rm.history)

    def dec_begin(first):
        """A speculative round starts with the pending token `first` (position len(history)): its rows are read in the background (flag KYOJIN_HY4_ENGRAM_DECPF)."""
        if not (EG.PREAD and EG.DECPF):
            return False
        rd.dec_begin(rm.history, len(rm.history))
        rd.dec_push(EG.compressed_ids(spec, [first])[0])
        return True

    def dec_push(toks):
        """The next tokens of the round (drafts) are known: the rows of their positions are read in the background."""
        if not (EG.PREAD and EG.DECPF) or len(toks) == 0:
            return False
        rd.dec_push(EG.compressed_ids(spec, toks)[0])
        return True
    src.hint, src.reader, src.dec_begin, src.dec_push = hint, rd, dec_begin, dec_push
    src.get_history = lambda: rm.history                         # per-session history: the serve backend parks / restores it when the live session changes (arrays are replaced, never edited in place)
    src.set_history = lambda h: setattr(rm, "history", h)
    return src


def _json_safe(v):
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_json_safe(x) for x in (sorted(v) if isinstance(v, set) else v)]
    if isinstance(v, np.generic):
        return v.item()
    return v


def serve_config(g) -> dict:
    """GgufFile (headers only) -> the config.json of the serve directory (foreign_names.gguf_serve_dir) for an architecture the skeleton serves with no kit
    table: the ModelArgs build_spec gives, the MoeSpec when the file is a MoE (detected as load_cpu does), `gguf_architecture`. No weight is read
    and no loader row is registered: the tokenizer files of the directory come from the metadata (foreign_names.write_gguf_tokenizer)."""
    from . import foreign_skeleton_moe as SM
    arch = g.kv["general.architecture"]
    ne = {k: ti.ne for k, ti in g.tensors.items()}
    moe = SM.detect(arch, g.kv, ne, ROWS.get(arch))
    sp = build_spec(arch, g.kv, ne, allow_moe=moe is not None)
    cfg = dataclasses.asdict(sp)
    if moe is not None:
        cfg["moe"] = dataclasses.asdict(moe.spec)
    nextn = int(g.kv.get(arch + ".nextn_predict_layers", 0) or 0)
    ctx = g.kv.get(arch + ".context_length")                        # serve.autoconfig reads the request context from max_position_embeddings
    extra = {"max_position_embeddings": int(ctx)} if ctx else {}
    return _json_safe({**cfg, "gguf_architecture": arch, "nextn_predict_layers": nextn, **extra})


def register(path, row=None):
    """Serve path (same contract as foreign_archkit_serve.register): put the architecture in foreign_names.ARCHS / _CONFIG with the skeleton's names, so
    foreign_linear.load_foreign builds `Model` and swaps its nn.Linear for the stored-format linears. Returns the architecture name."""
    from . import foreign_gguf as GF, foreign_names as N
    g = GF.GgufFile(GF.find_gguf(path))
    try:
        arch = g.kv["general.architecture"]
        if arch in N.ARCHS:
            return arch
        sp = build_spec(arch, g.kv, {k: ti.ne for k, ti in g.tensors.items()}, row=row)
        names = [k for k in g.tensors]
    finally:
        g.close()
    if sp.rope_factors:
        raise SkeletonError(f"{arch}: rope_freqs.weight is served by the CPU tier only (foreign_linear.install_rope_freqs is not wired to the skeleton)")
    install_module()
    top = {k: v for k, v in TOP_NAMES.items() if k in names}
    blk = {k: v for k, v in BLK_NAMES.items()}
    ckey = "skeleton:" + arch
    cfg = dataclasses.asdict(sp)
    N._CONFIG[ckey] = lambda gf, a, _c=cfg: {k: v for k, v in _c.items() if k != "model_type"}
    N.ARCHS[arch] = N.Arch("kyojin_skeleton", (), blk=blk, top=top, dense=(), xform="", config=ckey)
    return arch
