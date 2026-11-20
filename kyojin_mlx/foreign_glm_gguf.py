# SPDX-License-Identifier: Apache-2.0
"""GGUF side of GLM-5.3-Flash (llama.cpp arch `glm5-next`): rows only, no model code.

The file reaches the engine's own model code (mlx-vlm `glm5_next`, the class foreign_linear.load_foreign builds for the safetensors pack) under the
HF names of the pack. Everything llama.cpp's converter did on the way (conversion/glm.py `Glm5NextModel`, lines cited per rule):

  glm.py:525-540  per-expert [I, H] tensors stacked into one [E, I, H] tensor per projection      -> kept stacked: `mlp.experts.<proj>.weight`, read per expert as stored
  glm.py:543-552  kv_b_proj split: k_b = nope rows transposed [H, kv_lora, nope], v_b [H, v, kv_lora] -> exactly mlx-vlm sanitize's embed_q / unembed_out (same layout): rename only
  glm.py:554-561  q/k/v conv1d reshaped (1, C, 1, K); the same bytes as the HF [C, 1, K]            -> no transform (3-D dense tensor, never a linear)
  glm.py:563-565  A_log -> -exp(A_log) (`ssm_a`)                                                    -> POST "a_log" (foreign_xform), log(-x)
  glm.py:567-568  dt_bias renamed `ssm_dt.bias`                                                      -> exact name
  glm.py:570-572  hc_{attn,ffn}_{fn,base,scale} and index_kpool_compress_{ape,gate} get `.weight`   -> exact names without the suffix on the HF side
  glm.py:576-582  hc / indexer compressor / ssm_a / ssm_dt / exp_probs_b forced to F32              -> nothing to undo
No norm shift (TextModel has none for this class), no q / k permute (nope-only MLA, rope dimension 0), no scale folding.
The trunk is `block_count - nextn_predict_layers` blocks; the nextn block (blk.45: eh_proj, enorm, hnorm, shared_head_norm and a full MLA + MoE layer) is skipped
(skip_nextn): the serve path wires no draft head for this class yet."""
from __future__ import annotations

import numpy as np

from . import foreign_gguf as GF
from . import foreign_names as FN
from . import foreign_xform as XF

# GGUF kind (name after `blk.N.` without .weight / .bias) -> HF module under model.layers.N.
BLK = {
    "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm", "attn_output": "self_attn.o_proj",
    # KDA (linear attention) layers: conversion/glm.py names via gguf-py MODEL_ARCH.GLM5_NEXT (constants.py:5037-5100)
    "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
    "ssm_beta": "self_attn.b_proj", "ssm_f_a": "self_attn.f_a_proj", "ssm_f_b": "self_attn.f_b_proj", "ssm_g_a": "self_attn.g_a_proj", "ssm_g_b": "self_attn.g_b_proj",
    "ssm_norm": "self_attn.o_norm", "ssm_conv1d_q": "self_attn.q_conv1d", "ssm_conv1d_k": "self_attn.k_conv1d", "ssm_conv1d_v": "self_attn.v_conv1d",
    # MLA (nope) + DSA layers
    "attn_q_a": "self_attn.q_a_proj", "attn_q_b": "self_attn.q_b_proj", "attn_kv_a_mqa": "self_attn.kv_a_proj_with_mqa",
    "attn_q_a_norm": "self_attn.q_a_layernorm", "attn_kv_a_norm": "self_attn.kv_a_layernorm",
    "attn_k_b": "self_attn.embed_q", "attn_v_b": "self_attn.unembed_out",
    "indexer.attn_k": "self_attn.indexer.wk", "indexer.attn_q_b": "self_attn.indexer.wq_b", "indexer.proj": "self_attn.indexer.weights_proj",
    "indexer.k_norm": "self_attn.indexer.k_norm",
    # dense ffn (first leading_dense_block_count blocks), router, routed experts (stacked), shared expert
    "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj", "ffn_gate_inp": "mlp.gate",
    "ffn_gate_exps": "mlp.experts.gate_proj", "ffn_up_exps": "mlp.experts.up_proj", "ffn_down_exps": "mlp.experts.down_proj",
    "ffn_gate_shexp": "mlp.shared_experts.gate_proj", "ffn_up_shexp": "mlp.shared_experts.up_proj", "ffn_down_shexp": "mlp.shared_experts.down_proj",
}
# tensors whose GGUF name has no `.weight` form on the HF side (or the other way round)
EXACT = {
    "ssm_a": "self_attn.A_log", "ssm_dt.bias": "self_attn.dt_bias", "exp_probs_b.bias": "mlp.gate.e_score_correction_bias",
    "hc_attn_fn.weight": "hc_attn_fn", "hc_attn_base.weight": "hc_attn_base", "hc_attn_scale.weight": "hc_attn_scale",
    "hc_ffn_fn.weight": "hc_ffn_fn", "hc_ffn_base.weight": "hc_ffn_base", "hc_ffn_scale.weight": "hc_ffn_scale",
    "indexer_compressor_ape.weight": "self_attn.indexer.index_kpool_compress_ape", "indexer_compressor_gate.weight": "self_attn.indexer.index_kpool_compress_gate",
}
# never a quantised linear: 3-D convolutions and the MLA absorb matrices (batched GEMV in the model class, small), matched on the HF suffix
DENSE = ("conv1d.weight", "embed_q.weight", "unembed_out.weight")
# the nextn block is not loaded (skip_nextn) unless KYOJIN_FOREIGN_MTP is on; these four tensors are what a draft head needs (the names glm_nextn.NextN.build renames from: its
# regex drops `mtp.layers.0.`, enorm / hnorm / eh_proj stay, `shared_head.norm.` becomes `shared_head_norm.`); the block's attention / MoE follow BLK under `mtp.layers.0.`
NEXTN = ("nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight", "nextn.shared_head_norm.weight")

# mlx-vlm glm5_next.Model.sanitize wants the pack's names: `model.language_model.*` and a top-level `lm_head.weight`
LM = "model.language_model."
TOP = {"token_embd.weight": LM + "embed_tokens.weight", "output_norm.weight": LM + "norm.weight", "output.weight": "lm_head.weight"}
MTP = {"nextn.eh_proj.weight": "mtp.layers.0.eh_proj.weight", "nextn.enorm.weight": "mtp.layers.0.enorm.weight", "nextn.hnorm.weight": "mtp.layers.0.hnorm.weight",
       "nextn.shared_head_norm.weight": "mtp.layers.0.shared_head.norm.weight"}
_ROW = dict(blk=dict(BLK), exact=dict(EXACT), top=dict(TOP), dense=DENSE, xform="glm5next", config="glm5next", layer_prefix=LM + "layers.")
ARCH = FN.Arch("glm5_next", ("Glm5NextForConditionalGeneration",), skip_nextn=True, **_ROW)
ARCH_MTP = FN.Arch("glm5_next", ("Glm5NextForConditionalGeneration",), mtp=dict(MTP), **_ROW)    # head blocks come back as mtp.layers.0.<hf name> (load_foreign -> take_mtp)


def foreign_mtp() -> bool:
    """The existing foreign-MTP switch (KYOJIN_FOREIGN_MTP, default off): the nextn block is read from the GGUF; off = skipped as before."""
    import os
    return os.environ.get("KYOJIN_FOREIGN_MTP", "0") not in ("0", "")


def xform(name: str, kv: dict) -> XF.Xf | None:
    """Only ssm_a carries a value transform (glm.py:563-565)."""
    return XF.Xf(post="a_log") if name.startswith("blk.") and name.endswith(".ssm_a") else None


def _swiglu(a, key: str) -> float:
    v = a(key)
    if v is None:
        return 10.0                                     # mlx-vlm's own default (config.py swiglu_limit); the converter writes the key only when the HF config has the limit
    vs = {float(x) for x in (v if isinstance(v, list) else [v])}
    if len(vs) != 1:
        raise GF.GgufError(f"GGUF glm5-next: per-layer {key} {sorted(vs)} (one limit is mapped)")
    return vs.pop()


def config(g: GF.GgufFile, arch: str) -> dict:
    """mlx-vlm glm5_next config (text_config + top level) from the metadata. Refused: another gating function than sigmoid (llama.cpp enum 2), expert groups,
    rope. UNVERIFIED against a second file: only GLM-5.3-Flash was read."""
    a = FN._kv_getter(g.kv, arch)
    if int(a("expert_gating_func", 2)) != 2 or int(a("expert_group_count", 1) or 1) != 1:
        raise GF.GgufError(f"GGUF {arch}: expert_gating_func {a('expert_gating_func')} / expert_group_count {a('expert_group_count')}: only sigmoid without groups is mapped")
    if int(a("rope.dimension_count", 0)) != 0:
        raise GF.GgufError(f"GGUF {arch}: rope.dimension_count != 0 (the class is nope-only MLA)")
    nx = int(a("nextn_predict_layers", 0) or 0)
    n = int(a("block_count")) - nx
    hkv = list(a("attention.head_count_kv"))
    if len(hkv) != n + nx or any(x not in (0, 1) for x in hkv):
        raise GF.GgufError(f"GGUF {arch}: head_count_kv is not a 0 (KDA) / 1 (MLA) list of block_count entries")
    lt = ["linear_attention" if x == 0 else "deepseek_sparse_attention" for x in hkv[:n]]
    nh = int(a("attention.head_count"))
    vocab = int(a("vocab_size", 0) or g.tensors["token_embd.weight"].ne[1])
    tc = dict(
        model_type="glm5_next_text", vocab_size=vocab, hidden_size=int(a("embedding_length")), intermediate_size=int(a("feed_forward_length")),
        moe_intermediate_size=int(a("expert_feed_forward_length")), num_hidden_layers=n, num_nextn_predict_layers=nx, num_attention_heads=nh, num_key_value_heads=nh,
        n_shared_experts=int(a("expert_shared_count", 0) or 0), n_routed_experts=int(a("expert_count")), routed_scaling_factor=float(a("expert_weights_scale", 1.0)),
        kv_lora_rank=int(a("attention.kv_lora_rank")), q_lora_rank=int(a("attention.q_lora_rank")), qk_rope_head_dim=0,
        v_head_dim=int(a("attention.value_length_mla")), qk_nope_head_dim=int(a("attention.key_length_mla")), n_group=1, topk_group=1,
        num_experts_per_tok=int(a("expert_used_count")), norm_topk_prob=bool(a("expert_weights_norm", False)), max_position_embeddings=int(a("context_length")),
        rms_norm_eps=float(a("attention.layer_norm_rms_epsilon")), head_dim=0, qk_head_dim=int(a("attention.key_length_mla")),
        first_k_dense_replace=int(a("leading_dense_block_count", 0) or 0), scoring_func="sigmoid", topk_method="noaux_tc", mla_use_nope=True, mhc=True,
        index_topk=int(a("attention.indexer.top_k")), index_head_dim=int(a("attention.indexer.key_length")), index_n_heads=int(a("attention.indexer.head_count")),
        index_kpool=int(a("attention.indexer.kpool")), index_kpool_always_select_tail=bool(a("attention.indexer.kpool_select_tail", True)),
        indexer_types=["full"] * n,       # the file carries `indexer.types` only when the HF config had indexer_types; the pack of this model has 45 x "full"
        layer_types=lt, swiglu_limit=_swiglu(a, "swiglu_clamp_exp"),
        hc_mult=int(a("hyper_connection.count")), hc_eps=float(a("hyper_connection.epsilon")), hc_sinkhorn_iters=int(a("hyper_connection.sinkhorn_iterations")),
        linear_attn_config=dict(num_heads=nh, head_dim=int(a("kda.head_dim")), short_conv_kernel_size=int(a("ssm.conv_kernel")), gate_lower_bound=float(a("kda.gate_lower_bound")),
                                kda_layers=[i for i, x in enumerate(hkv[:n]) if x == 0], full_attn_layers=[i for i, x in enumerate(hkv[:n]) if x == 1]),
        linear_head_dim=int(a("kda.head_dim")), linear_num_heads=nh, linear_conv_kernel_dim=int(a("ssm.conv_kernel")), linear_lower_bound=float(a("kda.gate_lower_bound")),
        tie_word_embeddings="output.weight" not in g.tensors)
    return dict(text_config=tc, tie_word_embeddings=tc["tie_word_embeddings"], vocab_size=vocab, hidden_size=tc["hidden_size"])


def register() -> None:
    """Idempotent: the rows of glm5-next go in foreign_names.ARCHS / _CONFIG and foreign_xform._SETS. Proven on Metal (tools/glm_gguf_layer_proof.py), so they no longer
    wait for KYOJIN_KIT_UNPROVEN_ROWS; the switch still applies if the arch is put back in foreign_unproven.UNPROVEN_ARCHS."""
    from . import foreign_unproven as UP
    if "glm5-next" not in UP.UNPROVEN_ARCHS or UP.on("glm5-next"):
        FN.ARCHS["glm5-next"] = ARCH_MTP if foreign_mtp() else ARCH      # nextn head rows only with KYOJIN_FOREIGN_MTP
        FN._CONFIG.setdefault("glm5next", config)
        XF._SETS.setdefault("glm5next", xform)
    else:
        FN.ARCHS.pop("glm5-next", None)
        FN._CONFIG.pop("glm5next", None)
        XF._SETS.pop("glm5next", None)


register()
