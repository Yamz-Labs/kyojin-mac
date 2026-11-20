# SPDX-License-Identifier: Apache-2.0
"""Rows of the decoder skeleton (foreign_skeleton.py): the structural choices a GGUF does not carry, per GGUF architecture name.

Everything else comes from the file itself (tensor list = biases, gated or not, fused qkv, q/k norms, learned positions, tied head;
metadata = sizes, eps, rope base / dims / scaling, scales). A row is data read off llama.cpp's graph builder (src/models/<arch>.cpp and
llama_model_rope_type in llama-model.cpp); a new architecture of a known shape is one line here.
Row keys (all optional):
  norm              "rms" | "layer"             default "rms"
  act               key of foreign_skeleton.ACTS  default "silu"
  rope              "neox" (split halves) | "norm" (adjacent pairs) | "none"   default "neox"
  norm_pos          "pre" | "post" | "sandwich"   default "pre"
  qk_norm_after_rope  bool                       default False
  parallel          bool | None: None reads `<arch>.use_parallel_residual` (default False)
  logit_div         bool: the file's logit_scale divides the logits (llama.cpp granite) instead of multiplying
  moe               dict of foreign_moe_generic.MoeSpec fields for what the file's expert metadata does not say (the file wins), e.g. norm_weights
"""
ROWS = {
    # src/models/hunyuan-vl.cpp (hunyuan-dense inherits it): RMS pre-norm, SwiGLU, rope NEOX, then RMS q / k norms AFTER the rope
    "hunyuan-dense": dict(norm="rms", act="silu", rope="neox", qk_norm_after_rope=True),
    # src/models/olmo.cpp: LLM_NORM (mean, no weights in the file), SwiGLU, rope NORM, clamp_kqv from the metadata
    "olmo": dict(norm="layer", act="silu", rope="norm"),
    # src/models/gptneox.cpp: LayerNorm + bias, non-gated GELU, fused qkv, partial rope NEOX, parallel residual from use_parallel_residual
    "gptneox": dict(norm="layer", act="gelu", rope="neox"),
    # src/models/gpt2.cpp: LayerNorm + bias, learned positions (position_embd), non-gated GELU, fused qkv
    "gpt2": dict(norm="layer", act="gelu", rope="none"),
    # MoE rows (the experts come from the file: foreign_skeleton_moe). Both archs have model classes: the rows are only used when the skeleton is forced (KYOJIN_SKELETON)
    # src/models/granite-moe.cpp (llama graph, rope NORM): RMS pre-norm, SwiGLU experts, softmax over the logits then renormalised, logit_scale divides
    "granitemoe": dict(norm="rms", act="silu", rope="norm", logit_div=True, moe=dict(norm_weights=True)),
    # src/models/qwen2moe.cpp (rope NEOX): softmax gating, weights not renormalised (expert_weights_norm = false), sigmoid-gated shared expert
    "qwen2moe": dict(norm="rms", act="silu", rope="neox"),
    # hy-v4-flash.cpp of the hy4-flash-mac-minimal patch: RMS sandwich norms, SwiGLU, rope NORM (llama-model.cpp hunk). Unproven (foreign_unproven.UNPROVEN_ARCHS)
    "hy_v4_flash": dict(norm="rms", act="silu", rope="norm", norm_pos="sandwich", moe=dict(clamp_mode="pre_act")),   # routed swiglu_clamp: min(gate, c) before silu (llama-graph.cpp:2461)
}
