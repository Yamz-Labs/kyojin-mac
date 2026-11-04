# SPDX-License-Identifier: Apache-2.0
"""Tiny synthetic shapes (hidden 64) for families recent public releases use. Each entry is a config.json dict built only from the public config
field names of the model code (mlx-lm / mlx-vlm dataclasses); sizes are shrunk, no value comes from a checkpoint. Random weights: nothing here says
anything about real numerics. A shape is a dict, nothing else (about 5-12 lines each)."""
from __future__ import annotations

V = 96
_RT = dict(rope_theta=10000.0)


def _mamba2_like(**kw):
    return dict(hidden_size=64, vocab_size=V, num_hidden_layers=4, intermediate_size=128, state_size=16, num_heads=8, head_dim=16, n_groups=1,
                conv_kernel=4, layer_norm_epsilon=1e-5, use_bias=False, use_conv_bias=True, tie_word_embeddings=False, time_step_limit=[0.0, 1e9],
                time_step_rank=8, **kw)


SHAPES_BASE_GLM = dict(model_type="glm4_moe", architectures=["Glm4MoeForCausalLM"], hidden_size=64, num_hidden_layers=4, intermediate_size=128,
                       moe_intermediate_size=32, num_attention_heads=4, num_key_value_heads=2, head_dim=16, n_routed_experts=8, n_shared_experts=1,
                       num_experts_per_tok=2, n_group=1, topk_group=1, norm_topk_prob=True, routed_scaling_factor=1.5, first_k_dense_replace=1,
                       rms_norm_eps=1e-6, vocab_size=V, max_position_embeddings=512, rope_theta=10000.0, rope_scaling=None, use_qk_norm=False,
                       tie_word_embeddings=False, attention_bias=False, partial_rotary_factor=0.5)

SHAPES: dict = {
    # dense GQA transformer (the baseline every family shares)
    "dense_gqa": dict(model_type="llama", architectures=["LlamaForCausalLM"], hidden_size=64, num_hidden_layers=4, intermediate_size=128, num_attention_heads=4,
                      num_key_value_heads=2, rms_norm_eps=1e-6, vocab_size=V, max_position_embeddings=512, tie_word_embeddings=False),
    # dense + MoE mix (first layer dense, rest routed + shared expert)
    "dense_moe_mix": dict(model_type="glm4_moe", architectures=["Glm4MoeForCausalLM"], hidden_size=64, num_hidden_layers=4, intermediate_size=128,
                          moe_intermediate_size=32, num_attention_heads=4, num_key_value_heads=2, head_dim=16, n_routed_experts=8, n_shared_experts=1,
                          num_experts_per_tok=2, n_group=1, topk_group=1, norm_topk_prob=True, routed_scaling_factor=1.5, first_k_dense_replace=1,
                          rms_norm_eps=1e-6, vocab_size=V, max_position_embeddings=512, rope_theta=10000.0, rope_scaling=None, use_qk_norm=False,
                          tie_word_embeddings=False, attention_bias=False, partial_rotary_factor=0.5),
    # GLM-4.5 style next-n head declared (num_nextn_predict_layers): the head layer would be appended after the trunk's layers (mtpgen, flag mtpgen)
    "glm_nextn": dict(SHAPES_BASE_GLM, num_nextn_predict_layers=1),
    # Mamba2 / SSM only
    "mamba2": dict(model_type="mamba2", architectures=["Mamba2ForCausalLM"], max_position_embeddings=512, **_mamba2_like()),
    # SSM + attention + routed experts with the latent projection (hybrid_override_pattern: M mamba, * attention, E moe)
    "ssm_latent_moe": dict(model_type="nemotron_h", architectures=["NemotronHForCausalLM"], vocab_size=V, hidden_size=64, intermediate_size=128,
                           max_position_embeddings=512, num_attention_heads=4, num_key_value_heads=2, attention_bias=False, mamba_num_heads=8,
                           mamba_head_dim=16, mamba_proj_bias=False, ssm_state_size=16, conv_kernel=4, n_groups=1, mlp_bias=False,
                           layer_norm_epsilon=1e-5, use_bias=False, use_conv_bias=True, hybrid_override_pattern=["M", "E", "*", "E"],
                           n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=32, moe_shared_expert_intermediate_size=32,
                           moe_latent_size=32, n_group=1, topk_group=1, norm_topk_prob=True, routed_scaling_factor=1.0, num_hidden_layers=4),
    # Mamba + attention + MoE (jamba layout: attention every attn_layer_period, experts every expert_layer_period)
    "jamba_mix": dict(model_type="jamba", architectures=["JambaForCausalLM"], hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, attn_layer_offset=1, attn_layer_period=2, expert_layer_offset=1, expert_layer_period=2,
                      mamba_d_conv=4, mamba_d_state=16, mamba_expand=2, num_experts=4, num_experts_per_tok=2, rms_norm_eps=1e-6,
                      max_position_embeddings=512, vocab_size=V),
    # linear attention (gated delta net) + full attention + MoE
    "gdn_moe": dict(model_type="qwen3_next", architectures=["Qwen3NextForCausalLM"], hidden_size=64, num_hidden_layers=4, intermediate_size=128,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=16,
                    linear_value_head_dim=16, linear_conv_kernel_dim=4, num_experts=4, num_experts_per_tok=2, decoder_sparse_step=1, mlp_only_layers=[],
                    shared_expert_intermediate_size=32, moe_intermediate_size=32, rms_norm_eps=1e-6, vocab_size=V, rope_theta=10000.0,
                    partial_rotary_factor=0.5, max_position_embeddings=512, full_attention_interval=2, tie_word_embeddings=False),
    # MLA + multi-hyper-connection residual (hc_mult streams) + indexer + dense/MoE mix: Hy-v4-like from the public config field names
    "hyv4_like": dict(model_type="hy_v4", architectures=["HYV4ForCausalLM"], vocab_size=V, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
                      num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=1, n_shared_experts=1, n_routed_experts=8, routed_scaling_factor=1.5,
                      kv_lora_rank=32, q_lora_rank=32, qk_rope_head_dim=16, v_head_dim=16, qk_nope_head_dim=16, index_head_dim=16, index_n_heads=4,
                      index_topk=16, topk_method="noaux_tc", norm_topk_prob=True, n_group=1, topk_group=1, num_experts_per_tok=2, moe_layer_freq=1,
                      first_k_dense_replace=1, max_position_embeddings=512, rms_norm_eps=1e-6, rope_theta=10000.0, hc_mult=2),
    # the same linear attention + attention hybrid with a vision tower (image tokens replaced by tower rows) and a declared draft head
    "vl_gdn_mtp": dict(model_type="qwen3_5", architectures=["Qwen3_5ForConditionalGeneration"], image_token_id=90, video_token_id=91, vision_start_token_id=92,
                       vision_end_token_id=93, vocab_size=V,
                       text_config=dict(model_type="qwen3_5_text", hidden_size=64, num_hidden_layers=4, intermediate_size=96, num_attention_heads=4,
                                        num_key_value_heads=2, head_dim=16, linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=16,
                                        linear_value_head_dim=16, linear_conv_kernel_dim=4, vocab_size=V, rms_norm_eps=1e-6, full_attention_interval=2,
                                        max_position_embeddings=512, tie_word_embeddings=False, mtp_num_hidden_layers=1,
                                        rope_parameters=dict(rope_theta=10000.0, partial_rotary_factor=0.5, rope_type="default", mrope_section=[2, 1, 1],
                                                             mrope_interleaved=True)),
                       vision_config=dict(model_type="qwen3_5", depth=2, hidden_size=32, intermediate_size=64, out_hidden_size=64, num_heads=2, patch_size=4,
                                          spatial_merge_size=2, temporal_patch_size=2, in_channels=3, num_position_embeddings=64)),
}
