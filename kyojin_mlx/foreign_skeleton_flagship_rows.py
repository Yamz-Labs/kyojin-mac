# SPDX-License-Identifier: Apache-2.0
"""Skeleton rows (the choices a GGUF does not carry) of the flagship MoE architectures, as the llama.cpp graphs make them. Same row fields as
foreign_skeleton_rows.ROWS (norm, act, rope, norm_pos) plus the fields of the flagship spec: residual, rope_on, swa_period, rope_factors_swa,
attn_out_norm, embed_norm_plain, ffn_post_norm_mul_first, post_norm_eps. Not merged into ROWS here (that table is edited elsewhere); the skeleton
stays behind its switch KYOJIN_SKELETON and nothing imports this module unless asked.

Source of each row: src/models/<arch>.cpp of llama.cpp (beam, muse-glimmer, muse-spark, step35, mimo2, minimax-m3); all use NEOX rope (llama-model.cpp:3726-3848)."""
FLAGSHIP_ROWS = {
    "beam": dict(norm="rms", act="silu", rope="neox", norm_pos="sandwich", residual="depth", rope_on="swa"),
    "step35": dict(norm="rms", act="silu", rope="neox", rope_factors_swa=False),
    "mimo2": dict(norm="rms", act="silu", rope="neox"),
    "muse-glimmer": dict(norm="rms", act="silu", rope="neox", norm_pos="sandwich", rope_on="swa", swa_period=4, embed_norm_plain=True, post_norm_eps=1e-8),
    "muse-spark": dict(norm="rms", act="silu", rope="neox", norm_pos="sandwich", rope_on="swa", residual="vector", attn_out_norm=True, embed_norm_plain=True,
                       ffn_post_norm_mul_first=True),
    "minimax-m3": dict(norm="rms", act="swiglu_oai", rope="neox"),
}
