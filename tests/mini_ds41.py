"""Tiny random DS4.1 text model (mlx-vlm `deepseek_v41` LanguageModel, stock modules, no EXL3) for the CED / KV checkpoint logic tests.
Runs on any MLX backend (the Linux CPU one included). The layer layout has the CED shape: an encoder half with ratio 0 / 2 layers and a
ratio-1 half whose first layer owns the KV, index and candidate source (a reindex layer follows, then reuse layers)."""
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

CFG = dict(
    model_type="deepseek_v41", vocab_size=96, hidden_size=64, moe_intermediate_size=32, num_hidden_layers=8, num_attention_heads=4,
    num_key_value_heads=1, head_dim=64, qk_rope_head_dim=16, q_lora_rank=32, o_lora_rank=32, o_groups=2, swiglu_limit=10.0,
    rms_norm_eps=1e-6, n_routed_experts=8, n_shared_experts=1, num_experts_per_tok=2, routed_scaling_factor=1.5, sliding_window=16,
    compress_ratios=[0, 0, 2, 2, 1, 1, 1, 1], kv_source_layer_ids=[2, 4], index_source_layer_ids=[2, 4, 6], candidate_source_layer_id=4,
    index_n_heads=4, index_head_dim=32, index_topk=24, candidate_topk_blocks=12, candidate_block_size=4, hc_mult=2, hc_sinkhorn_iters=4,
    engram_layer_ids=[], engram_num_embeddings=[], max_position_embeddings=4096, rope_theta=10000.0, compress_rope_theta=160000.0)


def make(seed: int = 0, dtype=mx.float32, **over):
    """(LanguageModel, ModelConfig) with random weights."""
    from mlx_vlm.models.deepseek_v41.config import ModelConfig
    from mlx_vlm.models.deepseek_v41.language import LanguageModel
    cfg = ModelConfig.from_dict({**CFG, **over})
    lm = LanguageModel(cfg)
    mx.random.seed(seed)
    flat = []
    for k, v in tree_flatten(lm.parameters()):
        if k.endswith("norm.weight") or k.endswith("_norm.weight"):
            w = 1.0 + 0.1 * mx.random.normal(v.shape)
        elif k.endswith("attn_sink"):
            w = 0.5 * mx.random.normal(v.shape)
        elif "hc_" in k and k.endswith("_scale"):
            w = 0.5 + 0.1 * mx.random.normal(v.shape)
        elif "hc_" in k:
            w = 0.05 * mx.random.normal(v.shape)
        elif k.endswith("gate.weight") or k.endswith("gate.bias") or "e_score" in k:
            w = 0.3 * mx.random.normal(v.shape)
        elif k.endswith("embed_tokens.weight") or k.endswith("head.weight"):
            w = 0.5 * mx.random.normal(v.shape)
        else:
            fan_in = v.shape[-1] if v.ndim > 1 else 1
            w = mx.random.normal(v.shape) * (fan_in ** -0.5)
        flat.append((k, w.astype(v.dtype if v.dtype == mx.float32 else dtype)))
    lm.update(tree_unflatten(flat))
    mx.eval(lm.parameters())
    return lm, cfg


def tokens(n: int, seed: int = 1, vocab: int = CFG["vocab_size"]):
    import numpy as np
    return [int(x) for x in np.random.default_rng(seed).integers(3, vocab, n)]
