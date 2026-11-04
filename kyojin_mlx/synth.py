# SPDX-License-Identifier: Apache-2.0
"""Tiny synthetic models, written as real packs (config.json + safetensors + tokenizer files) and loaded through the loaders Kyojin uses, so the
descriptor fields, the pack checks and the CPU conformance checks (kyojin_mlx.conformance) run on any MLX backend, the Linux CPU one included.

Four shapes, one per cell that stock model code can build:

    ds_moe_ced     DeepSeek-style block (hyper-connections, compressed sparse attention, MoE) in the layer layout CED needs
    ds_moe_plain   the same block with a layer layout CED refuses (odd layer count): MoE, no CED
    q35_dense      hybrid linear attention + full attention, dense MLP: no MoE, no CED
    q35_moe        the same with routed experts: MoE, no CED

"CED without MoE" does not exist as a shape: every block of the CED-capable model code has a MoE FFN, so there is nothing to build it from.
A new architecture adds a `Shape` (its config + a `build`), nothing else; the checks in `conformance.py` do not name a model.
The random weights are float32 and small (hidden 64): nothing here says anything about a real checkpoint's numerics."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

DS_CFG = dict(
    model_type="deepseek_v41", vocab_size=96, hidden_size=64, moe_intermediate_size=32, num_hidden_layers=8, num_attention_heads=4,
    num_key_value_heads=1, head_dim=64, qk_rope_head_dim=16, q_lora_rank=32, o_lora_rank=32, o_groups=2, swiglu_limit=10.0,
    rms_norm_eps=1e-6, n_routed_experts=8, n_shared_experts=1, num_experts_per_tok=2, routed_scaling_factor=1.5, sliding_window=16,
    compress_ratios=[0, 0, 2, 2, 1, 1, 1, 1], kv_source_layer_ids=[2, 4], index_source_layer_ids=[2, 4, 6], candidate_source_layer_id=4,
    index_n_heads=4, index_head_dim=32, index_topk=24, candidate_topk_blocks=12, candidate_block_size=4, hc_mult=2, hc_sinkhorn_iters=4,
    engram_layer_ids=[], engram_num_embeddings=[], max_position_embeddings=4096, rope_theta=10000.0, compress_rope_theta=160000.0)

DS_PLAIN = dict(DS_CFG, num_hidden_layers=7, compress_ratios=[0, 0, 2, 2, 1, 1, 1])        # odd split: ced.explain refuses

Q_CFG = dict(
    model_type="qwen3_5_text", hidden_size=64, num_hidden_layers=4, intermediate_size=96, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4, vocab_size=64,
    rms_norm_eps=1e-6, full_attention_interval=2, max_position_embeddings=512, tie_word_embeddings=False,
    rope_parameters=dict(rope_theta=10000.0, partial_rotary_factor=0.5, rope_type="default"))
Q_MOE = dict(Q_CFG, model_type="qwen3_5_moe_text", num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32, shared_expert_intermediate_size=32)


@dataclass(frozen=True)
class Shape:
    name: str
    family: str                     # "ds" (mlx-vlm model code, patched loader) | "q35" (mlx-lm model code)
    moe: bool
    ced: bool                       # the layer layout is one CED supports
    config: dict                    # config.json
    stop_tokens: tuple = ()         # written into tokenizer.json so the stop-token check has something to find
    extra: dict = field(default_factory=dict)

    @property
    def text(self) -> dict:
        return self.config.get("text_config") or self.config


SHAPES = {
    "ds_moe_ced": Shape("ds_moe_ced", "ds", True, True, {"model_type": "deepseek_v41", "architectures": ["DeepseekV41ForCausalLM"], "text_config": DS_CFG},
                        ("<｜end▁of▁sentence｜>",)),
    "ds_moe_plain": Shape("ds_moe_plain", "ds", True, False, {"model_type": "deepseek_v41", "architectures": ["DeepseekV41ForCausalLM"], "text_config": DS_PLAIN},
                          ("<｜end▁of▁sentence｜>",)),
    "q35_dense": Shape("q35_dense", "q35", False, False, {"model_type": "qwen3_5", "architectures": ["Qwen3_5ForCausalLM"], "text_config": Q_CFG},
                       ("<|im_end|>", "<|endoftext|>")),
    "q35_moe": Shape("q35_moe", "q35", True, False, {"model_type": "qwen3_5_moe", "architectures": ["Qwen3_5MoeForCausalLM"], "text_config": Q_MOE},
                     ("<|im_end|>", "<|endoftext|>")),
}


# ------------------------------------------------------------------------------------------------------------------ building
def _init(params, seed: int):
    """Random float32 weights that keep the logits finite and distinct (same recipe as tests/mini_ds41.py)."""
    mx.random.seed(seed)
    flat = []
    for k, v in tree_flatten(params):
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
        elif k.endswith("embed_tokens.weight") or k.endswith("head.weight") or k.endswith("lm_head.weight"):
            w = 0.5 * mx.random.normal(v.shape)
        elif v.ndim <= 1:
            w = v if v.dtype != mx.float32 else v + 0.01 * mx.random.normal(v.shape)
        else:
            w = mx.random.normal(v.shape) * (v.shape[-1] ** -0.5)
        flat.append((k, w.astype(mx.float32)))
    return tree_unflatten(flat)


def build_reference(shape: Shape, seed: int = 0):
    """(model, lm, make_cache): the model with random weights, built straight from the stock classes. `lm` is what is called with ids + cache.
    The package is the one named by the shape's model_type (mlx-vlm for the vlm families, mlx-lm for the text only ones): no family name is written here."""
    import importlib
    if shape.family == "ds":
        mod = importlib.import_module("mlx_vlm.models." + shape.config["model_type"])
        m = mod.Model(mod.ModelConfig.from_dict(shape.config))
        m.language_model.update(_init(m.language_model.parameters(), seed))       # the text trunk only: the wrapper's default vision tower is 1.5 GB of random weights nobody reads
        mx.eval(m.language_model.parameters())
        return m, m.language_model, m.language_model.make_cache
    mod = importlib.import_module("mlx_lm.models." + shape.config["model_type"])
    from mlx_lm.models.cache import make_prompt_cache
    m = mod.Model(mod.ModelArgs.from_dict(shape.config))
    m.update(_init(m.parameters(), seed))
    mx.eval(m.parameters())
    return m, m, (lambda: make_prompt_cache(m))


def _tokenizer_files(path: str, shape: Shape) -> None:
    vocab = {f"t{i}": i for i in range(int(shape.text["vocab_size"]) - len(shape.stop_tokens))}
    added = [{"id": len(vocab) + i, "content": t, "special": True} for i, t in enumerate(shape.stop_tokens)]
    json.dump({"version": "1.0", "added_tokens": added, "model": {"type": "WordLevel", "vocab": vocab}}, open(os.path.join(path, "tokenizer.json"), "w"))
    json.dump({"chat_template": "{% for m in messages %}{{ m['content'] }}{% endfor %}"}, open(os.path.join(path, "tokenizer_config.json"), "w"))


def write_pack(shape: Shape, path: str, seed: int = 0):
    """Write `shape` as a pack directory (config.json, model.safetensors with the model's own tensor names, tokenizer files). Returns the reference
    (model, lm, make_cache) the pack was written from."""
    os.makedirs(path, exist_ok=True)
    model, lm, mk = build_reference(shape, seed)
    flat = {("language_model." + k if shape.family == "ds" else k): v for k, v in tree_flatten((model.language_model if shape.family == "ds" else model).parameters())}
    mx.save_safetensors(os.path.join(path, "model.safetensors"), flat, {"format": "mlx"})
    json.dump(shape.config, open(os.path.join(path, "config.json"), "w"))
    _tokenizer_files(path, shape)
    return model, lm, mk


def load_pack(shape: Shape, path: str):
    """Load the pack with the loader of its family, fresh weights from disk. ds: patch.build_model + load_dense + check_loaded (the loader the served DS4.1 / GLM
    path runs, its quantized-linear branch has nothing to do on a dense pack). q35: mlx_lm.utils.load_model (the loader mlxpack.load wraps; the Kyojin step regions
    and row kernels need Metal and are not installed here). Returns (model, lm, make_cache)."""
    if shape.family == "ds":
        from . import patch as P
        model, spec, cfg = P.build_model(path)
        plan = P.load_dense(model, path, spec, cfg)
        P.check_loaded(model, plan)
        mx.eval(model.language_model.parameters())
        return model, model.language_model, model.language_model.make_cache
    from pathlib import Path
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_model
    model, _ = load_model(Path(path), strict=True)
    mx.eval(model.parameters())
    return model, model, (lambda: make_prompt_cache(model))
