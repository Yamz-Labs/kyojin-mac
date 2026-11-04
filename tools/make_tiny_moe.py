#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Random tiny Qwen3.5-MoE checkpoint (HF layout) to exercise tools/conv_mlx.py on MoE blocks without a real MoE download.
usage: python tools/make_tiny_moe.py OUT_DIR [--layout fused|perexpert] [--experts 8] [--layers 4] [--cal CAL.safetensors --rows 16 --cols 128]
dims are multiples of 128 (the quantiser tiles); routing is random but non-degenerate (gate std 0.5)."""
import argparse, json, os, sys
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--layout", choices=("fused", "perexpert"), default="fused")
    ap.add_argument("--experts", type=int, default=8); ap.add_argument("--topk", type=int, default=2); ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=256); ap.add_argument("--inter", type=int, default=128); ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--cal", default=None); ap.add_argument("--rows", type=int, default=16); ap.add_argument("--cols", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from mlx_lm.models.qwen3_5 import DecoderLayer, TextModelArgs
    tc = {"model_type": "qwen3_5_moe_text", "hidden_size": a.hidden, "num_hidden_layers": a.layers, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64,
          "linear_num_key_heads": 4, "linear_num_value_heads": 4, "linear_key_head_dim": 64, "linear_value_head_dim": 64, "linear_conv_kernel_dim": 4,
          "full_attention_interval": 4, "vocab_size": a.vocab, "rms_norm_eps": 1e-6, "tie_word_embeddings": True, "attn_output_gate": True,
          "num_experts": a.experts, "num_experts_per_tok": a.topk, "moe_intermediate_size": a.inter, "shared_expert_intermediate_size": a.inter,
          "decoder_sparse_step": 1, "norm_topk_prob": True, "intermediate_size": a.inter, "max_position_embeddings": 4096,
          "rope_parameters": {"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10], "mrope_interleaved": True}}
    args = TextModelArgs.from_dict(tc)
    rng = np.random.default_rng(a.seed)
    T = {"model.language_model.embed_tokens.weight": rng.standard_normal((a.vocab, a.hidden)).astype(np.float32) * 0.5,
         "model.language_model.norm.weight": rng.standard_normal(a.hidden).astype(np.float32) * 0.1}
    for li in range(a.layers):
        lp = f"model.language_model.layers.{li}."
        flat = dict(tree_flatten(DecoderLayer(args, li).parameters()))
        for k, v in flat.items():
            sh = tuple(v.shape)
            if k.startswith("mlp.switch_mlp."):
                continue
            if k.endswith("norm.weight") or k.endswith(".norm.weight"):
                T[lp + k] = (rng.standard_normal(sh) * 0.1).astype(np.float32)          # stored as (w - 1) by HF, loaded as 1 + w
                continue
            if "conv1d.weight" in k:
                T[lp + k] = (rng.standard_normal((sh[0], sh[2], sh[1])) * 0.3).astype(np.float32)     # HF [C, 1, k]
                continue
            if k.endswith("A_log"):
                T[lp + k] = rng.uniform(0.0, 1.5, sh).astype(np.float32); continue
            if k.endswith("dt_bias"):
                T[lp + k] = np.ones(sh, np.float32); continue
            std = 0.5 if k == "mlp.gate.weight" else 0.9 / np.sqrt(sh[-1] if len(sh) > 1 else 1)
            T[lp + k] = (rng.standard_normal(sh) * std).astype(np.float32)
        E, I, H = a.experts, a.inter, a.hidden
        g, u, d = (rng.standard_normal((E, I, H)) / np.sqrt(H) * 0.9, rng.standard_normal((E, I, H)) / np.sqrt(H) * 0.9, rng.standard_normal((E, H, I)) / np.sqrt(I) * 0.9)
        if a.layout == "fused":
            T[lp + "mlp.experts.gate_up_proj"] = np.concatenate([g, u], axis=1).astype(np.float32)
            T[lp + "mlp.experts.down_proj"] = d.astype(np.float32)
        else:
            for e in range(E):
                T[f"{lp}mlp.experts.{e}.gate_proj.weight"] = g[e].astype(np.float32); T[f"{lp}mlp.experts.{e}.up_proj.weight"] = u[e].astype(np.float32)
                T[f"{lp}mlp.experts.{e}.down_proj.weight"] = d[e].astype(np.float32)
    os.makedirs(a.out, exist_ok=True)
    mx.save_safetensors(f"{a.out}/model.safetensors", {k: mx.array(v).astype(mx.bfloat16) for k, v in T.items()}, metadata={"format": "pt"})
    json.dump({"architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5_moe", "tie_word_embeddings": True, "text_config": tc}, open(f"{a.out}/config.json", "w"), indent=1)
    if a.cal:
        mx.save_safetensors(a.cal, {"input_ids": mx.array(rng.integers(0, a.vocab, (a.rows, a.cols)).astype(np.int32))})
    print(f"wrote {a.out}: {len(T)} tensors, layout {a.layout}")


if __name__ == "__main__":
    main()
