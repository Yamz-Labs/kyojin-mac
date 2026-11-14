# SPDX-License-Identifier: Apache-2.0
"""Serve a standard mlx-lm quantized pack (MLX affine 4/8-bit, e.g. Qwen3.5-4B-mlx4) through the Kyojin engine.

The linears stay on MLX own quantized kernels (qmv / qmm). Kyojin adds on top: compiled decode regions + live ICB replay (step_regions, KYOJIN_REPLAY=1),
and the engine KV options where they apply. KYOJIN_MLXPACK_REGIONS=0 keeps the stock forward.

    model, tok = mlxpack.load(path)
"""
from __future__ import annotations

import os


# text-only model_types written by transformers for the Qwen3.5 / 3.6 / 3.8 text classes: mlx-lm's qwen3_5 / qwen3_5_moe modules read
# them as flat configs (no text_config) but have no module of that name
TEXT_ALIASES = {"qwen3_5_text": "qwen3_5", "qwen3_5_moe_text": "qwen3_5_moe"}


# keys newer transformers configs write only at the top level, while mlx-lm's multimodal classes read them from text_config (Qwen3-VL:
# tie_word_embeddings -> "ModelArgs.__init__() missing ... 'tie_word_embeddings'")
INHERIT_TEXT_KEYS = ("tie_word_embeddings",)


def _stack_experts(weights: dict) -> dict:
    """Per-expert HF keys (...mlp.experts.<e>.{gate,up,down}_proj.weight, older transformers / optimum exports) -> the fused layout mlx-lm's
    qwen3_5_moe sanitize reads (...mlp.experts.gate_up_proj [E, 2I, H] and ...mlp.experts.down_proj [E, H, I])."""
    import re
    import mlx.core as mx
    pat = re.compile(r"^(.*\.mlp\.experts)\.(\d+)\.(gate|up|down)_proj\.weight$")
    groups: dict = {}
    for k in list(weights):
        m = pat.match(k)
        if m:
            groups.setdefault(m.group(1), {}).setdefault(m.group(3), {})[int(m.group(2))] = weights.pop(k)
    for base, g in groups.items():
        stack = lambda part: mx.stack([g[part][e] for e in sorted(g[part])])
        weights[base + ".gate_up_proj"] = mx.concatenate([stack("gate"), stack("up")], axis=-2)
        weights[base + ".down_proj"] = stack("down")
    return weights


def remap_text_types():
    """mlx-lm compatibility shims Kyojin applies before any load (idempotent): text-only model_type aliases, top-level keys inherited by
    text_config, per-expert Qwen3.5-MoE keys stacked into the fused layout."""
    from mlx_lm import utils
    for k, v in TEXT_ALIASES.items():
        utils.MODEL_REMAPPING.setdefault(k, v)
    if getattr(utils.load_config, "_kyojin", False):
        return
    orig = utils.load_config

    def load_config(model_path):
        cfg = orig(model_path)
        tc = cfg.get("text_config")
        if isinstance(tc, dict):
            for k in INHERIT_TEXT_KEYS:
                if tc.get(k) is None and k in cfg:
                    tc[k] = cfg[k]
        return cfg
    load_config._kyojin = True
    utils.load_config = load_config
    try:
        from mlx_lm.models import nemotron_h                    # transformers >= 5 writes backbone.embedding (mlx-lm reads backbone.embeddings)
        nsan = nemotron_h.Model.sanitize
        if not getattr(nsan, "_kyojin", False):
            def nemotron_sanitize(self, weights):
                return nsan(self, {("backbone.embeddings." + k[len("backbone.embedding."):] if k.startswith("backbone.embedding.") else k): v
                                   for k, v in weights.items()})
            nemotron_sanitize._kyojin = True
            nemotron_h.Model.sanitize = nemotron_sanitize
    except ImportError:
        pass
    try:
        from mlx_lm.models import qwen3_5_moe
        sanitize = qwen3_5_moe.Model.sanitize
        if not getattr(sanitize, "_kyojin", False):
            def patched(self, weights):
                return sanitize(self, _stack_experts(weights))
            patched._kyojin = True
            qwen3_5_moe.Model.sanitize = patched
    except ImportError:
        pass


def load(path, tokenizer_config=None):
    from mlx_lm import load as _load
    remap_text_types()

    from . import step_regions
    model, tok = _load(path, tokenizer_config=tokenizer_config) if tokenizer_config else _load(path)
    # the Qwen3.5 path (step regions, q35 backend) reads layer.is_linear; plain attention stacks routed to it by the "qwen3" prefix
    # (Qwen3-VL text: qwen3 TransformerBlock) have no such flag: every layer is attention
    inner = getattr(model, "language_model", model)
    for layer in getattr(getattr(inner, "model", inner), "layers", []) or []:
        if not hasattr(layer, "is_linear"):
            layer.is_linear = False
    if os.environ.get("KYOJIN_MLXPACK_REGIONS", "1") != "0":
        step_regions.enable(model)
        from . import decfuse, decrows, gdn_pf
        decrows.enable(model)                            # 1..8 rows (multi-user steps, chained verify): add + norm seams, fused projections, attention front end
        if decfuse.ENABLED and decfuse.GDN:
            gdn_pf.install(prefill=False, dec=True)      # verify rows (2..8) through the same fused gated-delta front end as the 1-row regions
    return model, tok


def is_mlx_pack(path) -> bool:
    import json
    try:
        return "quantization" in json.load(open(os.path.join(path, "config.json")))
    except Exception:
        return False


def is_plain_pack(path) -> bool:
    """A plain bf16 / fp16 / fp32 HF checkpoint: config.json without quantization / quantization_config (also in text_config), safetensors next to it.
    mlx-lm reads it as stored (no EXL3 / foreign-format loader involved)."""
    import glob
    import json
    try:
        cfg = json.load(open(os.path.join(path, "config.json")))
    except Exception:
        return False
    for c in (cfg, cfg.get("text_config") or {}):
        if "quantization" in c or "quantization_config" in c:
            return False
    return bool(glob.glob(os.path.join(path, "*.safetensors")))
