# SPDX-License-Identifier: Apache-2.0
"""Serve a standard mlx-lm quantized pack (MLX affine 4/8-bit, e.g. Qwen3.5-4B-mlx4) through the Kyojin engine.

The linears stay on MLX own quantized kernels (qmv / qmm). Kyojin adds on top: compiled decode regions + live ICB replay (step_regions, KYOJIN_REPLAY=1),
and the engine KV options where they apply. KYOJIN_MLXPACK_REGIONS=0 keeps the stock forward.

    model, tok = mlxpack.load(path)
"""
from __future__ import annotations

import os


def load(path, tokenizer_config=None):
    from mlx_lm import load as _load

    from . import step_regions
    model, tok = _load(path, tokenizer_config=tokenizer_config) if tokenizer_config else _load(path)
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
