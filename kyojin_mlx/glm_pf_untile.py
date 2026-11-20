# SPDX-License-Identifier: Apache-2.0
"""GLM-5.x prompt chunks: the 256-row tiling of `tiled_linear` (mlx_vlm.models.linear) off for batch 1 (flag glm_pf_untile, KYOJIN_GLM_PF_UNTILE, default 0).

mlx_vlm's glm5_next runs five projections through `tiled_linear`, which cuts a chunk into 256-row slices (padded, one matmul per slice, concatenated) so that every row sees the
same reduction geometry whatever the chunk length: the MoE router (fp32 x @ W.T, [4096 -> 288]), and the indexer's wk (4096 -> 128), the fp32 pool-gate projection (4096 -> 128),
wq_b (2048 -> 4096) and weights_proj (4096 -> 32). An 8192-row chunk is 32 tiny GEMMs per projection (G5g opprof, 8K cut: wq_b 7.8 ms, wk 7.0 ms, weights_proj 5.9 ms and the router
gate 4.4 ms PER LAYER at 1-13 TFLOPS; one 8192-row GEMM each is 5-10x faster). With the flag a prompt chunk of more than 256 rows (batch 1) runs one call: rows keep the same
reduction order along K (the kernel walks K in the same blocks whatever M), the engine's fixed chunk grid keeps the bits reproducible. Chunks of <= 256 rows and batches keep the stock code."""
from __future__ import annotations

import os

import mlx.core as mx

from . import archs as _A

ENABLED = os.environ.get("KYOJIN_GLM_PF_UNTILE", "0") != "0"
TILE = 256
STATS = {"untiled": 0, "tiled": 0}


def _wrap(orig):
    def tiled_linear(module, x):
        if ENABLED and x.ndim == 3 and x.shape[0] == 1 and x.shape[1] > TILE:     # ENABLED is read per call (the numerics check toggles it)
            STATS["untiled"] += 1
            return module(x)
        STATS["tiled"] += 1
        return orig(module, x)
    return tiled_linear


def install(model) -> int:
    """Replace `tiled_linear` in glm5_next.language (once per process). 1 when installed."""
    if not ENABLED:
        return 0
    try:
        L = _A.lang(_A.GLM5)
    except Exception:
        return 0
    if getattr(L, "_kyojin_untile", False):
        return 1
    L.tiled_linear = _wrap(L.tiled_linear)
    L._kyojin_untile = True
    return 1
