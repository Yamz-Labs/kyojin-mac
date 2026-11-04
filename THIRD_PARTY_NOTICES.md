# Third-party notices

Portions adapted from other projects keep their license. Files that contain adapted code say so in their header.

- beamivalice/sushi (MIT / Apache-2.0), commit 0f4f00b8a166fe8dc67bca05b34c732df7c7d176: the idea and chain layout of the shared expert riding the routed
  MoE decode launches (`src/exl3/rate_groups.zig`, `moeDecodeClampedSharedFused`). Used in `kyojin_mlx/moe_sumoe.py` and the sumoe kernels at the end of
  `kyojin_mlx/metal_src.py`.

- kyojin_mlx/suglue.py: role layout of the fused hyper-connection stage, from src/dsv41/dsv41_kernels.zig@0f4f00b (docs/dsv41-arch.md:122-127).

- mlx-serve (MIT, Copyright (c) 2026 David Dalcu), src/engines/mlx/spec/mtp_lookup.zig: the idea and gate of prompt-lookup drafts
  re-implemented in Python/numpy in `kyojin_mlx/pld.py` (4-gram index over committed tokens, suffix/candidate match, acceptance EMA
  with drift, cost gate). No Zig or Metal text is copied.

- llama.cpp / ggml (MIT, Copyright (c) 2023-2026 The ggml authors), ggml/src/ggml-common.h and ggml-quants.c: the GGUF block layouts and dequantisation
  formulas implemented again in numpy in `kyojin_mlx/foreign_gguf.py`, and the IQ3_XXS / IQ3_S / ksigns lookup tables in `kyojin_mlx/foreign_grids.py`
  (transcribed mechanically from ggml-common.h), and the iq2xs/iq3xs seed lists in `kyojin_mlx/foreign_seeds.py` (extracted by `tools/gen_gguf_seeds.py` from
  ggml-quants.c at llama.cpp 29c720ea). `tools/ggml_ref/*.c` are small drivers that call ggml's own functions; `tests/data/gguf_ggml_ref.npz` holds values it returned.
  No ggml C code is copied.
