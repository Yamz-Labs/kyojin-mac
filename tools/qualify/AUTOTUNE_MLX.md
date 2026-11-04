# Autotune on MLX / Metal (qualify --stage autotune, NORTH STAR pillar 3)

A new pack (head_dim, hidden size, expert count) must be fast on the M5 Ultra with no hand work: `qualify` times the kernel candidates of the pack's shapes on the
device ONCE and stores the choices next to the pack (`<pack>/tuning/<device>.json`); every load of the pack reads them (`kyojin_mlx/tuning.py`).
No file, another device, another kernel version (sha of metal_src / metal_nax / kernels / exl3_linear / tuning) or `KYOJIN_TUNING=0` = today's constants, bit-identical.

## 1. Inventory: choices keyed on a shape or a row count (state of feat/qualify-mlx)

| choice | where | keyed on | chosen how | in the stage |
|---|---|---|---|---|
| `kernels.pick_tgk` (k-splits of the decode GEMV: `tuning.fill_tgs` = 8 threadgroups per GPU core, never below the Air's measured 512, and `tuning.fill_min_tiles` k tiles per warp = 8 on 10 cores down to 2 from 64 up; the 512 / 8 pair of the Air sweeps is what `KYOJIN_FILL=0` gives every device, and a wide GPU needs the deeper split: 64 threadgroups 126 GB/s, 128 319, 512 897 of a ~1000 roof) | kernels.py, Exl3Linear / Mul1Gemv, fused_group members (each keeps its own) | (kt, column blocks) | heuristic, Air sweeps (macair2/3) | **tuned per (in, out, K)** |
| `DENSE_FUSED_MAX` = 1024 (fused trellis decode + NAX matmul vs decode + MLX matmul) | exl3_linear.py `_fused_ok` | rows (global) | S6 Air bench: "5-15 % faster up to 1024 rows, slower above" (macair24/26, faf2e42) | **tuned per shape** (`fused_max`) |
| NAX tile `_BLOCK_CFG` / `pick_cfg` (bmu, nmb, bns, bks; 16/32/64-row blocks) | metal_nax.py | rows per expert (cost model `_BLOCK_COST`, DS4.1 w1 5120x2304 K3, tools/sweep_nax_cfg.py) | one measured shape, cost model | **tuned per (shape, rows)** for the dense linears (`nax_cfg`); MoE experts keep the model |
| `GEMV_MAX_ROWS` 4, `kernels.MAX_ROWS` 4, `RG_ROWS` 8, `ROW_SPLIT_MAX`, `NAX_SMALL_WEIGHTS` 4 M | exl3_linear / kernels / rows | rows (decode band, multi-user) | Air sweeps, bit-exactness rule (a row keeps its solo bits) | not tuned: they define which bits a user gets (solo vs multi-user), not a speed knob |
| `ROWS_NAX`, `HOTK` 4, `GROUP_CLASSES` 1:1,2:2,3:3,4:4,5:8, `KYOJIN_MTP_GROUPED` | exl3_linear / moe / metal_src / spec | rows per expert | Air benches (macair10/29) | MoE: not tuned (expert shapes are stacked tensors, see 3) |
| `PF_CROSS` 14 / `PF_CROSS_NAX` 2.5 Air, 1.75 at >= 32 GPU cores (S3 vs decode path crossover = x * E / top-k rows) | patch.py | experts / top-k, GPU cores | Air bench (DS4.1 900 -> 160 tokens, GLM 500 -> 80); Ultra tools/pfcross.py (DS4.1 + GLM-5.3 dims) | MoE: not tuned |
| `PF_FUSED_MAX` 512, `PF_GROUP_GIB` 2, `PF_CHUNK` 4096, wcache cap 10 % | moe_prefill.py, wcache.py | rows per expert / memory | Air | not tuned |
| attention prefill `MIN_ROWS` 64, `CHUNK` 512; mHC `MIN_ROWS` 64; pf_glue `NH` 2 | attn_prefill / mhc_prefill / pf_glue | rows, head count | Air | DS4.1 only: not tuned (no Qwen3.5 equivalent: stock mlx-lm attention) |
| `--prefill-chunk` 256 (server) / `stages.chunk_for` 512 (small packs) 2048 (ds41, CED) | serve, qualify | pack family | Air | not tuned (changes cache layout / CED, not a pure speed choice) |
| GDN prefill recurrence (mlx_lm step kernel vs `mx.fast.gated_delta_update`, kyojin_mlx/gdn_fast.py) | gdn_fast.py | rows T (decode T=1 always stock) | microbench Air macgdn: fast -30..-40 % GDN time at T >= 512, whole prefill -2 % | **tuned per pack** (table entry `gdn: {min_t}`, written only when >= 5 % faster at every T >= min_t; `KYOJIN_GDN_FAST=0\|1\|N` overrides) |
| `NACC` 1, `FUSED_PREFETCH` 1, `ROT2_MIN_BLOCKS` off | metal_src / kernels | none (global) | "no gain measured on the Air" | not tuned: compile-time kernel variants, no shape key |

## 2. What the stage tunes (dense EXL3 linears, all families)

Shapes = the 3-D `*.trellis` tensors in the safetensors headers (kt, nt, 16K); the tuner builds random weights of that shape through the real `Exl3Linear`, so it
never loads the pack (footprint < 1 GiB for any model, the Ultra's pack included).
- decode, 1 row: `tgk` in powers of two up to kt / 8, chain of 24 dependent calls.
- prefill, rows 8 .. 2048: fused NAX with each of the 3 tiles vs decode + matmul -> `fused_max` (largest R of the contiguous fused-winning prefix; always written when a crossover was seen or fused wins up to the constant, so the table does not depend on the constants it was made against) and `nax_cfg[R]`.
- method: interleaved candidates in a shuffled order, 0.1 s idle before each timing (burst clock), `min` over the rounds (default 5); a candidate replaces the
  default only when >= 5 % faster (plateau/noise rule) and only when its output matches the default's to 5e-3 relative (k-split order rounds at ~1e-5, the tile at fp16-matmul level).

## 3. Limits
- Stacked tensors (MoE experts, DS4.1 wo_a groups: 4-D) are not tuned here; they keep the cost models above. Their shapes are known from the config, the same
  harness extends to them (`metal_nax.expert_gemm` on a `SingleStack`) but that needs the routed-count distribution, so it stays a day-1 item.
- A tuned `tgk` changes the order of the fp32 k-sum (relative 1e-5 on the linear): greedy tokens are checked in the A/B, not guaranteed bit-equal to the untuned run.
  The multi-user rule "a row keeps its solo bits" holds within one tuning table.
