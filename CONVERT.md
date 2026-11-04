# EXL3 conversion on MLX / Metal (groundwork for the Ultra)

Branch `feat/conv-mlx`. Files: `kyojin_mlx/conv_viterbi.py` (Metal trellis quantiser), `tools/conv_mlx.py` (pipeline), `tools/conv_eval_kl.py`,
`tools/conv_bench.py`, `tests/test_conv_viterbi.py`, fixtures in `results_mac/conv/`.

    python tools/conv_mlx.py SRC_HF_DIR OUT_DIR --cal results_mac/conv/cal_128x1024.safetensors [--k-attn 3 --k-mlp 2 --head-bits 6] [--strategy k.json] [--resume] [--fp-prefix]

Dense Qwen3.5-style models (GatedDeltaNet + attention + gated MLP, tied head). Output = `model.safetensors` + `config.json` with `quantization_config`,
same tensor names / dtypes / shapes as the CUDA converter (`<linear>.{trellis I16, suh F16, svh F16, mul1 I32}`), loadable by `load_exl3`.
Not done on the Mac path: MoE routing capture (per-expert H), MTP and vision towers (copied / skipped), `--hq`, bit allocation (K per tensor is given).

## Stages (exllamav3 `convert_model.py` / `quantize.py`) and where each runs

| # | stage | what | here |
|---|---|---|---|
| 1 | calibration forward ("capture") | FP block forward on the state, `H += x^T x` per distinct linear input | MLX / Metal |
| 2 | finalize H | `H/n`, damping 0.025 diag-mean, sign flips su, 128-Hadamards both sides, Cholesky, block LDL (16), identity blocks | numpy CPU fp32 (Cholesky) + MLX (Hadamard) |
| 3 | regularize | output scales (rms), `/sv`, Hadamard, input scales, `/su`, Hadamard, **global scale search** on ~900 sampled tiles x 15 scales | MLX + Viterbi |
| 4 | LDLQ | per 16-row block: error feedback matmul, then one Viterbi call over n/16 tiles | MLX matmul + **Metal Viterbi** |
| 5 | pack | 16-bit states -> K-bit tail-biting stream, int16 words | numpy |
| 6 | advance | quantised block forward gives the next block's state | MLX / Metal |

Viterbi = tail-biting trellis over 2^16 states, 256 steps x 2 passes, mul1 codebook, fp16 arithmetic. Port of `quantize_tiles_kernel.cuh`, one threadgroup per tile,
branch number stored instead of the 16-bit predecessor (1 byte per state-step), cost rows in device memory (Metal has 32 KB threadgroup memory), mul1 table lookup (+25 %).

## Correctness

* Viterbi vs the CUDA kernel (3090, `ext.quantize_tiles`, mul1, K = 1..4, 48 tiles each incl. tiny / outlier / all-zero tiles): `idx` and decoded values identical bit for bit (`tests/test_conv_viterbi.py`).
* Pack: round trip through `ref_numpy.decode_states` (the decoder validated against CUDA packs).
* Chunked head path (`quantize_big`) bit-identical to the in-memory path on a 512 x 2048 test.
* Not bit-identical to the CUDA converter end to end (random sign draws differ: numpy vs torch RNG; LDLQ matmul order). Equivalent statistically, see below.

Qwen3.5-0.8B-Base, 128 x 1024 calibration rows (same rows both sides), K=3 attention / K=2 MLP / head K=6, held-out 16 x 1023 tokens, BF16 teacher:

| pack | KL top-256 | top-1 agree | ppl (BF16 4.541) | median proxy_err ratio vs ROCm converter |
|---|---|---|---|---|
| Mac (this pipeline) | 0.2680 | 81.75 % | 5.689 | 1.002 (mean per K: K2 0.02661 vs 0.02660, K3 0.00410 vs 0.00402) |
| exllamav3 converter, Strix ROCm (`kyojin-rb`) | 0.2673 | 82.25 % | 5.687 | 1 |

global scale ratio median 1.001. Same K layout, same recipe.

## Time per stage, 0.8B (24 blocks + head, 4.35 M tile evaluations)

| | Air M5 (10 GPU cores, AC, LPM off) | Strix iGPU (ROCm, rdna kernel) |
|---|---|---|
| total | 1178 s | 277 s |
| Viterbi rate | 4.1-4.3 k tiles/s (3.5-3.7 k without the table; min over 7 reps, 2 interleaved rounds) | 31.8 k tiles/s |
| capture | 55 s (5 %) | 93 s (34 %) |
| finalize (Cholesky / LDL) | 8 s | 2 s |
| regularize incl. g-scale Viterbi | 324 s (27 %) | 41 s |
| LDLQ | 745 s (63 %) | 98 s |
| pack | 7 s | < 1 s |
| advance (quantised forward) | 32 s | in the 93 s |
| peak MLX memory | 4.8 GB (stage peaks 2.2-4.8) | rss 3.1 GB |

**Bottleneck = the Viterbi** (Air 90 % of wall time, Strix 50 %, and 34 % goes to the calibration forward there). It is ALU bound: 4.3 k tiles/s x 256 steps x 65536 edges = 7e10 edge evaluations/s.
The tile cost does not depend on K. The global scale search costs 46 % extra tiles on this small model (matrix samples are large relative to the matrices).

Original-engine check: the exllamav3 ROCm engine (Strix, `kyojin-rb`) loads the Mac pack and scores the same rows: ppl 5.6875, KL 0.2673, top-1 81.7 % against the converter's own pack 5.6879 / 0.2663 / 82.2 % (BF16 4.5386 in that engine).
`tensor_storage` of `quantization_config.json` (tools/conv_qconfig.py) identical to the converter's for all 267 modules.

## Quantised prefix or FP prefix (can blocks be pipelined?)

`--fp-prefix` feeds block l+1 the FP output of block l instead of the quantised one, so blocks no longer depend on each other. Qwen3.5-0.8B, same recipe and rows:
quantised prefix KL 0.2680 / ppl 5.689, FP prefix KL 0.2654 / ppl 5.623 (one seed each, no loss; DS4.1 needs its own A/B on day 1).

## Plan for the Ultra (DS4.1 experts: ~145 GiB at 2.3 bpw = 2.1e9 tiles, x1.25-1.46 for the global scale search = 2.7-3.2e9 Viterbi tile evaluations)

Measured rates: Air 4.3 k tiles/s, Strix 31.8 k tiles/s. Ultra 80 GPU cores vs 10: ALU-bound kernel, so estimate 29-34 k tiles/s (to measure: `python tools/conv_bench.py --K 2 3` under the lock, 2 min).
3090: not measured (busy with another job all night); the kernel is the same algorithm, assume 30-40 k.

| lane | role | share of tiles | alone |
|---|---|---|---|
| Ultra 256 GB | calibration forward + H capture (all experts active, FP weights), advance, Cholesky/LDL on the 16-core CPU, Metal Viterbi on a share | ~30 % | 22-26 h |
| 3090 | Viterbi (CUDA `quantize_tiles`), whole experts batched | ~35 % | 18-24 h |
| Strix | Viterbi (rdna kernel) + Cholesky on the 32 CPU threads | ~35 % | 24 h |
| all three | | | 7.4-9 h |

* Work unit = (layer, expert, projection): independent given its H, cost n*k/256 tiles x 1.25; static split by measured rate, workers receive W (bf16) + H, return the trellis (2-3 bit, small).
* The only serial chain is block l -> block l+1 through the calibration state. With a quantised prefix the Ultra capture of block l+1 waits for all workers (bubble ~ capture time, est. 20-30 % of a block's Viterbi time with all experts active). With `--fp-prefix` the Ultra runs the FP forward for all blocks first (or ahead of the workers) and the three lanes work on different blocks concurrently: no bubble.
* Small tensors must be stacked per Viterbi call (the Air reaches full speed from ~48 tiles per call; the 3090 and Strix need hundreds to thousands: batch whole experts, as `group_quant_linears` does).
* 10 h -> 5 h needs ~150 k tiles/s overall; the three lanes give ~95-105 k. Levers, in order of value: (1) one global-scale search per (layer, projection) instead of per expert (-20 to -30 % tiles, needs a KL A/B), (2) Metal kernel work on the Ultra (hist stores cost 22 %, decode table gave +25 %, packed-half2 not tried), (3) FP prefix (removes the capture bubble). With (1) and (3) the three lanes land at ~6 h; the last ~20 % to 5 h has to come from the kernels (Ultra Metal, 3090 CUDA).

## Update 26/10 (macair46): faster Viterbi, MoE capture

**Viterbi kernel (impl 2, default for K2-K5; K1 and K6-K8 keep the device-memory kernel).** Structure ported from the CUDA / HIP `quantize_tiles_optimized` kernel: the 2^K codebook values of every output pair live in registers for all 512 steps (no decode, no table load), the cost row sits in threadgroup memory (staged in registers, barrier, store; the argmin reads the registers), 512 threads per tile, runs of S consecutive output pairs share one predecessor-cost load (K2/K3). One lever per commit, all bit-exact with the CUDA oracle (`tests/test_conv_viterbi.py`, K1-4) and with the previous kernel on random / tiny / huge / flat tiles (K2-5).

| tiles/s, Air M5, AC, LPM off (`tools/conv_bench.py`, interleaved, best of 5) | K2 | K3 | K4 | K5 |
|---|---|---|---|---|
| impl 1 (device-memory costs, table decode) | 5.1 k | 4.6 k | 5.5 k | 5.8 k |
| impl 2 (registers + threadgroup costs, S pairs per load) | 11.6 k | 13.7 k | 13.6 k | 11.0 k |

2.3-3.0x. Remaining cost: the single-thread traceback and history stores are ~20-25 % of the kernel (K3 without them: 15.3 k). The kernel is now ALU / threadgroup-load bound; K6 (head) is slower in registers (64 values per thread), so it stays on impl 1. Packed integer min instead of compare / select: no gain (tried, reverted).
End to end (Qwen3.5-0.8B, 2 blocks, 16 x 256 rows, same seeds): trellis tensors identical between impl 1 and impl 2, 54.6 s -> 21.8 s (ldlq 37.3 -> 14.0 s, regularize 15.4 -> 6.0 s). Full 0.8B estimate: Viterbi 1036 s -> ~345 s; the rest (capture, finalize, advance, pack ~120 s) is now ~25 % of the Air run.

**MoE capture (`tools/conv_mlx.py`, Qwen3.5-MoE family, `tools/make_tiny_moe.py`, `tests/test_conv_moe.py`).** Per block the MoE input is tapped; experts are quantised as `mlp.experts.{e}.{gate,up,down}_proj` (exllamav3 naming), the shared expert as a dense MLP (`--k-mlp`), router and shared-expert gate copied.
* `--expert-mode all` (default): every expert sees all calibration tokens, so `H_in` is one matrix: one Cholesky for the layer, and the gate / up matrices of `--expert-batch` experts run as ONE LDLQ (thousands of tiles per Viterbi call). `H_down` is per expert (its own activations).
* `--expert-mode routed`: expert-wise Hessians from the tokens routed to it (same softmax / top-k as the model); experts under `--min-expert-tokens` fall back to all tokens (counted in the log).
* `--k-expert` (default 2) or `--strategy` per-name K. Both checkpoint layouts (fused `experts.gate_up_proj` / `down_proj`, per-expert keys). The advance forward uses the dequantised expert stacks.
* Checked on a random tiny MoE (4 blocks, 8 experts, top-2): expert Hessians equal a numpy brute force; every expert tensor present; packs decode (independent numpy decoder) to the source weights within K3 noise; fused + per-expert layouts, both modes. Not checked: a real MoE checkpoint (none small enough on the Air; DS4.1 FP8 / FP4 source needs its own dequant), quality vs the CUDA converter on MoE.

**Time plan (Viterbi tiles, 2.7-3.2e9 for DS4.1 experts incl. scale search; Ultra rate is an ESTIMATE = 8 x Air K2/K3 12.6 k = 100 k tiles/s, to measure day 1 with `conv_bench.py --K 2 3`; 3090 35 k assumed):**

| lanes | tiles/s | Viterbi time |
|---|---|---|
| Ultra alone | 100 k (75 k if scaling is 6x) | 7.4-8.8 h (9.9-11.8 h) |
| Ultra + Strix | 132 k | 5.7-6.7 h |
| Ultra + Strix + 3090 | 168 k (142 k) | 4.5-5.3 h (5.3-6.2 h) |

Adds the capture / finalize / advance chain (~25 % of the Air run, less on the Ultra relative to its Viterbi if the Viterbi scales 8x but the CPU Cholesky does not): the <= 5 h target needs all three lanes or a Ultra rate above 8x Air; FP prefix lets capture overlap the workers.

## Drill: bpw ladder, tokenizer-generic calibration, head lever (macair50)
* `tools/drill_mlx.py --hf ORG/MODEL --work DIR`: fetch -> calib -> ladder -> qualify. Default recipe for a small dense model: attention (full + GatedDeltaNet projections) K5, MLP K4, head K6 (about 4.1-4.5 bpw on the blocks, never 2).
* Ladder: every rung = ONE lock job (`tools/drill_rung.py`: convert with `--cache`, eval docs, BF16 reference once, parity). Gate = qualify's (mean KL <= 0.15, top-1 >= 0.75 per domain). On failure `next_cells` gives +1 bit to the (block x attention / MLP) cells with the highest measured local error per parameter (`conv_sens.json`: block output error with only that cell quantised, FP input) until `--step-frac` (0.25) of the block parameters is covered; stops at PASS, `--max-bpw` (5.0), `--k-max` 6, `--max-rungs`, or a plateau (< 5 % KL gain). `--cache` implies `--fp-prefix` (blocks independent: FP-prefix vs quantised-prefix KL 0.2654 vs 0.2680, eval_summary.txt), so a rung only quantises the blocks whose K changed; the head (47 % of a first convert) is loaded. The pack's `quantization_config.bits` is the real mean K of the blocks.
* Calibration: a raw calibration text file passed with `--text` (not shipped) (raw text: tool JSON conversations fr/en, code, wiki/c4, fr, multilingual, technical, chat; the spans read by the parity docs are excluded) -> `tools/calib_text.py tokens SRC --rows 128 --cols 1024` tokenizes with the model's own tokenizer and chat template (JSON fallback when it has none); a quarter of the rows are packed across domains. No token ids are stored.
* Head lever: the global-scale search of the head quantised ~52 K sampled tiles x 15 scales (40 % of the head time); `--head-sample-width 1` (was 3) cuts it to a third: head 320 s -> 247 s on the Air, scale +0.3 %, proxy unchanged. The K6 Viterbi itself is the floor (5.1 k tiles/s, `tools/head_bench.py`: NT, table, impl 2, chunk size all measured; chunk 512 for K6 +5 %).
