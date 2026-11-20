# Kyojin MLX

Kyojin for Apple Silicon. It runs large language models on a Mac with MLX and custom Metal kernels.
It serves an OpenAI-compatible API with continuous batching, prefix sharing, chunked prefill and speculative decode.

For AMD and NVIDIA GPUs see https://github.com/Yamz-Labs/kyojin

## Formats and models

- GGUF (llama.cpp quantisations, for example Q4_K_M and UD-Q4_K_XL)
- EXL3 (trellis-coded quantisation)
- MLX packs, including NVFP4

Tested families: DeepSeek V4.1 (Flash), GLM-5.3, Qwen3.5 and Qwen3.8 (dense and MoE).
Models are not in this repo. You bring the weights.

## Install

Python 3.11 or newer, Apple Silicon, macOS with Metal.

    git clone https://github.com/Yamz-Labs/kyojin-mac.git
    cd kyojin-mac
    python3 -m venv .venv && . .venv/bin/activate
    pip install -e .

DeepSeek and GLM packs need `mlx-vlm` (pinned to 0.7.12 in `pyproject.toml`).

## Serve

    python -m kyojin_mlx.serve.server /path/to/model --sessions 4 --ctx 32768 --port 8080

or `tools/mac_serve.sh /path/to/model ...` (same options, keeps the Mac awake).

    curl localhost:8080/v1/chat/completions -H 'content-type: application/json' \
         -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'

Options derive from the pack. `--help` lists them all. Details: `docs/SERVER.md`.
To add a model see `docs/ADD_A_MODEL.md` and `docs/NEW_MODEL.md`. To convert to EXL3 see `CONVERT.md`.

## Hy4-flash (GGUF)

Hy4-flash runs from the public Q3_K GGUF (2 files, 144 GiB) published by AngelSlim: [AngelSlim/Hy4-flash-GGUF](https://huggingface.co/AngelSlim/Hy4-flash-GGUF). The model keeps the publisher's licence.
Tested on a Mac Studio with 256 GB of unified memory.

    pip install -U "huggingface_hub[cli]"
    hf download AngelSlim/Hy4-flash-GGUF --include "Hy4-flash-Q3_K-*.gguf" --local-dir ./hy4-flash
    set -a; . tools/hy4_mac.env; set +a
    python -m kyojin_mlx.serve.server --model ./hy4-flash --port 8080 --model-id Hy4-flash \
        --sessions 1 --ctx 262144 --kv-mode fp16 --no-kv-disk --prefill-chunk 2048 --mtp-depth 3 --cache-limit-mb 16384

`tools/hy4_mac.env` holds the kernel flags of the measured line. The prefill speed depends on the MLX buffer cache limit: `--cache-limit-mb 16384` is the value of the measured line (without the flag the limit is derived from the weights and the free memory). Speculative decoding uses the model's own multi-token-prediction head (`--mtp-depth 3`).

Memory guard: `KYOJIN_MEM_BUDGET_GIB` (default 0.88 x physical RAM) caps the process. `KYOJIN_MEM_BUDGET_GIB=0` switches the guard off. A prompt whose projected memory is over the budget gets HTTP 413. The projection adds a slope per 1K prompt tokens to what the process holds: the Hy4 backend uses 0.12 GiB, the other backends use no slope (their prompts are not refused by length). `KYOJIN_MEM_GIB_PER_1K` sets the slope on every backend. A request that reaches the budget while running is stopped after the MLX cache and the prefix snapshots are dropped, so the server stays up.
Speed numbers: see the benchmark table.

- Context: 256K tokens (`--ctx 262144`). Prompts of 64K, 128K and 176K tokens are served (see "Long context" below). 1M tokens is not part of this release.
- One request decodes at a time; other requests queue. Batched multi-user decode of Hy4 is not in this release.
- The Engram tables (59 GiB) stay on disk and are read by row (parallel preads with a background prefetch, `KYOJIN_HY4_ENGRAM_PREAD` and `KYOJIN_HY4_ENGRAM_DECPF` in `tools/hy4_mac.env`); nothing is loaded or pinned in memory. The first request after a start is slower while rows are read. The read helper `tools/rowread/rowread.c` is compiled with `clang` on first use; without a compiler the rows are read through the memory map (same results, slower on a cold table).
- One model and one quantization (Q3_K). The model row is marked unproven in the loader, which is why `tools/hy4_mac.env` sets `KYOJIN_SKELETON=1` and `KYOJIN_KIT_UNPROVEN_ROWS=1`.
- Tests: `tests/test_hy4_*.py`, `tests/test_skeleton_hy4_*.py` and `tests/test_engram_hy4.py` (run one file at a time).

### Decensor spec (optional)

The server can remove a refusal direction from the Hy4-flash hidden stream at run time, without editing or re-quantising the weights. It is off by default. To arm it, point `KYOJIN_ABLIT_SPEC` at a spec directory before the serve command:

    export KYOJIN_ABLIT_SPEC=/path/to/spec_dir

The directory holds `edit_spec.json` (hidden size 3072, 42 layers, per-layer weights, `"r_per_layer": true`) and `edit_spec.safetensors` (tensor `r`, float32, shape `[42, 3072]`, one direction per layer). The path of `edit_spec.json` itself also works. A spec whose hidden size differs from the model is refused at load. To go back to the measured line, unset the variable.

- The hook edits the attention output (after its post norm) and the MLP / MoE output (after its post norm) of the trunk layers whose weight is not 0. The speculative draft head is never edited.
- Blocks with an armed hook run the unfused sub-layer path, so decode is slower than the unarmed line. The compiled decode step and the fused hyper-connection kernels are not armed.

A spec ships with the repository in `specs/hy4/` (off by default). To arm it, one line before the serve command:

    export KYOJIN_ABLIT_SPEC=specs/hy4

The sha256 of both files is listed in `specs/hy4/SHA256SUMS`. Measured cost of the armed hook on the Mac Studio (4K context, speculative decode, depth 3): about 20 to 30 % lower decode speed than the line in the table above, because armed blocks run the unfused path (see the second note above). Reasoning answers also get much longer on the refusal-style test prompts (in our check 4 of 17 answers reached the 4096-token limit and 3 came back empty). Treat the armed hook as experimental: it is not part of the measured table, and a faster fused version is in progress.

## Speed

DeepSeek and Qwen rows: Kyojin 1.8.9. Hy4-flash rows: Kyojin 1.10.3, see the Hy4-flash table below. Mac Studio M5 Ultra, 256 GB. Served through the real server, real text, tokens per second.
One warm-up discarded, then one measured run.

| Model and cell | tok/s |
|---|---|
| DeepSeek V4.1 Flash, prefill 8K | 3001 |
| DeepSeek V4.1 Flash, decode at 8K | 62 |
| DeepSeek V4.1 Flash, prefill 128K | 2681-2768 |
| DeepSeek V4.1 Flash, decode at 128K | 57 |
| DeepSeek V4.1 Flash, prefill at 983K | 2199-2291 |
| DeepSeek V4.1 Flash, decode at 983K | 48-49 |
| DeepSeek V4.1 Flash, 4 sessions, 8K, aggregate decode | 112 |
| Hy4-flash Q3_K (GGUF), decode at 4K to 128K, against llama.cpp | see the Hy4-flash table below |
| Qwen 35B-A3B UD-Q4_K_XL (GGUF), prefill 8K | ~7000 |
| Qwen 35B-A3B UD-Q4_K_XL (GGUF), prefill 32K | 6304 |
| Qwen 35B-A3B UD-Q4_K_XL (GGUF), decode at 1K | 185 |
| Qwen 35B-A3B UD-Q4_K_XL (GGUF), decode at 8K | 175 |
| Qwen 27B NVFP4, prefill 8K | 1577-1630 |
| Qwen 27B NVFP4, decode at 8K (no draft) | 41-43 |

Memory: one resident copy of the weights. The 35B GGUF is a 20.8 GiB file and uses 21.6-22.0 GB in MLX.
DeepSeek V4.1 uses 176-178 GB at 8K context and 187 GB at 983K.

### Hy4-flash Q3_K (GGUF) against llama.cpp

Decode speed in tokens per second, Mac Studio M5 Ultra 256 GB, same Q3_K GGUF, same fixed text, speculative decode with the model's own head on both sides (ours: `--mtp-depth 3`; llama.cpp: `draft-mtp`). Served through the real server, real text, median of repeated runs. "to follow" marks a cell whose paired release run is not in yet; the lead adds it.

| context | ours, speculative (depth 3) | ours, plain | llama.cpp, speculative | llama.cpp, plain |
|---|---|---|---|---|
| 4K | 64.20 | 44.38 | 50.92 | 48.66 |
| 16K | 56.88 | 42.42 | 57.17 | 48.85 |
| 24K | 55.58 | 41.14 | 52.78 | 48.92 |
| 32K | 46.08 | not measured | 50.4 | 48.7 |
| 64K | measured numbers to follow | measured numbers to follow | 52.69 | 48.72 |
| 128K | measured numbers to follow | measured numbers to follow | measured numbers to follow | measured numbers to follow |

Where llama.cpp leads:

- Speculative decode: ours is ahead at 4K (+26 %) and 24K (+5 %), level at 16K (57.17 against 56.88, llama.cpp is 0.5 % ahead), and behind at 32K (46.08 against 50.4). At 64K llama.cpp holds 52.69; our paired run is to follow.
- Plain decode (no draft): llama.cpp is faster at every context we measured (4K: 48.66 against 44.38; 16K: 48.85 against 42.42; 24K: 48.92 against 41.14). Our advantage comes from speculation, not from the plain step.
- Long context (1.10.x, Mac Studio M5 Ultra, 256 GB): 64K and 128K prompts are served. A 141K-token prompt was served at 128K context with passkey retrieval 3 of 3 (decode 31 tok/s on the leak-free lane stack). A 176K-token prompt was served with passkey 3 of 3 (decode 37 tok/s). The peak footprint in these runs was 143 GiB (`phys_footprint`). These are single served checks on one text, not the paired table run: the paired 64K and 128K cells are marked "measured numbers to follow".
- The speculative cells use the shipped serve line. Passkey retrieval passed 3 of 3 at 4K, 16K and 24K in both modes. The peak footprint is 150-151 GiB for a 144 GiB file (one resident copy of the weights).

## New in 1.10.3

Long-context and decode levers for Hy4-flash on the Mac, all in `tools/hy4_mac.env`. Measured numbers to follow.

- Q3_K expert decode (`KYOJIN_QDEC_FAST=1`): one fused decode and dot per 32-weight unit, without decoding the weights to floats first. The products match the old text up to float rounding. `KYOJIN_QDEC_FAST=0` restores the decode-to-float text.
- Absorb GEMV (`KYOJIN_ABSORB_GEMV=1`): the two MLA absorb products read one fp16 copy of `k_b` / `v_b` (the copy that the fp16 attention already makes, so there is one resident fp16 copy) in one launch per product. `KYOJIN_ABSORB_GEMV=0` runs the stock fp32 matmul.
- Long-context gather (`KYOJIN_HY4_FIXED_GATHER=16384`, `KYOJIN_HY4_FIXED_GATHER_DEC=8192`, `KYOJIN_HY4_FIXED_PKCACHE=1`): from that many key slots on, attention reads only the keys the indexer kept, and the pooled keys live in the cache state. Set each to `0` to opt out.
- The one-launch decode attention core keeps its row gate (`KYOJIN_HY4_ATTN_VEC_MIN_ROWS`, default 2): plain decode keeps the stock core, speculative verify rounds use the kernel.
- Fires log: the served log line `[flags 60s]` (one per minute of activity, plus one right after the first request settles) now ends with the lever totals, `| totals qdec_fast=N absorb_gemv=N fixed_gather=N`, and counts the kept-key gather per call (`hy4_gather fires/stock calls`, with the reasons `off`, `below_min`, `fp16_copy`) and the pooled-key cache (`hy4_pkcache`). A lever that is on but shows 0 did not run. `KYOJIN_HY4_FIRES_LOG=0` drops the totals; `KYOJIN_FLAG_AUDIT=0` drops the whole line.
- Code defaults are unchanged: every lever above is off unless the env file (or your own environment) turns it on.

## New in 1.10.2

Memory fix for long-running Hy4 servers (step compile of the fixed KV cache).

- Memory: the footprint no longer grows with earlier requests. The compiled layer programs were keyed on the per-request cache object and kept every finished request's KV buffers alive (about 0.11 GiB per 1K tokens of each earlier request); they are now keyed on shapes only and reused by every request. `KYOJIN_STEP_COMPILE_LEAK=1` restores the old key.

## New in 1.10.0

Hy4-flash (GGUF Q3_K) on a Mac Studio, up to 256K tokens of context. Numbers: see the benchmark table. Guide: the Hy4-flash section above.

What is new against 1.9.x:

- Hy4-flash (GGUF Q3_K) is served on the generic decoder skeleton, with speculative decode on the model's own head (MTP, depth 3).
- `attn_vec`: the MLA decode attention runs in one launch per layer (about 20 small launches before).
- Engram rows are read from disk by parallel pread, with a background prefetch during decode. The 59 GiB table is never loaded or pinned.
- Four kernels for the verify rows and the output head: `fmm8` (few-row Q8_0 matrix product), `head_q` (shared packed 8-bit output head), `nbr_small` (small 8-bit GEMVs), `moe_route_sel` (fused router selection).
- Context: 256K tokens are configured. 64K, 128K and 176K-token prompts are served (see "Long context" in the Hy4-flash table notes); the paired decode cells at 64K and 128K are to follow.

Run it (copy and paste; needs the install steps above and about 150 GiB of free memory):

    pip install -U "huggingface_hub[cli]"
    hf download AngelSlim/Hy4-flash-GGUF --include "Hy4-flash-Q3_K-*.gguf" --local-dir ./hy4-flash
    set -a; . tools/hy4_mac.env; set +a
    python -m kyojin_mlx.serve.server --model ./hy4-flash --port 8080 --model-id Hy4-flash \
        --sessions 1 --ctx 262144 --kv-mode fp16 --no-kv-disk --prefill-chunk 2048 --mtp-depth 3 --cache-limit-mb 16384

| change | effect | off |
|---|---|---|
| Hy4-flash serve backend on the generic decoder skeleton: prefill, decode, Engram rows read from the GGUF by row | serves the public Q3_K GGUF | - |
| multi-token-prediction speculation with the model's own head, depth 3 (depth 5 measured slower at 4K, 16K and 24K) | `--mtp-depth 3` | `--mtp-depth 0` |
| optional gather attention over the kept keys with pooled keys carried in the cache state (same speed and memory as the default path up to 40K tokens) | long context | off by default in the code; on in `tools/hy4_mac.env` since 1.10.3: `KYOJIN_HY4_FIXED_GATHER=16384`, `KYOJIN_HY4_FIXED_GATHER_DEC=8192`, `KYOJIN_HY4_FIXED_PKCACHE=1` |
| fused attention, hyper-connection and expert kernels for the skeleton (`tools/hy4_mac.env`) | decode and prefill speed | unset the flag |
| optional hierarchical candidate-restricted indexer for prompt chunks | `KYOJIN_HY4_HISA` (default off) | - |
| few-row Q8_0 simdgroup-matrix product for the verify rows (2 to 8 rows) of the dense Q8_0 linears | Hy4-flash speculative decode, the four 8-bit and router levers (FMM8, ROUTE_SEL, HEAD_Q, NBR_SMALL) measured together, no single-lever served number: 58.50 -> 60.94 tok/s at 4K (+4.2 %), 42.06 -> 42.70 at 16K (+1.5 %), 39.14 -> 40.21 at 24K (+2.7 %), speculative depth 3, same prompt repeated (bundle) | `KYOJIN_FOREIGN_FMM8=0` |
| fused MoE router selection: gating, bias, group limit, top-k, normalisation and scale in one launch | Hy4-flash speculative decode, the four 8-bit and router levers (FMM8, ROUTE_SEL, HEAD_Q, NBR_SMALL) measured together, no single-lever served number: 58.50 -> 60.94 tok/s at 4K (+4.2 %), 42.06 -> 42.70 at 16K (+1.5 %), 39.14 -> 40.21 at 24K (+2.7 %), speculative depth 3, same prompt repeated (bundle) | `KYOJIN_MOE_ROUTE_SEL=0` |
| output head of 1 to 16 row passes through one shared packed 8-bit copy | Hy4-flash speculative decode, the four 8-bit and router levers (FMM8, ROUTE_SEL, HEAD_Q, NBR_SMALL) measured together, no single-lever served number: 58.50 -> 60.94 tok/s at 4K (+4.2 %), 42.06 -> 42.70 at 16K (+1.5 %), 39.14 -> 40.21 at 24K (+2.7 %), speculative depth 3, same prompt repeated (bundle) | `KYOJIN_HEAD_Q=0` |
| one output row per simdgroup on small 8-bit GEMVs (twice the threadgroups) | Hy4-flash decode, output bit-equal; speed measured only inside the bundle of the four levers above (+4.2 % at 4K, +1.5 % at 16K, +2.7 % at 24K, speculative depth 3) | `KYOJIN_FOREIGN_NBR_SMALL=0` |
| one-launch flash-attention core for the MLA decode attention (windowed layers and pooled-key layers, 1 to 8 rows; replaces about 20 small launches per layer) | Hy4-flash speculative decode, depth 3, same prompt repeated: 60.94 -> 66.58 tok/s at 4K (+9 %), 42.70 -> 52.29 at 16K (+22 %), 40.21 -> 51.66 at 24K (+28 %), 31.50 -> 43.03 at 28K (+37 %); on its own it slows one-row plain decode by 8 % at 4K, 16 % at 16K and 20 % at 24K (see next row) | `KYOJIN_HY4_ATTN_VEC=0` |
| plain decode (one row) keeps the stock attention core; speculative verify rows (2 to 8) use the one-launch core | Hy4-flash plain decode keeps its speed with the gate: 44.66 / 43.14 / 40.33 tok/s at 4K / 16K / 24K (44.70 / 43.16 / 40.51 without the one-launch core), against 41.3 / 36.1 / 32.5 without the gate; speculative speed is unchanged | `KYOJIN_HY4_ATTN_VEC_MIN_ROWS=1` |
| Engram rows read by deduplicated parallel preads instead of one page-in per row from the memory map (the table stays on disk: no warm-up, no pinning) | Hy4-flash first request and prompt chunks: 8K prefill 2016 -> 2068 tok/s (+2.6 %, loss bit-equal); decode on new text at 4K, speculative depth 3: 55.0 -> 49.7 ms per round (44.6 -> 50.7 tok/s), at 24K 73.6 -> 71.3 ms (33.8 -> 36.7 tok/s) | `KYOJIN_HY4_ENGRAM_PREAD=0` |
| Engram rows of a speculative round read in the background while the draft head runs (rows stay on disk, only the cold rows are read) | Hy4-flash speculative decode on new text, depth 3: on top of the pread row, 49.7 -> 45.7 ms per round at 4K (-4.0 ms) and 71.3 -> 68.5 ms at 24K; with the pread row together 49.0 -> 41.1 ms per round at 4K (51.7 -> 60.8 tok/s) and 55.6 -> 50.9 ms at 24K (47.2 -> 49.6 tok/s); no change when the prompt repeats (rows already warm) | `KYOJIN_HY4_ENGRAM_DECPF=0` |

- Hy4 runs one session at a time; extra requests queue.

## New in 1.9.8

Faster decode for several users on dense models; FP8 packs with any block size load. Measured on a Mac Studio.

| change | effect | off |
|---|---|---|
| users without a shared prefix decode their attention in one pooled kernel instead of one kernel per user | 8 users on Qwen3-8B UD-IQ3_XXS GGUF: N8 646-649 -> 701-704 tok/s (+8 %), N8 with 8 different prompts 522-524 -> 548-551 (+5 %) | `KYOJIN_MU_POOL_ZERO=0` |
| FP8 packs dequantize with their own `weight_block_size` (mlx-lm assumed 128); the model class keeps its own `sanitize` after the load | FP8 DeepSeek-V3 packs with another block size load and serve | - |

- Same session as llama.cpp Metal (`-np 8`) on the 8B: N8 701-704 vs 677-703 tok/s.
- Release gate against 1.9.7 (4 passes on the 8B, 2 on the 35B, swapped order, real server, temperature 0, md5 per prompt): same outputs on
  Qwen3-8B UD-IQ3_XXS GGUF and Qwen 35B-A3B UD-Q4_K_XL GGUF; 1-user speed and prefill 8K unchanged; the 35B (MoE) multi-user cells unchanged.

## New in 1.9.7

Qwen3 MoE GGUF files load. Measured on a Mac Studio.

| change | effect | off |
|---|---|---|
| GGUF architecture `qwen3moe` maps to mlx-lm's `qwen3_moe` model (it named a `qwen3moe` module that does not exist) | Qwen3-30B-A3B and other Qwen3 MoE GGUF files load and serve | - |
| the cached serve directory of a GGUF file is rebuilt once (key v6) | a machine that tried such a file before gets the corrected config | - |

- Qwen3-30B-A3B-Instruct-2507 Q8_0 GGUF, 8 sessions, same session as llama.cpp Metal (`-np 8`), tok/s: N1 160-162 vs 138-139, N4 346-361 vs
  292, N8 421-471 vs 388, N8 with 8 different prompts 319-333 vs 316-317, prefill 8K 2.02-2.23 s vs 3.28 s.
- Release gate against 1.9.6 (2 passes in swapped order, real server, temperature 0, md5 per prompt): Qwen 35B-A3B UD-Q4_K_XL GGUF same
  outputs and same speed for 1, 4 and 8 users; the Qwen3-30B-A3B Q8_0 GGUF loads (1.9.6 refused it) with stable outputs.

## New in 1.9.6

Lower load memory for GGUF K-quant models. Measured on a Mac Studio.

| change | effect | off |
|---|---|---|
| big GGUF K-quant tensors are repacked in row slabs of 4M weights (the bytes are the same as the whole-tensor repack for all 18 types) | the host temporaries of the repack (4-16 bytes per weight, ~8 GiB on a 150K-vocabulary head) are gone: Qwen3-8B Q4_K_M load 13.0 -> 6.4 GiB, Qwen 27B Q4_K_M 23-26 -> 19-20 GB after load | - |
| the draft (MTP) head of a GGUF is not decoded when MTP is off | less memory and a faster load with MTP off | - |

- Release gate against 1.9.5 (2 passes in swapped order, real server, temperature 0, md5 per prompt, memory after load, at the end
  and at the peak): Qwen 35B-A3B UD-Q4_K_XL GGUF, Qwen 27B NVFP4 and Qwen 27B Q4_K_M GGUF (MTP on and off) same outputs and same speed
  for 1, 4 and 8 users; steady and peak memory not higher; DeepSeek V4.1 Flash same speed.

## New in 1.9.5

mlx-vlm 0.7.12 architectures. Measured on a Mac Studio.

Built on mlx-vlm 0.7.12 (was 0.7.10; mlx-lm 0.32.2 unchanged).

| change | effect | off |
|---|---|---|
| the generic backend takes the per-user caches from the language model on every path (a vision wrapper without its own cache factory included) | Qwen 4 experimental (`qwen4_exp`) decodes instead of failing on the first step (checked on a tiny random pack; no full pack fits the disk yet); its draft head is not used (speculative depth 0) | - |
| the TurboQuant attention wrapper forwards extra positional arguments | LFM3 serves: LFM3-1.7B 4-bit coherent, tool call parsed, 1 user 168.6 tok/s, 4 users 507 tok/s | - |
| the tokenizer loads with `trust_remote_code=False` | a pack with custom tokenizer code no longer waits for a yes/no answer on the terminal | - |

- Release gate against fleet-1.9.4 (3 passes, real server, temperature 0, md5 per prompt): Qwen 35B-A3B UD-Q4_K_XL GGUF and Qwen 27B NVFP4 same
  outputs and same speed; DeepSeek V4.1 Flash same outputs at 131K and 983K context; Qwen3.5-2B 4-bit and gpt-oss-20b same outputs.

## New in 1.9.4

Any architecture, served by default. Measured on a Mac Studio.

A model type without its own row is now served by default through the generic backend (the stock mlx-lm / mlx-vlm model class with the
engine's per-user caches and multi-user batching) and the Jinja chat family (the pack's own chat template). Before 1.9.4 it needed
`KYOJIN_BACKEND_FALLBACK=1` / `KYOJIN_CHAT_FALLBACK=1`; models that have a row take the same path as in 1.9.3 (same outputs).

| change | effect | off |
|---|---|---|
| generic backend + Jinja chat family on by default; the per-user attention hook covers every attention module; mlx-lm weight remap on the generic load; plain bf16 packs (no quantization config) load | Ministral 3 3B, Phi-4-mini, Llama 3.2 1B bf16, SmolLM3 3B, Gemma 3 270M bf16, Gemma 4 E2B, Nemotron 3 Nano 4B bf16 serve (1 user 95 to 572 tok/s, 4 users 357 to 1064); 4-bit packs give the same first 96 greedy tokens as stock mlx_lm | `KYOJIN_BACKEND_FALLBACK=0`, `KYOJIN_CHAT_FALLBACK=0` |
| caches the code does not list (a model's own rotating cache, cache lists, state-only caches) fork and persist through their own state | Gemma / SSM caches serve several users | `KYOJIN_GENERIC_CACHE=0` |
| TurboQuant KV keeps a plain cache on layers whose head size has no codebook | Gemma 4 (global layers of head size 512) no longer fails on the first token | - |
| tool calls of the Jinja family: Mistral `[TOOL_CALLS]`, Llama `<\|python_tag\|>`, Gemma 4 `<\|tool_call>`; `tools` passed as none when the request has none; a thinking block the model opens itself goes to `reasoning` | tool calls parsed on Ministral 3, Llama 3.2, Gemma 4, Nemotron 3 Nano | - |

- Release gate against fleet-1.9.3 (real server, temperature 0, 3 fixed prompts, md5 per prompt): Qwen 35B-A3B UD-Q4_K_XL GGUF and Qwen 27B NVFP4
  same outputs and same speed (3 passes on the 27B), DeepSeek V4.1 Flash same outputs at 131K and 983K context; gpt-oss-20b same outputs.
- bf16 Gemma 3 / Nemotron outputs drift from stock mlx_lm after 20 to 70 tokens (different attention accumulation order); outputs stay coherent.

## New in 1.9.3

MLX-format checkpoints, 8-bit linears with several rows, day-one rows. Measured on a Mac Studio through the real server (temperature 0, 3 fixed
prompts, one warm-up discarded, 2 passes in swapped order), against 1.9.2:

| change | effect (served) | off |
|---|---|---|
| MLX-format quantized Qwen3 directories (mlx-community affine 4-bit, MXFP4, NVFP4; dense and MoE) load as a plain Qwen backend | same files against mlx_lm.server, 1 user: Qwen3-8B 4-bit 177.3 vs 152.8 tok/s, MXFP4 183.6 vs 160.4, NVFP4 176.6 vs 148.0, Qwen3-30B-A3B 4-bit 170.9 vs 130.6; prefill still 2 to 19 % slower | - |
| Q8_0 / FP8 matrices of at least 16M weights take the wide product from 5 rows | Qwen3.6-27B FP8, 8 users 101.8 -> 149.6, 8 different prompts 92.3 -> 125.5; Qwen 35B-A3B UD-Q4_K_XL (Q8_0 attention), 8 users 530 -> 562, 1 user equal | `KYOJIN_FOREIGN_WIDE_MIN=9` |
| ERNIE 4.5 MoE row (generic backend, Jinja chat family); a vision wrapper without its own cache factory serves with its language model's caches | loads instead of refused | - |

- The wide product changes the summation order on 5 to 8-row steps (speculative verify, several users): greedy outputs of models with large 8-bit
  matrices can differ from 1.9.2 after a few dozen tokens. Qwen 35B-A3B UD-Q4_K_XL: GSM8K (200 problems) 93.0 % (1.9.2) vs 93.5 %, perplexity equal.
- DeepSeek V4.1 Flash: unchanged (same outputs at 131K and 983K context). Qwen 27B NVFP4: same outputs, 1 user +12 %.

## New in 1.9.2

Foreign checkpoints (GGUF, NVFP4, FP8, MXFP4) and several users at once. Measured on a Mac Studio through the real server
(temperature 0, 3 fixed prompts, one warm-up discarded), against 1.9.1. Each change can be turned off:

| change | effect (served) | off |
|---|---|---|
| fused TurboQuant decode step, plain and per user | Qwen3-8B UD-IQ3_XXS, 1 user 122 -> 139 tok/s, 4 users +47 % | `KYOJIN_TQ_STEP=0` |
| KV format by depth: a session whose prompt fits in 4096 tokens keeps a plain fp16 cache and turns TurboQuant when it crosses 4096 (between two decode steps); a longer prompt starts in TurboQuant | Qwen3-8B, clients that send no `max_tokens`: 4 users 265 -> 320, 8 users 256 -> 540, 8 different prompts 208 -> 394; decode at 32K stays TurboQuant (82 vs 61 tok/s fp16) | `KYOJIN_TQ_FROM=0` |
| `--sessions` derived from memory (one session's KV at 4096 tokens measured after load, floor 4, cap 8; an explicit value wins) | Qwen3-8B: 8 sessions, 8 users ~620 tok/s | `--sessions 4` |
| no shared-prefix pool below 256 tokens (a prefix made of the chat template alone made 8 different users slower) | Qwen 35B-A3B UD-Q4_K_XL, 8 users 371 -> 469, 8 different prompts 320 -> 407; Qwen 27B NVFP4, 8 users 219 -> 257 | `KYOJIN_MU_POOL_MIN=0` |
| MLX buffer cache floor capped at 2 GiB (was working set / 32) | memory gpt-oss-20b 22 -> 16 GB, Qwen 35B-A3B 33 -> 28 GB, Qwen 27B NVFP4 35 -> 29 GB, Qwen3-30B-A3B FP8 43 -> 38 GB; speed equal | `KYOJIN_CACHE_FLOOR_MAX_MB=1000000` |

- Per model, 1 user / 8 users: Qwen3-30B-A3B FP8 97 -> 129 / 161 -> 300; Qwen 35B-A3B UD-Q4_K_XL +6 to 11 % / +27 %; Qwen 27B NVFP4 +2 to 7 % / +17 %;
  gpt-oss-20b equal. DeepSeek V4.1 Flash: unchanged (it keeps 4 sessions).
- The depth rule and the in-session conversion apply to the mlx-lm backends (Qwen family, GLM, gpt-oss); the DeepSeek V4.1 path is unchanged. The conversion
  ends one decode epoch (all sessions rebatch once).

## New in 1.9.1

Sampled sessions (temperature > 0) are what agents send. These flags are runtime (`POST /admin/flags`) and read from `KYOJIN_*` at
start. The code defaults stay off: turn them on per pack after an A/B and a quality check of your own. Numbers are DeepSeek V4.1
Flash on a Mac Studio, T=1, top_p 0.95, a replay of 14 real agent turns.

| env | value | kind | effect |
|---|---|---|---|
| `KYOJIN_MOE_CHAIN` | 1 | exact | routed MoE decode in 3 launches per layer: +2-3 % decode |
| `KYOJIN_DENSE_DEC` | 1 | exact | dense trellis GEMV lane sum in one multiply-add: +1.5 % decode |
| `KYOJIN_ENGRAM_ADOPT` | 1 | exact | Engram row jobs of the first chunk hashed at admission: -1 to -4 % turn latency |
| `KYOJIN_PF_CUT_MERGE` | 3 | rounding | last chat cut and generation prompt in one prefill pass: first token 659 -> 574 ms on two-pass turns |
| `KYOJIN_SPEC_TYPICAL` | 1 | **not exact** | a draft is also accepted when the target gives it enough probability: 62 -> 75 tok/s |
| `KYOJIN_SPEC_TYPICAL_PHASE` | answer | exact inside thinking | the relaxed rule only after the think block closes: about 10 % slower than `all`, no reasoning loops in our probes |
| `KYOJIN_DSPARK_CONF_PCT` | 40 | changes the drafts only | draft confidence gate at 40 %: +4.7 % on top |

- Greedy sessions are untouched: they stay bit-identical to plain decode.
- Under the relaxed rule everywhere (`phase=all`), long sampled reasoning can drift into a repeating span. Use `phase=answer` for
  agents. The server also has a loop guard (now active on prompts with images), per-request `kyojin_flags` and
  `continue_final_message` on a reasoning prefix, so a client that cuts a loop can resume from before it.
- Measured and closed without a gain on this model: Gumbel-coupled drafts (full or truncated support), a power-law acceptance rule
  inside thinking, rejection sampling from the draft head. The draft head is a block predictor: its argmax is good, its distribution
  is not close enough to the target for exact sampled-draft schemes to pay.
- Where the time goes: a decode step is bound by the EXL3 trellis decode (ALU), not by bandwidth or launches.

## Tests

    python -m pytest tests

Some tests need a calibration text (`KYOJIN_CALIB_TEXT`) or a model; they skip without it.

## Licence

Apache-2.0, see `LICENSE`. Third-party code and tables: `THIRD_PARTY_NOTICES.md`.
