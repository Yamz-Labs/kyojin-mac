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

    git clone <this repo> kyojin-mlx && cd kyojin-mlx
    python3 -m venv .venv && . .venv/bin/activate
    pip install -e .

DeepSeek and GLM packs need `mlx-vlm` (pinned to 0.7.10 in `pyproject.toml`).

## Serve

    python -m kyojin_mlx.serve.server /path/to/model --sessions 4 --ctx 32768 --port 8080

or `tools/mac_serve.sh /path/to/model ...` (same options, keeps the Mac awake).

    curl localhost:8080/v1/chat/completions -H 'content-type: application/json' \
         -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'

Options derive from the pack. `--help` lists them all. Details: `docs/SERVER.md`.
To add a model see `docs/ADD_A_MODEL.md` and `docs/NEW_MODEL.md`. To convert to EXL3 see `CONVERT.md`.

## Speed

Kyojin 1.8.9 on a Mac Studio M3 Ultra, 256 GB. Served through the real server, real text, tokens per second.
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
| Qwen 35B-A3B UD-Q4_K_XL (GGUF), prefill 8K | ~7000 |
| Qwen 35B-A3B UD-Q4_K_XL (GGUF), prefill 32K | 6304 |
| Qwen 35B-A3B UD-Q4_K_XL (GGUF), decode at 1K | 185 |
| Qwen 35B-A3B UD-Q4_K_XL (GGUF), decode at 8K | 175 |
| Qwen 27B NVFP4, prefill 8K | 1577-1630 |
| Qwen 27B NVFP4, decode at 8K (no draft) | 41-43 |

Memory: one resident copy of the weights. The 35B GGUF is a 20.8 GiB file and uses 21.6-22.0 GB in MLX.
DeepSeek V4.1 uses 176-178 GB at 8K context and 187 GB at 983K.

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
