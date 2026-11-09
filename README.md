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
