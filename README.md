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

## Tests

    python -m pytest tests

Some tests need a calibration text (`KYOJIN_CALIB_TEXT`) or a model; they skip without it.

## Licence

Apache-2.0, see `LICENSE`. Third-party code and tables: `THIRD_PARTY_NOTICES.md`.
