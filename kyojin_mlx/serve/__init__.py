# SPDX-License-Identifier: Apache-2.0
"""Multi-session OpenAI-compatible server for kyojin-mlx (continuous batching, prefix sharing, chunked prefill).

    backends.py   model adapters (Qwen3.5 EXL3 through multiuser.py; DS4.1 / GLM through rows.py)
    engine.py     the scheduler: one thread owns every MLX call, sessions join and leave the batched step between steps
    sampling.py   per-session sampling on batched logits
    chat.py       chat templates and reply parsing (reasoning, tool calls) per model family
    server.py     the HTTP layer (stdlib), /v1/chat/completions, /v1/completions, /v1/models, /health
"""
