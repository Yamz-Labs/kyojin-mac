#!/usr/bin/env python3
"""Snapshot of llama.cpp's HF -> GGUF tensor-name map (gguf-py TensorNameMap) for the architectures kyojin_mlx.foreign_names maps.

    python tools/gen_gguf_names.py /path/to/llama.cpp/gguf-py <commit> > tests/data/gguf_names_llamacpp.json

For each architecture: {gguf tensor name of blocks 0 and 3 (no .weight/.bias) and of the non-block tensors: [every HF name llama.cpp maps to it]}.
tests/test_gguf_arch.py checks that each row of foreign_names.ARCHS lands on a name of that list, so the tables are not checked against themselves."""
import json
import sys

sys.path.insert(0, sys.argv[1])
import gguf  # noqa: E402
from gguf.constants import MODEL_ARCH_NAMES  # noqa: E402

ARCHS = ["qwen3", "qwen3moe", "qwen35", "qwen35moe", "qwen2", "qwen2moe", "llama", "mistral3", "mistral4", "gemma3", "gemma4", "glm4", "glm4moe", "deepseek2"]
inv = {v: k for k, v in MODEL_ARCH_NAMES.items()}
out = {"_commit": sys.argv[2] if len(sys.argv) > 2 else "unknown"}
for name in ARCHS:
    m = gguf.get_tensor_name_map(inv[name], 5)
    d = {}
    for hf, (_, g) in m.mapping.items():
        if g.startswith("blk.") and not (g.startswith("blk.0.") or g.startswith("blk.3.")):
            continue
        if hf.startswith(("model.", "lm_head")):                 # the HF layouts mlx-lm classes use (the table also lists other frameworks' names)
            d.setdefault(g, []).append(hf)
    out[name] = {k: sorted(v) for k, v in sorted(d.items())}
json.dump(out, sys.stdout, indent=0, sort_keys=True, ensure_ascii=False)
