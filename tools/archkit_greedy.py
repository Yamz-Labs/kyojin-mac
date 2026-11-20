#!/usr/bin/env python3
"""Greedy token identity of the serve path (foreign_linear.load_foreign, Metal, stored kernels) against the CPU day-0 loader, on the oracle's prompts.
usage: python tools/archkit_greedy.py cpu|metal|map GGUF REF.bin OUT.json [N_PROMPT_TOKENS=32] [N_NEW=24]
cpu: foreign_archkit_cpu.load_cpu (fp32 dense). metal: foreign_linear.load_foreign (fp16 activations, stored K-quant kernels). map: dry name check, no pack built.
The prompt tokens are the first N_PROMPT_TOKENS of each chunk of the llama-perplexity --kl-divergence-base file (no tokenizer involved)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

mode, gguf, refp, outp = sys.argv[1:5]
npt = int(sys.argv[5]) if len(sys.argv) > 5 else 32
nnew = int(sys.argv[6]) if len(sys.argv) > 6 else 24
if mode in ("cpu", "map"):
    mx.set_default_device(mx.cpu)
if mode == "map":
    os.environ["KYOJIN_FOREIGN_MAPONLY"] = "1"
import archkit_logit_check as L  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

_, _, toks, _, _ = L.read_ref(refp)
if mode == "map":
    from kyojin_mlx import foreign_linear as FL
    r, cfg = FL.load_foreign(gguf)
    print(len(r["where"]), "linears mapped; fused", r["fused"], "dropped", r["dropped"], "\nconfig", cfg)
    sys.exit(0)
if mode == "cpu":
    from kyojin_mlx import foreign_archkit_cpu as C
    model, _ = C.load_cpu(gguf)
else:
    from kyojin_mlx import foreign_linear as FL
    model, _ = FL.load_foreign(gguf, dtype=mx.float16)
res = []
for c in range(toks.shape[0]):
    cache = make_prompt_cache(model)
    x = mx.array(toks[c][:npt][None].astype(np.int32))
    out, gaps = [], []
    for _ in range(nnew):
        lg = model(x, cache=cache)[0, -1].astype(mx.float32)
        top2 = mx.topk(lg, 2)
        t = int(mx.argmax(lg).item())
        out.append(t)
        gaps.append(round(float((top2[1] - top2[0]).item()), 4))   # top-1 minus top-2 logit
        x = mx.array([[t]], dtype=mx.int32)
    res.append(dict(prompt=toks[c][:npt].tolist(), tokens=out, gap=gaps))
    print(f"prompt {c}: {out}")
json.dump(res, open(outp, "w"))
