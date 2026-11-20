#!/usr/bin/env python3
"""Logits of the first N prompt tokens of each oracle chunk through one loader, for a loader-to-loader comparison (the greedy tool only keeps tokens).
usage: python tools/archkit_logits_dump.py cpu|metal|metal-bf16 GGUF REF.bin OUT.npz [N=32]
Stores per prompt: `last` (V logits at the last position, fp32), `argmax` (N), `gap` (N, top1 - top2 logit).  Compare with tools/archkit_logits_cmp.py."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

mode, gguf, refp, outp = sys.argv[1:5]
n = int(sys.argv[5]) if len(sys.argv) > 5 else 32
if mode == "cpu":
    mx.set_default_device(mx.cpu)
import archkit_logit_check as L  # noqa: E402

_, _, toks, _, _ = L.read_ref(refp)
if mode == "cpu":
    from kyojin_mlx import foreign_archkit_cpu as C
    model, _ = C.load_cpu(gguf)
else:
    from kyojin_mlx import foreign_linear as FL
    model, _ = FL.load_foreign(gguf, dtype=mx.bfloat16 if mode.endswith("bf16") else mx.float16)
last, am, gap = [], [], []
for c in range(toks.shape[0]):
    lg = model(mx.array(toks[c][:n][None].astype(np.int32)))[0].astype(mx.float32)
    mx.eval(lg)
    a = np.array(lg)
    last.append(a[-1])
    am.append(a.argmax(-1))
    t2 = np.sort(a, -1)[:, -2:]
    gap.append(t2[:, 1] - t2[:, 0])
np.savez(outp, last=np.stack(last), argmax=np.stack(am), gap=np.stack(gap))
print("saved", outp, np.stack(last).shape)
