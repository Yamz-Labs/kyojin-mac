#!/usr/bin/env python3
"""Our dequantization of a quantized GGUF vs the F32 GGUF that `llama-quantize --allow-requantize Q.gguf F.gguf F32` wrote (ggml's own to_float).
usage: python -I tools/archkit_dq_vs_file.py Q.gguf F32.gguf"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from kyojin_mlx import foreign_gguf as GF
q = GF.GgufFile(sys.argv[1]); f = GF.GgufFile(sys.argv[2])
per = {}
for n, t in q.tensors.items():
    d = float(np.abs(q.tensor_f32(n).astype(np.float32) - f.tensor_f32(n).astype(np.float32)).max())
    e = per.setdefault(t.tname, [0, 0, 0.0]); e[0] += 1; e[1] += int(d == 0); e[2] = max(e[2], d)
for k, (n, b, d) in per.items():
    print(f"{k:5s} tensors {n:3d} bit-equal {b:3d} max|diff| {d:.3e}")
