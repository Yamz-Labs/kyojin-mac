#!/usr/bin/env python3
"""Our GGUF dequantization vs gguf-py's reference (numpy), every tensor of a file, max abs difference per quant type.
usage: python -I tools/archkit_dq_check.py FILE.gguf LLAMACPP_CHECKOUT"""
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(sys.argv[2], "gguf-py"))
sys.path.insert(0, os.path.join(here, ".."))
import numpy as np
import gguf
from kyojin_mlx import foreign_gguf as GF

r = gguf.GGUFReader(sys.argv[1]); g = GF.GgufFile(sys.argv[1])
per = {}
for t in r.tensors:
    tn = t.tensor_type.name
    if tn == "F32":
        continue
    ref = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32).reshape(g.tensors[t.name].shape)
    ours = g.tensor_f32(t.name).astype(np.float32)
    d = float(np.abs(ref - ours).max())
    e = per.setdefault(tn, dict(n=0, elems=0, maxdiff=0.0, maxref=0.0, bitequal=0))
    e["n"] += 1; e["elems"] += ref.size; e["maxdiff"] = max(e["maxdiff"], d); e["maxref"] = max(e["maxref"], float(np.abs(ref).max()))
    e["bitequal"] += int(d == 0.0)
for tn, e in per.items():
    print(f"{tn:6s} tensors {e['n']:3d} (bit-equal {e['bitequal']:3d}) elems {e['elems']:>11d} max|diff| {e['maxdiff']:.3e} max|ref| {e['maxref']:.3f}")
