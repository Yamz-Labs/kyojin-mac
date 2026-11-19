#!/usr/bin/env python3
"""Compare two archkit_logits_dump files: python tools/archkit_logits_cmp.py A.npz B.npz  (A = reference).
Per prompt: relative max logit error at the last position (to the logit scale), argmax agreement over all positions, and the gaps where they disagree."""
import sys

import numpy as np

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
for i in range(a["last"].shape[0]):
    x, y = a["last"][i], b["last"][i]
    err = np.abs(x - y).max() / max(1.0, np.abs(x).max())
    dis = np.where(a["argmax"][i] != b["argmax"][i])[0]
    print(f"prompt {i}: last-position rel err {err:.4f} (mean |d| {np.abs(x - y).mean():.4f}), argmax agree {a['argmax'].shape[1] - len(dis)}/{a['argmax'].shape[1]}, "
          f"disagree at {dis.tolist()[:8]} with ref gap {[round(float(a['gap'][i][j]), 3) for j in dis[:8]]}, top5 equal {set(np.argsort(-x)[:5]) == set(np.argsort(-y)[:5])}")
