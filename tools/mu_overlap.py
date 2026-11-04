"""Expert overlap f of consecutive tokens of one session, from the ids captured by tools/mu_route.py (the f of ULTRA.md s.17 / s.18 and tools/mtp_ultra_est.py).
usage: mu_overlap.py OUT.npz
f = mean over layers, users and consecutive step pairs of |experts(t) & experts(t - 1)| / top-k. Also prints the mean for the chained rows of a verify (t, t + 1, t + 2)
against the independent-routing value top-k / E that the estimate takes as 'no sharing'."""
import re
import sys

import numpy as np

z = np.load(sys.argv[1])
layers = sorted(int(m.group(1)) for k in z.files if (m := re.fullmatch(r"L(\d+)_ids", k)))
assert layers, "no L<layer>_ids arrays: is this a mu_route.py output?"
per, lag2 = [], []
for l in layers:
    ids = z[f"L{l}_ids"]                                   # [users, steps, topk]
    u, s, k = ids.shape
    if s < 3:
        continue
    a = np.array([[len(set(ids[i, t]) & set(ids[i, t - 1])) / k for t in range(1, s)] for i in range(u)])
    b = np.array([[len(set(ids[i, t]) & set(ids[i, t - 2])) / k for t in range(2, s)] for i in range(u)])
    per.append((l, float(a.mean()), float(b.mean())))
    lag2.append(float(b.mean()))
f1 = float(np.mean([p[1] for p in per]))
print(f"layers {len(per)}  users {z[f'L{layers[0]}_ids'].shape[0]}  steps {z[f'L{layers[0]}_ids'].shape[1]}  topk {z[f'L{layers[0]}_ids'].shape[2]}")
print("layer   f(t-1)  f(t-2)")
for l, a, b in per:
    print(f"{l:5d}   {a:.3f}   {b:.3f}")
print(f"f = {f1:.3f}   (lag 2: {float(np.mean(lag2)):.3f})")
