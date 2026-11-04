"""kl87_cmp.py A.npz B.npz: mean KL(A||B) per position for tq6 (old vs new), plus the quantization floor KL(fp16 || tq6) of each, top-1 agreement."""
import sys, numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
def kl(p, q):
    p, q = p.astype(np.float64), q.astype(np.float64); return (np.exp(p) * (p - q)).sum(-1)
def top1(p, q): return float((p.argmax(-1) == q.argmax(-1)).mean())
for lbl, x, y in (("old tq6 -> new tq6 (layout change)", a["tq6"], b["tq6"]), ("floor: fp16 -> old tq6", a["fp16"], a["tq6"]), ("floor: fp16 -> new tq6", b["fp16"], b["tq6"]), ("fp16 old vs new (must be 0)", a["fp16"], b["fp16"])):
    k = kl(x, y); print(f"{lbl:40s} KL mean {k.mean():.3e} max {k.max():.3e}  top1 {top1(x, y):.4f}  (n={len(k)})")
