"""Greedy decode of a DS4.1 slice with the env switches as given, logits of every step saved: bitwise A/B of a cut across processes (a mx.compile region keeps the
choice it was traced with, so a switch flipped inside one process may not reach it).
usage: dump_logits.py PACK OUT.npz [--prompt 150] [--steps 24] [--seed 5]      then   dump_logits.py --cmp A.npz B.npz
       dump_logits.py --kl A.npz B.npz [KL_MAX]   KL(A||B) per step over the steps whose prefix tokens agree; exit 1 when the mean KL > KL_MAX (default 0.02)"""
import argparse, sys
import numpy as np
if "--kl" in sys.argv:
    i = sys.argv.index("--kl"); a, b = np.load(sys.argv[i + 1]), np.load(sys.argv[i + 2])
    kl_max = float(sys.argv[i + 3]) if len(sys.argv) > i + 3 else 0.02
    ta, tb = a["tokens"], b["tokens"]
    diff = np.nonzero(ta != tb)[0]; n = int(diff[0]) + 1 if diff.size else len(ta)   # step k is comparable while tokens[:k] agree
    la, lb = a["logits"][:n].astype(np.float64), b["logits"][:n].astype(np.float64)
    la -= la.max(1, keepdims=True); lb -= lb.max(1, keepdims=True)
    la -= np.log(np.exp(la).sum(1, keepdims=True)); lb -= np.log(np.exp(lb).sum(1, keepdims=True))
    kl = (np.exp(la) * (la - lb)).sum(1); top1 = (la.argmax(1) == lb.argmax(1)).mean()
    ok = kl.mean() <= kl_max
    print(f"KL mean {kl.mean():.3e} max {kl.max():.3e} over {n}/{len(ta)} steps; top1 agree {top1:.3f}; first token diff "
          f"{int(diff[0]) if diff.size else None}; {'PASS' if ok else 'FAIL'} (KL_MAX {kl_max})")
    sys.exit(0 if ok else 1)
if "--cmp" in sys.argv:
    a, b = np.load(sys.argv[-2]), np.load(sys.argv[-1])
    same = np.array_equal(a["logits"].view(np.uint32), b["logits"].view(np.uint32))
    d = np.abs(a["logits"] - b["logits"]).max()
    print(f"logits bitwise equal: {same}; max |diff| {d:.3e}; tokens equal: {np.array_equal(a['tokens'], b['tokens'])}")
    sys.exit(0 if same else 1)
import mlx.core as mx
from kyojin_mlx import patch as P, rows as R
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("out"); ap.add_argument("--prompt", type=int, default=150)
ap.add_argument("--steps", type=int, default=24); ap.add_argument("--seed", type=int, default=5); a = ap.parse_args()
model, spec, cfg = P.build_model(a.pack); P.patch_model(model, a.pack, spec, cfg); R.enable(model, spec)
V = cfg["text_config"]["vocab_size"]
ids = np.random.default_rng(a.seed).integers(16, V - 16, a.prompt).tolist()
cache, lg = R.prefill(model, ids)
tok = int(mx.argmax(lg[0]).item()); toks, logs = [], []
for _ in range(a.steps):
    lg = R.step(model, cache, mx.array([tok])); mx.eval(lg)
    x = np.array(lg.astype(mx.float32)).reshape(-1)
    logs.append(x); tok = int(np.argmax(x)); toks.append(tok)
np.savez(a.out, logits=np.stack(logs), tokens=np.array(toks))
print("saved", a.out, "tokens", toks[:8], "...")
