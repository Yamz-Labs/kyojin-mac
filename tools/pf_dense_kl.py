"""KL gate of the folded dense prefill (dense_pf) vs the compressed EXL3 prefill path: last-row logits of N real-text prompts, same backend, weights folded on / off.
usage: pf_dense_kl.py EXL3_DIR [--T 512] [--n 8]  -> mean / max KL(ref || folded), top-1 agreement, max |dlogit|"""
import argparse, glob, os, mlx.core as mx
ap = argparse.ArgumentParser(); ap.add_argument("exl3"); ap.add_argument("--T", type=int, default=512); ap.add_argument("--n", type=int, default=8); a = ap.parse_args()
from transformers import AutoTokenizer
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx import wcache as WC, dense_pf as DP
tok = AutoTokenizer.from_pretrained(a.exl3)
files = sorted(glob.glob(os.path.join(os.path.dirname(DP.__file__), "*.py")), key=os.path.getsize, reverse=True)[:a.n]
prompts = [tok.encode(open(f).read())[:a.T] for f in files]
be, _ = load_backend(a.exl3)
lins = WC._linears(be.model)
weff = {id(q): q._weff if q._weff is not None else DP.fold(q) for q in lins}; mx.eval(list(weff.values()))   # the loader may have folded them already
def logits(ids, on):
    for q in lins: q._weff = weff[id(q)] if on else None
    lg = be.prefill_chunk(be.make_cache(), ids)
    return lg.reshape(-1).astype(mx.float32)
kls, agree, dmax = [], 0, 0.0
for ids in prompts:
    r, f = logits(ids, False), logits(ids, True)
    lr, lf = r - mx.logsumexp(r), f - mx.logsumexp(f)
    kl = float(mx.sum(mx.exp(lr) * (lr - lf))); kls.append(kl)
    agree += int(mx.argmax(r).item() == mx.argmax(f).item()); dmax = max(dmax, float(mx.abs(r - f).max()))
print(f"n={len(kls)} T={a.T}: KL mean {sum(kls)/len(kls):.2e} max {max(kls):.2e}  top1 agree {agree}/{len(kls)}  max|dlogit| {dmax:.3f}")
