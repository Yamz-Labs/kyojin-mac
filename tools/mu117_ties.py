"""Near-tie check of spec-vs-plain divergences: at the first differing token of each stream, the plain logits (solo decode path) margin between the plain token and the spec token.
usage: mu117_ties.py MODEL PLAIN.json SPEC.json KEYS(comma, e.g. 0,1)  (streams named as in mu117 --tokens-out; P=4096)"""
import sys, os, json
os.environ.setdefault("KYOJIN_TQ", "6:6:4:256")
import mlx.core as mx
from transformers import AutoTokenizer
from kyojin_mlx.serve.backends import load_backend
import importlib.util
mdir, pf, sf, keys = sys.argv[1], json.load(open(sys.argv[2])), json.load(open(sys.argv[3])), sys.argv[4].split(",")
be, cfg = load_backend(mdir, mtp=False)
tok = AutoTokenizer.from_pretrained(mdir)
src = open(os.path.join(os.path.dirname(__file__), "mu117.py")).read()
ns = {}
exec(src[src.index("base = ("):src.index("Ns = ")], {"tok": tok, "a": type("A", (), {"P": 4096})()}, ns)
prefix, Q = ns["prefix"], ns["Q"]
base = be.make_cache()
for s in range(0, len(prefix), 256):
    be.prefill_chunk(base, prefix[s:s + 256])
for k in keys:
    i = int(k.split("_")[-1]); ref, got = pf[k.split("_")[-1] if k not in pf else k], sf[k]
    j = next((j for j, (x, y) in enumerate(zip(ref, got)) if x != y), -1)
    if j < 0:
        print(k, "equal"); continue
    c = be.fork(base)
    ctx = tok.encode(" " + Q[i]) + ref[:j]
    lg = be.prefill_chunk(c, ctx)[0]
    top = mx.sort(lg)[-3:].tolist(); a, b = lg[ref[j]].item(), lg[got[j]].item()
    print(k, "first diff at", j, "plain tok logit %.3f spec tok logit %.3f gap %.3f (bf16 ulp at this magnitude ~%.3f)" % (a, b, a - b, 2 ** (int(max(abs(a), 1)).bit_length() - 1 - 7)), "rank of spec tok:", int((lg > b).sum().item()))
