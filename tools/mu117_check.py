"""Batched verify vs per-session verify on the same state: N sessions share a P-token prefix, each has its own suffix; verify rows [t, d1..dD] with
arbitrary token drafts. Compares the logits row by row (max abs diff, argmax equal, smallest top1-top2 margin where argmax differs), then the
caches after commit with a partial accept (the next token logits must agree). usage: mu117_check.py MODEL [N] [D] [P]"""
import sys, os
os.environ.setdefault("KYOJIN_TQ", "6:6:4:256")
import mlx.core as mx
from transformers import AutoTokenizer
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx.serve import q35mlx as QM

mdir, N, D, P = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4, int(sys.argv[3]) if len(sys.argv) > 3 else 4, int(sys.argv[4]) if len(sys.argv) > 4 else 1024
be, cfg = load_backend(mdir)
tok = AutoTokenizer.from_pretrained(mdir)
ids = tok.encode("The history of the Roman Empire begins with the founding of Rome and the long struggle between patricians and plebeians. " * 200)
pre = ids[:P]
base = be.make_cache()
for s in range(0, P, 256):
    be.prefill_chunk(base, pre[s:s + 256])
sufs = [tok.encode(" Question number %d: explain how the legions were organised and paid in detail." % (i * 7)) for i in range(N)]
users = []
for sf in sufs:
    c = be.fork(base)
    be.prefill_chunk(c, sf[:-1])
    users.append(c)
mx.eval(*[a for c in users for a in be.state(c)])
import random
random.seed(1)
toks = [[sf[-1]] + [random.randrange(1000, 20000) for _ in range(D if os.environ.get("MIXED") != "1" else (D - i) % (D + 1))] for i, sf in enumerate(sufs)]   # MIXED=1: a different row count per session
res = {}
for mode in (False, True):
    QM._BATCH_VERIFY = mode
    cs = [be.fork(c) for c in users]
    bc = be.batch_spec(cs)
    p = be.spec_verify_bc(bc, toks)
    mx.eval(p.logits)
    res[mode] = (p, cs)
a, b = res[False][0].logits.astype(mx.float32), res[True][0].logits.astype(mx.float32)
d = mx.abs(a - b).max(axis=-1); am = mx.argmax(a, -1) == mx.argmax(b, -1)
srt = mx.sort(a, axis=-1)
margin = srt[:, -1] - srt[:, -2]
mx.eval(d, am, margin)
print("rows", a.shape[0], "max|dlogit| per row", [round(x, 4) for x in d.tolist()])
print("argmax equal all:", bool(am.all().item()), " min margin", round(margin.min().item(), 4))
# partial accept on every session: accepted = (k mod (D+1)); next-step logits from each cache
outs = {}
for mode in (False, True):
    p, cs = res[mode]
    p.commit([i % (D + 1) for i in range(N)])
    nx = mx.concatenate([be.MU.logits_last(be.model, mx.array([[7 + i]]), c).astype(mx.float32) for i, c in enumerate(cs)])
    mx.eval(nx); outs[mode] = nx
dn = mx.abs(outs[False] - outs[True]).max(axis=-1)
print("post-commit next-step max|dlogit| per session", [round(x, 4) for x in dn.tolist()], "argmax equal", bool((mx.argmax(outs[False], -1) == mx.argmax(outs[True], -1)).all().item()))
