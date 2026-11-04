"""Shared-prefix verify attention vs the per-session path: same sessions, same drafted rows, logits compared.
usage: mu2_attn_check.py MODEL P N D [spread]"""
import sys, os, random
os.environ.setdefault("KYOJIN_TQ", "6:6:4:256")
import mlx.core as mx
from transformers import AutoTokenizer
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx.serve import q35mlx as QM
from kyojin_mlx import tq as TQ

mdir, P, N, D = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
spread = int(sys.argv[5]) if len(sys.argv) > 5 else 6
be, cfg = load_backend(mdir)
tok = AutoTokenizer.from_pretrained(mdir)
ids = tok.encode("The history of the Roman Empire begins with the founding of Rome and the long struggle between patricians and plebeians. " * 400)[:P]
base = be.make_cache(); d0 = be.new_draft()
for s in range(0, P, 256):
    be.prefill_chunk(base, ids[s:s + 256], draft=d0)
random.seed(3)
users, last = [], []
for i in range(N):
    sf = [random.randrange(1000, 20000) for _ in range(8 + (i * spread) % (spread + 1) + i)]
    c = be.fork(base); dd = be.draft_restore(be.draft_snapshot(d0))
    be.prefill_chunk(c, sf[:-1], draft=dd); users.append(c); last.append(sf[-1])
mx.eval(*[a for c in users for a in be.state(c)])
toks = [[last[i]] + [random.randrange(1000, 20000) for _ in range(D)] for i in range(N)]
res = {}
for shared in (False, True):
    QM._ATT_SHARED = shared
    TQ.STATS.clear()
    bc = be.batch_spec([be.fork(c) for c in users])
    p = be.spec_verify_bc(bc, toks)
    mx.eval(p.logits)
    res[shared] = p.logits
    print("shared=%s calls=%s offsets=%s" % (shared, TQ.STATS, [c[3].offset for c in bc.caches] if hasattr(bc.caches[0][3], "offset") else "?"))
a, b = res[False].astype(mx.float32), res[True].astype(mx.float32)
print("max abs logit diff %.4f  (logit scale %.1f)  argmax equal %d/%d rows" % (mx.max(mx.abs(a - b)).item(), mx.max(mx.abs(a)).item(), int(mx.sum(mx.argmax(a, -1) == mx.argmax(b, -1)).item()), a.shape[0]))
