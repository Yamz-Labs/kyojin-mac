"""Per-phase ms of a multi-session spec round (Qwen3.5 mlx4 pack): verify (batched vs per session), commit rollback, MTP draft chains.
usage: mu117_prof.py MODEL [N] [D] [P]"""
import sys, os, time
os.environ.setdefault("KYOJIN_TQ", "6:6:4:256")
import mlx.core as mx
from transformers import AutoTokenizer
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx.serve import q35mlx as QM

mdir, N, D, P = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
be, cfg = load_backend(mdir)
tok = AutoTokenizer.from_pretrained(mdir)
ids = tok.encode("The history of the Roman Empire begins with the founding of Rome and the long struggle between patricians and plebeians. " * 200)[:P]
base = be.make_cache(); drafts0 = be.new_draft()
for s in range(0, P, 256):
    be.prefill_chunk(base, ids[s:s + 256], draft=drafts0)
sufs = [tok.encode(" Question number %d: explain how the legions were organised." % (i * 7)) for i in range(N)]
users, drs = [], []
for sf in sufs:
    c = be.fork(base); d = be.draft_restore(be.draft_snapshot(drafts0))
    be.prefill_chunk(c, sf[:-1], draft=d)
    users.append(c); drs.append(d)
mx.eval(*[a for c in users for a in be.state(c)], *[a for d in drs for a in be.draft_arrays(d)])
import random
random.seed(1)
toks = [[sf[-1]] + [random.randrange(1000, 20000) for _ in range(D)] for sf in sufs]


def t(f, reps=6):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    ts.sort(); return ts[len(ts) // 2] * 1e3


def verify(mode, commit=None):
    QM._BATCH_VERIFY = mode
    cs = [be.fork(c) for c in users]
    mx.eval(*[a for c in cs for a in be.state(c)])
    def f():
        p = be.spec_verify_bc(be.batch_spec([be.fork(c) for c in users]), toks)
        mx.eval(p.logits)
    return f
def build(mode):
    QM._BATCH_VERIFY = mode
    ts = []
    for _ in range(6):
        bc = be.batch_spec([be.fork(c) for c in users]); mx.eval(*[a for c in bc.caches for a in be.state(c)])
        t0 = time.perf_counter(); p = be.spec_verify_bc(bc, toks); t1 = time.perf_counter(); mx.eval(p.logits); t2 = time.perf_counter()
        ts.append((t1 - t0, t2 - t1))
    ts.sort(); return ts[3]
for mode in (False, True):
    b, e = build(mode); print("graph build batched=%s: build %.1f ms, then eval %.1f ms" % (mode, b * 1e3, e * 1e3), flush=True)
for mode in (False, True):
    print("verify batched=%s: %.1f ms" % (mode, t(verify(mode))), flush=True)
QM._BATCH_VERIFY = True
def vc(acc):
    def f():
        bc = be.batch_spec([be.fork(c) for c in users])
        p = be.spec_verify_bc(bc, toks)
        p.commit(acc)
        mx.eval(p.logits, *[a for c in bc.caches for a in be.state(c)])
    return f
base_ms = t(vc([D] * N))
print("verify+commit all accepted: %.1f ms" % base_ms)
print("verify+commit partial accept (1): %.1f ms" % t(vc([1] * N)))
print("verify+commit partial accept (mixed): %.1f ms" % t(vc([i % (D + 1) for i in range(N)])), flush=True)
# draft chains
jobs = [(d, sf[-1], len(ids) + len(sf) - 1, D, (ids, sf)) for d, sf in zip(drs, sufs)]
def dr():
    for d, sf in zip(drs, sufs):
        d.pend = d.pend[:1] if d.pend else d.pend
    be.draft_many(jobs)
print("draft_many N=%d depth %d: %.1f ms" % (N, D, t(dr)))
# plain batched step for reference
bcp = be.batch([be.fork(c) for c in users]); tk = mx.array([sf[-1] for sf in sufs], dtype=mx.int32)
def ps():
    lg = be.step(bcp, tk); mx.eval(lg)
print("plain batched step N=%d: %.1f ms" % (N, t(ps, 12)))
