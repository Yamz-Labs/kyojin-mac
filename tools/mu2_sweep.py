"""Verify cost sweep (Qwen3.5 mlx4 pack): ms of one batched verify pass for N sessions x rows (D+1) at P tokens of shared prefix, plus the plain batched step.
usage: mu2_sweep.py MODEL P N1,N2.. D1,D2.. [--prof]"""
import sys, os, time, random
os.environ.setdefault("KYOJIN_TQ", "6:6:4:256")
import mlx.core as mx
from transformers import AutoTokenizer
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx.serve import q35mlx as QM

mdir, P = sys.argv[1], int(sys.argv[2])
Ns = [int(v) for v in sys.argv[3].split(",")]; Ds = [int(v) for v in sys.argv[4].split(",")]
be, cfg = load_backend(mdir)
tok = AutoTokenizer.from_pretrained(mdir)
ids = tok.encode("The history of the Roman Empire begins with the founding of Rome and the long struggle between patricians and plebeians. " * 200)[:P]
base = be.make_cache(); drafts0 = be.new_draft()
for s in range(0, P, 256):
    be.prefill_chunk(base, ids[s:s + 256], draft=drafts0)
NM = max(Ns)
sufs = [tok.encode(" Question number %d: explain how the legions were organised." % (i * 7)) for i in range(NM)]
users = []
for sf in sufs:
    c = be.fork(base); d = be.draft_restore(be.draft_snapshot(drafts0))
    be.prefill_chunk(c, sf[:-1], draft=d)
    users.append(c)
mx.eval(*[a for c in users for a in be.state(c)])
random.seed(1)


def med(f, reps=7):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    ts.sort(); return ts[len(ts) // 2] * 1e3


for N in Ns:
    row = []
    for D in Ds:
        toks = [[sufs[i][-1]] + [random.randrange(1000, 20000) for _ in range(D)] for i in range(N)]
        def f():
            bc = be.batch_spec([be.fork(c) for c in users[:N]])
            p = be.spec_verify_bc(bc, toks)
            mx.eval(p.logits)
        row.append("D%d %.1f" % (D, med(f)))
    bcp = be.batch([be.fork(c) for c in users[:N]]); tk = mx.array([sf[-1] for sf in sufs[:N]], dtype=mx.int32)
    def ps():
        mx.eval(be.step(bcp, tk))
    print("N=%d verify ms: %s | plain batched step %.1f" % (N, "  ".join(row), med(ps, 12)), flush=True)
