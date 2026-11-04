"""Batched drafting vs per-session drafting: same proposals? ms per draft_many call.
usage: mu3_draft_check.py MODEL [N] [D] [P]"""
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
base = be.make_cache(); d0 = be.new_draft()
for s in range(0, P, 256):
    be.prefill_chunk(base, ids[s:s + 256], draft=d0)
sufs = [tok.encode(" Question number %d: explain how the legions were organised." % (i * 7))[: 8 + i % 5] for i in range(N)]
def build():
    out = []
    for sf in sufs:
        c = be.fork(base); d = be.draft_restore(be.draft_snapshot(d0))
        be.prefill_chunk(c, sf[:-1], draft=d)
        out.append(d)
    mx.eval(*[a for d in out for a in be.draft_arrays(d)])
    return out
def jobs_of(drs, depths):
    return [(d, sf[-1], len(ids) + len(sf) - 1, dd, (ids, sf)) for d, sf, dd in zip(drs, sufs, depths)]
for depths in ([D] * N, [1 + i % D for i in range(N)]):
    res = {}
    for mode in (False, True):
        QM._BATCH_DRAFT = mode
        drs = build(); ts = []
        for _ in range(5):
            drs2 = build(); jb = jobs_of(drs2, depths)
            t0 = time.perf_counter(); r = be.draft_many(jb); ts.append(time.perf_counter() - t0)
        res[mode] = r; ts.sort()
        print("depths %s batched=%s: %.1f ms" % (depths, mode, ts[2] * 1e3), flush=True)
    same = sum(a == b for a, b in zip(res[False], res[True]))
    print("proposals equal for %d/%d sessions; ex: %s | %s" % (same, N, res[False][0], res[True][0]), flush=True)
# round 2: several pending rows per session (k = 1..4), as after a partly accepted verify
import random
random.seed(3)
H = d0.pend[-1].shape[-1]
ks = [1 + i % 4 for i in range(N)]
hids = [mx.random.normal((1, k, H)).astype(d0.pend[-1].dtype) for k in ks]
res = {}
for mode in (False, True):
    QM._BATCH_DRAFT = mode
    drs = build()
    be.draft_many(jobs_of(drs, [D] * N))
    jb = []
    for d, sf, k, h in zip(drs, sufs, ks, hids):
        d.pend = [h]
        hist = (ids, sf + [random.Random(5).randrange(1000, 9000) for _ in range(k)])
        jb.append((d, 0, d.mv + k, D, hist))
    res[mode] = be.draft_many(jb)
print("round2 (k=1..4) equal %d/%d" % (sum(a == b for a, b in zip(res[False], res[True])), N), res[False][:3], res[True][:3])
# host cost split of draft_lazy at round-2 shapes
import cProfile, pstats
QM._BATCH_DRAFT = True
def one():
    drs = build()
    be.draft_many(jobs_of(drs, [D] * N))
    jb = []
    for d, sf, k, h in zip(drs, sufs, ks, hids):
        d.pend = [h]
        jb.append((d, 0, d.mv + k, D, (ids, sf + [7] * k)))
    return jb
jb = one()
pr = cProfile.Profile(); pr.enable(); lz = be.draft_lazy(jb); pr.disable()
mx.eval(*lz.arrays.values())
pstats.Stats(pr).sort_stats("tottime").print_stats(8)
