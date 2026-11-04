"""specround_cpu.py PACK [--users 1] [--depth 3] [--rounds 40] [--lazy 0|1] [--acc 99]: critical path of one served MTP round on a DS pack with a draft head,
split into draft (build + encode + GPU wait), verify build, verify encode, verify GPU wait, accept + commit + settle. Greedy, natural acceptance (ACC caps it).
M10_LIGHT=F (tools/light.py) divides the custom-kernel grids: the GPU is never the limit, the CPU path is what a wide GPU sees.
Prints one ROUND line per run and the token stream digest, so two code paths can be compared bit for bit."""
import argparse, hashlib, os, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("M10_LIGHT"):
    import light
import mlx.core as mx
import numpy as np
from kyojin_mlx.serve.backends import load_backend

ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("--users", type=int, default=1); ap.add_argument("--depth", type=int, default=3)
ap.add_argument("--rounds", type=int, default=40); ap.add_argument("--len", type=int, default=140); ap.add_argument("--lazy", type=int, default=0)
ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()
be, cfg = load_backend(a.pack, mtp=True, ced=False, log=lambda *x: None)
vocab = cfg.get("text_config", cfg)["vocab_size"]
rng = np.random.default_rng(7)
N, D = a.users, a.depth
caches, drafts, lasts = [], [], []
base = rng.integers(16, vocab - 16, 61).tolist()
for u in range(N):
    ids = ([16 + u] + (base * 4)[:a.len - 1])                    # a repeated slice: the draft head finds real structure, so acceptance is not zero
    c = be.make_cache(); d = be.new_draft()
    lg = be.prefill_chunk(c, ids, draft=d)
    caches.append(c); drafts.append(d); lasts.append(int(mx.argmax(lg[0]).item()))
bc = be.batch(caches)
pos = [a.len] * N
outs = [[l] for l in lasts]
hist = [list(ids) for _ in range(N)]
lazy = getattr(be, "draft_lazy", None) if a.lazy else None


def one_round():
    t0 = time.perf_counter()
    jobs = [(drafts[u], lasts[u], pos[u], D, (hist[u], outs[u])) for u in range(N)]
    lh = lazy(jobs) if lazy else None
    if lh is not None:
        res = be.spec_verify_lazy(bc, lh, lasts)
        tb = time.perf_counter()
        mx.async_eval(res.logits); te = time.perf_counter()
        vals = mx.argmax(res.logits, -1).tolist(); tw = time.perf_counter()
        toks = [[lasts[u]] + dr[:lh.keep[u]] for u, dr in enumerate(lh.tolist())]
        td = tb                                                  # draft and verify build overlap: the draft column is only the draft CPU
    else:
        dr = be.draft_many(jobs)
        td = time.perf_counter()
        toks = [[lasts[u]] + [int(t) for t in dr[u]] for u in range(N)]
        res = be.spec_verify_bc(bc, toks)
        tb = time.perf_counter()
        mx.async_eval(res.logits); te = time.perf_counter()
        vals = mx.argmax(res.logits, -1).tolist(); tw = time.perf_counter()
    acc = []
    for u in range(N):
        r0 = res.starts[u]; d_ = toks[u][1:]; k = 0
        while k < len(d_) and vals[r0 + k] == d_[k]:
            k += 1
        acc.append(k)
        outs[u].extend(d_[:k] + [vals[r0 + k]]); lasts[u] = vals[r0 + k]; pos[u] += k + 1
    res.commit(acc, drafts, [p - k - 1 for p, k in zip(pos, acc)])
    be.settle(bc, drafts)
    t5 = time.perf_counter()
    return (td - t0, tb - td, te - tb, tw - te, t5 - tw, t5 - t0, sum(acc) / N)


for _ in range(4):
    one_round()
for rep in range(a.reps):
    rows = [one_round() for _ in range(a.rounds)]
    m = [1e3 * statistics.median(r[i] for r in rows) for i in range(6)]
    toks = sum(len(o) for o in outs)
    print("ROUND users %d depth %d lazy %d light %s: draft %.2f vbuild %.2f vencode %.2f vwait %.2f commit %.2f total %.2f ms  acc/round %.2f"
          % (N, D, a.lazy, os.environ.get("M10_LIGHT", "1"), *m, statistics.mean(r[6] for r in rows)), flush=True)
dig = hashlib.sha1(repr(outs).encode()).hexdigest()[:12]
print("DIGEST", dig, "tokens", sum(len(o) for o in outs), flush=True)
try:
    from kyojin_mlx import verify_replay as _VR
    if _VR.ENABLED:
        print("verify_replay STATS", _VR.STATS, flush=True)
except Exception:
    pass
