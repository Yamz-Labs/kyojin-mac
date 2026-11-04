"""snap90.py CTX SNAP [REPS]: verify-step ms (plain, v5) of a cache prefilled straight vs one restored from a SNAP-token state snapshot (the server's prefix
cache hit) then prefilled to CTX, interleaved in one process. KYOJIN_TQ selects the TurboQuant pool. Shows a cost a restored cache keeps paying."""
import os, sys, json, time, statistics
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
import mlx.core as mx
import test_q35mlx as T
from kyojin_mlx.serve.backends import load_backend
CTX, SNAP = int(sys.argv[1]), int(sys.argv[2]); REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 6
ids = json.load(open(os.path.expanduser("~/rivals/corpus_ids.json"))); ids = ids["ids"] if isinstance(ids, dict) else ids
pids = ids[300000:300000 + CTX]
be, _ = load_backend(T.PACK, mtp_head=T.HEAD, log=lambda *a: None)
def build(snap):
    cache = be.make_cache(); d = be.new_draft()
    if snap:
        for i in range(0, snap, 2048): be.prefill_chunk(cache, pids[i:min(i + 2048, snap)], draft=d)
        st = [c.state for c in cache]; mx.eval(st)
        cache = be.make_cache()
        for c, s in zip(cache, st): c.state = s
        keep.append(st)                                     # the prefix cache keeps the snapshot alive
    for i in range(snap, len(pids), 2048): be.prefill_chunk(cache, pids[i:i + 2048], draft=d)
    return cache
keep = []
C = {"straight": build(0), "restored": build(SNAP)}
att = {k: [c for l, c in zip(be.lm.model.layers, v) if not l.is_linear] for k, v in C.items()}
def step(k, n):
    if n == 1:
        h = be.lm.model(mx.array([[1000]], dtype=mx.int32), C[k]); mx.eval(be._head(h)[0])
    else:
        p = be._verify(be.batch_spec([C[k]]), [mx.array([1000 + i for i in range(n)], dtype=mx.int32)]); mx.eval(p.logits)
    for c in att[k]: c.trim(n)
res = {}
for r in range(REPS + 1):
    for k in C:
        for n in (1, 5):
            t0 = time.perf_counter(); step(k, n); dt = (time.perf_counter() - t0) * 1e3
            if r: res.setdefault(f"{k}:{'plain' if n == 1 else 'v5'}", []).append(dt)
print("SNAP90", CTX, SNAP, os.environ.get("KYOJIN_TQ", "fp16"), json.dumps({k: [round(min(v), 1), round(statistics.median(v), 1)] for k, v in res.items()}), flush=True)
