"""hostbuild112.py CTX [REPS]: host-side time to BUILD the d4 verify graph (no eval) vs the GPU time, per KYOJIN_TQ setting; cProfile top of the build under PROF=1."""
import os, sys, json, time, statistics
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
import mlx.core as mx
import test_q35mlx as T
from kyojin_mlx.serve.backends import load_backend
CTX = int(sys.argv[1]); REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 15
ids = json.load(open(os.path.expanduser("~/rivals/corpus_ids.json"))); ids = ids["ids"] if isinstance(ids, dict) else ids
pids = ids[300000:300000 + CTX]
be, _ = load_backend(T.PACK, mtp_head=T.HEAD, log=lambda *a: None)
cache = be.make_cache(); d = be.new_draft()
for i in range(0, len(pids), 2048): be.prefill_chunk(cache, pids[i:i + 2048], draft=d)
bc = be.batch_spec([cache])
caches_a = [c for l, c in zip(be.lm.model.layers, cache) if not l.is_linear]
toks = [mx.array([1000 + i for i in range(5)], dtype=mx.int32)]
def build():
    return be._verify(bc, toks)
def full():
    p = build(); mx.eval(p.logits); [c.trim(5) for c in caches_a]
for _ in range(3): full()
hb, hf = [], []
for _ in range(REPS):
    t0 = time.perf_counter(); p = build(); t1 = time.perf_counter(); mx.eval(p.logits); t2 = time.perf_counter(); [c.trim(5) for c in caches_a]
    hb.append((t1 - t0) * 1e3); hf.append((t2 - t0) * 1e3)
print("HOSTBUILD", CTX, os.environ.get("KYOJIN_TQ", "off"), "build_ms med", round(statistics.median(hb), 2), "total_ms med", round(statistics.median(hf), 2), flush=True)
if os.environ.get("PROF"):
    import cProfile, pstats
    pr = cProfile.Profile(); pr.enable()
    for _ in range(5): p = build(); mx.eval(p.logits); [c.trim(5) for c in caches_a]
    pr.disable(); pstats.Stats(pr).sort_stats("tottime").print_stats(14)
