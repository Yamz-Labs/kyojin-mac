"""prof89.py CTX [REPS]: ms of one plain step, one T-row verify pass (T=2..5) and one MTP-layer single-row draft step at CTX tokens of context
(synced, interleaved reps, best + median). KYOJIN_TQ selects the TurboQuant pool. The cache is trimmed back after each pass."""
import os, sys, json, time, statistics
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
import mlx.core as mx
import test_q35mlx as T
from kyojin_mlx.serve.backends import load_backend
from transformers import AutoTokenizer
CTX = int(sys.argv[1]); REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 7
tok = AutoTokenizer.from_pretrained(T.PACK)
ids = json.load(open(os.path.expanduser("~/rivals/corpus_ids.json"))); ids = ids["ids"] if isinstance(ids, dict) else ids
pids = ids[300000:300000 + CTX]
be, _ = load_backend(T.PACK, mtp_head=T.HEAD, log=lambda *a: None)
cache = be.make_cache(); d = be.new_draft()
SNAP = int(os.environ.get("PROF_SNAP", "0"))             # > 0: prefill SNAP tokens, keep their state as the server's prefix cache does, restore it into a new cache, prefill the rest
if SNAP:
    for i in range(0, SNAP, 2048): be.prefill_chunk(cache, pids[i:min(i + 2048, SNAP)], draft=d)
    snap = [c.state for c in cache]; mx.eval(snap)
    cache = be.make_cache()
    for c, st in zip(cache, snap): c.state = st
for i in range(SNAP, len(pids), 2048): be.prefill_chunk(cache, pids[i:i + 2048], draft=d)
bc = be.batch_spec([cache])
layers = [l for l in be.lm.model.layers if not l.is_linear]
caches_a = [c for l, c in zip(be.lm.model.layers, cache) if not l.is_linear]
def trim(n):
    for c in caches_a: c.trim(n)
def plain():
    h = be.lm.model(mx.array([[1000]], dtype=mx.int32), cache); lg = be._head(h)[0].astype(mx.float32); mx.eval(lg); trim(1)
def verify(T_):
    def f():
        p = be._verify(bc, [mx.array([1000 + i for i in range(T_)], dtype=mx.int32)]); mx.eval(p.logits); trim(T_)
    return f
def draft1():
    o = mx.zeros((1, 1, be.lm.args.hidden_size), dtype=mx.float16) if hasattr(be.lm, "args") else d.pend[-1]
    y = be._mtp_rows(d, mx.array([1000], dtype=mx.int32), d.pend[-1][:, -1:]); head = be.dhead if be.dhead is not None else be._head
    mx.eval(mx.argmax(head(y)[0, 0], -1)); d.mc.trim(1)
def draft1s():
    n = d.mc.offset; d.mc.trim(n - 256)
    y = be._mtp_rows(d, mx.array([1000], dtype=mx.int32), d.pend[-1][:, -1:]); head = be.dhead if be.dhead is not None else be._head
    mx.eval(mx.argmax(head(y)[0, 0], -1)); d.mc.trim(1); d.mc.offset = n
def mtp_nohead():
    y = be._mtp_rows(d, mx.array([1000], dtype=mx.int32), d.pend[-1][:, -1:]); mx.eval(y); d.mc.trim(1)
def rnd(T_, keep):
    def f():
        pend0 = list(d.pend)
        p = be._verify(bc, [mx.array([1000 + i for i in range(T_)], dtype=mx.int32)]); mx.eval(p.logits)
        t0 = time.perf_counter()
        p.commit([keep - 1], [d], [0]); mx.eval(*be.state(cache)); d.pend = pend0
        COMMIT.append((time.perf_counter() - t0) * 1e3)
        trim(max(0, keep - keep))
        for c in caches_a: c.trim(keep)
    return f
COMMIT = []
fs = {"commit_v5k2": rnd(5, 2), "draft1s": draft1s, "mtp_nohead": mtp_nohead, "plain": plain, "v2": verify(2), "v3": verify(3), "v4": verify(4), "v5": verify(5), "draft1": draft1}
for f in fs.values(): f(); f()
res = {k: [] for k in fs}
for _ in range(REPS):
    for k, f in fs.items():
        t0 = time.perf_counter(); f(); res[k].append((time.perf_counter() - t0) * 1e3)
print("COMMIT_ms", round(min(COMMIT), 1), round(statistics.median(COMMIT), 1))
print("PROF", CTX, os.environ.get("KYOJIN_TQ", "fp16"), json.dumps({k: [round(min(v), 1), round(statistics.median(v), 1)] for k, v in res.items()}), flush=True)
