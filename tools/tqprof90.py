"""tqprof90.py CTX [REPS]: in-model split of the TQ verify cost at CTX: ms of plain / v5 steps as is, with the TQ attend stubbed (returns q) and with the pool write stubbed
(KYOJIN_TQ selects the pool; under fp16 only the attend stub applies). Shows how much of the verify step is attention vs cache write vs the rest."""
import os, sys, json, time, statistics
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
import mlx.core as mx
import test_q35mlx as T
from transformers import AutoTokenizer
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx import tq, kvwin
CTX = int(sys.argv[1]); REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
ids = json.load(open(os.path.expanduser("~/rivals/corpus_ids.json"))); ids = ids["ids"] if isinstance(ids, dict) else ids
pids = ids[300000:300000 + CTX]
be, _ = load_backend(T.PACK, mtp_head=T.HEAD, log=lambda *a: None)
cache = be.make_cache(); d = be.new_draft()
for i in range(0, len(pids), 2048): be.prefill_chunk(cache, pids[i:i + 2048], draft=d)
bc = be.batch_spec([cache])
caches_a = [c for l, c in zip(be.lm.model.layers, cache) if not l.is_linear]
print("cache types", sorted({type(c).__name__ for c in caches_a}), flush=True)
def trim(n):
    for c in caches_a: c.trim(n)
def plain():
    h = be.lm.model(mx.array([[1000]], dtype=mx.int32), cache); mx.eval(be._head(h)[0]); trim(1)
def v5():
    p = be._verify(bc, [mx.array([1000 + i for i in range(5)], dtype=mx.int32)]); mx.eval(p.logits); trim(5)
cls = type(caches_a[0])
att0, wp0 = getattr(cls, "attend", None), getattr(cls, "_write_pool", None)
modes = {"base": lambda: None}
def stub_att(): cls.attend = lambda self, q, s: q
if att0 is not None: modes["no_attend"] = stub_att
if wp0 is not None:
    def stub_wp(): cls._write_pool = lambda self, prev, k, v: None
    modes["no_poolwrite"] = stub_wp
res = {}
for m, setup in modes.items():
    if att0 is not None: cls.attend = att0
    if wp0 is not None: cls._write_pool = wp0
    setup()
    for f in (plain, v5): f(); f()
    for name, f in (("plain", plain), ("v5", v5)):
        xs = []
        for _ in range(REPS):
            t0 = time.perf_counter(); f(); xs.append((time.perf_counter() - t0) * 1e3)
        res[m + ":" + name] = [round(min(xs), 1), round(statistics.median(xs), 1)]
if att0 is not None: cls.attend = att0
print("TQPROF", CTX, os.environ.get("KYOJIN_TQ", "fp16"), json.dumps(res), flush=True)
