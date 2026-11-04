"""trim90.py CTX [REPS]: interleaved A/B in one process (the fanless Air drifts between runs) of the kvwin ring refill on rollback: refill up to
window + margin rows (before: once the ring is full every trim dequantizes rows lazily from the pool, so the next pool write cannot donate and
MLX copies every K and V word pool) vs refill only below the window (current). ms of plain and v2..v5 verify steps at CTX, each followed by trim."""
import os, sys, json, time, inspect, statistics
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
os.environ.setdefault("KYOJIN_TQ", "6:6:4:256")
import mlx.core as mx
import test_q35mlx as T
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx import kvwin
CTX = int(sys.argv[1]); REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
ids = json.load(open(os.path.expanduser("~/rivals/corpus_ids.json"))); ids = ids["ids"] if isinstance(ids, dict) else ids
be, _ = load_backend(T.PACK, mtp_head=T.HEAD, log=lambda *a: None)
cache = be.make_cache(); d = be.new_draft()
pids = ids[300000:300000 + CTX]
for i in range(0, len(pids), 2048): be.prefill_chunk(cache, pids[i:i + 2048], draft=d)
att = [c for l, c in zip(be.lm.model.layers, cache) if not l.is_linear]
new = kvwin.WinKVCache.trim
src = inspect.getsource(new).replace("need = min(self.window, self.offset)", "need = min(self.window + self.margin, self.offset)")
assert "self.window + self.margin, self.offset" in src
ns = {}; exec("import mlx.core as mx\nif 1:\n" + src, ns); old = ns["trim"]
def step(n):
    if n == 1:
        h = be.lm.model(mx.array([[1000]], dtype=mx.int32), cache); mx.eval(be._head(h)[0])
    else:
        p = be._verify(be.batch_spec([cache]), [mx.array([1000 + i for i in range(n)], dtype=mx.int32)]); mx.eval(p.logits)
    for c in att: c.trim(n)
res = {}
for r in range(REPS + 1):
    for name, f in (("old", old), ("new", new)):
        kvwin.WinKVCache.trim = f
        step(1)                                             # first step after the switch pays the other variant's pending refill
        for n in (1, 2, 3, 5):
            t0 = time.perf_counter(); step(n); dt = (time.perf_counter() - t0) * 1e3
            if r: res.setdefault(f"{name}:{n}", []).append(dt)
print("TRIM90", CTX, json.dumps({k: [round(min(v), 1), round(statistics.median(v), 1)] for k, v in res.items()}), flush=True)
