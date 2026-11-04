"""poolw90.py CTX [REPS]: interleaved A/B in one process (the fanless Air drifts between runs) of the TQ pool write for verify chunks: in-place strided
tile columns (current) vs the rows-view patch (before 932460b, copies the whole K pool) vs no pool write (timing floor, wrong output). ms of plain and v2..v5 verify steps at CTX."""
import os, sys, json, time, statistics
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
os.environ.setdefault("KYOJIN_TQ", "6:6:4:256")
import mlx.core as mx
import test_q35mlx as T
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx import tq
CTX = int(sys.argv[1]); REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
ids = json.load(open(os.path.expanduser("~/rivals/corpus_ids.json"))); ids = ids["ids"] if isinstance(ids, dict) else ids
be, _ = load_backend(T.PACK, mtp_head=T.HEAD, log=lambda *a: None)
cache = be.make_cache(); d = be.new_draft()
pids = ids[300000:300000 + CTX]
for i in range(0, len(pids), 2048): be.prefill_chunk(cache, pids[i:i + 2048], draft=d)
att = [c for l, c in zip(be.lm.model.layers, cache) if not l.is_linear]
new = tq.TQKVCache._write_pool
def old(self, prev, keys, values):
    B, H, n, _ = keys.shape
    keys, values = self._src
    for dst, x, bits, tiled in ((self.kq, keys, self.kbits, True), (self.vq, values, self.vbits, False)):
        w, s = tq.tq_write(x.astype(mx.float32).reshape(-1, x.shape[-1]), bits)
        w = w.reshape(B, H, n, -1)
        if not tiled:
            dst[0][:, :, prev:self.offset] = w
        elif n == 1:
            dst[0][:, :, prev >> 5, :, prev & 31] = w[:, :, 0]
        else:
            t0, t1, a = prev >> 5, (self.offset + 31) >> 5, prev & 31
            blk = tq.tiles_to_rows(dst[0][:, :, t0:t1]); blk[:, :, a:a + n] = w
            dst[0][:, :, t0:t1] = tq.rows_to_tiles(blk)
        dst[1][:, :, prev:self.offset] = s.reshape(B, H, n)
def step(n):
    if n == 1:
        h = be.lm.model(mx.array([[1000]], dtype=mx.int32), cache); mx.eval(be._head(h)[0])
    else:
        p = be._verify(be.batch_spec([cache]), [mx.array([1000 + i for i in range(n)], dtype=mx.int32)]); mx.eval(p.logits)
    for c in att: c.trim(n)
res = {}
for r in range(REPS + 1):
    for name, f in (("old", old), ("new", new), ("skip", lambda self, prev, k, v: None)):
        tq.TQKVCache._write_pool = f
        for n in (1, 2, 3, 5):
            t0 = time.perf_counter(); step(n); dt = (time.perf_counter() - t0) * 1e3
            if r: res.setdefault(f"{name}:{n}", []).append(dt)
print("POOLW90", CTX, json.dumps({k: [round(min(v), 1), round(statistics.median(v), 1)] for k, v in res.items()}), flush=True)
