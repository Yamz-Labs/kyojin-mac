"""ttft94.py SNAP TAIL [REPS]: cached-prompt TTFT parts in process, fp16 KV vs TurboQuant tq6w (KYOJIN_TQ_PF=0 old score path / 1 fused prefill):
prefill SNAP tokens once per KV kind and keep the state (the server's prefix snapshot), then per rep: fork it (the server's prefix hit), prefill the
TAIL tokens that follow (2048-row chunks) and take the last logits; plus the checkpoint restore (`state` setter: K rows -> tiles copy) per layer.
Interleaved reps, best + median ms."""
import os, sys, json, time, statistics
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
import mlx.core as mx
import test_q35mlx as T
from kyojin_mlx import tq
from kyojin_mlx.serve.backends import load_backend
SNAP, TAIL = int(sys.argv[1]), int(sys.argv[2]); REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
ids = json.load(open(os.path.expanduser("~/rivals/corpus_ids.json"))); ids = ids["ids"] if isinstance(ids, dict) else ids
pids = ids[300000:300000 + SNAP + TAIL]
be, _ = load_backend(T.PACK, mtp_head=T.HEAD, log=lambda *a: None)


def kind(tqon):
    if tqon:
        os.environ["KYOJIN_TQ"] = "6:6:4:256"
    else:
        os.environ.pop("KYOJIN_TQ", None)


snaps = {}
for tqon in (False, True):
    kind(tqon); tq.PF = 1
    c = be.make_cache()
    for i in range(0, SNAP, 2048):
        be.prefill_chunk(c, pids[i:min(i + 2048, SNAP)])
    snaps[tqon] = c
res = {}
cfgs = [("fp16", False, 1), ("tq-pf0", True, 0), ("tq-pf1", True, 1)]
for r in range(REPS + 1):
    for name, tqon, pf in (cfgs if r % 2 == 0 else cfgs[::-1]):
        kind(tqon); tq.PF = pf
        t0 = time.perf_counter()
        c = be.fork(snaps[tqon])
        for i in range(SNAP, SNAP + TAIL, 2048):
            lg = be.prefill_chunk(c, pids[i:min(i + 2048, SNAP + TAIL)])
        dt = (time.perf_counter() - t0) * 1e3
        del c, lg
        if r:
            res.setdefault(name, []).append(dt)
tqc = [c for c in snaps[True] if isinstance(c, tq.TQKVCache)]
st = [c.state for c in tqc]; mx.eval(st)
t0 = time.perf_counter()
for c, s in zip(tqc, st):
    n = tq.TQKVCache(6, 6, 4, 256); n.state = s; mx.eval(n.kq, n.vq)
rst = (time.perf_counter() - t0) * 1e3
print("TTFT94", SNAP, TAIL, json.dumps({k: [round(min(v)), round(statistics.median(v))] for k, v in res.items()}), f"tq checkpoint restore (all {len(tqc)} layers) {rst:.0f} ms", flush=True)
