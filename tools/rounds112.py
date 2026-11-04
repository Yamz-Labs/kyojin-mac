"""rounds112.py CTX [ROUNDS]: served-like MTP rounds (draft_lazy -> spec_verify_lazy -> argmax accept -> commit -> settle) in-process, depth 4, greedy;
prints tok/s and ms/round with host split (draft+verify build / wait / commit). KYOJIN_TQ selects the cache. SKIP=trim stubs the TQ commit trim."""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
import mlx.core as mx
import test_q35mlx as T
from kyojin_mlx.serve.backends import load_backend
CTX = int(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
ids = json.load(open(os.path.expanduser("~/rivals/corpus_ids.json"))); ids = ids["ids"] if isinstance(ids, dict) else ids
pids = ids[300000:300000 + CTX]
be, _ = load_backend(T.PACK, mtp_head=T.HEAD, log=lambda *a: None)
cache = be.make_cache(); d = be.new_draft()
for i in range(0, len(pids), 2048): lg = be.prefill_chunk(cache, pids[i:i + 2048], draft=d)
bc = be.batch_spec([cache])
last = int(mx.argmax(lg[0]).item()); out = [last]
P0 = CTX
def run(n):
    global last, P0
    th = [0.0, 0.0, 0.0]; toks = 0; p0 = P0; t00 = time.perf_counter(); acc = 0
    for _ in range(n):
        t0 = time.perf_counter()
        lh = be.draft_lazy([(d, last, p0, 4, (pids, out))])
        res = be.spec_verify_lazy(bc, lh, [last])
        t1 = time.perf_counter()
        vals = mx.argmax(res.logits, -1).tolist(); drafts = lh.tolist()[0]
        t2 = time.perf_counter()
        a = min(int(os.environ.get("ACC", "2")), len(drafts))        # forced acceptance: timing only (the corpus slice gives 0 natural accepts)
        res.commit([a], [d], [p0]); be.settle(bc, [d])
        out.extend(drafts[:a] + [vals[a]]); last = vals[a]; p0 += a + 1; toks += a + 1; acc += a
        t3 = time.perf_counter(); th[0] += t1 - t0; th[1] += t2 - t1; th[2] += t3 - t2
    P0 = p0; mx.eval(*be.state(cache)); dt = time.perf_counter() - t00
    return toks / dt, dt / n * 1e3, [round(x / n * 1e3, 2) for x in th], acc / n
run(8)
for k in range(3):
    r = run(N)
    print("ROUNDS", CTX, os.environ.get("KYOJIN_TQ", "off"), "tok/s %.1f ms/round %.1f build/wait/commit %s acc/round %.2f" % r, flush=True)
