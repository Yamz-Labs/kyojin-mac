"""Step compile default (program key without the cache object) on the synthetic Hy4 server (CPU, in a subprocess because the deployed flags are read at import): a ladder of requests with the deployed CPU-able
set (fixed KV, pooled-key cache, step compile + attn split, key cache, SWA slice, skinny rows) returns the tokens of a full-recompute greedy reference for every request, a repeated
prompt included (prefix hit = fork path), and the memory held after the requests stops growing (prefix budget fixed)."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROBE = r'''
import os, sys, gc, json, tempfile
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import numpy as np
import mlx.core as mx
mx.set_default_device(mx.cpu)
import test_hy4_serve_backend as T
os.environ.update({"KYOJIN_SKELETON": "1", "KYOJIN_KIT_UNPROVEN_ROWS": "1", "KYOJIN_TQ": "0", "KYOJIN_WIRED_MB": "0"})
root = tempfile.mkdtemp()
g = T._write(root + "/model/hy4_2l.gguf")
ln = T.Lane(g, root + "/cache"); ln.start()
eng = ln.app.engine
eng.prefix.budget = 500000
rng = np.random.default_rng(3)
V = T.E2E.V
prompts = [[int(x) for x in rng.integers(1, V - 1, n)] for n in (16, 24, 32, 40, 48, 44)]
prompts += [prompts[5], prompts[4], prompts[3], prompts[5]]                 # repeats: prefix-cache hits
out = {"ok": [], "mem": [], "hits": None}
for ids in prompts:
    toks, reason = T._collect(eng.submit(ids, max_new=8))
    out["ok"].append(toks == T._reference(g, ids, 8, {12}))
    gc.collect(); out["mem"].append(mx.get_active_memory())
out["hits"] = eng.prefix.hits
print("RESULT " + json.dumps(out))
'''


def _run(leak):
    env = dict(os.environ, KYOJIN_HY4_FIXED_KV="1", KYOJIN_HY4_FIXED_PKCACHE="1", KYOJIN_STEP_COMPILE="1", KYOJIN_STEP_COMPILE_ATTN="1", KYOJIN_KPOOL_KEYCACHE="1",
               KYOJIN_HY4_SWA_SLICE="1", KYOJIN_SKINNY_ROWS="8", KYOJIN_STEP_COMPILE_LEAK="1" if leak else "0")
    r = subprocess.run([sys.executable, "-I", "-c", f"HERE={HERE!r}; ROOT={ROOT!r}\n" + PROBE], env=env, capture_output=True, text=True, timeout=600)
    line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
    assert line, r.stdout[-800:] + r.stderr[-800:]
    return json.loads(line[0][7:])


def test_default_tokens_equal_reference_and_memory_flat():
    o = _run(False)
    assert all(o["ok"]), o["ok"]
    assert o["hits"] >= 1                                       # the repeated prompts hit the prefix cache
    assert o["mem"][-1] - o["mem"][3] <= 64 * 1024, o["mem"]    # nothing piles up from the 4th request on


def test_escape_hatch_still_leaks_on_the_same_ladder():
    o = _run(True)
    assert all(o["ok"]), o["ok"]
    assert o["mem"][-1] - o["mem"][3] >= 500 * 1024, o["mem"]
