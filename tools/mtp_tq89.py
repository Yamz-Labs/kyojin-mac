"""mtp_tq89.py CTX [N]: engine-level greedy plain vs MTP at CTX tokens of context (set KYOJIN_TQ=6:6:4:256 for the TurboQuant pool): first divergence index per question."""
import os, sys, json, inspect, threading
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
import test_q35mlx as T
CH = int(os.environ.get("CHUNK", "64"))
src = inspect.getsource(T.run).replace("ctx=2048", "ctx=%d" % (int(sys.argv[1]) + 400)).replace("chunk=64, block=64", "chunk=%d, block=%d" % (CH, CH))
ns = dict(T.__dict__); exec(src, ns); run = ns["run"]
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(T.PACK); RIV = os.path.expanduser("~/rivals")
ids = json.load(open(RIV + "/corpus_ids.json")); ids = ids["ids"] if isinstance(ids, dict) else ids
doc = tok.decode(ids[300000:300000 + int(sys.argv[1]) - 120]); n = int(sys.argv[2]) if len(sys.argv) > 2 else 64
QS = ["Explain how a hash map handles collisions and compare chaining with open addressing.", "Write a Python function that merges overlapping intervals, then explain its complexity.",
      "Summarize the causes of the French Revolution in five bullet points.", "What is the difference between TCP and UDP? Give examples of when to use each."]
def pr(q):
    r = tok.apply_chat_template([{"role": "user", "content": doc + "\n\n---\nIgnore the text above. " + q}], add_generation_prompt=True, enable_thinking=False)
    return r["input_ids"] if not isinstance(r, list) else r
P = [pr(q) for q in QS]
outs = {}
for spec in (False, True):
    res, st = run(spec, [[p] for p in P], n)
    outs[spec] = [w[0] for w in res]
    if spec: print("stats", {k: v for k, v in st.items() if "spec" in k or "draft" in k or "accept" in k})
for i in range(len(P)):
    a, b = outs[False][i], outs[True][i]
    d = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), None)
    print(f"q{i}: tokens {len(a)}/{len(b)} first divergence {d}")
