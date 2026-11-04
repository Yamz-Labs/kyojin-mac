"""dacc89.py CTX [N]: depth-1 draft accuracy of the MTP layer on the model's own greedy text (teacher forced, one pass): top-1 of the draft head and of the full head vs the next trunk token."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
import numpy as np
import mlx.core as mx
import test_q35mlx as T
from kyojin_mlx.serve.backends import load_backend
from transformers import AutoTokenizer
CTX = int(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 96
tok = AutoTokenizer.from_pretrained(T.PACK)
ids = json.load(open(os.path.expanduser("~/rivals/corpus_ids.json"))); ids = ids["ids"] if isinstance(ids, dict) else ids
doc = tok.decode(ids[300000:300000 + CTX - 120])
QS = ["Explain how a hash map handles collisions and compare chaining with open addressing.", "Write a Python function that merges overlapping intervals, then explain its complexity.",
      "Summarize the causes of the French Revolution in five bullet points.", "What is the difference between TCP and UDP? Give examples of when to use each."]
be, _ = load_backend(T.PACK, mtp_head=T.HEAD, log=lambda *a: None)
tot = {"dhead": [0, 0], "full": [0, 0]}
for q in QS:
    r = tok.apply_chat_template([{"role": "user", "content": doc + "\n\n---\nIgnore the text above. " + q}], add_generation_prompt=True, enable_thinking=False)
    p = r["input_ids"] if not isinstance(r, list) else r
    cache = be.make_cache()
    for i in range(0, len(p), 2048): lg = be.prefill_chunk(cache, p[i:i + 2048])
    t = int(mx.argmax(lg[0])); gen = [t]
    for _ in range(N - 1):
        h = be.lm.model(mx.array([[t]], dtype=mx.int32), cache); t = int(mx.argmax(be._head(h)[0, -1])); gen.append(t)
    full = p[-64:] + gen                                   # context of 64 prompt tokens is enough for the trunk hiddens only if re-run: re-run the whole sequence instead
    cache2 = be.make_cache(); seq = p + gen; hs = []
    for i in range(0, len(seq), 2048): hs.append(be.lm.model(mx.array(seq[i:i + 2048], dtype=mx.int32)[None], cache2))
    h = mx.concatenate(hs, axis=1)
    d = be.new_draft()
    y = be._mtp_rows(d, mx.array(seq[1:], dtype=mx.int32), h[:, :-1])          # row j: token j+1, hidden j -> predicts token j+2
    s0 = len(p) - 1
    ys = y[:, s0 - 0:len(seq) - 2]                                             # rows whose target is a generated token (positions >= len(p))
    tgt = np.array(seq[s0 + 2:len(seq)])
    for name, head in (("dhead", be.dhead if be.dhead is not None else be._head), ("full", be._head)):
        pred = np.array(mx.argmax(head(ys)[0], -1)); n = min(len(pred), len(tgt))
        tot[name][0] += int((pred[:n] == tgt[:n]).sum()); tot[name][1] += n
print("DACC ctx", CTX, {k: round(a / b, 3) for k, (a, b) in tot.items()}, tot, flush=True)
