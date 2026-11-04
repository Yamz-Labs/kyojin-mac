"""klrows89.py CTX [N]: teacher-forced KL of the R=4 / R=5 verify blocks against the 1-row plain steps at CTX tokens of context (KYOJIN_TQ=6:6:4:256 for the TurboQuant pool).
KL_REF=stock: the 1-row reference runs the stock MLX attention (sdpa_dec off) and an R=1 pass is added, so every line is the split-K kernel vs stock."""
import os, sys, json, inspect
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
import numpy as np
import mlx.core as mx
import test_q35mlx as T
TQV = os.environ.get("KYOJIN_TQ") or "-"
CTX = int(sys.argv[1]); n = int(sys.argv[2]) if len(sys.argv) > 2 else 64
src = inspect.getsource(T.run).replace("ctx=2048", "ctx=%d" % (CTX + 400)).replace("chunk=64, block=64", "chunk=2048, block=2048")
ns = dict(T.__dict__); exec(src, ns); run = ns["run"]
from transformers import AutoTokenizer
from kyojin_mlx.serve.backends import load_backend
tok = AutoTokenizer.from_pretrained(T.PACK); RIV = os.path.expanduser("~/rivals")
ids = json.load(open(RIV + "/corpus_ids.json")); ids = ids["ids"] if isinstance(ids, dict) else ids
doc = tok.decode(ids[300000:300000 + CTX - 120])
r = tok.apply_chat_template([{"role": "user", "content": doc + "\n\n---\nIgnore the text above. Explain how a hash map handles collisions and compare chaining with open addressing."}], add_generation_prompt=True, enable_thinking=False)
pids = r["input_ids"] if not isinstance(r, list) else r
(nat,), _ = run(False, [[pids]], n)
text = nat[0]
be, _ = load_backend(T.PACK, mtp_head=T.HEAD, log=lambda *a: None)
ca, cb4, cb5 = be.make_cache(), be.make_cache(), be.make_cache()
for c in (ca, cb4, cb5):
    for i in range(0, len(pids), 2048): be.prefill_chunk(c, pids[i:i + 2048])
from kyojin_mlx import sdpa_dec as SD
STOCK = os.environ.get("KL_REF") == "stock"
SD.ENABLE = not STOCK
ref, bc = [], be.batch([ca])
for t in text: ref.append(be.step(bc, mx.array([t], dtype=mx.int32))[0]); mx.eval(ref[-1])
ref = mx.stack(ref)
SD.ENABLE = True
runs = ((4, cb4), (5, cb5))
if STOCK:
    cb1 = be.make_cache()
    for i in range(0, len(pids), 2048): be.prefill_chunk(cb1, pids[i:i + 2048])
    runs = ((1, cb1),) + runs
for R, cache in runs:
    out = []
    if R == 1:                          # the served one-row step (the direct model call at T = 1 does not reach the sdpa wrapper)
        b1 = be.batch([cache]); out = [mx.stack([be.step(b1, mx.array([t], dtype=mx.int32))[0] for t in text]).astype(mx.float32)]
    for i in range(0, len(text) if R > 1 else 0, R):
        h = be.lm.model(mx.array(text[i:i + R], dtype=mx.int32)[None], cache); out.append(be._head(h)[0].astype(mx.float32))
    got = mx.concatenate(out, axis=0)
    lp, lq = ref - mx.logsumexp(ref, axis=-1, keepdims=True), got - mx.logsumexp(got, axis=-1, keepdims=True)
    kl = mx.sum(mx.exp(lp) * (lp - lq), axis=-1); mx.eval(kl)
    top1 = float(np.mean(np.array(mx.argmax(ref, -1)) == np.array(mx.argmax(got, -1))))
    print(f"KLSTAT ctx {CTX} tq={TQV} ref={'stock' if STOCK else 'ours'} R={R} mean {float(kl.mean()):.2e} max {float(kl.max()):.2e} top1 {top1:.3f}", flush=True)
