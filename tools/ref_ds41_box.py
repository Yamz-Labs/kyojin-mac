#!/usr/bin/env python3
"""Reference capture of DS4.1 on the box engine (CUDA/ROCm, SSD expert tier): greedy token ids for fixed chat prompts.
Run through tools/ref_ds41_box.sh (it sets the engine env and the gpurun ticket). Output: OUT_DIR/ref.json with, per prompt, the rendered
prompt, its token ids, the greedy ids (stops after EOS), the text, and the top-5 ids / log-probs of every step (margin = top1 - top2:
a Mac flip at a tiny margin is rounding, a flip at a big margin is a bug). One pass: prefill + decode steps on model.forward (the _score path).
usage: ref_ds41_box.py PACK OUT_DIR [--engram-dir D] [--new 64] [--chunk 4096] [--prompts tools/ref_prompts.json]"""
import argparse, hashlib, json, os, sys, time

ROOT = os.environ.get("DS41_ROOT", os.path.expanduser("~/kyojin-work/ds41golive"))
sys.path.insert(0, os.path.join(ROOT, "tools", "ds41"))
ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("out")
ap.add_argument("--engram-dir"); ap.add_argument("--new", type=int, default=64); ap.add_argument("--chunk", type=int, default=4096)
ap.add_argument("--prompts", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref_prompts.json"))
a = ap.parse_args()
from ds41_serving import build_model
from ds41_engine import HFTokenizer
from ds41_model import PortEngine, TopPSampler
from ds41_chat import render_chat

t0 = time.perf_counter()
m = build_model(a.pack, device="cuda:0", dtype="float16", layers=None, mtp=False, moe="auto", prefill_chunk=a.chunk, engram_dir=a.engram_dir)
load_s = time.perf_counter() - t0
tk = HFTokenizer(os.path.join(a.pack, "tokenizer.json"))
eng = PortEngine(m, tk, ctx=4096, num_draft=0, eos_id=None, prefix_cache=False)
os.makedirs(a.out, exist_ok=True)
res = {"pack": a.pack, "engram_dir": a.engram_dir, "engine": os.popen(f"git -C {ROOT} describe --tags --always").read().strip(),
       "ablit_runtime": os.environ.get("EXL3_ABLIT_RUNTIME", ""), "new": a.new, "load_s": round(load_s, 1), "prompts": []}
import torch
EOS = tk._tk.token_to_id("<｜end▁of▁sentence｜>")
for spec in json.load(open(a.prompts)):
    text = render_chat(spec["messages"], thinking=False)
    ids = eng.prompt_ids(text)
    st = m.new_state(1, len(ids) + a.new + 2)
    outt, top5, pos = [], [], len(ids)
    t1 = time.perf_counter(); tf = None
    logits, _ = m.forward(torch.tensor([ids]), 0, st)              # prefill, logits of the last row
    for _ in range(a.new):
        lp = torch.log_softmax(logits[-1].float(), dim=-1)
        v, i = lp.topk(5)
        tok = int(i[0])
        if tf is None: tf = time.perf_counter()
        outt.append(tok); top5.append([[int(x), round(float(y), 5)] for x, y in zip(i, v)])
        if tok == EOS or len(outt) >= a.new: break
        logits, _ = m.forward(torch.tensor([[tok]]), pos, st); pos += 1
    t2 = time.perf_counter()
    rec = {"name": spec["name"], "prompt_text": text, "prompt_ids": ids, "out_ids": outt, "eos_id": EOS, "finished": outt[-1] == EOS,
           "out_text": tk._tk.decode(outt, skip_special_tokens=False), "ttft_s": round(tf - t1, 2),
           "dec_tok_s": round((len(outt) - 1) / max(t2 - tf, 1e-9), 2), "tokhash": hashlib.sha1(str(outt).encode()).hexdigest()[:12],
           "top5": top5, "margin": [round(t[0][1] - t[1][1], 5) for t in top5]}
    res["prompts"].append(rec)
    json.dump(res, open(os.path.join(a.out, "ref.json"), "w"), indent=1, ensure_ascii=False)       # saved before the next prompt
    print(spec["name"], {k: rec[k] for k in ("ttft_s", "dec_tok_s", "tokhash", "finished")}, repr(rec["out_text"][:80]), flush=True)
print("done", flush=True)
