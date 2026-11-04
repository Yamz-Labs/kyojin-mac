"""Greedy text gate: stock vs chunked GDN prefill (KYOJIN_GDN_FAST=64) on short prompts of 100-600 tokens, 32 new tokens each.
usage: ab_gdn_text.py PACK(exl3)"""
import sys, os
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from transformers import AutoTokenizer
from kyojin_mlx.exl3_linear import load_exl3
d = sys.argv[1]
model, _ = load_exl3(d); lm = model.language_model
tok = AutoTokenizer.from_pretrained(d)
base = ["Explain how a gated delta network differs from softmax attention, with an example.",
        "Write a short story about a lighthouse keeper who finds a message in a bottle.",
        "List the steps to make sourdough bread and the common mistakes beginners make.",
        "Summarize the causes of the French Revolution and its main consequences for Europe."]
def gen(ids, n=32):
    c = make_prompt_cache(model); x = mx.array(ids)[None]; out = []
    for _ in range(n):
        h = lm.model(x, cache=c)[:, -1]
        l = lm.lm_head(h) if getattr(lm, "lm_head", None) is not None else lm.model.embed_tokens.as_linear(h)
        t = mx.argmax(l, -1); mx.eval(t); out.append(t.item()); x = t[:, None]
    return out
tot = same = 0
for i, p in enumerate(base):
    for rep in (1, 8):
        ids = tok.encode((("Context note. " + p + " ") * rep))[:600]
        os.environ["KYOJIN_GDN_FAST"] = "0"; a = gen(ids)
        os.environ["KYOJIN_GDN_FAST"] = "64"; b = gen(ids)
        n = next((j for j in range(len(a)) if a[j] != b[j]), len(a))
        tot += 1; same += a == b
        print(f"prompt {i} len {len(ids)}: identical {a == b} (first diff at {n}/{len(a)})", flush=True)
print(f"greedy identical {same}/{tot}")
