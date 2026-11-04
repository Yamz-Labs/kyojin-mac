"""mlxlm96.py PACK CTX [N]: mlx-lm stream_generate decode tok/s (its own generation_tps) after a CTX-token document + question, the same
prompts as depth89 (greedy, thinking off), plain decode; the rival line of the served table."""
import json, os, statistics, sys
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
PACK, CTX = sys.argv[1], int(sys.argv[2]); N = int(sys.argv[3]) if len(sys.argv) > 3 else 96
RIV = os.path.expanduser("~/rivals")
model, tok = load(PACK)
ids = json.load(open(RIV + "/corpus_ids.json")); ids = ids["ids"] if isinstance(ids, dict) else ids
doc = tok.decode(ids[300000:300000 + CTX - 120])
QS = ["Explain how a hash map handles collisions and compare chaining with open addressing.", "Write a Python function that merges overlapping intervals, then explain its complexity."]
tps = []
for q in QS + QS:
    p = tok.apply_chat_template([{"role": "user", "content": doc + "\n\n---\nIgnore the text above. " + q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    last = None
    for r in stream_generate(model, tok, p, max_tokens=N, sampler=make_sampler(temp=0.0)):
        last = r
    tps.append(round(last.generation_tps, 1))
print("MLXLM96", os.path.basename(PACK), CTX, tps, "best", max(tps), "median", statistics.median(tps), flush=True)
