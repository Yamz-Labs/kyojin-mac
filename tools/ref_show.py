#!/usr/bin/env python3
"""Readable text of a reference json (tools/ref_ds41_box.py output): provenance header, per prompt the ids, the text, log-probs / margins.
usage: ref_show.py REF.json [OUT.txt]"""
import json
import sys

r = json.load(open(sys.argv[1]))
L = [f"DS4.1 reference, box engine {r['engine']}, pack {r['pack']}, Engram {r['engram_dir']}, runtime ablit '{r['ablit_runtime']}' "
     f"({'OFF' if r['ablit_runtime'] == 'off' else 'as set'}; the production lane has it ON), greedy, fp16, up to {r['new']} new tokens per prompt, load {r['load_s']} s.", ""]
for p in r["prompts"]:
    n = len(p["out_ids"])
    L.append(f"=== {p['name']}  prompt {len(p['prompt_ids'])} ids, {n} generated{' (EOS reached)' if p.get('finished') else ''}, ttft {p['ttft_s']} s, "
             f"decode {p['dec_tok_s']} tok/s, tokhash {p['tokhash']}")
    L += ["PROMPT: " + repr(p["prompt_text"]), "PROMPT IDS: " + json.dumps(p["prompt_ids"]), "OUT IDS: " + json.dumps(p["out_ids"]), "OUT TEXT: " + repr(p["out_text"])]
    if p.get("chosen_logp"):
        L.append("CHOSEN LOGP: " + json.dumps([round(x, 3) for x in p["chosen_logp"]]))
    if p.get("margin"):
        m = p["margin"]
        L.append("MARGIN top1-top2: " + json.dumps([round(x, 3) for x in m]) + f"   steps below 0.05: {sum(x < 0.05 for x in m)}, below 0.5: {sum(x < 0.5 for x in m)}")
    L.append("")
text = "\n".join(L)
if len(sys.argv) > 2:
    open(sys.argv[2], "w").write(text)
else:
    print(text)
