"""Write the multi-user benchmark prompts (token ids) once: usage make_mu_prompts.py MODEL_DIR OUT.json
4 independent prompts of different lengths, plus a shared-prefix scenario (one common prefix + 4 different suffixes)."""
import json, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
topics = [
    "The history of the Roman Empire begins with the founding of Rome in 753 BC and the long struggle between patricians and plebeians. ",
    "def merge_sort(items):\n    if len(items) <= 1:\n        return items\n    mid = len(items) // 2\n    return merge(merge_sort(items[:mid]), merge_sort(items[mid:]))\n\n",
    "Le petit village de montagne se réveillait doucement sous la brume, tandis que les cloches de l'église sonnaient sept heures du matin. ",
    "Photosynthesis converts light energy into chemical energy stored in glucose, using carbon dioxide and water, and releases oxygen as a by-product. ",
]
lens = [517, 389, 640, 455]
prompts = []
for t, L in zip(topics, lens):
    ids = tok.encode(t * 200)[:L]
    prompts.append(ids)
prefix = tok.encode("You are a careful research assistant working in a team. Shared context follows. " + topics[0] * 60)[:384]
suffix_txt = ["\nTask A: summarise the early republic in three sentences.", "\nTask B: list the main causes of the fall of the western empire, with dates and the names of the emperors involved.",
              "\nTask C: translate the previous paragraph into French.", "\nTask D: write a short poem about Rome, then explain its rhyme scheme in detail, line by line, for a high school class."]
suffixes = [tok.encode(s) for s in suffix_txt]
json.dump({"prompts": prompts, "prefix": prefix, "suffixes": suffixes}, open(sys.argv[2], "w"))
print([len(p) for p in prompts], len(prefix), [len(s) for s in suffixes])
