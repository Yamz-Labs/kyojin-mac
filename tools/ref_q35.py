"""Box CPU reference for an EXL3 Qwen3.5 pack: dequantize every trellis linear with the numpy reference
(validated against the engine in tests/test_ref_engine.py), run a dense fp32 torch model on CPU.
Writes OUT/fixtures.npz (per-linear), OUT/layers.npz (hidden states of prompt 0), OUT/ref_tokens.json
(greedy tokens + top-5 ids/logits per step for every prompt).
usage: ref_q35.py PACK_DIR TOKENIZER_DIR OUT_DIR [N=64]"""
import sys, json, time
import numpy as np, torch
sys.path.insert(0, __file__.rsplit("/tools/", 1)[0])
from kyojin_mlx import ref_numpy as R
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig

PROMPTS = ["The history of the Roman Empire begins with",
           "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
           "Question: What is the capital of Australia, and why was it chosen?\nAnswer:"]
pack, tokdir, out = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 64
tok = AutoTokenizer.from_pretrained(tokdir)
cfg = json.load(open(f"{pack}/config.json")); cb = cfg["quantization_config"]["codebook"]
f = safe_open(f"{pack}/model.safetensors", "pt")
keys = list(f.keys())
pref = sorted({k[:-8] for k in keys if k.endswith(".trellis")})
rng = np.random.default_rng(1234)
fix = {}; seenK = {}; sd = {}
t0 = time.time()
for p in pref:
    tr = f.get_tensor(p + ".trellis").numpy(); suh = f.get_tensor(p + ".suh").numpy(); svh = f.get_tensor(p + ".svh").numpy()
    K = tr.shape[2] // 16
    W = R.reconstruct(tr, suh, svh, cb)                         # fp32 [k, n]
    name = p.replace("model.language_model.", "model.") if p != "lm_head" else p
    sd[name + ".weight"] = torch.from_numpy(np.ascontiguousarray(W.T))
    if (seenK.get(K, 0) < 2 and tr.shape[0] * tr.shape[1] < 2e6) or K == 6:
        seenK[K] = seenK.get(K, 0) + 1
        x = (rng.standard_normal((8, W.shape[0])) * 0.7).astype(np.float16)
        fix[p] = (x, R.gemv(x, tr, suh, svh, np.float64, cb))
    if len(sd) % 20 == 0: print(p, K, round(time.time() - t0), flush=True)
np.savez(f"{out}/fixtures.npz", **{f"{k}|x": v[0] for k, v in fix.items()}, **{f"{k}|y": v[1] for k, v in fix.items()})
print("fixtures", len(fix), flush=True)
for k in keys:
    if k.startswith("model.language_model") and not any(k.endswith(s) for s in (".trellis", ".suh", ".svh", ".mcg", ".mul1")):
        sd[k.replace("model.language_model.", "model.")] = f.get_tensor(k).float()
tc = cfg["text_config"]; tc["tie_word_embeddings"] = False
m = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**{k: v for k, v in tc.items() if k != "model_type"}))
print(m.load_state_dict(sd, strict=False))
m.tie_weights = lambda: None
m = m.float().eval()
hs = {}
for i, l in enumerate(m.model.layers):
    l.register_forward_hook(lambda mod, a, o, i=i: hs.__setitem__(i, (o[0] if isinstance(o, tuple) else o).detach().numpy()))
res = []
for pi, text in enumerate(PROMPTS):
    ids = torch.tensor([tok.encode(text)])
    with torch.no_grad():
        if pi == 0:
            lg = m(ids).logits[0, -1].numpy()
            np.savez(f"{out}/layers.npz", logits=lg, **{f"h{i}": v[0] for i, v in hs.items()})
        g = m.generate(ids, max_new_tokens=N, do_sample=False, output_scores=True, return_dict_in_generate=True)
    seq = g.sequences[0, ids.shape[1]:].tolist()
    top = [torch.topk(s[0], 5) for s in g.scores]
    res.append({"prompt": text, "prompt_ids": ids[0].tolist(), "tokens": seq,
                "top5_ids": [t.indices.tolist() for t in top], "top5_logits": [t.values.tolist() for t in top]})
    print(pi, repr(tok.decode(seq)), flush=True)
json.dump(res, open(f"{out}/ref_tokens.json", "w"))
print("done", round(time.time() - t0))
