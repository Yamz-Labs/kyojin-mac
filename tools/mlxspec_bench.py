"""Speculative decode on a Qwen3.5 mlx-lm quantized pack (MLX qmv/qmm): exactness vs plain greedy, tokens/s and acceptance per position for depths.
usage: mlxspec_bench.py PACK BF16_DIR [--depths 1,2,3,4,6,8] [--n 96] [--prompts 4] [--rowsplit 16] [--qbits 0]"""
import argparse, time, json, statistics
import mlx.core as mx
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("bf16")
ap.add_argument("--depths", default="1,2,3,4,6,8"); ap.add_argument("--n", type=int, default=96)
ap.add_argument("--prompts", type=int, default=4); ap.add_argument("--rowsplit", type=int, default=16)
ap.add_argument("--qbits", type=int, default=4); ap.add_argument("--dhead", type=int, default=32768); ap.add_argument("--json", default=""); a = ap.parse_args()
from kyojin_mlx import qspec as QS
from kyojin_mlx import mlxpack
import mlx.nn as nn
model, _ = mlxpack.load(a.pack)
lm = model.language_model
if not hasattr(lm, "lm_head") or lm.args.tie_word_embeddings:
    class _TiedHead(nn.Module):
        def __init__(s, emb): super().__init__(); s.__dict__["_emb"] = emb
        def __call__(s, h): return s._emb.as_linear(h)
    lm.__dict__["lm_head"] = _TiedHead(lm.model.embed_tokens)
print("row_linears", QS.row_linears(model), flush=True)
mtp = QS.load_mtp(model, a.bf16, quant_bits=a.qbits)
dh = QS.load_draft_head(a.bf16, a.dhead) if a.dhead else None
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.pack)
QUESTIONS = ["Explain how a hash map handles collisions and compare chaining with open addressing.",
             "Write a Python function that merges overlapping intervals, then explain its complexity.",
             "Summarize the causes of the French Revolution in five bullet points.",
             "What is the difference between TCP and UDP? Give examples of when to use each.",
             "Describe how a transformer attention layer works, step by step.",
             "Translate to French: The committee will review the proposal next week and send its answer by Friday."]
prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=True, add_generation_prompt=True, enable_thinking=False) for q in QUESTIONS[:a.prompts]]
prompts = [p["input_ids"] if not isinstance(p, list) else p for p in prompts]
res = {"plain": {}, "spec": {}}
# plain
QS.uninstall_capture()
tp = []; refs = []
for p in prompts:
    QS.plain_generate(model, p, 8)
    t0 = time.perf_counter(); r = QS.plain_generate(model, p, a.n); dt = time.perf_counter() - t0
    refs.append(r); tp.append(a.n / dt)
print(f"plain: {statistics.mean(tp):.2f} tok/s  per prompt {[round(x,1) for x in tp]}", flush=True)
res["plain"] = tp
for d in [int(x) for x in a.depths.split(",")]:
    sp = QS.QSpec(model, mtp, depth=d, dhead=dh); tps = []; ok = []
    for p, ref in zip(prompts, refs):
        sp.stats = QS.Stats()
        sp.prefill(p); sp.generate(8)
        sp.stats = QS.Stats()
        sp.prefill(p)
        t0 = time.perf_counter(); out = sp.generate(a.n); dt = time.perf_counter() - t0
        tps.append(len(out) / dt); ok.append(out == ref)
        st = sp.stats
        if out != ref:
            i = next(j for j in range(min(len(out), len(ref))) if out[j] != ref[j])
            from mlx_lm.models.cache import make_prompt_cache
            c = make_prompt_cache(model); h = lm.model(mx.array(p + ref[:i])[None], c)
            lg = lm.lm_head(h[:, -1:])[0, 0].astype(mx.float32); top = mx.sort(lg)[-2:]
            print(f"  diverge at {i}: plain {ref[i]} spec {out[i]} top-2 logit gap {float(top[1]-top[0]):.3f} (logit of spec token {float(lg[out[i]]):.3f}, plain {float(lg[ref[i]]):.3f})", flush=True)
    acc = [round(st.accept_pos[i] / max(1, st.reach_pos[i]), 2) for i in range(d)]
    print(f"d={d}: {statistics.mean(tps):.2f} tok/s (x{statistics.mean(tps)/statistics.mean(tp):.2f}) exact {sum(ok)}/{len(ok)} "
          f"last prompt: tok/round {st.emitted/st.rounds:.2f} verify {st.t_verify/st.rounds*1e3:.1f} ms draft+rb {st.t_rollback_draft/st.rounds*1e3:.1f} ms cond-accept {acc}", flush=True)
    res["spec"][d] = {"tps": tps, "exact": ok, "tok_per_round": st.emitted / st.rounds, "acc": acc}
if a.json: json.dump(res, open(a.json, "w"))
