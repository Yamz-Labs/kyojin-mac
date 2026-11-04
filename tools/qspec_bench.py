"""Speculative decode on Qwen3.5 EXL3: exactness vs plain greedy, tokens/s and acceptance per position for depths.
usage: qspec_bench.py PACK BF16_DIR [--depths 1,2,3,4,6,8] [--n 96] [--prompts 4] [--rowsplit 16] [--qbits 0]"""
import argparse, time, json, statistics
import mlx.core as mx
from kyojin_mlx import exl3_linear as EL
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("bf16")
ap.add_argument("--depths", default="1,2,3,4,6,8"); ap.add_argument("--n", type=int, default=96)
ap.add_argument("--prompts", type=int, default=4); ap.add_argument("--rowsplit", type=int, default=16)
ap.add_argument("--qbits", type=int, default=0); ap.add_argument("--headk", type=int, default=0); ap.add_argument("--maxrows", type=int, default=0); ap.add_argument("--auto", type=int, default=0); ap.add_argument("--overlap", type=int, default=1); ap.add_argument("--json", default=""); a = ap.parse_args()
EL.ROW_SPLIT_MAX = a.rowsplit
from kyojin_mlx import kernels as KN
if a.maxrows: KN.MAX_ROWS = a.maxrows; EL.GEMV_MAX_ROWS = a.maxrows
from kyojin_mlx import qspec as QS
model, _ = EL.load_exl3(a.pack)
print("row linears", QS.row_linears(model))
mtp = QS.load_mtp(model, a.bf16, quant_bits=a.qbits)
dhead = QS.load_draft_head(a.bf16, a.headk) if a.headk else None
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.pack)
QUESTIONS = ["Explain how a hash map handles collisions and compare chaining with open addressing.",
             "Write a Python function that merges overlapping intervals, then explain its complexity.",
             "Summarize the causes of the French Revolution in five bullet points.",
             "What is the difference between TCP and UDP? Give examples of when to use each.",
             "Describe how a transformer attention layer works, step by step.",
             "Translate to French: The committee will review the proposal next week and send its answer by Friday."]
prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for q in QUESTIONS[:a.prompts]]
prompts = [tok.encode(p) for p in prompts]
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
    sp = QS.QSpec(model, mtp, depth=d, dhead=dhead, auto=bool(a.auto), overlap=bool(a.overlap)); tps = []; ok = []
    tot = QS.Stats()
    for p, ref in zip(prompts, refs):
        sp.stats = QS.Stats()
        sp.prefill(p); sp.generate(8)
        sp.stats = QS.Stats()
        sp.prefill(p)
        t0 = time.perf_counter(); out = sp.generate(a.n); dt = time.perf_counter() - t0
        tps.append(len(out) / dt); ok.append(out == ref)
        st = sp.stats
        for f in ('rounds', 'emitted', 't_verify', 't_rollback_draft'): setattr(tot, f, getattr(tot, f) + getattr(st, f))
        for i in range(32): tot.accept_pos[i] += st.accept_pos[i]; tot.reach_pos[i] += st.reach_pos[i]
    st = tot
    acc = [round(st.accept_pos[i] / max(1, st.reach_pos[i]), 2) for i in range(d)]
    print(f"d={d}: {statistics.mean(tps):.2f} tok/s (x{statistics.mean(tps)/statistics.mean(tp):.2f}) exact {sum(ok)}/{len(ok)} "
          f"all prompts: tok/round {st.emitted/st.rounds:.2f} verify {st.t_verify/st.rounds*1e3:.1f} ms draft+rb {st.t_rollback_draft/st.rounds*1e3:.1f} ms cond-accept {acc}", flush=True)
    res["spec"][d] = {"tps": tps, "exact": ok, "tok_per_round": st.emitted / st.rounds, "acc": acc}
if a.json: json.dump(res, open(a.json, "w"))
