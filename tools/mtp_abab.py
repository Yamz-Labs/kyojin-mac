"""Interleaved ABAB: plain vs MTP spec (stock MLX qmv rows vs q4mma rows) on a Qwen3.5 mlx pack.
usage: mtp_abab.py PACK BF16 [--depths 3] [--rounds 3] [--n 96] [--prompts 4] [--variants stock,mma] [--qbits 4] [--dhead 32768]"""
import argparse, time, statistics, json
import mlx.core as mx, mlx.nn as nn
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("bf16")
ap.add_argument("--depths", default="3"); ap.add_argument("--rounds", type=int, default=3); ap.add_argument("--n", type=int, default=96); ap.add_argument("--prompts", type=int, default=4)
ap.add_argument("--dhead", type=int, default=32768); ap.add_argument("--json", default=""); ap.add_argument("--variants", default="stock"); ap.add_argument("--qbits", type=int, default=4)
ap.add_argument("--thinking", action="store_true")
a = ap.parse_args()
from kyojin_mlx import qspec as QS, mlxpack
model, tok = mlxpack.load(a.pack); lm = model.language_model
class _T(nn.Module):
    def __init__(s, e): super().__init__(); s.__dict__["_e"] = e
    def __call__(s, h): return s._e.as_linear(h)
lm.__dict__["lm_head"] = _T(lm.model.embed_tokens)
mtp = QS.load_mtp(model, a.bf16, quant_bits=a.qbits); dh = QS.load_draft_head(a.bf16, a.dhead) if a.dhead else None
variants = a.variants.split(",")
if "mma" in variants:
    from kyojin_mlx import q4mma as QM
    print("q4mma linears", QM.install(model), "mtp", QM.install_mtp(mtp), flush=True)
    if dh is not None: dh = QM.Q4Linear(dh)
Q = ["Explain how a hash map handles collisions and compare chaining with open addressing.", "Write a Python function that merges overlapping intervals, then explain its complexity.",
     "Summarize the causes of the French Revolution in five bullet points.", "What is the difference between TCP and UDP? Give examples of when to use each."]
ps = []
for q in Q[:a.prompts]:
    p = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=True, add_generation_prompt=True, enable_thinking=a.thinking)
    ps.append(p["input_ids"] if not isinstance(p, list) else p)
depths = [int(x) for x in a.depths.split(",")]
def setv(v):
    if "mma" in variants: QM.MIN_ROWS = 2 if v == "mma" else 99
sps = {d: QS.QSpec(model, mtp, depth=d, dhead=dh) for d in depths}
refs = []
setv("stock")
for p in ps:
    QS.uninstall_capture(); QS.plain_generate(model, p, 8); refs.append(QS.plain_generate(model, p, a.n))
for d in depths:
    for v in variants:
        setv(v); QS.install_capture(); sps[d].prefill(ps[0]); sps[d].generate(32)
keys = [f"d{d}{v}" for d in depths for v in variants]
R = {"plain": [], **{k: [] for k in keys}}; ex = {k: [] for k in keys}; tpc = {k: [] for k in keys}
for r in range(a.rounds):
    for i, p in enumerate(ps):
        setv("stock"); QS.uninstall_capture()
        t0 = time.perf_counter(); out = QS.plain_generate(model, p, a.n); R["plain"].append(a.n / (time.perf_counter() - t0))
        for d in depths:
            for v in variants:
                k = f"d{d}{v}"; setv(v); sp = sps[d]; QS.install_capture(); sp.stats = QS.Stats(); sp.prefill(p)
                t0 = time.perf_counter(); out = sp.generate(a.n); dt = time.perf_counter() - t0
                R[k].append(len(out) / dt); ex[k].append(out == refs[i]); tpc[k].append(sp.stats.emitted / sp.stats.rounds)
                if out != refs[i] and r == 0:
                    j = next(q for q in range(min(len(out), len(refs[i]))) if out[q] != refs[i][q]); from mlx_lm.models.cache import make_prompt_cache
                    setv("stock"); QS.uninstall_capture(); c = make_prompt_cache(model); hh = lm.model(mx.array(p + refs[i][:j])[None], c)
                    lg = lm.lm_head(hh[:, -1:])[0, 0].astype(mx.float32); t2 = mx.sort(lg)[-2:]
                    print(f"  prompt {i} {k} diverge at {j}/{a.n}: top-2 gap {float(t2[1]-t2[0]):.3f}", flush=True)
mp = statistics.median(R["plain"]); print(f"plain median {mp:.1f} tok/s  {[round(x,1) for x in R['plain']]}")
for k in keys:
    m = statistics.median(R[k]); print(f"{k} median {m:.1f} tok/s x{m/mp:.2f} exact {sum(ex[k])}/{len(ex[k])} tok/cycle {statistics.mean(tpc[k]):.2f}  {[round(x,1) for x in R[k]]}")
if a.json: json.dump(R, open(a.json, "w"))
