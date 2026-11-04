"""Exactness and speed of MTP spec vs plain greedy for several q4mma row thresholds, same process, interleaved per prompt.
usage: mtp_exact.py PACK BF16 [--mins 99,5,1] [--depths 3,4] [--n 96] [--prompts 8]
MIN_ROWS m: verify/plain linears with R >= m rows run q4mma (row invariant, bitwise); 99 = MLX qmv/qmm everywhere. Plain is run with the SAME setting (the exactness reference)."""
import argparse, time, statistics
import mlx.core as mx, mlx.nn as nn
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("bf16")
ap.add_argument("--mins", default="99,5,1"); ap.add_argument("--depths", default="3,4"); ap.add_argument("--n", type=int, default=96); ap.add_argument("--prompts", type=int, default=8)
ap.add_argument("--dhead", type=int, default=32768); a = ap.parse_args()
from kyojin_mlx import qspec as QS, mlxpack, q4mma as QM
model, tok = mlxpack.load(a.pack); lm = model.language_model
class _T(nn.Module):
    def __init__(s, e): super().__init__(); s.__dict__["_e"] = e
    def __call__(s, h): return s._e.as_linear(h)
lm.__dict__["lm_head"] = _T(lm.model.embed_tokens)
mtp = QS.load_mtp(model, a.bf16, quant_bits=4); dh = QS.load_draft_head(a.bf16, a.dhead)
print("q4mma linears", QM.install(model), "mtp", QM.install_mtp(mtp), flush=True)
Q = ["Explain how a hash map handles collisions and compare chaining with open addressing.", "Write a Python function that merges overlapping intervals, then explain its complexity.",
     "Summarize the causes of the French Revolution in five bullet points.", "What is the difference between TCP and UDP? Give examples of when to use each.",
     "Describe how a transformer attention layer works, step by step.", "Translate to French: The committee will review the proposal next week and send its answer by Friday.",
     "Write a short story about a lighthouse keeper who finds a message in a bottle.", "List the planets of the solar system with one fact about each.",
     "Explain the difference between a process and a thread.", "How does public key cryptography work?"]
ps = []
for q in Q[:a.prompts]:
    p = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=True, add_generation_prompt=True, enable_thinking=False)
    ps.append(p["input_ids"] if not isinstance(p, list) else p)
mins = [int(x) for x in a.mins.split(",")]; depths = [int(x) for x in a.depths.split(",")]
sps = {d: QS.QSpec(model, mtp, depth=d, dhead=dh) for d in depths}
R = {}; ex = {}
for i, p in enumerate(ps):
    for m in mins:
        QM.MIN_ROWS = m
        QS.uninstall_capture(); QS.plain_generate(model, p, 4)
        t0 = time.perf_counter(); ref = QS.plain_generate(model, p, a.n); R.setdefault((m, 0), []).append(a.n / (time.perf_counter() - t0))
        QS.install_capture()
        for d in depths:
            sp = sps[d]; sp.stats = QS.Stats(); sp.prefill(p)
            t0 = time.perf_counter(); out = sp.generate(a.n); dt = time.perf_counter() - t0
            R.setdefault((m, d), []).append(len(out) / dt); ex.setdefault((m, d), []).append(out == ref)
            if out != ref:
                j = next(q for q in range(min(len(out), len(ref))) if out[q] != ref[q])
                from mlx_lm.models.cache import make_prompt_cache
                QS.uninstall_capture(); c = make_prompt_cache(model); hh = lm.model(mx.array(p + ref[:j])[None], c)
                lg = lm.lm_head(hh[:, -1:])[0, 0].astype(mx.float32); t2 = mx.sort(lg)[-2:]; QS.install_capture()
                print(f"  prompt {i} min={m} d={d} diverge at {j}: top-2 gap {float(t2[1]-t2[0]):.3f}", flush=True)
for m in mins:
    pm = statistics.median(R[(m, 0)])
    print(f"MIN_ROWS={m}: plain {pm:.1f} " + " ".join(f"d{d} {statistics.median(R[(m, d)]):.1f} (x{statistics.median(R[(m, d)])/pm:.2f}, exact {sum(ex[(m, d)])}/{len(ex[(m, d)])})" for d in depths))
