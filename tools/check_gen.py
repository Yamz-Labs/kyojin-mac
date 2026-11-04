"""Greedy-generation check against the box CPU reference (tools/ref_q35.py output).
usage: check_gen.py MODEL_DIR REF_DIR [s5]   (s5: install the S5 grouped same-input linears first; the output must not change)
Per prompt: (a) free-running greedy: first divergence + matching prefix length; (b) teacher-forced top-1 agreement over all
generated positions, through the decode path (one row = Metal GEMV) and through one prefill pass (decode + matmul);
(c) at a free-running divergence: reference top-2 logit gap (a small gap = near-tie flip from fp16 vs fp32 rounding)."""
import sys, json
import numpy as np
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from kyojin_mlx.exl3_linear import load_exl3
model_dir, ref = sys.argv[1], sys.argv[2]
cb = json.load(open(f"{model_dir}/config.json"))["quantization_config"]["codebook"]
model, _ = load_exl3(model_dir, cb=cb)
if len(sys.argv) > 3 and sys.argv[3] == "s5":
    from kyojin_mlx import fused_group as FG
    print("S5:", {k: v for k, v in FG.install(model).items() if k != "names"})
R = json.load(open(f"{ref}/ref_tokens.json"))
tot_ok = tot_n = tot_pf_ok = 0
for pi, r in enumerate(R):
    ids, rt = r["prompt_ids"], r["tokens"]; n = len(rt)
    # (a) free running
    cache = make_prompt_cache(model)
    lg = model(mx.array(ids)[None], cache=cache)[:, -1]
    out = []
    for i in range(n):
        t = mx.argmax(lg, axis=-1); mx.eval(t); out.append(int(t.item()))
        lg = model(t[:, None], cache=cache)[:, -1]
    div = next((i for i in range(n) if out[i] != rt[i]), None)
    # (b) teacher forced, decode path
    cache = make_prompt_cache(model)
    lg = model(mx.array(ids)[None], cache=cache)[:, -1]
    ok = 0
    for i in range(n):
        t = int(mx.argmax(lg, axis=-1).item()); ok += t == rt[i]
        lg = model(mx.array([[rt[i]]]), cache=cache)[:, -1]
    # (b2) teacher forced, one prefill pass over prompt + reference tokens
    full = mx.array(ids + rt[:-1])[None]
    lgf = model(full, cache=make_prompt_cache(model))[0, len(ids) - 1:]
    pf = np.array(mx.argmax(lgf, axis=-1)).tolist(); pf_ok = sum(int(a == b) for a, b in zip(pf, rt))
    tot_ok += ok; tot_n += n; tot_pf_ok += pf_ok
    gap = ""
    if div is not None:
        l = r["top5_logits"][div]; gap = f" ref top1-top2 logit gap at divergence {l[0] - l[1]:.3f} (ours picked ref rank {r['top5_ids'][div].index(out[div]) + 1 if out[div] in r['top5_ids'][div] else '>5'})"
    print(f"prompt {pi}: free-run match {n if div is None else div}/{n}" + (f", first divergence at token {div}{gap}" if div is not None else ", identical")
          + f" | teacher-forced top-1 decode {ok}/{n}, prefill {pf_ok}/{n}", flush=True)
print(f"TOTAL teacher-forced top-1: decode {tot_ok}/{tot_n} = {100*tot_ok/tot_n:.1f}%, prefill {tot_pf_ok}/{tot_n} = {100*tot_pf_ok/tot_n:.1f}%")
