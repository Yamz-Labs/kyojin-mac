#!/usr/bin/env python3
"""S4: greedy generation of a Kyojin-patched DS4.1 pack on MLX against the box engine's reference (tools/ref_ds41_box.py -> results_mac/ref_ds41/ref.json).
usage: check_ds41_ref.py MODEL_DIR REF.json [--new 64] [--prompts sky,code] [--dump OUT.json] [--memmap-gather] [--self-ref] [--s5]
Per prompt, from the reference prompt ids (no tokenizer or chat template on the Mac):
  (a) free running greedy: matching prefix length and first divergence; at that position the reference margin (top1 - top2 log-prob; a flip below
      ~0.05 is fp16 vs fp32 rounding, a flip above 0.5 is a bug) and whether the Mac token is in the reference top-5
  (b) teacher forced through one prefill pass over prompt + reference tokens: top-1 agreement at every position (no cascade after a flip) and
      the mean |log p(reference token) Mac - box|  (when the reference holds log-probs)
  (c) teacher forced through the decode path (one row per step): top-1 agreement
The verdict line says PASS when (b) and (c) agree on >= 95 % of the positions and every free-running flip has a margin < 0.5.
--self-ref: no reference file needed (REF.json may be '-'): the model's own free-running greedy output is the reference, to prove the pipeline
(a) and (c) must read 100 % then; (b) can lose a near-tie or two (the prefill rows use another kernel path). The Engram tables (when the pack has an Engram layer) are validated and read with parallel preads (kyojin_mlx/engram_mlx.py)."""
import argparse
import json
import os
import sys
import time

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kyojin_mlx import patch as P

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("ref")
ap.add_argument("--new", type=int, default=64)
ap.add_argument("--prompts", default="")
ap.add_argument("--dump")
ap.add_argument("--memmap-gather", action="store_true", help="keep the mlx-vlm memmap gather instead of the parallel pread one")
ap.add_argument("--self-ref", action="store_true")
ap.add_argument("--s5", action="store_true", help="group same-input linears (kyojin_mlx/fused_group.py: fewer launches per token, bit-identical output)")
a = ap.parse_args()

t0 = time.time()
model, spec, cfg = P.build_model(a.model)
tables = None
try:
    from kyojin_mlx import engram_rows as ER
    es = ER.EngramSpec.from_config(cfg)
    if es is not None:
        tables = ER.EngramTables(a.model, es)
        bad = tables.validate()
        assert not bad, "Engram tables do not match the config:\n  " + "\n  ".join(bad)
except ImportError:
    pass
rep = P.patch_model(model, a.model, spec, cfg)
if tables is not None and not a.memmap_gather:
    from kyojin_mlx import engram_mlx
    engram_mlx.install_parallel_gather(model, tables)
lm = model.language_model
if a.s5:
    from kyojin_mlx import fused_group as FG
    _i = FG.install(model)
    print("S5:", {k: v for k, v in _i.items() if k != "names"}, flush=True)
print(f"loaded in {time.time() - t0:.1f}s  {rep}  peak {mx.get_peak_memory() / 2**30:.1f} GiB  engram={'yes' if tables else 'no'}", flush=True)


def logits_of(ids, cache):
    out = lm(mx.array([ids]), cache=cache)
    return (out.logits if hasattr(out, "logits") else out).astype(mx.float32)


def greedy(prompt, n, eos):
    cache = lm.make_cache()
    lg = logits_of(prompt, cache)[:, -1]
    out = []
    for _ in range(n):
        t = int(mx.argmax(lg, axis=-1).item())
        out.append(t)
        if eos is not None and t == eos:
            break
        lg = logits_of([t], cache)[:, -1]
    return out


refs = [] if a.self_ref else json.load(open(a.ref))["prompts"]
if a.self_ref:
    vocab = cfg.get("text_config", cfg)["vocab_size"]
    rng = np.random.default_rng(0)
    for nm in ("selfA", "selfB"):
        ids = [int(x) for x in rng.integers(3, vocab, 12)]
        refs.append({"name": nm, "prompt_ids": ids, "out_ids": greedy(ids, a.new, None), "eos_id": None})
if a.prompts:
    refs = [r for r in refs if r["name"] in a.prompts.split(",")]
res, tot = [], dict(n=0, b=0, c=0, flips=0, big=0)
for r in refs:
    prompt, ref, eos = r["prompt_ids"], r["out_ids"][:a.new], r.get("eos_id")
    n = len(ref)
    top5 = r.get("top5")
    margin = r.get("margin")
    if r.get("chosen_logp") is None and top5:                                   # greedy: the chosen token is the top-1 entry
        r["chosen_logp"] = [t[0][1] for t in top5]
    t1 = time.time()
    out = greedy(prompt, n, eos)
    dt = time.time() - t1
    div = next((i for i in range(min(n, len(out))) if out[i] != ref[i]), None)
    if div is None and len(out) < n:
        div = len(out)
    match = n if div is None else div
    # (b) one prefill over prompt + reference, argmax at the positions that predict reference tokens
    full = prompt + ref
    lg = logits_of(full, lm.make_cache())[0]
    pred = np.array(mx.argmax(lg[len(prompt) - 1: len(full) - 1], axis=-1))
    b_ok = int((pred == np.array(ref)).sum())
    dlp = None
    if r.get("chosen_logp") is not None:
        lp = np.array(lg[len(prompt) - 1: len(full) - 1] - mx.logsumexp(lg[len(prompt) - 1: len(full) - 1], axis=-1, keepdims=True))
        mine = lp[np.arange(n), np.array(ref)]
        dlp = float(np.mean(np.abs(mine - np.array(r["chosen_logp"][:n]))))
    # (c) teacher forced, decode path
    cache = lm.make_cache()
    l1 = logits_of(prompt, cache)[:, -1]
    c_ok = 0
    for i in range(n):
        c_ok += int(int(mx.argmax(l1, axis=-1).item()) == ref[i])
        if i + 1 < n:
            l1 = logits_of([ref[i]], cache)[:, -1]
    info = dict(name=r["name"], new=n, free_match=match, first_div=div, prefill_top1=b_ok, decode_top1=c_ok, mean_abs_dlogp=dlp, sec=round(dt, 1))
    if div is not None and div < n:
        info["mac_token"], info["ref_token"] = out[div] if div < len(out) else None, ref[div]
        if margin:
            info["ref_margin"] = margin[div]
            info["mac_in_ref_top5"] = bool(top5 and out[div] in [t[0] for t in top5[div]])
            tot["big"] += int(margin[div] >= 0.5)
        tot["flips"] += 1
    res.append(info)
    tot["n"] += n; tot["b"] += b_ok; tot["c"] += c_ok
    print(json.dumps(info), flush=True)
pb, pc = tot["b"] / max(tot["n"], 1), tot["c"] / max(tot["n"], 1)
ok = pb >= 0.95 and pc >= 0.95 and tot["big"] == 0
if a.self_ref:
    ok = pc == 1.0 and pb >= 0.9 and all(r["first_div"] is None for r in res)           # prefill rows run another kernel path: near-ties may flip
has_margin = any(r.get("margin") for r in refs)
print(f"VERDICT {'PASS' if ok else 'FAIL'}{'' if has_margin or a.self_ref else ' (reference has no margins: top-1 gates only, flips cannot be sorted into rounding or bug)'}: prefill top-1 {tot['b']}/{tot['n']} ({100 * pb:.1f} %), decode top-1 {tot['c']}/{tot['n']} ({100 * pc:.1f} %), "
      f"free-running flips {tot['flips']}/{len(res)} prompts, flips at margin >= 0.5: {tot['big']}, peak {mx.get_peak_memory() / 2**30:.1f} GiB")
if a.dump:
    json.dump(dict(prompts=res, totals=tot, verdict=ok), open(a.dump, "w"), indent=1)
if tables is not None:
    tables.close()
sys.exit(0 if ok else 1)
