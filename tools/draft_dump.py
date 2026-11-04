"""MAC side of the draft numerics check: one draft call of the DSpark head on a pack, with every input and intermediate saved for the CPU reference
(tools/draft_ref.py, box, runs the checkpoint's own model.py DSparkBlock / forward_head on the same dequantized weights).
usage: draft_dump.py PACK OUT.npz [--len 150] [--split 100] [--chunk 64] [--seed 0] [--depth 4]

The trunk (the pack's own layers) runs the prompt and gives the taps of every position. The drafter is fed them the way the speculative loop does
(--split positions at once, then one row at a time), then drafts after the pending token (argmax of the last prefill logits) at position --len.
OUT.npz: ids, taps [len, n_taps * H] f32 (fp16 values), pending, split, rows<s> [<= window, 512] f32 (main-kv rows of stage s, position order),
         attn [stages, block, H] (attention output of every stage), logits [block, V] (markov bias included), x [block, H] (pre-norm collapsed state), conf_logit [block], out_ids [block]."""
import argparse
import time

import numpy as np
import mlx.core as mx

from kyojin_mlx import patch as P, rows as R, spec as S, dspark as D

ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("out")
ap.add_argument("--len", type=int, default=150); ap.add_argument("--split", type=int, default=100)
ap.add_argument("--chunk", type=int, default=64); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--no-fq-out", action="store_true", help="draft attention without the fp8 round trip before wo_a (what the trunk of mlx-vlm does)")
ap.add_argument("--p0-shift", type=int, default=0, help="negative control: draft with the block start position moved by this many positions")
ap.add_argument("--no-fq-main", action="store_true", help="main_proj without the fp8 round trip of its input")
a = ap.parse_args()
model, spec, cfg = P.build_model(a.pack)
P.patch_model(model, a.pack, spec, cfg)
print("trunk:", R.enable(model, spec), flush=True)
t0 = time.time()
drafter = D.build_drafter(model, a.pack, spec, cfg)
drafter.fq_main = not a.no_fq_main
if a.no_fq_out:
    for blk in drafter.lm.layers:
        blk.attn.fq_out = False
print(f"drafter loaded in {time.time() - t0:.1f}s: stages {drafter.n_stages}, taps {drafter.tap_ids}, block {drafter.block}, window {drafter.window}", flush=True)
print("spec.install:", S.install(model, spec, tap_layers=set(drafter.tap_ids)), flush=True)
vocab = cfg["text_config"]["vocab_size"]
prompt = np.random.default_rng(a.seed).integers(16, vocab - 96, a.len).tolist()
cache, lg, taps, first = S.prefill(model, prompt, drafter.tap_ids, a.chunk, keep=10 ** 9)
assert first == 0, first
tl = [taps[l] for l in drafter.tap_ids]                                            # one [len, H] array per target layer, in config order
pending = int(mx.argmax(lg[0]).item())
drafter.reset()
drafter.append([t[:a.split] for t in tl], 0)
for i in range(a.split, a.len):
    drafter.append([t[i:i + 1] for t in tl], i)
drafter.debug = {}
toks = drafter.draft(pending, a.len + a.p0_shift, a.depth)
d = drafter.debug
rows = [np.array(r[0].astype(mx.float32)) for r in drafter.ctx.rows]
out = dict(ids=np.array(prompt), pending=np.array(pending), split=np.array(a.split),
           taps=np.array(mx.concatenate([t.astype(mx.float32) for t in tl], axis=-1)),
           logits=np.array(d["logits"]), attn=np.array(d["attn"].astype(mx.float32)), x=np.array(d["x"].astype(mx.float32)), conf_logit=np.array(d["conf_logit"]), out_ids=np.array(d["ids"]))
for s, r in enumerate(rows):
    out[f"rows{s}"] = r
np.savez(a.out, **out)
print(f"pending {pending} at position {a.len}; block ids {np.array(d['ids']).tolist()}; confidence {np.round(1 / (1 + np.exp(-np.array(d['conf_logit']))), 3).tolist()}; "
      f"proposals {toks}; rows per stage {[r.shape for r in rows]}", flush=True)
