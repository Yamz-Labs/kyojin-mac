"""Calibration forward throughput of ONE DS4.1 decoder layer (attention + MoE + mHC, unquantized bf16 weights, stock mlx-vlm code, no patch): tok/s and peak memory.
usage: calib_layer_bench.py MODEL_DIR [--layer N] [--rows 250] [--cols 2048] [--batch 1] [--reps 10] [--warmup 2] [--set key=json ...] [--json F]

What it times: the quantizer (exllamav3 conversion, convert_model.py) keeps the calibration set as `cal_rows` rows of `cal_cols` tokens (defaults 250 x 2048, the
DS4.1 recipe packs 250 x 2048 in data/cal3.safetensors) and runs module.forward on ONE row at a time ([1, 2048, hidden]) for every module. So the measured unit is one
forward of the layer over --batch x --cols tokens (default 1 x 2048); the line "all rows" extrapolates it to --rows rows (one pass over the set, capture or state advance).
MODEL_DIR only supplies config.json (the pack weights are not read): the layer is built from the config with random bf16 weights and N(0, 1) hidden states, so
the routing is near uniform over the experts (real calibration text is more skewed: read the figure as the speed of a balanced layer). The first layer whose attention
reads its own kv / index sources is the default (--layer to choose). Peak memory is mx.get_peak_memory() over the whole run (weights included).
Wrap the run in lockf on a shared machine (the Air); on the Ultra nothing else should run. --set n_routed_experts=16 shrinks the experts to fit the 4 GB rule of the Air."""
import argparse, json, statistics, sys, time

import mlx.core as mx

from kyojin_mlx import patch as P

ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("--layer", type=int); ap.add_argument("--rows", type=int, default=250); ap.add_argument("--cols", type=int, default=2048)
ap.add_argument("--batch", type=int, default=1); ap.add_argument("--reps", type=int, default=10); ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--set", nargs="*", default=[]); ap.add_argument("--json")
a = ap.parse_args()
over = {k: json.loads(v) for k, v in (s.split("=", 1) for s in a.set)}
over = {"engram_layer_ids": [], "engram_num_embeddings": [], **over}            # the Engram tables are not part of a layer forward (and need the pack shards)
model, spec, cfg = P.build_model(a.pack, over)
lm = model.language_model
tc = cfg["text_config"]
src = [i for i in tc.get("kv_source_layer_ids", [0]) if i in tc.get("index_source_layer_ids", [i])]
li = a.layer if a.layer is not None else (src[0] if src else 0)
blk = lm.layers[li]
hc, H = int(tc["hc_mult"]), int(tc["hidden_size"])
blk.set_dtype(mx.bfloat16, lambda dt: dt == mx.float32)                # weights bf16 (what the unquantized checkpoint is dequantized to)
for n in ("hc_attn_fn", "hc_ffn_fn", "hc_attn_base", "hc_ffn_base"):  # the mHC mixes stay fp32 as built
    setattr(blk, n, getattr(blk, n).astype(mx.float32))
mx.eval(blk.parameters())
nbytes = sum(v.nbytes for _, v in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(blk.parameters()))
print(f"layer {li} of {len(lm.layers)}: {nbytes / 2 ** 30:.2f} GiB weights (bf16), {tc.get('n_routed_experts')} routed experts, hidden {H}, hc_mult {hc}; "
      f"batch {a.batch} x {a.cols} tokens per forward", flush=True)
mx.random.seed(0)
h = mx.random.normal((a.batch, a.cols, hc, H)).astype(mx.bfloat16)
pre = lm.__class__.__module__ and __import__("importlib").import_module(type(blk).__module__).make_identity_pre_mix(a.batch, a.cols, hc)
mx.eval(h, pre)


def fwd():
    cache = lm.make_cache()[0]
    out = blk(h, pre, None, cache)
    mx.eval(out)
    return out


mx.reset_peak_memory()
for _ in range(a.warmup):
    out = fwd()
assert bool(mx.all(mx.isfinite(out[0].astype(mx.float32))).item()), "non-finite layer output"
ts = []
for _ in range(a.reps):
    mx.synchronize()
    t = time.perf_counter()
    fwd()
    mx.synchronize()
    ts.append(time.perf_counter() - t)
tok = a.batch * a.cols
tps = [tok / t for t in ts]
peak = mx.get_peak_memory() / 2 ** 30
rep = dict(layer=li, batch=a.batch, cols=a.cols, rows=a.rows, reps=a.reps, tok_s_min=min(tps), tok_s_p25=sorted(tps)[len(tps) // 4], tok_s_median=statistics.median(tps), tok_s_max=max(tps),
           ms_median=statistics.median(ts) * 1e3, peak_gib=peak, weights_gib=nbytes / 2 ** 30,
           all_rows_s_median=a.rows * a.cols / statistics.median(tps))
print(f"forward ms: min {min(ts) * 1e3:.1f} median {statistics.median(ts) * 1e3:.1f} max {max(ts) * 1e3:.1f}   tok/s: min {rep['tok_s_min']:.0f} p25 {rep['tok_s_p25']:.0f} "
      f"median {rep['tok_s_median']:.0f} max {rep['tok_s_max']:.0f}", flush=True)
print(f"all rows ({a.rows} x {a.cols} = {a.rows * a.cols} tokens, one pass over the calibration set): {rep['all_rows_s_median']:.1f} s at the median", flush=True)
print(f"peak memory (mx.get_peak_memory, weights included): {peak:.2f} GiB", flush=True)
if a.json:
    json.dump(rep, open(a.json, "w"), indent=1)
