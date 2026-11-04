"""Reference engine bar: launches and CPU ms per layer of the stock mlx-lm DeepSeek-V4.1 class (mlx_lm.models.deepseek_v41, the class oMLX / mlx-lm run),
random weights, the DS4.1 layer widths, quantized the way the checkpoint is stored (mxfp8 dense, mxfp4 routed experts: the formats `Model.sanitize` keeps packed).
usage: ref_mlxlm_cost.py [--reps 15] [--len 135] [--experts 32] [--json F]

Layers of the test model (all ratio 2 except the last, like the ds2 slices): 0 = full (compressor + indexer + candidate source), 1 = reuse (reads the shared
compressed KV and top-k), 2 = reindex (own indexer, shared compressed KV, consumes the candidate mask of layer 0), 3 = reuse, 4 = reindex again, 5 = local
(window only, ratio 0). The lazy graph only holds what the output reads: the candidate mask of layer 0 is in the graph only when a later indexer consumes it, so
reindex #1 pays it and reindex #2 does not (candidate path = reindex #1 - reindex #2; in the real 40-layer model it is paid once).
Per layer kind: graph dispatches (lazy graph nodes minus view primitives, as tools/count_launches.py) and CPU ms (python graph build + async_eval encode,
medians and mins) by differencing the cost of a step that runs a subset of the layers, same method as tools/layer_cpu.py. Both cache parities are timed
(ratio 2 closes a group every second token)."""
import argparse, copy, json, os, statistics, sys, time
if "--default-buffers" not in sys.argv:        # measuring device: one command buffer per step, else async_eval blocks on the GPU once >10 buffers are in flight (sample trace, macair12)
    os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "100000"); os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "1000000")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models import deepseek_v41 as D
from mlx_lm.models.switch_layers import SwitchLinear
from count_launches import graph_census

ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=15); ap.add_argument("--len", type=int, default=135)
ap.add_argument("--experts", type=int, default=32); ap.add_argument("--json"); ap.add_argument("--default-buffers", action="store_true"); ap.add_argument("--loop", help="only loop this case (for sampling with `sample`)"); ap.add_argument("--secs", type=float, default=15)
a = ap.parse_args(); a_json = a.json

ARGS = D.ModelArgs(vocab_size=4096, hidden_size=5120, num_hidden_layers=6, num_attention_heads=64, head_dim=512, q_lora_rank=1280, qk_rope_head_dim=64, o_groups=8,
                   o_lora_rank=1024, moe_intermediate_size=2304, n_routed_experts=a.experts, num_experts_per_tok=6, sliding_window=128,
                   compress_ratios=(2, 2, 2, 2, 2, 0), kv_source_layer_ids=(0,), index_source_layer_ids=(0, 2, 4), candidate_source_layer_id=0, candidate_topk_blocks=2048,
                   rope_scaling={"rope_type": "yarn", "factor": 16, "beta_fast": 32, "beta_slow": 1, "original_max_position_embeddings": 65536}, max_position_embeddings=1048576)
mx.random.seed(0)
model = D.Model(ARGS)


def small_random(layer):
    """Quantize one layer in place (dense mxfp8, switch experts mxfp4), evaluating parameter by parameter to keep the dense temporaries small."""
    def pred(path, m):
        if isinstance(m, (nn.Linear, SwitchLinear)):
            return {"group_size": 32, "bits": 4, "mode": "mxfp4"} if "switch_mlp" in path else {"group_size": 32, "bits": 8, "mode": "mxfp8"}
        return False
    nn.quantize(layer, class_predicate=pred)
    for _, v in tree_flatten(layer.parameters()):
        mx.eval(v)


for l in model.model.layers:
    small_random(l)
for p in (model.model.embed_tokens, model.lm_head):
    for _, v in tree_flatten(p.parameters()):
        mx.eval(v)
for l in model.model.layers:                                     # router weights are fp32 parameters that quantize left alone: small random values
    l.mlp.gate.weight = mx.random.normal(l.mlp.gate.weight.shape) * 0.02
    mx.eval(l.mlp.gate.weight)
full_layers = list(model.model.layers)
ids = np.random.default_rng(0).integers(16, ARGS.vocab_size - 16, a.len + 2).tolist()


def prefill(n):
    cache = model.make_cache()
    out = model(mx.array(ids[:n])[None], cache=cache)[:, -1]
    mx.eval(out)
    return cache, mx.argmax(out.astype(mx.float32), axis=-1)


def fork(cache):
    return [copy.copy(c) for c in cache]


CASES = {"none": [], "full": [0], "f+r": [0, 1], "f+r+x1": [0, 1, 2], "f+r+x1+r": [0, 1, 2, 3], "f+r+x1+r+x2": [0, 1, 2, 3, 4], "local": [5]}
BASES = {n: prefill(n) for n in (a.len, a.len + 1)}               # both parities of the open compression group
def run(k, cache, tok):
    """One decode step through the layers of case k (body of TextModel.__call__ on a cache subset); returns the lazy next token."""
    idx = CASES[k]
    model.model.layers = [full_layers[i] for i in idx]
    c = fork(cache)
    sub = [c[i] for i in idx]
    h = model.model.embed_tokens(tok[:, None])
    batch, length, _ = h.shape
    start_pos = c[0].offset
    prev = 0 if c[0].window is None else c[0].window.shape[1]
    widx = mx.broadcast_to(D._window_indices(previous=prev, length=length, window=ARGS.sliding_window), (batch, length, min(ARGS.sliding_window, prev + length)))
    streams = mx.broadcast_to(h[:, :, None, :], (batch, length, ARGS.hc_mult, ARGS.hidden_size))
    pre_mix = mx.zeros((batch, length, ARGS.hc_mult), dtype=mx.float32); pre_mix[..., 0] = 1.0
    shared = {}
    for layer, lc in zip(model.model.layers, sub):
        streams, pre_mix = layer(streams, pre_mix, start_pos=start_pos, window_idx=widx, cache=lc, shared=shared)
    out = model.model.norm(D._collapse(streams, pre_mix))
    return mx.argmax(model.lm_head(out)[:, -1].astype(mx.float32), axis=-1)


if a.loop:
    cache, tok = BASES[a.len]
    print("looping", a.loop, flush=True); t_end = time.time() + a.secs
    while time.time() < t_end:
        out = run(a.loop, cache, tok); mx.async_eval(out); mx.eval(out)
    sys.exit()
res = {k: {par: dict(build=[], enc=[]) for par in BASES} for k in CASES}
census = {k: graph_census(run(k, BASES[a.len][0], BASES[a.len][1])) for k in CASES}
for rep in range(a.reps + 2):
    for par, (cache, tok) in BASES.items():
        for k in CASES:
            t0 = time.perf_counter(); out = run(k, cache, tok); t1 = time.perf_counter(); mx.async_eval(out); t2 = time.perf_counter(); mx.eval(out)
            if rep >= 2:
                res[k][par]["build"].append(t1 - t0); res[k][par]["enc"].append(t2 - t1)
model.model.layers = full_layers

def tot(k, q, par):
    v = sorted(b + e for b, e in zip(res[k][par]["build"], res[k][par]["enc"])); return 1e3 * v[int(q * (len(v) - 1))]


def avg(k, q):
    """Mean over the two parities of the open compression group (ratio 2 closes a group every second token) of the q-quantile of build+encode, ms."""
    return sum(tot(k, q, par) for par in BASES) / len(BASES)


print(f"{'case':14s} {'p25':>7s} {'min':>7s} | per-parity p25   dispatches   (CPU build+encode ms, parity mean)")
for k in CASES:
    print(f"{k:14s} {avg(k, .25):7.3f} {avg(k, 0):7.3f} | " + " ".join(f"{tot(k, .25, par):7.3f}" for par in BASES) + f"   {census[k]['dispatches']}")
out = {"cases": {k: dict(dispatches=census[k]["dispatches"], by_type=census[k]["by_type"], p25=avg(k, .25), min=avg(k, 0)) for k in CASES}}
KINDS = (("full", "full", "none"), ("reuse", "f+r", "full"), ("reindex1", "f+r+x1", "f+r"), ("reuse2", "f+r+x1+r", "f+r+x1"), ("reindex2", "f+r+x1+r+x2", "f+r+x1+r"), ("local", "local", "none"))
for name, k1, k0 in KINDS:
    dl = census[k1]["dispatches"] - census[k0]["dispatches"]
    r = {q: avg(k1, qq) - avg(k0, qq) for q, qq in (("p25", .25), ("min", 0))}
    out[name] = dict(dispatches=dl, **{"ms_" + q: v for q, v in r.items()})
    print(f"per {name:9s} layer: {dl:4d} dispatches | CPU build+encode ms (parity mean): p25 {r['p25']:.3f}  min {r['min']:.3f}  | {1e3 * r['p25'] / max(dl, 1):.1f} us/launch (p25)")
print("candidate path (reindex1 - reindex2):", out["reindex1"]["dispatches"] - out["reindex2"]["dispatches"], "dispatches,", f"{out['reindex1']['ms_p25'] - out['reindex2']['ms_p25']:.3f} ms (p25)")
for name, k1, k0 in KINDS:
    d = {t: census[k1]["by_type"].get(t, 0) - census[k0]["by_type"].get(t, 0) for t in census[k1]["by_type"]}
    out[name]["types"] = {t: v for t, v in d.items() if v}
if a_json:
    json.dump(out, open(a_json, "w"), indent=1)
