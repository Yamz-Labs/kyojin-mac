"""Multi-user decode on a patched mlx-vlm slice (DS4.1 / GLM): ms/step, launches/step, per-user cost at N = 1, 2, 4. Component timings (moe block, routed gather, attention layer) are the minimum of 60 synced calls.
usage: bench_mu_vlm.py PACK [--ns 1,2,4] [--steps 24] [--rounds 5] [--lens 37,64,90,131] [--json results_mac/macair7/out.json]
ms/step: interleaved rounds (N=1,2,4,1,2,4,...), median of `rounds`, each round = `steps` timed steps after a warm-up (mx.eval per step).
solo*N = N sequential solo steps (the engine without batching). launches = graph nodes of one step that dispatch GPU work (view ops
Reshape/Transpose/Broadcast/Slice/... excluded), from mx.export_to_dot of the step output; moe_launches = the same for the MoE block alone."""
import argparse, collections, json, os, re, statistics, time

import numpy as np
import mlx.core as mx

from kyojin_mlx import patch as P, rows as R, moe as M

VIEWS = {"Reshape", "Transpose", "Broadcast", "Squeeze", "ExpandDims", "Flatten", "Slice", "AsStrided", "View", "Unflatten", "StopGradient", "Depends"}

ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("--ns", default="1,2,4"); ap.add_argument("--steps", type=int, default=24); ap.add_argument("--rounds", type=int, default=5)
ap.add_argument("--lens", default="37,64,90,131"); ap.add_argument("--json"); ap.add_argument("--union", action="store_true"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--core", default=None, help="user = per-user DS4.1 attention cores (before the batched core)")
ap.add_argument("--count-steps", type=int, default=8)
a = ap.parse_args()
ns = [int(v) for v in a.ns.split(",")]
model, spec, cfg = P.build_model(a.pack)
P.patch_model(model, a.pack, spec, cfg)
R.enable(model, spec, union=a.union, core=a.core)
vocab = cfg["text_config"]["vocab_size"]
rng = np.random.default_rng(a.seed)
lens = [int(v) for v in a.lens.split(",")][:max(ns)]
users = [R.prefill(model, rng.integers(16, vocab - 16, n).tolist()) for n in lens]
first = [int(mx.argmax(lg, axis=-1).item()) for _, lg in users]


def count(out) -> tuple[int, int, dict]:
    os.makedirs("results_mac/macair7", exist_ok=True)
    dot = "results_mac/macair7/_graph.dot"
    mx.export_to_dot(dot, out)
    labels = re.findall(r'label ?= ?"([^"]*)"', open(dot).read())
    os.remove(dot)
    c = collections.Counter(l for l in labels if l not in VIEWS)
    global LAST_COUNTER
    LAST_COUNTER = c
    return sum(c.values()), len(labels), dict(c.most_common(8))


def fresh(N):
    return R.batch([R.fork(c) for c, _ in users[:N]])


def eval_caches(bc):
    arrs = []
    for c in bc:
        for v in vars(c).values():
            if isinstance(v, mx.array):
                arrs.append(v)
            elif isinstance(v, (list, tuple)):
                arrs += [x for x in v if isinstance(x, mx.array)]
    mx.eval(arrs)


res = {"pack": a.pack, "lens": lens, "union": a.union, "core": a.core or "batched", "n": {}}
state = {}
for N in ns:
    bc = fresh(N)
    cur = mx.array(first[:N])
    for _ in range(2):                                   # warm-up: compile kernels, settle lazy cache updates
        lg = R.step(model, bc, cur); mx.eval(lg); eval_caches(bc)
        cur = mx.argmax(lg, axis=-1)
    # launches of consecutive steps (a compression group completes on some users at some steps only): mean, min, max over `count_steps`
    series = []
    for _ in range(a.count_steps):
        lg = R.step(model, bc, cur)
        series.append(count(mx.argmax(lg, axis=-1)))
        full = dict(LAST_COUNTER)
        mx.eval(lg); eval_caches(bc)
        cur = mx.argmax(lg, axis=-1)
    launches, nodes, top = round(statistics.mean(c[0] for c in series), 1), series[-1][1], series[-1][2]
    res['n'][N] = dict(full=full, launches_min=min(c[0] for c in series), launches_max=max(c[0] for c in series))
    state[N] = [bc, cur]
    res["n"][N].update(launches=launches, graph_nodes=nodes, top=top)

# MoE block alone (per layer): launches and time
blocks = [(n, m) for n, m in model.language_model.named_modules() if type(m).__name__ in ("DeepseekV41MoE", "Glm5NextMoE")]
H = cfg["text_config"]["hidden_size"]
for N in ns:
    x = mx.array(rng.standard_normal((N, 1, H)).astype(np.float16))
    for name, blk in blocks[:1]:
        y = blk(x); mx.eval(y)
        y = blk(x)
        cnt = count(y)
        mx.eval(y)
        ts = []
        for _ in range(60):
            t = time.perf_counter(); mx.eval(blk(x)); ts.append(time.perf_counter() - t)
        res["n"][N]["moe_launches"] = cnt[0]; res["n"][N]["moe_top"] = cnt[2]; res["n"][N]["moe_kernels"] = cnt[2].get("CustomKernel"); res["n"][N]["moe_ms"] = round(1e3 * min(ts), 3)
        res["n"][N]["moe_block"] = name

# the routed-expert dispatch alone (MoeGather: one set of kernels for all rows of the layer)
mg = [m.moe for _, m in model.language_model.named_modules() if type(m).__name__ == "KyojinMoe"][0]
E = mg.g.experts
for N in ns:
    x = mx.array(rng.standard_normal((N, H)).astype(np.float16))
    ids = mx.array(np.stack([rng.permutation(E)[:mg.topk] for _ in range(N)]).astype(np.int32))
    wts = mx.array(rng.random((N, mg.topk)).astype(np.float32))
    mx.eval(mg(x, ids, wts))
    res["n"][N]["gather_launches"] = count(mg(x, ids, wts))[0]
    ts = []
    for _ in range(60):
        t = time.perf_counter(); mx.eval(mg(x, ids, wts)); ts.append(time.perf_counter() - t)
    res["n"][N]["gather_ms"] = round(1e3 * min(ts), 3)

# DS4.1: the attention module of layer 0 alone (projections batched, per-user cores): ms and launches per layer
layer0 = model.language_model.layers[0]
if hasattr(layer0, "attn"):
    for N in ns:
        bc = fresh(N)
        x = mx.array(rng.standard_normal((N, 1, H)).astype(np.float16))
        mx.eval(layer0.attn(x, bc[0]))
        o = layer0.attn(x, bc[0])
        res["n"][N]["attn_launches"] = count(o)[0]
        mx.eval(o)
        ts = []
        for _ in range(60):
            t = time.perf_counter(); mx.eval(layer0.attn(x, bc[0])); ts.append(time.perf_counter() - t)
        res["n"][N]["attn_ms"] = round(1e3 * min(ts), 3)

times = {N: [] for N in ns}
solo_times = []
for r in range(a.rounds):
    for N in ns:
        bc, cur = state[N]
        t = time.perf_counter()
        for _ in range(a.steps):
            lg = R.step(model, bc, cur); mx.eval(lg); cur = mx.argmax(lg, axis=-1)
        times[N].append(1e3 * (time.perf_counter() - t) / a.steps)
        state[N][1] = cur
# Python graph build alone, and the pipelined step (async_eval of the next step queued while the GPU runs the current one, as a serving loop does)
for N in ns:
    bc, cur = state[N]
    bt = []
    for _ in range(12):
        t = time.perf_counter(); lg = R.step(model, bc, cur); bt.append(time.perf_counter() - t); cur = mx.argmax(lg, axis=-1); mx.eval(cur)
    res["n"][N]["build_ms"] = round(1e3 * statistics.median(bt), 3)
    pt = []
    for _ in range(a.rounds):
        t = time.perf_counter()
        nxt = mx.argmax(R.step(model, bc, cur), axis=-1); mx.async_eval(nxt)
        for _ in range(a.steps - 1):
            cur = nxt
            nxt = mx.argmax(R.step(model, bc, cur), axis=-1); mx.async_eval(nxt)
            mx.eval(cur)
        mx.eval(nxt); cur = nxt
        pt.append(1e3 * (time.perf_counter() - t) / a.steps)
    res["n"][N]["ms_step_pipelined"] = round(statistics.median(pt), 3)
    state[N][1] = cur
for N in ns:
    ms = statistics.median(times[N])
    res["n"][N]["ms_step"] = round(ms, 3)
    res["n"][N]["ms_step_rounds"] = [round(v, 3) for v in times[N]]
base = res["n"][ns[0]]["ms_step"]
for N in ns:
    d = res["n"][N]
    d["ms_per_user"] = round(d["ms_step"] / N, 3)
    d["aggregate_vs_N_solo"] = round(N * base / d["ms_step"], 3)
res["peak_gb"] = round(mx.get_peak_memory() / 2**30, 2)
for N in ns:
    d = res["n"][N]
    print(f"N={N}: {d['ms_step']} ms/step ({d['ms_per_user']} ms/user, {d['aggregate_vs_N_solo']}x vs {N} solo steps), launches/step {d['launches']} "
          f"[{d['launches_min']}..{d['launches_max']}] (graph nodes {d['graph_nodes']}; python build {d['build_ms']} ms, pipelined {d['ms_step_pipelined']} ms/step), moe block {d.get('moe_launches')} launches {d.get('moe_ms')} ms, routed gather {d['gather_launches']} launches {d['gather_ms']} ms"
          + (f", attention layer {d['attn_launches']} launches {d['attn_ms']} ms" if "attn_ms" in d else ""))
print("peak GB", res["peak_gb"])
if a.json:
    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    json.dump(res, open(a.json, "w"), indent=1)
