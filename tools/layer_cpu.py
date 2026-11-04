"""CPU cost (python graph build + async_eval encode) and launches of ONE decode layer of each kind, from the real patched model, per token: K = --chain
decode steps run back to back as one lazy graph (each takes the lazy argmax token of the previous one), costs divided by K, against the same with no layer.
K = 2 from an even length covers both parities of the open compression group (ratio 2 closes a group every second token; the state writes of the step that
does not close are consumed by the one that does).
usage: layer_cpu.py PACK [--reps 15] [--len 136] [--chain 2] [--json F]    layers of the slice: 0 = full layer (compressor + indexer; the candidate-source
selection is not in the graph of a slice: nothing reads the mask), 1 = reuse layer (ds2 / ds2e)
Per layer kind: build ms, encode ms (async_eval return), launches (graph dispatches) per layer. Variants run interleaved (the Air is shared)."""
import argparse, json, os, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if "--default-buffers" not in sys.argv:        # measuring device: one command buffer per step, else async_eval blocks on the GPU once >10 buffers are in flight (sample trace, macair12)
    os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "100000"); os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "1000000")
import numpy as np
import mlx.core as mx
from kyojin_mlx import patch as P, rows as R, spec as S
from count_launches import graph_census

ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--reps", type=int, default=15); ap.add_argument("--len", type=int, default=136); ap.add_argument("--chain", type=int, default=2); ap.add_argument("--json"); ap.add_argument("--default-buffers", action="store_true")
ap.add_argument("--variant", default="full", choices=["full", "reindex", "cand"],
                help="what layer 0 is: full = compressor + indexer; reindex = indexer without the compressor (own indexer, shared compressed KV); cand = full + the candidate-block selection (nothing in a slice reads the mask, so it is added to the step outputs)")
ap.add_argument("--engram-resident", action="store_true", help="swap the Engram row read (host sync: np.asarray of the hash ids) for a device gather on a resident table, as a resident table on a big machine does: the CPU side of the Engram layers without the GPU wait")
ap.add_argument("--verify", type=int, default=0, help="N: cost of a chained verify of N rows (spec.verify, all accepted, K rounds back to back) instead of a solo step")
ap.add_argument("--ratios", help="compress ratios of the two layers, e.g. 2,0: a copy of the slice config (weights symlinked) with layer 1 local (window only)")
a = ap.parse_args()
if a.ratios:                                                    # config copy, weights linked
    import atexit, shutil, tempfile
    src, a.pack = a.pack, tempfile.mkdtemp(prefix="layer_cpu_")
    atexit.register(shutil.rmtree, a.pack, True)
    for f in os.listdir(src):
        if f != "config.json":
            os.symlink(os.path.abspath(os.path.join(src, f)), os.path.join(a.pack, f))
    c = json.load(open(os.path.join(src, "config.json"))); (c.get("text_config") or c)["compress_ratios"] = [int(v) for v in a.ratios.split(",")]
    json.dump(c, open(os.path.join(a.pack, "config.json"), "w"))
model, spec, cfg = P.build_model(a.pack); P.patch_model(model, a.pack, spec, cfg); R.enable(model, spec)
if a.verify:
    S.install(model, spec)
lm = model.language_model
if a.engram_resident:
    import mlx.nn as nn

    class ResidentRows(nn.Module):
        def __init__(self, dims):
            super().__init__(); self.table = mx.random.normal((1024, dims)).astype(mx.bfloat16)

        def __call__(self, idx):
            return self.table[idx % 1024]
    for l in lm.layers:
        if getattr(l, "engram", None) is not None:
            l.engram.embed = ResidentRows(l.engram.head_dim)
ids = np.random.default_rng(0).integers(16, cfg["text_config"]["vocab_size"] - 16, a.len).tolist()
cache = lm.make_cache()
lg = lm(mx.array(ids)[None], cache=cache).logits[:, -1].astype(mx.float32); mx.eval(lg)
tok = mx.argmax(lg, axis=-1)
if a.variant == "reindex":                                     # after the prefill: the cache holds the compressed KV and the index keys the indexer reads
    import importlib
    importlib.import_module("mlx_vlm.models.deepseek_v41.language").Compressor.__call__ = lambda self, x, start_pos, cache: None
layers = list(lm.layers)
L0, L1 = layers[0], layers[1]
CASES = {"none": [], "full": [L0], "both": [L0, L1]}           # layer 1 reads the KV layer 0 writes in the same step: it cannot run alone
K = a.chain
VTOK = [int(tok.item())] + np.random.default_rng(3).integers(16, cfg["text_config"]["vocab_size"] - 16, 8).tolist()
if a.variant == "cand":
    import importlib
    CAND = importlib.import_module("mlx_vlm.models.deepseek_v41.language").select_candidate_blocks
    _c = json.load(open(os.path.join(a.pack, "config.json"))); _c = _c.get("text_config") or _c
    cfg_topk, cfg_bs = _c["candidate_topk_blocks"], _c["candidate_block_size"]
    SC = mx.random.normal((1, 1, (a.len + 1) // 2)); mx.eval(SC)


def run(ls):
    """K chained decode steps (each step takes the lazy argmax token of the previous one) over the layers `ls`; returns the last logits. K = 2 from an even
    length covers both parities of the open compression group, and the state writes of the step that does not close a group are consumed by the one that does:
    per token = total / K (a single forked step misses the write nodes nobody reads yet, and bills a closing step with only its own)."""
    lm.layers = ls
    c = R.fork(cache); t = tok; outs = []
    for _ in range(K):
        if a.verify:                                                     # n rows of one sequence per round, all accepted: the next round starts from the last row's cache
            out, _, chain = S.verify(model, c, VTOK[:a.verify]); c = S.commit(chain, a.verify - 1)
        else:
            out = R.step(model, c, t); t = mx.argmax(out, axis=-1)
        if a.variant == "cand" and ls:                                  # the mask a slice never reads: build the candidate-block selection on a score row of the same shape
            outs.append(CAND(SC, SC.shape[-1], cfg_topk, cfg_bs))
    return [out] + outs if outs else out


res = {k: dict(build=[], enc=[]) for k in CASES}
for rep in range(a.reps + 2):
    for k, ls in CASES.items():
        t0 = time.perf_counter(); out = run(ls); t1 = time.perf_counter(); mx.async_eval(out); t2 = time.perf_counter(); mx.eval(out)
        if rep >= 2:
            res[k]["build"].append((t1 - t0) / K); res[k]["enc"].append((t2 - t1) / K)
cen, types = {}, {}
for k, ls in CASES.items():                                       # launches: graph dispatches of the same step after the warm-up above (the first call of a fused block runs the stock code)
    g = graph_census(run(ls)); cen[k] = g["dispatches"] / K; types[k] = {t: n / K for t, n in g["by_type"].items()}
lm.layers = layers
m = lambda k, f: 1e3 * statistics.median(res[k][f])
mn = lambda k, f: 1e3 * min(res[k][f])
p25 = lambda k, f: 1e3 * sorted(res[k][f])[len(res[k][f]) // 4]
tot = lambda k, q: 1e3 * sorted(b + e for b, e in zip(res[k]["build"], res[k]["enc"]))[int(q * (len(res[k]["build"]) - 1))]
base_b, base_e = m("none", "build"), m("none", "enc")
out = {"none": dict(build=base_b, enc=base_e, build_min=mn("none", "build"), enc_min=mn("none", "enc"), launches=cen["none"])}
print(f"{'case':12s} {'build med':>9s} {'enc med':>8s} | {'build min':>9s} {'enc min':>8s}   (ms)")
for k in CASES:
    print(f"{k:12s} {m(k, 'build'):9.3f} {m(k, 'enc'):8.3f} | {mn(k, 'build'):9.3f} {mn(k, 'enc'):8.3f}")
for name, k1, k0 in (("full", "full", "none"), ("reuse", "both", "full")):
    pb, pe = m(k1, "build") - m(k0, "build"), m(k1, "enc") - m(k0, "enc")
    pbm, pem = mn(k1, "build") - mn(k0, "build"), mn(k1, "enc") - mn(k0, "enc")
    out[name] = dict(launches=cen[k1] - cen[k0], build_ms=pb, enc_ms=pe, build_min_ms=pbm, enc_min_ms=pem, tot_p25_ms=tot(k1, .25) - tot(k0, .25), tot_min_ms=tot(k1, 0) - tot(k0, 0))
    print(f"   p25 of build+encode: {tot(k1, .25) - tot(k0, .25):.3f} ms   min: {tot(k1, 0) - tot(k0, 0):.3f} ms")
    print(f"per {name:8s} layer: {cen[k1] - cen[k0]:.1f} launches | build {pb:.3f} + encode {pe:.3f} = {pb + pe:.3f} ms (median) | min-based {pbm:.3f} + {pem:.3f} = {pbm + pem:.3f} ms")
out["types"] = {k: {t: n for t, n in types[k].items()} for k in ("none", "full", "both")}
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
