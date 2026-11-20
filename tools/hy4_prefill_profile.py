"""Prefill-chunk cost of the Hy4 indexer (the only term of a DSA layer that grows with depth) versus context depth S: one chunk of L rows at the end of an S-token context.

Why: prefill must stay >= 90 % of the 8K value up to 1M. Per chunk and DSA layer the scorer is L x J x 16 heads x 128 x 2 FLOP (J = S / 4 pooled keys), 12 layers; the top-k reads
[L, J] fp32 scores. This times the production scorer (kpool_score.scores, fp32 pooled keys) and the same kernel on fp16 pooled keys (half the key bytes), then the radix top-k
(idxfast.topk_mt, 511 pools) on the result, in ms per chunk per layer, so (12 x sum) / chunk time says what the indexer alone costs at that depth.
Usage (Metal, chain cell only): python tools/hy4_prefill_profile.py [L=4096] [S list, comma]; prints one RESULTS line per S (json), file HY4P_OUT when set."""
import os, sys, time, json
sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, mlx.core as mx
from kyojin_mlx import kpool_score as KS, idxfast as IX, kpool_hisa as HI
mx.set_default_device(mx.gpu)
LQ = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
SS = [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "8192,32768,131072,262144,524288,1048576").split(",")]
H, D, KP, TOPK = 16, 128, 4, 511
rng = np.random.default_rng(1)


def T(fn, k=2, w=1):
    for _ in range(w):
        mx.eval(fn(0))
    t = time.perf_counter(); mx.eval([fn(i) for i in range(k)]); return (time.perf_counter() - t) / k * 1e3


out_all = []
for S in SS:
    J = S // KP
    o = {"S": S, "L": LQ, "J": J}
    try:
        pk = mx.array(rng.standard_normal((1, J, D)).astype(np.float32)); pk16 = pk.astype(mx.float16)
        iq = mx.array(rng.standard_normal((1, H, LQ, D)).astype(np.float32)); w = mx.array(rng.standard_normal((1, LQ, H)).astype(np.float32))
        valid = mx.ones((1, J), dtype=mx.bool_); off = mx.array([S - LQ], dtype=mx.int32)
        mx.eval(pk, pk16, iq, w, valid)
        sc = lambda p: (lambda i: KS.scores(iq * (1.0 + 1e-6 * i), w, p, valid, off, KP, True))
        def part(name, fn):                                      # every measure on its own: a failing variant (the half-key kernel) never hides the others
            try:
                o[name] = fn()
            except Exception as e:
                o[name + "_error"] = repr(e)[:160]
        sc = lambda p: (lambda i: KS.scores(iq * (1.0 + 1e-6 * i), w, p, valid, off, KP, True))
        part("score_f32_ms", lambda: round(T(sc(pk)), 2))
        def mm():                                                # the path the served lane takes (KYOJIN_KPOOL_SCORE is off in flags.lc.env): 256-row blocks, [1, H, 256, J] fp32 product
            def blk(i):
                kq = mx.maximum(iq[:, :, :256] * (1.0 + 1e-6 * i) @ pk[:, None].swapaxes(-1, -2), 0.0)
                return (kq * w[:, :256].transpose(0, 2, 1)[..., None]).sum(axis=1)
            return round(T(blk) * (LQ / 256), 2)
        part("score_matmul_blocks_ms", mm)
        for nm, mb in (("hisa64_ms", 64), ("hisa256_ms", 256)):          # KYOJIN_HY4_HISA: block bounds + exact scoring of the tile candidates + their top-k (whole selection, ms per chunk and layer; random keys: timing only)
            if J // 32 > mb:
                prm = dict(bp=32, m=mb, tile=64, first=2, last=2)
                part(nm, lambda prm=prm: round(T(lambda i: HI.top(iq * (1.0 + 1e-6 * i), w, pk, valid, off, KP, TOPK, True, prm)), 2))
                if nm in o: o[nm.replace("_ms", "_12_layers_ms")] = round(12 * o[nm], 1)
        part("score_f16pk_ms", lambda: round(T(sc(pk16)), 2))
        s0 = KS.scores(iq, w, pk, valid, off, KP, True); mx.eval(s0)
        def rel():
            s1 = KS.scores(iq, w, pk16, valid, off, KP, True); mx.eval(s1)
            return float(mx.max(mx.abs(s1 - s0)) / (mx.max(mx.abs(s0)) + 1e-9))
        part("f16pk_max_rel", rel)
        lens = mx.full((LQ,), J, dtype=mx.int32)
        part("topk_ms", lambda: round(T(lambda i: IX.topk_mt(s0.reshape(LQ, J), lens, TOPK)), 2))   # the 4 GB add of a perturbation is avoided: the same input twice may be shared by MLX, so this is a floor
        if "topk_ms" in o:
            for k_, n_ in (("score_f32_ms", "indexer_12_layers_ms"), ("score_matmul_blocks_ms", "indexer_12_layers_matmul_ms"), ("score_f16pk_ms", "indexer_12_layers_f16pk_ms")):
                if k_ in o: o[n_] = round(12 * (o[k_] + o["topk_ms"]), 1)
        del s0
    except Exception as e:
        o["error"] = repr(e)[:200]
    print("RESULTS hy4_prefill_profile " + json.dumps(o), flush=True)
    out_all.append(o)
    mx.clear_cache()
if os.environ.get("HY4P_OUT"):
    json.dump(out_all, open(os.environ["HY4P_OUT"], "w"), indent=1)
