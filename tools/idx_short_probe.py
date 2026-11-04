#!/usr/bin/env python3
# idx_short on the Mac (needs Metal). Memory: a few MB of synthetic tensors (well under 3.5 GiB). Not run where it was written (no Metal).
#
# usage: python3 -I tools/idx_short_probe.py [--reps 15] [--json out.json]
#
# 1. GATE (exactness at kernel level): for lens in {1, 7, 64, 255, 511, 512} and rows in {1, 2, 4} (rows with different lens), the selection of the
#    real kernels (idxfuse.fused plain and candidate-source, dsv4_core.index_topk_rt / candidate_mask_rt) equals idxshort.short_idx / all_mask on
#    t < lens, bit for bit; lens = 513 must differ from the identity (the gate boundary). Exit code 1 on any mismatch.
# 2. TIMING of the indexer part of a plain step: a chain of L dependent layers (query, head weights and keys at DS4.1 dims: 32 heads x 128, keys
#    capacity 1024), stock (fused scores + select + mask) against the skip, slope between 8 and 32 layers, medians of --reps, interleaved arms,
#    per layer in us and for the 8 indexer layers of a token. The wq_b / head-weights / rotary launches the skip also removes are NOT in this
#    chain (they need the model): the number is the select part only, a floor of the lever.
# The end-to-end logits gate (flag on vs off across the threshold, text equal) needs a model: use the existing teacher-forced gates
# (tools/mu_check.py on the shrunk pack with --set index_topk=16 so a short context crosses the threshold, KYOJIN_IDX_SHORT=1 vs 0).
import argparse, json, statistics, sys, time

import numpy as np
import mlx.core as mx

from kyojin_mlx import dsv4_core as C, idxfuse as IXF, idxshort as S

NH, DI, K, CAP, BS, TB = 32, 128, 512, 1024, 8, 2048


def mk(n, seed=0):
    rng = np.random.default_rng(seed)
    q = mx.array(rng.standard_normal((n, NH, DI)).astype(np.float16) * 0.2)
    keys = mx.array(rng.standard_normal((n, CAP, DI)).astype(np.float16) * 0.2)
    wts = mx.array(np.abs(rng.standard_normal((n, NH))).astype(np.float16))
    return q, keys, wts


def gate():
    bad = 0
    for n in (1, 2, 4):
        for L in (1, 7, 64, 255, 511, 512):
            lens_np = np.array([max(1, L - 37 * i) for i in range(n)], dtype=np.int32)
            lens = mx.array(lens_np)
            q, keys, wts = mk(n, L + n)
            assert IXF.supported(CAP, NH, q.dtype)
            for src in (False, True):
                idx, m, _ = IXF.fused(q, keys, wts, lens, CAP, K, None, None, TB if src else 0, BS)
                sc = C.index_scores(q, keys, wts, lens, CAP, None, None, rt=True)
                idx2 = C.index_topk_rt(sc, lens, K)
                want = S.short_idx(lens, K)
                m2 = C.candidate_mask_rt(sc, lens, TB, BS) if src else None
                mx.eval(idx, idx2, want, *( [m, m2] if src else []))
                ok = bool((np.array(idx) == np.array(want)).all() and (np.array(idx2) == np.array(want)).all())
                if src:
                    wm = np.array(S.all_mask(n, CAP))[:, 0, :]
                    for u, Lu in enumerate(lens_np):
                        ok = ok and bool(np.array(m)[u, :Lu].all() == wm[u, :Lu].all() and np.array(m2)[u, :Lu].all())
                print(f"gate rows={n} lens<={L} source={src}: {'OK' if ok else 'MISMATCH'}")
                bad += (not ok)
    lens = mx.array([513], dtype=mx.int32)
    q, keys, wts = mk(1, 99)
    idx, _, _ = IXF.fused(q, keys, wts, lens, CAP, K, None, None, 0, BS)
    differs = not (np.array(idx) == np.array(S.short_idx(lens, K))).all()
    print(f"boundary lens=513: real selection differs from the identity: {differs} (expected True; the gate refuses it)")
    return bad + (0 if differs else 1)


def chain(layers, short, cached):
    """L dependent layers. Stock: the next layer's lens waits for this select. Cached skip: one idx per pass shared by the layers (as the model does,
    one per ratio per pass), so no per-layer dependency exists; uncached skip rebuilds it per layer."""
    q, keys, wts = mk(1, 3)
    lens = mx.array([300], dtype=mx.int32)
    em = type("E", (), {})(); em.lens_arr = lens
    acc, keep = None, None
    for _ in range(layers):
        if short:
            if not cached:
                em = type("E", (), {})(); em.lens_arr = lens
            idx = S.idx_for(em, K)
            m = S.all_mask(1, CAP)[:, 0, :]
        else:
            idx, m, _ = IXF.fused(q, keys, wts, lens, CAP, K, None, None, TB, BS)
        lens = lens + (idx[:, :1].sum(-1) * 0).astype(mx.int32)
        acc = idx if acc is None else acc + idx
        keep = m
    return acc, keep


def t_of(layers, short, cached, reps):
    ts = []
    for _ in range(reps):
        a, k = chain(layers, short, cached)
        t0 = time.perf_counter(); mx.eval(a, k); ts.append(time.perf_counter() - t0)
    return statistics.median(ts[2:] if len(ts) > 4 else ts)


def timing(reps):
    out = {}
    for name, short, cached in (("stock_fused", False, False), ("short_uncached", True, False), ("short_cached", True, True)):
        t_of(8, short, cached, 3)                                    # warm (kernel build)
    res = {k: [] for k in ("stock_fused", "short_uncached", "short_cached")}
    for r in range(3):
        for name, short, cached in (("stock_fused", False, False), ("short_uncached", True, False), ("short_cached", True, True)):
            lo, hi = t_of(8, short, cached, reps), t_of(32, short, cached, reps)
            res[name].append((hi - lo) / 24 * 1e6)
    for k, v in res.items():
        out[k] = dict(us_per_layer_median=statistics.median(v), runs=v, ms_for_8_layers=statistics.median(v) * 8 / 1000)
        print(f"{k:16s} {statistics.median(v):8.1f} us/layer (3 slopes {['%.1f' % x for x in v]}) -> {statistics.median(v) * 8 / 1000:.3f} ms for 8 layers")
    out["saved_ms_8_layers_select_part_only"] = out["stock_fused"]["ms_for_8_layers"] - out["short_cached"]["ms_for_8_layers"]
    print(f"select part saved per token (8 indexer layers): {out['saved_ms_8_layers_select_part_only']:.3f} ms; the wq_b / weights / rotary launches come on top")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=15); ap.add_argument("--json")
    a = ap.parse_args()
    if not mx.metal.is_available():
        sys.exit("no Metal")
    bad = gate()
    res = {"gate_bad": bad}
    if bad == 0:
        res["timing"] = timing(a.reps)
    else:
        print("gate failed: no timing")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    sys.exit(1 if bad else 0)
