"""Per-term cost of one batched-core round on Metal, synthetic keys, no model and no pack (flatdec3).

A verify round of R rows at L tokens of context runs 8 index layers: three ratio-2 layers score L/2 keys, the candidate source scores L, and the four
consumer layers walk all L keys of the pass to score the 16384 the source's block mask kept. This prints the cost of each of those terms plus the
selection, the source mask and the cache append, with the two levers of this track on and off, so the round can be summed from measured pieces
instead of assumed:

  score_chain / score_mma     the 8 layers' scores of every row, stock chain (_ISCORE) / dsa_decode.scores_rows   [KYOJIN_ISCORE_MMA=1]
  cons_masked / cons_blocks   the 4 consumer layers, masked full-width scores + select / idxblocks              [KYOJIN_ISCORE_BLOCKS=1]
  select                      index_topk_rt over each layer's width (8 layers)
  src_mask                    candidate_mask_rt of the source layer, one call
  append                      the whole-buffer copy _put makes per emitting layer under the runtime-width / explicit state (pool + key rows); it
                              is NOT part of the round sum because a plain pass lets MLX scatter in place, and it is printed so the size is known

The device is shared with the served lane, so every A/B pair alternates inside each repetition and BOTH the median and the minimum are printed: the
minimum is the estimate that external load cannot inflate, the median says how loaded the box was. A ratio is quoted from the minima, and nothing
here is a served number - the owner's A/B on the idle lane is the measurement that decides.

Bounded: in-process memory limit 2 GiB, widths up to 262144 keys, one pass per group. The 1M column is MODEL, linear in the key count from the widest
measured width: a 1M-key run does not belong in a sandbox that must not touch the lane.

usage: PYTHONPATH=. timeout 300 .venv/bin/python tools/flatdec3_terms.py [--lens 65536,131072,262144] [--rows 6] [--json out.json]
"""
import argparse
import json
import statistics
import time

import mlx.core as mx
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--lens", default="65536,131072,262144")
ap.add_argument("--rows", type=int, default=6)
ap.add_argument("--reps", type=int, default=15)
ap.add_argument("--warm", type=int, default=3)
ap.add_argument("--topk", type=int, default=512)
ap.add_argument("--json", default="")
a = ap.parse_args()

mx.set_memory_limit(int(2.0 * 2 ** 30))
from kyojin_mlx import dsa_decode as D, dsv4_core as C, idxblocks as B     # noqa: E402

NH, DI, POOL, BS, NBK = 32, 128, 512, 8, 2048
R = a.rows


def pair(fa, fb, reps=None, warm=None):
    """A and B alternate inside every repetition (a busy device costs both arms equally) -> (min a, min b, median a, median b)."""
    for _ in range(warm or a.warm):
        mx.eval(fa())
        mx.eval(fb())
    ta, tb = [], []
    for i in range(reps or a.reps):
        for f, ts in (((fa, ta), (fb, tb)) if i % 2 == 0 else ((fb, tb), (fa, ta))):
            t0 = time.perf_counter()
            mx.eval(f())
            ts.append((time.perf_counter() - t0) * 1e3)
    return min(ta), min(tb), statistics.median(ta), statistics.median(tb)


def one(fn, reps=None, warm=None):
    """(min, median) of one op: no pair to alternate with."""
    for _ in range(warm or a.warm):
        mx.eval(fn())
    ts = []
    for _ in range(reps or a.reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts), statistics.median(ts)


def keys_of(cap, seed):
    rng = np.random.default_rng(seed)
    k = rng.standard_normal((1, cap, DI)).astype(np.float32)
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    return mx.array(k.astype(np.float16))


out = []
rng = np.random.default_rng(3)
for L in [int(v) for v in a.lens.split(",")]:
    row = {"L": L, "rows": R}
    q = mx.array(rng.standard_normal((R, NH, DI)).astype(np.float16))
    w = mx.array(rng.standard_normal((R, NH)).astype(np.float16))
    k1, k2 = keys_of(L, 5), keys_of(L // 2, 6)                                   # ratio-1 and ratio-2 key buffers
    lens1 = mx.array([L] * R, dtype=mx.int32)
    lens2 = mx.array([L // 2] * R, dtype=mx.int32)
    S0 = mx.zeros((R,), dtype=mx.int32)
    starts = np.sort(rng.choice(L // BS, NBK, replace=False))                    # the source layer's kept blocks
    m = np.zeros(L, dtype=bool)
    m[(starts[:, None] * BS + np.arange(BS)[None, :]).reshape(-1)] = True
    mask = mx.array(np.broadcast_to(m[None, :], (R, L)))

    chain = lambda: [C.index_scores(q, k2, w, lens2, L // 2, None, S0, rt=True) for _ in range(3)] + \
        [C.index_scores(q, k1, w, lens1, L, None, S0, rt=True)]
    mma = lambda: [D.scores_rows(q, w, k2, lens2, L // 2, None, S0) for _ in range(3)] + \
        [D.scores_rows(q, w, k1, lens1, L, None, S0)]
    row["score_chain_ms"], row["score_mma_ms"], row["score_chain_med"], row["score_mma_med"] = [round(v, 3) for v in pair(chain, mma)]

    ma, mb, meda, medb = pair(lambda: [C.index_topk_rt(D.scores_rows(q, w, k1, lens1, L, mask, S0), lens1, a.topk) for _ in range(4)],
                              lambda: [B.select(q, w, k1, lens1, mask, S0, L, a.topk, BS, NBK) for _ in range(4)])
    row["cons_masked_ms"], row["cons_blocks_ms"], row["cons_masked_med"], row["cons_blocks_med"] = [round(v, 3) for v in (ma, mb, meda, medb)]

    s2 = C.index_scores(q, k2, w, lens2, L // 2, None, S0, rt=True)
    s1 = C.index_scores(q, k1, w, lens1, L, None, S0, rt=True)
    mx.eval(s2, s1)
    row["select_ms"], row["select_med"] = [round(v, 3) for v in one(lambda: [C.index_topk_rt(s2, lens2, a.topk) for _ in range(3)]
                                                                   + [C.index_topk_rt(s1, lens1, a.topk)])]   # the 4 layers cons_* does not cover
    row["src_mask_ms"], row["src_mask_med"] = [round(v, 3) for v in one(lambda: C.candidate_mask_rt(s1, lens1, NBK, BS))]
    pb2, pb1 = mx.zeros((1, L // 2, POOL), dtype=mx.float16), mx.zeros((1, L, POOL), dtype=mx.float16)
    kb2, kb1 = mx.zeros((1, L // 2, DI), dtype=mx.float16), mx.zeros((1, L, DI), dtype=mx.float16)
    row["append_ms"], row["append_med"] = [round(v, 3) for v in one(lambda: [pb2[mx.arange(1)] for _ in range(3)] + [pb1[mx.arange(1)]]
                                                                  + [kb2[mx.arange(1)] for _ in range(3)] + [kb1[mx.arange(1)]])]
    del pb1, kb1, s1, s2
    mx.clear_cache()

    base = row["score_chain_ms"] + row["cons_masked_ms"] + row["select_ms"] + row["src_mask_ms"]
    levers = row["score_mma_ms"] + row["cons_blocks_ms"] + row["select_ms"] + row["src_mask_ms"]
    row["round_ms"] = round(base, 3)
    row["round_levers_ms"] = round(levers, 3)
    row["round_x"] = round(base / levers, 2)
    row["scale_to_1M"] = round(1048576 / L, 2)
    row["round_1M_model_ms"] = round(base * 1048576 / L, 1)
    row["round_1M_levers_model_ms"] = round(levers * 1048576 / L, 1)
    out.append(row)
    print(json.dumps(row))
    del k1, k2, mask
    mx.clear_cache()

if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
print(f"\nms per round of {R} rows; min of {a.reps} interleaved reps (median next to it). "
      "The 1M column is MODEL: linear in the key count from the widest measured L.")
