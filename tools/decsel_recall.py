"""decsel: recall of the reused decode selection on pack-shaped keys, and the cost of a candidate round on this Mac.

`tests/test_decsel.py` pins the mechanism (identity at EVERY = 1, the candidate set, the rollback, the gates). This tool answers the two
questions the lever lives or dies on:

  recall   on keys shaped like the pack's index keys (fp4 ue8m0 codes on 32-value power-of-two blocks around a few learned-looking
           centroids, the shape `tests/test_pf_cluster.py` uses) with successive queries in near-duplicate groups (a decode step), for
           EVERY = 2 / 3 / 4 / 8: the mean top-k overlap of the reused round with the FULL scan of the same row, and how many rounds the
           selection is identical. Plus a random-query control: with unrelated queries the candidate set cannot carry over, so that number
           is the floor the lever has to beat.
  cost     one candidate round of one layer at n = 6 rows, k = 512 (index_topk on DS4.1), the appended keys of a 6-row round: the same
           `decsel.layer` the core calls, against `decsel.full_select` on the same buffer. The candidate path does not read the buffer
           beyond its candidates, so this cost is the same at 8K keys and at 1M.

usage: python tools/decsel_recall.py [--keys 16384] [--rows 32] [--topk 512] [--reps 9]
Nothing here starts a server, touches the lane or allocates more than one 64K-key buffer.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))

import mlx.core as mx                                                     # noqa: E402

from kyojin_mlx import decsel as DS                                       # noqa: E402
from kyojin_mlx import dsv4_core as C                                     # noqa: E402

NH, D = 32, 128
CODES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def fp4_keys(rng, n, centroids=24, spread=0.35, mag=0.0):
    """The pack's index keys: fp4 codes times a power-of-two scale per 32-value block, drawn around `centroids` centres."""
    c = rng.standard_normal((centroids, D)).astype(np.float32)
    own = rng.integers(0, centroids, n)
    k = c[own] + spread * rng.standard_normal((n, D)).astype(np.float32)
    k = np.array(mx.array(k).astype(mx.float16), dtype=np.float32)
    blk = k.reshape(n, D // 32, 32)
    amax = np.abs(blk).max(-1, keepdims=True) + 1e-9
    scale = np.exp2(np.ceil(np.log2(amax)) - 2.0)
    q = np.clip(blk / scale, -6, 6)
    val = np.sign(q) * np.abs(CODES)[np.abs(np.abs(q)[..., None] - np.abs(CODES)).argmin(-1)]
    k = (val * scale).reshape(n, D)
    if mag:                                                                   # per-key magnitude: what the per-block fp4 scale leaves free
        k = k * np.exp(rng.standard_normal(n) * mag)[:, None].astype(np.float32)
    return mx.array(k.astype(np.float16))


def queries(rng, T, templates=6, spread=0.15):
    """Successive decode queries: `templates` near-duplicate groups (a template plus a small drift per row), plus the head weights."""
    c = rng.standard_normal((templates, NH, D)).astype(np.float32)
    own = np.arange(T) * templates // T
    q = c[own] + spread * rng.standard_normal((T, NH, D)).astype(np.float32)
    w = (rng.standard_normal((T, NH)) * 0.05).astype(np.float16)
    return mx.array(q).astype(mx.float16), mx.array(w)


def rounds(keys, q, w, k, every, lens0, T):
    """One session decoded one plain step at a time: round t has `lens0 + t` visible keys. Returns (mean overlap, identical-round fraction)."""
    row = object()
    st = _St(row)
    DS.EVERY, DS.CHECK = every, False
    DS.HITS.clear()
    DS.STATS.clear()
    idxr = type("I", (), {"index_topk": k, "is_candidate_source": False, "uses_candidates": False, "n_heads": NH, "index_head_dim": D})()
    ov, exact = [], []
    for t in range(T):
        n = lens0 + t
        plan = C.Plan([n - 1], [1])
        em = type("E", (), {"lens": [n], "lens_arr": mx.array([n], dtype=mx.int32), "tmax": n})()
        full = DS.full_select(q[t:t + 1], w[t:t + 1], keys, em.lens_arr, plan.sess_idx, keys.shape[1], None, k)
        got = DS.layer(st, 5, idxr, keys, None, q[t:t + 1], w[t:t + 1], em, keys.shape[1], plan)
        mx.eval(full)
        if got is not None:
            mx.eval(got)
            ov.append(DS.overlap(got, full, keys.shape[1])[0] / k)
            exact.append(bool(np.array_equal(np.array(got), np.array(full))))
            DS.publish(DS.store_of(st), 5, st.rows, (got, em.lens, False), None)
        else:
            DS.publish(DS.store_of(st), 5, st.rows, (full, em.lens, True), None)
    return float(np.mean(ov)) if ov else float("nan"), float(np.mean(exact)) if exact else float("nan")


class _St:
    def __init__(self, row):
        self.rows = [row]


def cost(keys, k, rows=6, reps=9):
    """ms per call of one layer's candidate round (n = `rows`, k + the appended keys) and of the full scan on the same buffer."""
    lens = keys.shape[1]
    q = mx.array(np.random.default_rng(1).standard_normal((rows, NH, D)).astype(np.float16))
    w = mx.array((np.random.default_rng(2).standard_normal((rows, NH)) * 0.05).astype(np.float16))
    em = type("E", (), {"lens": [lens - rows + 1 + u for u in range(rows)],
                        "lens_arr": mx.array([lens - rows + 1 + u for u in range(rows)], dtype=mx.int32), "tmax": lens})()
    plan = C.Plan([lens - rows + u for u in range(rows)], [rows])
    idxr = type("I", (), {"index_topk": k, "is_candidate_source": False, "uses_candidates": False, "n_heads": NH, "index_head_dim": D})()
    row = object()
    st = _St(row)
    DS.EVERY = 2
    full = DS.full_select(q, w, keys, em.lens_arr, plan.sess_idx, lens, None, k)
    DS.publish(DS.store_of(st), 5, st.rows, (full, em.lens, True), None)
    mx.eval(full)
    out = {}
    for name, fn in (("candidate", lambda: DS.layer(st, 5, idxr, keys, None, q, w, em, lens, plan)),
                     ("full", lambda: DS.full_select(q, w, keys, em.lens_arr, plan.sess_idx, lens, None, k))):
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            r = fn()
            mx.eval(r)
            ts.append((time.perf_counter() - t0) * 1e3)
        out[name] = min(ts)
    DS.EVERY = 0
    return out


def ceiling(keys, q, w, k, T, lens0):
    """The ceiling of ANY "reuse the previous selection" scheme: the overlap of two CONSECUTIVE full selections. A candidate set made of
    the previous answer + the appended keys cannot do better than this, so it is the number the lever's overlap has to be read against."""
    sels = []
    for t in range(T):
        n = lens0 + t
        s = DS.full_select(q[t:t + 1], w[t:t + 1], keys, mx.array([n], dtype=mx.int32), mx.array([0], dtype=mx.int32), keys.shape[1], None, k)
        mx.eval(s)
        sels.append(s)
    return float(np.mean([DS.overlap(sels[t], sels[t - 1], keys.shape[1])[0] / k for t in range(1, T)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", type=int, default=16384)
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--topk", type=int, default=512)
    ap.add_argument("--reps", type=int, default=9)
    a = ap.parse_args()
    assert a.keys <= 65536, "the house limit is 64K keys per array"
    rng = np.random.default_rng(11)
    lens0 = a.keys - a.rows
    for world, mag in (("flat", 0.0), ("wide", 2.0)):
        keys = fp4_keys(rng, a.keys, mag=mag)[None]
        q, w = queries(rng, a.rows)
        qr, wr = queries(rng, a.rows, templates=a.rows, spread=1.0)           # control: every query unrelated to the last
        mx.eval(keys, q, w, qr, wr)
        ceil = ceiling(keys, q, w, a.topk, a.rows, lens0)
        print(f"\nkeys {a.keys} x {D} fp16, world `{world}`, index_topk {a.topk}, {a.rows} rounds of one session "
              f"(lens {lens0}..{a.keys - 1}), key/topk {a.keys / a.topk:.1f}, ceiling {ceil:.3f}")
        print("| EVERY | mean overlap `ov/k` | identical rounds | control (unrelated queries) |")
        print("|---|---|---|---|")
        for every in (2, 3, 4, 8):
            ov, ex = rounds(keys, q, w, a.topk, every, lens0, a.rows)
            cv, _ = rounds(keys, qr, wr, a.topk, every, lens0, a.rows)
            print(f"| {every} | {ov:.3f} | {ex:.2f} | {cv:.3f} |")
        keys = keys
    c = cost(keys, a.topk, reps=a.reps)
    print(f"\ncost per layer, n = 6, k = {a.topk}, candidate width {a.topk + 6}: candidate {c['candidate']:.3f} ms, "
          f"full scan over {a.keys} keys {c['full']:.3f} ms (this Mac, min of {a.reps})")


if __name__ == "__main__":
    main()
