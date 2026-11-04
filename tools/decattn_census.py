"""Synthetic census of one DS4.1 decode step's attention core (no pack, < 4 GiB).

8 index layers (ratios 2,2,2 then 1,1,1,1,1; layer 3 is the candidate source) + 32 reuse layers,
same tensors for every arm. Reports the in-chain split (score kernel / top-k select / window+pool
gather+merge) and the lever arms (idx_fuse, idx_score_fast) as ms per step.

usage: PYTHONPATH=. python tools/decattn_census.py [pos=4096] [reps=12]
"""
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx.core as mx

from kyojin_mlx import dsv4_core as C
from kyojin_mlx import idxfast as IX
from kyojin_mlx import idxfuse as IF

NH, DI, H, D, W, K = 32, 128, 64, 512, 128, 512
RATIOS = [2, 2, 2, 1, 1, 1, 1, 1]
CAND_I = 3
FAST_MIN = C.IDX_FAST_MIN
FUSE_MAX = C.IDX_FUSE_MAX
POS = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 12


def make_layer(rng, tm, keys_buf=None):
    q = mx.array(rng.standard_normal((1, NH, DI)).astype(np.float16))
    keys = mx.array((rng.standard_normal((1, tm, DI)) * 0.5).astype(np.float16)) if keys_buf is None else keys_buf
    w = mx.array((rng.standard_normal((1, NH)) * 0.05).astype(np.float16))
    qa = mx.array(rng.standard_normal((1, H, D)).astype(np.float16))
    win = mx.array(rng.standard_normal((1, W, D)).astype(np.float16))
    pool = mx.array((rng.standard_normal((1, tm, D)) * 0.5).astype(np.float16))
    sink = mx.array(rng.standard_normal(H).astype(np.float32))
    return q, keys, w, qa, win, pool, sink, mx.array([POS], dtype=mx.int32)


def build(seed=0):
    rng = np.random.default_rng(seed)
    layers, shared = [], None
    for i, r in enumerate(RATIOS):
        tm = max(1, (POS + 1) // r)
        layers.append(make_layer(rng, tm) if i <= CAND_I else make_layer(rng, tm, keys_buf=shared))
        if i == CAND_I:
            shared = layers[i][1]
    return layers


def select_base(sc, ln, tm, cand):
    if tm >= FAST_MIN:
        if cand is None:
            return IX.topk_mt(sc, ln, K), None
        return IX.select_fused(sc, ln, K, 2048, 8)
    topk = min(K, tm)
    ids = mx.sort(mx.argsort(-sc, axis=-1)[..., :topk], axis=-1).astype(mx.int32)
    return mx.where(ids < ln[:, None], ids, -1).astype(mx.int32), None


def main():
    layers = build(0)
    reuse = make_layer(np.random.default_rng(7), POS)
    reuse_idx = mx.array(np.sort(np.random.default_rng(8).choice(POS, K, replace=False))[None].astype(np.int32))
    mx.eval(*[a for l in layers for a in l[:1]], *layers[0][2:], reuse_idx)

    def t(fn, reps=REPS):
        ts = []
        for _ in range(reps + 2):
            mx.synchronize()
            t0 = time.perf_counter()
            mx.eval(fn())
            mx.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        return statistics.mean(ts[2:]), statistics.stdev(ts[2:]) / (reps - 1) ** 0.5

    def attends():
        out = [C.attend(qa, win, pool, mx.array(np.sort(np.random.default_rng(keys.shape[1]).choice(keys.shape[1], K, replace=False))[None].astype(np.int32)),
                        pos, sink, D ** -0.5, K) for q, keys, w, qa, win, pool, sink, pos in layers]
        out += [C.attend(reuse[3], reuse[4], reuse[5], reuse_idx, reuse[7], reuse[6], D ** -0.5, K) for _ in range(32)]
        return out

    scs, cand = [], None
    for i, (q, keys, w, qa, win, pool, sink, pos) in enumerate(layers):
        tm = keys.shape[1]
        ln = mx.array([tm], dtype=mx.int32)
        sc = C.index_scores(q, keys, w, ln, tm, cand, None, rt=tm >= FAST_MIN)
        if i == CAND_I:
            if tm < FAST_MIN:
                cand = C._lang().select_candidate_blocks(sc[:, None, :], ln[:, None, None], 2048, 8)[:, 0, :]
            else:
                _, cand = select_base(sc, ln, tm, cand)
        scs.append((sc, ln, tm, cand))

    def scores():
        return [C.index_scores(q, keys, w, ln, tm, cd, None, rt=tm >= FAST_MIN)
                for (q, keys, w, *_), (sc, ln, tm, cd) in zip(layers, scs)]

    def selects():
        return [select_base(sc, ln, tm, cd) for sc, ln, tm, cd in scs]

    def chain(fuse):
        out, cand = [], None
        for i, (q, keys, w, qa, win, pool, sink, pos) in enumerate(layers):
            tm = keys.shape[1]
            ln = mx.array([tm], dtype=mx.int32)
            if fuse and tm * 1 <= FUSE_MAX and IF.supported(tm, NH, q.dtype):
                kb = 2048 if i == CAND_I else 0
                idx, cand, _ = IF.fused(q, keys, w, ln, tm, K, cand, None, kb, 8, lmax=tm)
            else:
                sc = IF.scores_fast(q, keys, w, ln, tm, cand, None) if tm <= C.IDX_SCORE_FAST_MAX else \
                    C.index_scores(q, keys, w, ln, tm, cand, None, rt=tm >= FAST_MIN)
                idx, m = select_base(sc, ln, tm, cand)
                if i == CAND_I:
                    cand = m if m is not None else C._lang().select_candidate_blocks(sc[:, None, :], ln[:, None, None], 2048, 8)[:, 0, :]
            out.append(C.attend(qa, win, pool, idx[:, 0, :] if idx.ndim == 3 else idx, pos, sink, D ** -0.5, K))
        out += [C.attend(reuse[3], reuse[4], reuse[5], reuse_idx, reuse[7], reuse[6], D ** -0.5, K) for _ in range(32)]
        return out

    print(f"attention core, pos {POS}, keys {[l[1].shape[1] for l in layers]}, 8 index + 32 reuse layers")
    for name, fn in (("score kernel (8)", scores), ("top-k select (8)", selects), ("gather+merge (40)", attends)):
        m, e = t(fn)
        print(f"  {name:20s} {m:7.3f} +- {e:.3f} ms")
    for name, fn in (("chain base", lambda: chain(False)), ("chain idx_fuse", lambda: chain(True))):
        m, e = t(fn)
        print(f"  {name:20s} {m:7.3f} +- {e:.3f} ms")


if __name__ == "__main__":
    main()
