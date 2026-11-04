"""Decode indexer top-k at long context, old vs new, synthetic fp16 scores of the real geometry (k 512, 2048 kept 8-key blocks, N rows). Run on the Mac:

    python tools/bench_topk_tree.py [--layers 8] [--scores] [--quick]

Widths are INDEX KEYS (context / compress ratio 4): 32768 = 128K ctx, 131072 = 512K, 262144 = 1M, 524288 = 2M. Per width and row count it prints ms per call of

  plain top-k (non-source indexer layers)   old = idxfast._run radix chain (what topk_mt runs with every lever off)
                                           blk = idxblk.topk_blocks (lever dec_topk_blocks)      new = topk_tree.topk_tree (lever topk_tree)
  source select (top-k + candidate mask)    old = idxfast._run fused radix chain (select_fused)   new = topk_tree.topk_tree with the block level
  consumer-like scores (94 % of the 8-key blocks -inf) through the plain variants

`same` = the new indices (and mask on t < lens) equal the old ones bit for bit. Timing: 8 calls chained by a data dependence (so MLX cannot overlap them,
like the layers of a step) inside one eval, minus the same chain with a trivial op in place of the select, divided by 8; `single` is one call per eval
(includes the sync). `--scores` adds the index score kernels at the same widths (index_scores one thread per key, iscore_kb, scores_fast), which says
whether KYOJIN_IDX_SCORE_FAST_MAX / KYOJIN_ISCORE_KB should move. Memory: under 0.6 GiB. Time: under 60 s with --quick, about 90 s without.
Needs Metal; nothing here has been run by the author."""
import argparse
import time

import mlx.core as mx
import numpy as np

from kyojin_mlx import idxblk as B
from kyojin_mlx import idxfast as F
from kyojin_mlx import topk_tree as TT

ap = argparse.ArgumentParser()
ap.add_argument("--layers", type=int, default=8, help="indexer layers of a step, for the per-token line")
ap.add_argument("--scores", action="store_true")
ap.add_argument("--quick", action="store_true")
ap.add_argument("--k", type=int, default=512)
ap.add_argument("--kb", type=int, default=2048)
args = ap.parse_args()
K, KB, BS, CH = args.k, args.kb, 8, 8
TT.MIN_KEYS = 0
B.MIN_ELEMS = 0


def chain_ms(step, s0, n=CH, reps=6):
    """ms per call of `step(scores) -> idx` when n calls are chained by a dependence on the previous result."""
    def run():
        s = s0
        for _ in range(n):
            idx = step(s)
            s = s0 + (idx[:, :1] * 0).astype(s0.dtype)
        return s
    mx.eval(run())
    t = time.perf_counter()
    for _ in range(reps):
        mx.eval(run())
    return (time.perf_counter() - t) / reps / n * 1e3


def single_ms(fn, reps=12):
    for _ in range(3):
        mx.eval(fn())
    t = time.perf_counter()
    for _ in range(reps):
        mx.eval(fn())
    return (time.perf_counter() - t) / reps * 1e3


def scores_of(N, Tm, kind):
    x = mx.random.normal((N, Tm)) * 3
    if kind == "masked":
        nb = Tm // BS
        keep = mx.random.uniform(shape=(N, nb)) < (2048 / nb)
        x = mx.where(mx.repeat(keep, BS, axis=1), x, -mx.inf)
    return x.astype(mx.float16)


def row(tag, N, Tm, olds, news, same, extra=""):
    print(f"{tag:22s} N={N} keys={Tm:7d}: " + "  ".join(f"{n} {v:7.3f}" for n, v in olds + news) + f"   x{olds[0][1] / news[-1][1]:5.2f}  same={same}{extra}", flush=True)
    return olds[0][1] - news[-1][1]


mx.random.seed(0)
widths = (32768, 262144) if args.quick else (32768, 131072, 262144, 524288)
saved = {}
for Tm in widths:
    for N in ((1,) if args.quick else (1, 4)):
        ln = mx.full((N,), Tm, dtype=mx.int32)
        for kind in ("normal", "masked"):
            s = scores_of(N, Tm, kind)
            mx.eval(s)
            ref = F._run(s, ln, K, 2, BS, False)[0]
            new = TT.topk_tree(s, ln, K)[0]
            blk = B.topk_blocks(s, ln, K)
            mx.eval(ref, new, blk)
            same = bool(mx.array_equal(ref, new).item())
            ms_ref = chain_ms(lambda x: F._run(x, ln, K, 2, BS, False)[0], s)
            ms_blk = chain_ms(lambda x: B.topk_blocks(x, ln, K), s)
            ms_new = chain_ms(lambda x: TT.topk_tree(x, ln, K)[0], s)
            ms_base = chain_ms(lambda x: x[:, :K].astype(mx.int32), s)
            d = row(f"plain/{kind}", N, Tm, [("old", ms_ref - ms_base), ("blk", ms_blk - ms_base)], [("new", ms_new - ms_base)], same,
                    f"   single old {single_ms(lambda: F._run(s, ln, K, 2, BS, False)[0]):6.3f} new {single_ms(lambda: TT.topk_tree(s, ln, K)[0]):6.3f}")
            if kind == "normal":
                saved[(Tm, N)] = d
        s = scores_of(N, Tm, "normal")
        mx.eval(s)
        if TT.applies(N, Tm, K, s.dtype, KB, BS):
            ri, rm = F._run(s, ln, K, KB, BS, True)
            ni, nm = TT.topk_tree(s, ln, K, KB, BS)
            mx.eval(ri, rm, ni, nm)
            same = bool(mx.array_equal(ri, ni).item()) and bool(mx.array_equal(rm, nm).item())
            ms_ref = chain_ms(lambda x: F._run(x, ln, K, KB, BS, True)[0], s)
            ms_new = chain_ms(lambda x: TT.topk_tree(x, ln, K, KB, BS)[0], s)
            ms_base = chain_ms(lambda x: x[:, :K].astype(mx.int32), s)
            row("source (k + mask)", N, Tm, [("old", ms_ref - ms_base)], [("new", ms_new - ms_base)], same)
        else:
            print(f"{'source (k + mask)':22s} N={N} keys={Tm:7d}: tree does not apply (block words past RB_MAX): radix chain stays", flush=True)
print(f"\nms saved per token (plain variant, normal scores, {args.layers} layers, if every layer ran it; the source layers save about the same):")
for (Tm, N), d in sorted(saved.items()):
    print(f"  keys={Tm:7d} (ctx {Tm * 4 // 1024}K) N={N}: {d * args.layers:6.2f} ms")

if args.scores:
    from kyojin_mlx import dsv4_core as C
    from kyojin_mlx import idxfuse as IXF
    print("\nindex score kernels (q [N,32,128], keys [1,C,128] fp16), ms per call, chained:")
    for Tm in ((32768, 131072) if args.quick else (32768, 65536, 131072, 262144)):
        N = 2
        q = (mx.random.normal((N, 32, 128)) * 0.3).astype(mx.float16)
        keys = (mx.random.normal((1, Tm, 128)) * 0.3).astype(mx.float16)
        w = mx.random.normal((N, 32)).astype(mx.float16)
        ln = mx.full((N,), Tm, dtype=mx.int32)
        sess = mx.zeros((N,), dtype=mx.uint32)
        mx.eval(q, keys, w, ln, sess)

        def timed(fn):
            def run():
                x = q
                for _ in range(CH):
                    sc = fn(x)
                    x = q + (sc[:, :1, None] * 0).astype(q.dtype)
                return x
            mx.eval(run())
            t = time.perf_counter()
            for _ in range(5):
                mx.eval(run())
            return (time.perf_counter() - t) / 5 / CH * 1e3
        a = C.index_scores(q, keys, w, ln, Tm, None, sess, rt=True)
        b = IXF.scores_fast(q, keys, w, ln, Tm, None, sess)
        mx.eval(a, b)
        t_old = timed(lambda x: C.index_scores(x, keys, w, ln, Tm, None, sess, rt=True))
        t_fast = timed(lambda x: IXF.scores_fast(x, keys, w, ln, Tm, None, sess))
        was = (C.ISCORE_KB, C.ISCORE_KB_MIN)
        C.ISCORE_KB, C.ISCORE_KB_MIN = True, 0
        try:
            t_kb = timed(lambda x: C.index_scores(x, keys, w, ln, Tm, None, sess, rt=True))
        finally:
            C.ISCORE_KB, C.ISCORE_KB_MIN = was
        print(f"  keys={Tm:7d}: index_scores {t_old:7.3f}  iscore_kb {t_kb:7.3f}  scores_fast {t_fast:7.3f}  fast==old {bool(mx.array_equal(a, b).item())}", flush=True)
