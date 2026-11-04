#!/usr/bin/env python3
"""Mixed-K penalty of the MoE verify band (rows 2..16 of one MoE call: MTP verify rows, batched decode) on a stacked DS4.1 layer slice.
For each rows R and path (grouped GEMV = default, bitwise; nax = KYOJIN_ROWS_GROUPED=nax, groups of >= 4 rows on the NAX GEMM), the same routed
work is timed with the layer's kernel (all bitrates of the stack compiled in) and with a kernel that holds only the bitrate of the routed experts.
Routing: every pair on experts of one bitrate class (--kclass), `share` of the rows reuse the experts of row 0 (share 1: every expert has R rows).
Time per call: slope of chains of n and 4n calls with fresh inputs in one eval, min of --reps. Bit-identity of the two kernels is checked per case.
usage: vband108.py [--dir models/ds_l10] [--layer 10] [--rows 2,4,8,16] [--share 0,1] [--reps 5]"""
import argparse, os, sys, time

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kyojin_mlx import patch as P, moe as M, metal_nax as NX  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--dir", default=os.path.expanduser("~/kyojin-mlx/models/ds_l10"))
ap.add_argument("--layer", type=int, default=10)
ap.add_argument("--rows", default="2,4,8,16")
ap.add_argument("--share", default="0,1")
ap.add_argument("--kclass", default="2,3")
ap.add_argument("--topk", type=int, default=6)
ap.add_argument("--chain", type=int, default=8)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--paths", default="grp,nax")
ap.add_argument("--splitab", action="store_true", help="nax path, routing over all experts: one mixed-K hot launch vs per-bitrate passes (moe.VB_SPLIT_K)")
ap.add_argument("--mixcheck", action="store_true", help="routing mixed K2 + K3 vs pure K2 / pure K3 at equal work, layer kernels")
a = ap.parse_args()
mx.set_cache_limit(256 << 20)
pre = f"layers.{a.layer}"
t = P.load_pack(a.dir, prefix=pre + ".")
st = [M.Stack.from_arrays(t, f"{pre}.ffn.experts.{w}", "mul1") for w in ("w1", "w3", "w2")]
mx.eval(*[x for s in st for x in (s.words, s.offset, s.k, s.suh, s.svh)])
kh = st[0].kh
print(f"{os.path.basename(a.dir)}: {st[0].experts} experts, bitrates {dict(zip(*np.unique(kh, return_counts=True)))}, stack ks {st[0].ks}", flush=True)
E, H = st[0].experts, st[0].kin
rng = np.random.default_rng(0)
_orig = NX.expert_gemm


def make(path, ks):
    m = M.MoeGather(*st, a.topk, limit=10.0, wpre=True, fq=True, grouped=path != "gather", hotk=4 if path == "nax" else 0)
    if ks is not None:
        m.ks_gu = m.ks_dn = ks
    return m


def call(m, ks):
    def f(x, ids, w):
        NX.expert_gemm = (lambda *p, **k: _orig(*p, **k, ks=ks)) if ks is not None else _orig
        try:
            return m(x, ids, w)
        finally:
            NX.expert_gemm = _orig
    return f


def timeit(f, ins):
    n = len(ins)
    def run(c):
        outs = [f(*ins[i % n]) for i in range(c)]
        t0 = time.perf_counter(); mx.eval(*outs); return time.perf_counter() - t0
    T = {n: [], 4 * n: []}
    for r in range(a.reps + 1):
        for c in T:
            dt = run(c)
            if r:
                T[c].append(dt)
    return (min(T[4 * n]) - min(T[n])) / (3 * n) * 1e3


def pick(kc):
    if kc != "mix":
        return rng.choice(np.flatnonzero(kh == int(kc)), a.topk, replace=False)
    h = a.topk // 2                                                       # half the pairs on K2 experts, half on K3, interleaved
    p2, p3 = rng.choice(np.flatnonzero(kh == 2), h, replace=False), rng.choice(np.flatnonzero(kh == 3), a.topk - h, replace=False)
    return np.array([v for q in zip(p2, p3) for v in q])


if a.splitab:                                                             # the hot-path fix: KYOJIN_VB_SPLIT_K off / on, interleaved, same inputs
    for R in [int(v) for v in a.rows.split(",")]:
        for share in [float(v) for v in a.share.split(",")]:
            base = rng.choice(E, a.topk, replace=False)
            rows = [base if r == 0 or rng.random() < share else rng.choice(E, a.topk, replace=False) for r in range(R)]
            ids = mx.array(np.stack(rows).astype(np.int32))
            ins = [(mx.array((rng.standard_normal((R, H)) * 0.5).astype(np.float16)), ids,
                    mx.array(rng.random((R, a.topk)).astype(np.float32))) for _ in range(a.chain)]
            mx.eval(*[v for i in ins for v in i])
            nk = len(set(kh[np.array(rows).ravel()].tolist()))
            for path in a.paths.split(","):
                m = make(path, None)
                def f(on, m=m):
                    def g(*x):
                        M.VB_SPLIT_K = on
                        return m(*x)
                    return g
                same = bool(mx.array_equal(f(False)(*ins[0]), f(True)(*ins[0])).item())
                T = {False: [], True: []}
                for _ in range(3):
                    for on in T:
                        T[on].append(timeit(f(on), ins))
                print(f"R{R} share{share:g} {path} ({nk} bitrates routed): one launch {min(T[False]) * 1e3:.1f} us (med {np.median(T[False]) * 1e3:.1f}) -> per-K "
                      f"{min(T[True]) * 1e3:.1f} us (med {np.median(T[True]) * 1e3:.1f}) ({(min(T[True]) / min(T[False]) - 1) * 100:+.1f}%), bit-identical {same}", flush=True)
    sys.exit(0)

if a.mixcheck:                                                            # same work, routing mixed K2 + K3 vs pure K2 / pure K3: the runtime mixing cost
    for R in [int(v) for v in a.rows.split(",")]:
        for share in [float(v) for v in a.share.split(",")]:
            for path in a.paths.split(","):
                if path == "nax" and R < 4:
                    continue
                m = make(path, None)
                f = call(m, None)
                res = {}
                cases = {}
                for kc in ("2", "3", "mix"):
                    base = pick(kc)
                    rows = [base if r == 0 or rng.random() < share else pick(kc) for r in range(R)]
                    ids = mx.array(np.stack(rows).astype(np.int32))
                    cases[kc] = [(mx.array((rng.standard_normal((R, H)) * 0.5).astype(np.float16)), ids,
                                  mx.array(rng.random((R, a.topk)).astype(np.float32))) for _ in range(a.chain)]
                    mx.eval(*[v for i in cases[kc] for v in i])
                    res[kc] = []
                for _ in range(2):
                    for kc in cases:
                        res[kc].append(timeit(f, cases[kc]))
                b = {k: min(v) * 1e3 for k, v in res.items()}
                avg = (b["2"] + b["3"]) / 2
                print(f"R{R} share{share:g} {path}: pure K2 {b['2']:.1f} us, pure K3 {b['3']:.1f} us, mixed {b['mix']:.1f} us vs mean {avg:.1f} ({(b['mix'] / avg - 1) * 100:+.1f}%)", flush=True)
    sys.exit(0)

for kc in [int(v) for v in a.kclass.split(",")]:
    pool = np.flatnonzero(kh == kc)
    for R in [int(v) for v in a.rows.split(",")]:
        for share in [float(v) for v in a.share.split(",")]:
            base = rng.choice(pool, a.topk, replace=False)
            rows = [base if r == 0 or rng.random() < share else rng.choice(pool, a.topk, replace=False) for r in range(R)]
            ids = mx.array(np.stack(rows).astype(np.int32))
            ins = [(mx.array((rng.standard_normal((R, H)) * 0.5).astype(np.float16)), ids,
                    mx.array(rng.random((R, a.topk)).astype(np.float32))) for _ in range(a.chain)]
            mx.eval(*[v for i in ins for v in i])
            for path in a.paths.split(","):
                mm, m1 = make(path, None), make(path, (kc,))
                f_mix, f_one = call(mm, None), call(m1, (kc,))
                same = bool(mx.array_equal(f_mix(*ins[0]), f_one(*ins[0])).item())
                tm, to = [], []
                for _ in range(2):                                       # interleaved
                    tm.append(timeit(f_mix, ins)); to.append(timeit(f_one, ins))
                b0, b1 = min(tm), min(to)
                print(f"K{kc} R{R} share{share:g} {path}: mixed-K kernel {b0 * 1e3:.1f} us -> K{kc}-only {b1 * 1e3:.1f} us ({(b0 / b1 - 1) * 100:+.1f}%), "
                      f"bit-identical {same}", flush=True)
