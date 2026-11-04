#!/usr/bin/env python3
"""Where the mixed-K penalty of the NAX expert GEMM comes from, at verify-band rows (1..16 rows per expert), real experts of a stacked layer slice.
G experts (half K2, half K3, alternating ids) of --proj, M rows each, cfg of the verify band (moe.MoeGather.nax_cfg):
  one      : one launch over all G experts, kernel with both decode bodies (what moe._call_hot runs)
  onesort  : the same launch with the experts ordered by bitrate (K2 slots first)
  same2    : two launches (K2 experts, then K3 experts), the same two-body kernel   -> runtime mixing of K in one launch
  split    : two launches, a K2-only and a K3-only kernel                            -> per-bitrate passes (moe_prefill PF_SPLIT_K)
ms per (pair of) launch(es), min / median over interleaved reps; bit-identity of split vs one.
usage: vmix108.py [--dir models/ds_l10] [--proj w1] [--rows 1,4,16] [--experts 24]"""
import argparse, os, statistics, sys, time

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kyojin_mlx import metal_nax as NX  # noqa: E402
from nax_common import mini_stack  # noqa: E402
from safetensors import safe_open  # noqa: E402
import glob  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--dir", default=os.path.expanduser("~/kyojin-mlx/models/ds_l10"))
ap.add_argument("--proj", default="w1")
ap.add_argument("--rows", default="1,4,16")
ap.add_argument("--experts", type=int, default=24)
ap.add_argument("--reps", type=int, default=9)
ap.add_argument("--kpair", default="2,3", help="bitrates of the two halves (control: 2,2 or 3,3 = uniform K split in two launches)")
a = ap.parse_args()
mx.set_cache_limit(256 << 20)
f = sorted(glob.glob(os.path.join(a.dir, "stacked.*.safetensors")))[0]
with safe_open(f, "np") as s:
    kv = s.get_tensor([x for x in s.keys() if x.endswith(f".experts.{a.proj}.k")][0])
h = a.experts // 2
ka, kb = [int(v) for v in a.kpair.split(",")]
e2 = [i for i in range(len(kv)) if kv[i] == ka][:h]
e3 = [i for i in range(len(kv)) if kv[i] == kb and i not in e2][:h]
exps = [e for p in zip(e2, e3) for e in p]                       # alternating K2, K3
st = mini_stack(a.dir, a.proj, exps, "mul1")
mx.eval(st.words, st.offset, st.k)
G, KD, N = len(exps), st.kin, st.nout
loc2 = np.array([i for i, e in enumerate(exps) if e in e2], np.int32)
loc3 = np.array([i for i, e in enumerate(exps) if e in e3], np.int32)
rng = np.random.default_rng(0)
for M in [int(v) for v in a.rows.split(",")]:
    cfg = NX._BLOCK_CFG[16 if M <= 16 else 32]
    A = mx.array(NX.permute_k(rng.standard_normal((G * M, KD)).astype(np.float16)))
    order = np.concatenate([loc2, loc3])
    A_s = A.reshape(G, M, KD)[mx.array(order)].reshape(G * M, KD)          # rows of the K2 experts first, then K3
    mx.eval(A_s)
    ro = lambda g: mx.array((np.arange(g + 1) * M).astype(np.uint32))
    eall, e2a, e3a = mx.array(np.arange(G, dtype=np.int32)), mx.array(loc2), mx.array(loc3)
    A2, A3 = A_s[:h * M], A_s[h * M:]
    mx.eval(A2, A3)

    def one():
        return [NX.expert_gemm((st,), eall, ro(G), A, G * M, M, cfg)]

    def onesort():
        return [NX.expert_gemm((st,), mx.array(order.astype(np.int32)), ro(G), A_s, G * M, M, cfg)]

    def same2():
        return [NX.expert_gemm((st,), e2a, ro(h), A2, h * M, M, cfg), NX.expert_gemm((st,), e3a, ro(h), A3, h * M, M, cfg)]

    def split():
        return [NX.expert_gemm((st,), e2a, ro(h), A2, h * M, M, cfg, ks=(ka,)), NX.expert_gemm((st,), e3a, ro(h), A3, h * M, M, cfg, ks=(kb,))]
    fns = dict(one=one, onesort=onesort, same2=same2, split=split)
    y1 = one()[0].reshape(G, M, N)[mx.array(order)].reshape(G * M, N)
    ys = mx.concatenate(split(), 0)
    same = bool(mx.array_equal(y1, ys).item())
    for fn in fns.values():
        mx.eval(*fn())
    T = {k: [] for k in fns}
    for _ in range(a.reps):
        for k, fn in fns.items():
            o = fn(); t0 = time.perf_counter(); mx.eval(*o); T[k].append((time.perf_counter() - t0) * 1e3)
    print(f"{os.path.basename(a.dir)} {a.proj} {KD}x{N} G{G} (K{ka} {h} + K{kb} {h}) M{M}: " +
          "  ".join(f"{k} {min(v):.3f} (med {statistics.median(v):.3f})" for k, v in T.items()) + f"  split==one {same}", flush=True)
