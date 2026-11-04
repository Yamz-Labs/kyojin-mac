#!/usr/bin/env python3
"""MoE prefill root cause (macds105): split the fused EXL3 + NAX expert GEMM (metal_nax.expert_gemm, the T2048 MoE prefill path) into trellis
decode and matmul by swapping the per-weight decode (same kernel, same loads, same matmul2d tiles), real experts of a stacked layer slice.
  mul1 : the engine's codebook (reference)
  v1na : timing only: hash + byte sum, no affine (what folding the affine out of the kernel would save)
  null : timing only: the 16-bit window is the weight (load + window + matmul floor)
ms per launch of --experts experts at M rows each (pick_cfg config), min / median over interleaved reps.
usage: pfdec105.py [--dir models/ds_l10] [--proj w1] [--rows 1,16,32,64] [--experts 8] [--mix]"""
import argparse, glob, os, statistics, sys, time

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kyojin_mlx import metal_nax as NX  # noqa: E402
from nax_common import mini_stack  # noqa: E402
from safetensors import safe_open  # noqa: E402

NX.HEADER_NAX += r"""
inline half dec_v1na(uint s) { uint h = s * 0x83DCD12Du; uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu); return as_type<half2>(t * 0x10001u + 0x64000000u).y; }
inline half dec_null(uint s) { return as_type<half>(ushort(s)); }
"""
CBS = ["mul1", "v1na", "null"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.path.expanduser("~/kyojin-mlx/models/ds_l10"))
    ap.add_argument("--proj", default="w1")
    ap.add_argument("--rows", default="1,16,32,64")
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--cbs", default=",".join(CBS))
    ap.add_argument("--mix", action="store_true", help="one stack of the first --experts experts, all bitrates (the real one-launch prefill)")
    a = ap.parse_args()
    mx.set_cache_limit(256 << 20)
    f = sorted(glob.glob(os.path.join(a.dir, "stacked.*.safetensors")))[0]
    with safe_open(f, "np") as s:
        kv = s.get_tensor([x for x in s.keys() if x.endswith(f".experts.{a.proj}.k")][0])
    rng = np.random.default_rng(0)
    for K in (["mix"] if a.mix else sorted(set(kv.tolist()))):
        exps = list(range(min(a.experts, len(kv)))) if a.mix else [i for i in range(len(kv)) if kv[i] == K][:a.experts]
        cbs = a.cbs.split(",")
        st = {cb: mini_stack(a.dir, a.proj, exps, cb) for cb in cbs}
        G, KD, N = len(exps), st["mul1"].kin, st["mul1"].nout
        eids = mx.array(np.arange(G, dtype=np.int32))
        for M in [int(v) for v in a.rows.split(",")]:
            A = mx.array(NX.permute_k(rng.standard_normal((G * M, KD)).astype(np.float16)))
            rowoff = mx.array((np.arange(G + 1) * M).astype(np.uint32))
            cfg = NX.pick_cfg([M] * G)
            run = lambda cb: NX.expert_gemm((st[cb],), eids, rowoff, A, G * M, M, cfg)
            for cb in cbs:
                mx.eval(run(cb))
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 0.5:
                mx.eval(run("mul1"))
            T = {cb: [] for cb in cbs}
            for _ in range(a.reps):
                for cb in cbs:
                    o = run(cb); t0 = time.perf_counter(); mx.eval(o); T[cb].append((time.perf_counter() - t0) * 1e3)
            ref = min(T["mul1"])
            print(f"{os.path.basename(a.dir)} {a.proj} {KD}x{N} K{K} G{G} M{M:3d} cfg {cfg}: " + "  ".join(
                f"{cb} {min(T[cb]):.3f} (med {statistics.median(T[cb]):.3f}, {100 * min(T[cb]) / ref:.0f}%)" for cb in cbs), flush=True)


if __name__ == "__main__":
    main()
