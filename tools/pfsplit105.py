#!/usr/bin/env python3
"""A/B of the per-bitrate fused MoE prefill passes (moe_prefill.PF_SPLIT_K) vs one launch for all bitrates, interleaved in one process,
bit-identity of the layer output checked per T. Uniform random routing (every expert active).
  --dir DIR --prefix P   a stacked layer slice (e.g. models/ds_l10, layers.10.ffn.experts; models/glm_l10 ...mlp.experts with --names gate_proj,up_proj,down_proj)
  --synth H,I,E,KS        synthetic stack: hidden H, inner I, E experts with bitrates cycling over KS (e.g. 4096,1536,160,2:3 = GLM-4.5-Air shapes)
usage: pfsplit105.py (--dir DIR --prefix P | --synth H,I,E,KS) [--topk 6] [--T 512,2048] [--reps 4]"""
import argparse, os, statistics, sys, time

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
from kyojin_mlx import patch as P, moe as M, moe_prefill as MP  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="")
    ap.add_argument("--prefix", default="layers.10.ffn.experts")
    ap.add_argument("--names", default="w1,w3,w2")
    ap.add_argument("--synth", default="")
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--T", default="512,2048")
    ap.add_argument("--reps", type=int, default=4)
    a = ap.parse_args()
    mx.set_cache_limit(256 << 20)
    if a.synth:
        from bench_moe import synth_stack
        H, I, E, ks = a.synth.split(",")
        H, I, E = int(H), int(I), int(E)
        kl = [int(k) for k in ks.split(":")]
        ks_e = [kl[e % len(kl)] for e in range(E)]
        st = [synth_stack(H, I, ks_e, "mul1", 1), synth_stack(H, I, ks_e, "mul1", 2), synth_stack(I, H, ks_e, "mul1", 3)]
        tag = f"synth H{H} I{I} E{E} K{ks}"
    else:
        pre = a.prefix.split(".ffn.")[0].split(".mlp.")[0] + "."
        t = P.load_pack(a.dir, prefix=pre)
        st = [M.Stack.from_arrays(t, f"{a.prefix}.{w}", "mul1") for w in a.names.split(",")]
        tag = f"{os.path.basename(a.dir)} {a.prefix}"
    mx.eval(*[x for s in st for x in (s.words, s.offset, s.k, s.suh, s.svh)])
    moe = M.MoeGather(*st, a.topk, limit=10.0, wpre=True, fq=True)
    pf = MP.MoePrefill(moe)
    E, H = st[0].experts, st[0].kin
    rng = np.random.default_rng(0)
    for T in [int(v) for v in a.T.split(",")]:
        ids = mx.array(np.stack([rng.choice(E, a.topk, replace=False) for _ in range(T)]).astype(np.int32))
        x = mx.array((rng.standard_normal((T, H)) * 0.5).astype(np.float16))
        w = mx.array(rng.random((T, a.topk)).astype(np.float32))
        mx.eval(x, ids, w)

        def run(split):
            MP.PF_SPLIT_K = split
            return pf(x, ids, w)
        y1, y0 = run(True), run(False)
        same = bool(mx.array_equal(y1, y0).item())
        D = {False: [], True: []}
        for _ in range(a.reps):
            for v in (False, True):
                t0 = time.perf_counter(); mx.eval(run(v)); D[v].append((time.perf_counter() - t0) * 1e3)
        b0, b1 = min(D[False]), min(D[True])
        print(f"{tag} T{T}: one launch {b0:.1f} ms (med {statistics.median(D[False]):.1f}) -> per bitrate {b1:.1f} ms "
              f"(med {statistics.median(D[True]):.1f}), {T / b0 * 1e3:.0f} -> {T / b1 * 1e3:.0f} tok/s/layer, bit-identical {same}", flush=True)


if __name__ == "__main__":
    main()
