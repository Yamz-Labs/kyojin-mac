#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Decode stages k1-k2 on synthetic weights of the pack's real geometry: old path vs new path, ms (printed in us) per op. Run on the Mac; nothing here was
timed by its author, the sandbox it was written in has no Metal GPU.

Arms (old = what the lane / decstage-2 runs today, new = the lever; every pair is measured A B B A, one variable at a time):
  mix      dec_stage_glue  mx-op chain of the mHC mix + collapse + RMSNorm + expand math (the lane's chain launches, reference math) vs stagek.mix_stage + expand_stage
  qkv      dec_qkv         mx.fast.rms_norm(q latent) + dsv4_core.kv_chain  vs  stagek2.qkv_stage                      (R rows, window ring W x 512)
  idxq     dec_idxq        attention wq_b and indexer wq_b as two fp8-folded linears  vs  one same-input Exl3Group launch
  cproj    dec_cproj       x.astype(f32) + two f32 nn.Linear matmuls (weights stored f32, then f16)  vs  stagek2.xgemv
Timing: slope of chains of n and 4n calls on rotating inputs in one eval (no per-eval floor), min over --reps; this prices a call in a pipeline, the way a
step issues them. It does not price the gap a dependent chain of launches pays (the served A/B decides that).
Numerics: every arm's output against a float64 numpy reference (max abs error and relative L2) for old and new side by side: the bar is "new no worse than old".
Memory: weights are counted before anything is allocated and the run stops above --cap-gib (3.5). Whole run is a few seconds per arm (budget 60 s).

  python tools/k1k2_bench.py [--rows 1,2,4] [--reps 7] [--arms mix,qkv,idxq,cproj] [--no-num]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kyojin_mlx import ref_numpy as R  # noqa: E402

# real geometry of the pack (verify_row_bench.py / config.json): hidden 5120, q latent 1280, kv 512, 64 heads x 512 q, indexer 32 heads x 128, window 128
HIDDEN, QLORA, KV, QOUT, IDXOUT, WIN, RD = 5120, 1280, 512, 64 * 512, 32 * 128, 128, 64
KBITS = 3


def exl3_bytes(kin, nout, K=KBITS):
    return kin * nout // 16 * K * 2 + (kin + nout) * 2


def err(y, ref):
    y, ref = np.asarray(y, np.float64), np.asarray(ref, np.float64)
    return float(np.abs(y - ref).max()), float(np.linalg.norm(y - ref) / max(np.linalg.norm(ref), 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1,2,4")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--arms", default="mix,qkv,idxq,cproj")
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--no-num", action="store_true")
    ap.add_argument("--cap-gib", type=float, default=3.5)
    a = ap.parse_args()
    rows_l = [int(v) for v in a.rows.split(",")]
    arms = a.arms.split(",")
    need = exl3_bytes(QLORA, QOUT) + exl3_bytes(QLORA, IDXOUT) + 2 * 512 * HIDDEN * 4 + 24 * 4 * HIDDEN * 4 + 4 * rows_l[-1] * WIN * KV * 2 * 4
    print(f"weights + buffers: {need / 2**30:.3f} GiB (cap {a.cap_gib})")
    if need > a.cap_gib * 2**30:
        raise SystemExit("over the cap")
    import mlx.core as mx
    if not mx.metal.is_available():
        raise SystemExit("k1k2_bench: needs a Metal GPU")
    from kyojin_mlx import dsv4_core as DC, exl3_linear as EL, fused_group as FG, stagek as SK, stagek2 as SK2

    def timeit(f, ins):
        n = len(ins)

        def run(c):
            outs = [f(*ins[i % n]) for i in range(c)]
            flat = [o for out in outs for o in (out if isinstance(out, (list, tuple)) else [out])]
            t0 = time.perf_counter()
            mx.eval(*flat)
            return time.perf_counter() - t0
        T = {n: [], 4 * n: []}
        for r in range(a.reps + 1):
            for c in T:
                dt = run(c)
                if r:
                    T[c].append(dt)
        return (min(T[4 * n]) - min(T[n])) / (3 * n) * 1e6

    def abba(old, new, ins):
        cur = {"old": [], "new": []}
        for lab in ("old", "new", "new", "old"):
            f = old if lab == "old" else new
            mx.eval(*[o for o in (lambda r: r if isinstance(r, (list, tuple)) else [r])(f(*ins[0]))])
            cur[lab].append(timeit(f, ins))
        return min(cur["old"]), min(cur["new"])

    def report(name, rows, o, n, per_layer):
        print(f"{name:<8} rows {rows}: old {o:8.1f} us  new {n:8.1f} us  delta {n - o:+8.1f} us  (x{per_layer} layers/step: {(n - o) * per_layer / 1000:+.3f} ms)")

    rng = np.random.default_rng(3)
    nums = []

    # ------------------------------------------------------------------------------------------------ mix
    if "mix" in arms:
        D = HIDDEN
        fn = mx.array((rng.standard_normal((SK.NMIX, 4 * D)) / np.sqrt(4 * D)).astype(np.float32))
        scale, base = mx.array([0.7, 0.9, 1.1], dtype=mx.float32), mx.array((rng.standard_normal(SK.NMIX) * 0.3).astype(np.float32))
        w = mx.array(rng.uniform(0.5, 1.5, D).astype(np.float16))
        for r in rows_l:
            ins = [(mx.array((rng.standard_normal((r, 1, 4, D)) * 0.5).astype(np.float16)), mx.array(rng.uniform(0.1, 1, (r, 1, 4)).astype(np.float32)),
                    mx.array((rng.standard_normal((r, 1, D)) * 0.5).astype(np.float16))) for _ in range(3)]

            def old(x, c, y):
                h, z, pre = SK.mix_ref(x, c, fn, scale, base, w, 1e-6, 1e-6, 1e-6)       # ~10 launches in mx ops
                return [h, SK.expand_ref(y, x, z, scale, base, 20, 1e-6)]

            def new(x, c, y):
                h, z, pre = SK.mix_stage(x, c, fn, scale, base, w, 1e-6, 1e-6, 1e-6)
                return [h, SK.expand_stage(y, x, z, scale, base, 20, 1e-6)[0]]
            o, n = abba(old, new, ins)
            report("mix", r, o, n, 2 * a.layers)
            if not a.no_num:
                ho, eo = old(*ins[0]); hn, en = new(*ins[0]); mx.eval(ho, eo, hn, en)
                print(f"         numerics h: old vs new max|d| {err(ho, hn)[0]:.2e}; expand {err(eo, en)[0]:.2e} (both f32 inside, one rounding)")

    # ------------------------------------------------------------------------------------------------ qkv
    if "qkv" in arms:
        wq = mx.array(rng.uniform(0.5, 1.5, QLORA).astype(np.float16))
        wk = mx.array(rng.uniform(0.5, 1.5, KV).astype(np.float16))
        for r in rows_l:
            ring = mx.array((rng.standard_normal((r, WIN, KV)) * 0.5).astype(np.float16))
            sess = mx.array(np.arange(r), dtype=mx.uint32)
            wm = -np.ones(r * WIN, np.int32)
            for i in range(r):
                wm[i * WIN + 17] = i
            wmap = mx.array(wm)
            rt = mx.array(rng.uniform(-1, 1, (r, RD)).astype(np.float32))
            ins = [(mx.array((rng.standard_normal((r, QLORA)) * 1.2).astype(np.float16)), mx.array((rng.standard_normal((r, KV)) * 0.8).astype(np.float16)))
                   for _ in range(3)]

            def old(qa, kv):
                return [mx.fast.rms_norm(qa, wq, 1e-6), DC.kv_chain(kv, wk, ring, sess, wmap, rt, 1e-6)]

            def new(qa, kv):
                return list(SK2.qkv_stage(qa, wq, 1e-6, kv, wk, ring, sess, wmap, rt, 1e-6))
            o, n = abba(old, new, ins)
            report("qkv", r, o, n, a.layers)
            if not a.no_num:
                qo, wo = old(*ins[0]); qn, wn = new(*ins[0]); mx.eval(qo, wo, qn, wn)
                x = np.array(ins[0][0]).astype(np.float64)
                ref = np.array(wq).astype(np.float64) * x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6)
                print(f"         q_norm vs f64: old max|d| {err(qo, ref)[0]:.2e} rel {err(qo, ref)[1]:.2e} | new {err(qn, ref)[0]:.2e} rel {err(qn, ref)[1]:.2e}; "
                      f"kv ring new == kv_chain: {bool(mx.array_equal(wo, wn))}")

    # ------------------------------------------------------------------------------------------------ idxq
    if "idxq" in arms:
        def lin(kin, nout, seed):
            tr, su, sv = R.random_linear(kin, nout, KBITS, seed=seed)
            m = EL.Exl3Linear(mx.array(tr), mx.array(su), mx.array(sv), cb="mul1", out_dtype=mx.float16)
            mx.eval(m.parameters())
            return m, (tr, su, sv)
        wq_b, wq_np = lin(QLORA, QOUT, 21)
        iq, iq_np = lin(QLORA, IDXOUT, 22)
        grp = FG.Exl3Group([wq_b, iq])
        for r in rows_l:
            ins = [(mx.array((rng.standard_normal((r, QLORA)) * 0.7).astype(np.float16)),) for _ in range(3)]

            def old(x):
                return [wq_b(x, fq8=True), iq(x, fq8=True)]

            def new(x):
                return grp.run(x, fq8=True)
            o, n = abba(old, new, ins)
            report("idxq", r, o, n, a.layers)
            if not a.no_num:
                from mlx_vlm.models.deepseek_v41.fakequant import fake_quant_fp8_ue8m0 as fq
                x = fq(ins[0][0])
                oo, nn_ = old(*ins[0]), new(*ins[0])
                for nm, (tr, su, sv), yo, yn in (("wq_b", wq_np, oo[0], nn_[0]), ("idx wq_b", iq_np, oo[1], nn_[1])):
                    mx.eval(yo, yn)
                    ref = R.gemv(np.array(x).astype(np.float32), tr, su, sv, dtype=np.float32)
                    print(f"         {nm:<9} vs f32 ref: old rel {err(yo, ref)[1]:.2e} | new rel {err(yn, ref)[1]:.2e}; new == old bitwise: {bool(mx.array_equal(yo, yn))}")

    # ------------------------------------------------------------------------------------------------ cproj
    if "cproj" in arms:
        for wdt, npdt in (("f32", np.float32), ("f16", np.float16)):
            w0 = mx.array((rng.standard_normal((KV, HIDDEN)) * 0.02).astype(npdt))
            w1 = mx.array((rng.standard_normal((KV, HIDDEN)) * 0.02).astype(npdt))
            for r in rows_l:
                ins = [(mx.array((rng.standard_normal((r, HIDDEN)) * 0.7).astype(np.float16)),) for _ in range(3)]

                def old(x):
                    xf = x.astype(mx.float32)
                    return [xf @ w0.T, xf @ w1.T]

                def new(x):
                    return list(SK2.xgemv(x, w0, w1))
                o, n = abba(old, new, ins)
                report(f"cproj {wdt}", r, o, n, 8)
                if not a.no_num:
                    yo, yn = old(*ins[0]), new(*ins[0]); mx.eval(*yo, *yn)
                    x = np.array(ins[0][0]).astype(np.float64)
                    ref = x @ np.array(w0).astype(np.float64).T
                    print(f"         vs f64: old rel {err(yo[0], ref)[1]:.2e} | new rel {err(yn[0], ref)[1]:.2e}")
    print("\nper-step figures assume the layer counts given (mix: 2 per layer x --layers; qkv/idxq: --layers; cproj: 8 kv-source layers, edit report() for the pack); "
          "a pipelined-slope price ignores the launch gap a dependent chain pays: the served A/B decides.")


if __name__ == "__main__":
    main()
