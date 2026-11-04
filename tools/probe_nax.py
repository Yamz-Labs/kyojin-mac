"""Fused EXL3 decode + NAX GEMM probe vs today's path (decode_pf + gather_mm) on real DS4.1 experts.
   PYTHONPATH=$PWD:$PWD/bench:$PWD/tools python tools/probe_nax.py [--proj w1|w2] [--rows 1,8,64] [--experts 8] [--variants today,coop] [--reps 5]"""
import argparse, os, sys, time
import numpy as np, mlx.core as mx
from kyojin_mlx import moe_prefill as MP, metal_nax as NX
import nax_arms as ARMS
from nax_common import mini_stack, kperm_cols, synth_stack

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.path.expanduser("~/kyojin-mlx/models/ds_l3"))
ap.add_argument("--proj", default="w1"); ap.add_argument("--rows", default="1,8,64"); ap.add_argument("--experts", type=int, default=8)
ap.add_argument("--variants", default="today,coop"); ap.add_argument("--reps", type=int, default=5); ap.add_argument("--cfg", default="", help="coop config bmu,nmb,bns,bks")
ap.add_argument("--synth", default="", help="kin,nout,K (random trellis, dense-like shape) instead of real experts")
ap.add_argument("--check", action="store_true")
a = ap.parse_args()
G = a.experts
if a.synth:
    kin, nout, K = (int(v) for v in a.synth.split(","))
    st = synth_stack(kin, nout, [K] * G)
else:
    from safetensors import safe_open
    import glob
    f = sorted(glob.glob(os.path.join(a.dir, "stacked.*.safetensors")))[0]
    with safe_open(f, "np") as s:
        k = [x for x in s.keys() if x.endswith(f".experts.{a.proj}.k")][0]
        kv = s.get_tensor(k)
    k2 = [i for i in range(len(kv)) if kv[i] == 2][:G // 2]; k3 = [i for i in range(len(kv)) if kv[i] == 3][:G - len(k2)]
    exps = sorted(k2 + k3)
    st = mini_stack(a.dir, a.proj, exps)
print(f"proj {a.proj} kin {st.kin} nout {st.nout} ks {st.ks} experts {G} weights/expert {st.kin*st.nout/1e6:.1f}M")
eids = mx.array(np.arange(G, dtype=np.int32))
rng = np.random.default_rng(0)
KD, N = st.kin, st.nout
vs = a.variants.split(",")

def timeit(fn, reps):
    r = fn(); mx.eval(r)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); r = fn(); mx.eval(r); ts.append(time.perf_counter() - t0)
    return min(ts), r

for Mr in [int(v) for v in a.rows.split(",")]:
    A = rng.standard_normal((G * Mr, KD)).astype(np.float16)
    flop = 2.0 * G * Mr * KD * N
    res = {}
    Ad = mx.array(A)
    lidx = mx.array(np.repeat(np.arange(G, dtype=np.uint32), Mr))
    out = {}
    if "today" in vs:
        td, w = timeit(lambda: MP.decode_group((st,), eids, st.cb), a.reps)
        tg, y = timeit(lambda: mx.gather_mm(Ad.reshape(G * Mr, 1, KD), w.swapaxes(-1, -2), rhs_indices=lidx, sorted_indices=True), a.reps)
        tt, y = timeit(lambda: mx.gather_mm(Ad.reshape(G * Mr, 1, KD), MP.decode_group((st,), eids, st.cb).swapaxes(-1, -2), rhs_indices=lidx, sorted_indices=True), a.reps)
        out["today"] = tt
        yref = np.array(y).reshape(G * Mr, N).astype(np.float32)
        print(f"M={Mr:4d} today: decode {td*1e3:7.2f} ms  gather_mm {tg*1e3:7.2f} ms ({flop/tg/1e12:5.2f} TFLOPS)  together {tt*1e3:7.2f} ms ({flop/tt/1e12:5.2f} TFLOPS)")
    if "dense" in vs:       # today's dense path (Exl3Linear._dense): decode_wr -> fp16 weights, then mx.matmul (needs one expert)
        from kyojin_mlx import kernels as KN
        assert G == 1
        wr_t, wr = timeit(lambda: KN.decode_wr(st.words, st.ks[0], st.kt, st.nt, st.cb), a.reps)
        mm_t, y = timeit(lambda: Ad @ wr.T, a.reps)
        tt, y = timeit(lambda: Ad @ KN.decode_wr(st.words, st.ks[0], st.kt, st.nt, st.cb).T, a.reps)
        out["today"] = tt
        yref = np.array(y).astype(np.float32)
        print(f"M={Mr:4d} dense today: decode_wr {wr_t*1e3:7.2f} ms  matmul {mm_t*1e3:7.2f} ms ({flop/mm_t/1e12:5.2f} TFLOPS)  together {tt*1e3:7.2f} ms ({flop/tt/1e12:5.2f} TFLOPS)")
    for v in vs:
        if v.startswith("stg"):       # stg:nax | stg:simd [:rows[:mode]]
            parts = v.split(":"); eng = parts[1]; rws = int(parts[2]) if len(parts) > 2 else 64; mode = parts[3] if len(parts) > 3 else "full"
            src = ARMS.staged_source(st.ks, st.cb, eng, rws, True, mode)
            kern = mx.fast.metal_kernel(name=f"stg_{eng}_{rws}_{mode}", input_names=["eids", "rowoff", "tr", "off", "kk", "A", "params"], output_names=["Y"], header=NX.HEADER_NAX, source=src)
            rowoff = mx.array((np.arange(G + 1) * Mr).astype(np.uint32))
            Ap = mx.array(np.concatenate([A, np.zeros((64, KD), np.float16)])); params = mx.array([st.kt, st.nt], dtype=mx.uint32)
            def go():
                return kern(inputs=[eids, rowoff, st.words, st.offset, st.k, Ap, params],
                            grid=(((st.nt + 7) // 8) * 128, (Mr + rws - 1) // rws, G), threadgroup=(128, 1, 1),
                            output_shapes=[(G * Mr, N)], output_dtypes=[mx.float16])[0]
            t, y2 = timeit(go, a.reps)
            msg = f"M={Mr:4d} {v}: {t*1e3:7.2f} ms ({flop/t/1e12:5.2f} TFLOPS)"
            if "today" in out: msg += f"  x{out['today']/t:4.2f} vs today"
            if a.check and mode == "full" and "today" in out:
                y2n = np.array(y2).astype(np.float32)
                msg += f"  rel diff vs today {np.abs(y2n-yref).max()/np.abs(yref).max():.1e}"
            print(msg)
        if v.startswith("coop"):
            mode = v.split(":")[1] if ":" in v else "full"
            bmu, nmb, bns, bks = (int(x) for x in (a.cfg or ("32,1,32,16" if Mr <= 32 else "32,2,32,16")).split(","))
            rowoff = mx.array((np.arange(G + 1) * Mr).astype(np.uint32))
            Ap = mx.array(kperm_cols(A))
            def go():
                return NX.expert_gemm((st,), eids, rowoff, Ap, G * Mr, Mr, (bmu, nmb, bns, bks), True, mode)
            t, y2 = timeit(go, a.reps)
            msg = f"M={Mr:4d} {v} cfg={bmu},{nmb},{bns},{bks}: {t*1e3:7.2f} ms ({flop/t/1e12:5.2f} TFLOPS)"
            if "today" in out: msg += f"  x{out['today']/t:4.2f} vs today"
            if a.check and mode == "full" and "today" in out:
                y2n = np.array(y2).astype(np.float32)
                msg += f"  rel diff vs today {np.abs(y2n-yref).max()/np.abs(yref).max():.1e}"
            print(msg)
