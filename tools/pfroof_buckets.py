#!/usr/bin/env python3
"""pfroof step 1: production-scale routed-expert GEMM per M bucket (gate+up and down).

384 virtual experts (pool of 48 synth K=3 experts aliased, real DS4.1 dims:
gate/up 5120x2304, down 2304x5120), uniform M rows each, sustained clocks
(burn-in, then interleaved reps, median). Compares Air tiles vs bns=16 tiles
vs nodec (matmul floor) / nomm (decode cost). Bit-identity across full-mode
cfgs asserted (same tile math must give identical bits).

usage: flock /tmp/ultra_bench.lock python tools/pfroof_buckets.py [--half]
"""
import argparse, os, sys, time
import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from bench_moe import synth_stack  # noqa: E402
from kyojin_mlx import metal_nax as NX  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--half", action="store_true", help="96 experts (half-size first)")
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--burn", type=int, default=8)
ap.add_argument("--ms", default="16,32,64,128,256,512", help="rows per expert")
ap.add_argument("--pad", type=int, default=0, help="maxrows = M + pad (pad 65: never the <= 64-row decode-once TG path)")
ap.add_argument("--cfgs", default="", help="extra name=bmu:nmb:bns:bks,... (e.g. c128=32:4:16:64)")
a = ap.parse_args()
mx.set_cache_limit(16 << 30)  # production default: a smaller cap makes every big output a fresh (page-faulting) buffer

POOL = 48
H, I, K = 5120, 2304, 3
g0 = synth_stack(H, I, [K] * POOL, "mul1", 1)
u0 = synth_stack(H, I, [K] * POOL, "mul1", 2)
d0 = synth_stack(I, H, [K] * POOL, "mul1", 3)


def virtual(st, V):
    pool = st.experts
    phys = np.arange(V) % pool
    off = np.array(st.offset)[:pool][phys]
    off = np.concatenate([off, off[-1:] + 1]).astype("<u4")
    return type(st)(st.words, mx.array(off), mx.array(np.array(st.k)[phys]),
                    mx.array(np.array(st.suh)[phys]), mx.array(np.array(st.svh)[phys]),
                    st.kt, st.nt, st.cb)


mx.eval(g0.words, u0.words, d0.words)
print(f"pool {POOL} synth K=3 experts; gate/up {H}x{I}, down {I}x{H}", flush=True)

CFGS = {
    "air16": (16, 1, 32, 64),
    "air32": (32, 1, 32, 64),
    "air64": (32, 2, 32, 64),
    "b16": (16, 1, 16, 64),
    "b32": (32, 1, 16, 64),
    "b64": (32, 2, 16, 64),
    "big": NX._BIG_CFG_ULTRA,
}
for kv in filter(None, a.cfgs.split(",")):
    k, v = kv.split("=")
    CFGS[k] = tuple(int(t) for t in v.split(":"))
rng = np.random.default_rng(0)

# fp16 NAX roof anchor (peak84 methodology: single eval, min over many, interleaved)
rops = []
for (M_, K_, N_) in ((2048, 5120, 2304), (2048, 2304, 5120)):
    xa = mx.array(rng.standard_normal((M_, K_)).astype(np.float16))
    wa = mx.array(rng.standard_normal((N_, K_)).astype(np.float16))
    mx.eval(xa, wa)
    rops.append((f"fp16 {M_}x{K_}x{N_}", 2 * M_ * K_ * N_, lambda xa=xa, wa=wa: xa @ wa.T))
for _ in range(3):
    for _, _, f in rops:
        mx.eval(f())
T = {n: [] for n, _, _ in rops}
for _ in range(30):
    for n, _, f in rops:
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(f()); T[n].append(time.perf_counter() - t0)
for n, fl, _ in rops:
    v = sorted(T[n])
    print(f"roof {n}: min {fl / v[0] / 1e12:.1f} TF med {fl / v[len(v) // 2] / 1e12:.1f} TF",
          flush=True)

for M in (int(v) for v in a.ms.split(",")):
    E = 96 if a.half else (192 if M == 512 else 384)  # M=512x384 would near the 12 GiB cap
    g, u, d = (virtual(s, E) for s in (g0, u0, d0))
    eids = mx.array(np.arange(E, dtype=np.int32))
    print(f"M={M}: E={E}", flush=True)
    n = E * M
    A = mx.array(NX.permute_k(rng.standard_normal((2 * n, H)).astype(np.float16)))
    Ad = mx.array(NX.permute_k(rng.standard_normal((n, I)).astype(np.float16)))
    cl = (np.arange(E + 1) * M).astype(np.uint32)
    ro2 = mx.array(np.concatenate([cl[:-1], n + cl]))
    ro1 = mx.array(cl)
    mx.eval(A, Ad)
    fl1 = 2.0 * E * M * H * I  # per projection pair-half (gate or up); gate+up = 2x, down = 2x

    def run(proj, A_, ro, rows, cfg, mode="full"):
        return NX.expert_gemm(proj, eids, ro, A_, rows, M + a.pad, cfg, True, mode)

    # bit-identity across full-mode cfgs (gate+up)
    ref = None
    for name, cfg in CFGS.items():
        y = np.array(run((g, u), A, ro2, 2 * n, cfg)).astype(np.float32)
        if ref is None:
            ref = y
        elif not np.array_equal(y, ref):
            print(f"M={M} {name} DIFFERS rel "
                  f"{np.linalg.norm(y - ref) / np.linalg.norm(ref):.2e}", flush=True)
    # burn-in (sustained clocks)
    for _ in range(a.burn):
        mx.eval(run((g, u), A, ro2, 2 * n, CFGS["air32"]))
        mx.eval(run((d,), Ad, ro1, n, CFGS["air32"]))
    row = {}
    for tag, proj, A_, ro, rows, mult in (("gu", (g, u), A, ro2, 2 * n, 4.0),
                                          ("dn", (d,), Ad, ro1, n, 2.0)):
        fl = mult * E * M * H * I  # gu = gate+up = 2 projs x 2*M*N*K...; dn = 1 proj
        for name, cfg in CFGS.items():
            ts = []
            for _ in range(a.reps):
                mx.synchronize(); t0 = time.perf_counter()
                mx.eval(run(proj, A_, ro, rows, cfg))
                ts.append(time.perf_counter() - t0)
            vs = sorted(ts)
            row[(tag, name)] = (float(np.median(ts)), fl / np.median(ts) / 1e12,
                                fl / vs[0] / 1e12)
        for mode in ("nodec", "nomm"):
            ts = []
            for _ in range(a.reps):
                mx.synchronize(); t0 = time.perf_counter()
                mx.eval(run(proj, A_, ro, rows, CFGS["air32"], mode))
                ts.append(time.perf_counter() - t0)
            vs = sorted(ts)
            row[(tag, mode)] = (float(np.median(ts)), fl / np.median(ts) / 1e12,
                                fl / vs[0] / 1e12)
    parts = []
    for tag in ("gu", "dn"):
        for name in list(CFGS) + ["nodec", "nomm"]:
            ms, tf, tfmin = row[(tag, name)]
            parts.append(f"{tag}/{name} {ms * 1e3:7.2f}ms {tf:5.1f}/{tfmin:5.1f}TF")
    print(f"M={M:4d} E={E} med/min " + " | ".join(parts), flush=True)
print(f"peak {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)
