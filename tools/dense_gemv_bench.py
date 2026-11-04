#!/usr/bin/env python3
"""Dense decode GEMV bench at the real dense shapes of the current trellis pack: one synthetic linear per kernel, R1 (plain step),
ms / GB/s / memory floor / decode-ALU floor per call, per layer and summed over the layers, plus the served in-chain time from the
op profile for reference. Nothing is loaded from the pack; weights are random trellis of the same shape and bitrate.

  python tools/dense_gemv_bench.py [--bw 720] [--rounds 5] [--calls 120] [--dec 0,1,2] [--no-alu] [--warm-s 1.0]

--bw: memory floor GB/s (720 = the brief's floor, 956 = this box's measured streaming read roof; tools/roofline.py).
--dec: KYOJIN_DENSE_DEC arms to run (0 today, 1 dec_mul1r1 bit-identical, 2 dec_mul1fd fast not bit-identical). The flag is read when
the kernel is built, so each arm builds its own kernel from the same bytes. Arms are interleaved per kernel (one variable, n rounds).
Decode-ALU floor: a separate pure-decode microkernel (32 states and 32 FMAs per iteration, two accumulator chains, no loads in the
loop): the ALU is far below the memory floor at every shape, i.e. these kernels are not decode-ALU bound.
Served in-chain ms/step: exclusive GPU time of the same kernel names in the served plain step of the current pack
(~/kyojin-work/oc-runs/bis/opprof8_full.out, 40 layers). The two S5 x384 groups share one kernel name there; their 2.059 ms is split in the ratio the bench measures.
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlx.core as mx

from kyojin_mlx import metal_src as S
from kyojin_mlx.exl3_linear import Exl3Linear
from kyojin_mlx.fused_group import Exl3Group
from kyojin_mlx.grouped import GroupedGemv

ap = argparse.ArgumentParser()
ap.add_argument("--bw", type=float, default=720.0, help="memory floor GB/s (720 brief floor, 956 measured stream roof)")
ap.add_argument("--rounds", type=int, default=5)
ap.add_argument("--calls", type=int, default=120)
ap.add_argument("--dec", default="0,1,2")
ap.add_argument("--no-alu", action="store_true")
ap.add_argument("--warm-s", type=float, default=1.0)
ap.add_argument("--cap-gib", type=float, default=3.5)
a = ap.parse_args()

# Real dense decode shapes of the pack (config.json: hidden 5120, q_lora 1280, o_lora 1024 x 8 groups, kv 512, moe inter 2304,
# head 129280; plan.bits attn/shared 5, head 6). K = trellis words per 16 columns.
SH = {
    "wq_a":  (5120, 1280, 5), "wkv": (5120, 512, 5), "wq_b": (1280, 32768, 5), "idx.wq_b": (1280, 4096, 5),
    "wo_b":  (8192, 5120, 5), "wo_a_g": (4096, 1024, 5), "sh_gate": (5120, 2304, 5), "sh_up": (5120, 2304, 5),
    "sh_down": (2304, 5120, 5), "head": (5120, 129280, 6),
}
# (label, kind, members, calls per step over the 40 trunk layers, served in-chain ms/step) from opprof8 plain:
# 40 wo_b, 32 single wq_b + 8 grouped wq_b+idx.wq_b, 40 wq_a+wkv groups, 40 gate+up groups, 40 sh_down, 40 grouped wo_a, 1 head.
LAYERS = [
    ("mul1_x512_p1   wo_b",       "single", ("wo_b",), 40, 1.540),
    ("x640_p1        wq_b",       "single", ("wq_b",), 32, 1.203),
    ("s5_x640        wq_b+idx",   "group",  ("wq_b", "idx.wq_b"), 8, 0.315),
    ("s5_x384        wq_a+wkv",   "group",  ("wq_a", "wkv"), 40, 0.780),
    ("s5_x384        gate+up",    "group",  ("sh_gate", "sh_up"), 40, 1.279),
    ("x384_p1        sh_down",    "single", ("sh_down",), 40, 0.427),
    ("grp_fused_x512 wo_a (8g)",  "grouped", ("wo_a_g",), 40, 1.116),
    ("mul1_x5120_p1  head",       "single", ("head",), 1, 0.549),
]
rng = np.random.default_rng(0)
ARRS = {}
TOTAL = [0]


def shape_arrays(name):
    if name in ARRS:
        return ARRS[name]
    kin, nout, K = SH[name]
    tr = rng.integers(-32768, 32768, (kin // 16, nout // 16, 16 * K)).astype(np.int16)
    suh = (rng.choice([-1.0, 1.0], kin) * rng.uniform(0.5, 1.5, kin)).astype(np.float16)
    svh = (rng.choice([-1.0, 1.0], nout) * rng.uniform(0.5, 1.5, nout)).astype(np.float16)
    t, u, v = mx.array(tr), mx.array(suh), mx.array(svh)
    mx.eval(t, u, v)
    nb = tr.nbytes + suh.nbytes + svh.nbytes
    TOTAL[0] += nb
    ARRS[name] = (t, u, v, nb)
    if TOTAL[0] > a.cap_gib * 2**30:
        raise SystemExit(f"dense_gemv_bench: {TOTAL[0]/2**30:.2f} GiB over the {a.cap_gib} GiB cap")
    return ARRS[name]


def make_single(name):
    t, u, v, nb = shape_arrays(name)
    return Exl3Linear(t, u, v, cb="mul1", out_dtype=mx.float16), nb


def make_group(names):
    lins = [make_single(n)[0] for n in names]
    return Exl3Group(lins), sum(shape_arrays(n)[3] for n in names)


def make_grouped(name):
    t, u, v, nb = shape_arrays(name)
    G = 8
    kt, nt, w = t.shape
    tt = mx.broadcast_to(t.reshape(1, kt, nt, w), (G, kt, nt, w))
    uu = mx.broadcast_to(u.reshape(1, -1), (G, u.size))
    vv = mx.broadcast_to(v.reshape(1, -1), (G, v.size))
    mx.eval(tt, uu, vv)
    TOTAL[0] += 7 * nb
    if TOTAL[0] > a.cap_gib * 2**30:
        raise SystemExit(f"dense_gemv_bench: {TOTAL[0]/2**30:.2f} GiB over the {a.cap_gib} GiB cap")
    return GroupedGemv(tt, uu, vv), nb * G


def timed(thunk, n):
    t0 = time.perf_counter()
    outs = [thunk() for _ in range(n)]
    mx.eval(*outs); mx.synchronize()
    return (time.perf_counter() - t0) / n


class Arm:
    def __init__(self, kind, names):
        self.kind = kind
        if kind == "single":
            self.obj, self.nb = make_single(names[0])
            kin = SH[names[0]][0]
            self.call = lambda x: self.obj(x)
            self.xs = [mx.array((rng.standard_normal((1, kin)) * .3).astype(np.float16)) for _ in range(8)]
        elif kind == "group":
            self.obj, self.nb = make_group(names)
            kin = SH[names[0]][0]
            self.call = lambda x: self.obj.run(x)[0]
            self.xs = [mx.array((rng.standard_normal((1, kin)) * .3).astype(np.float16)) for _ in range(8)]
        else:
            self.obj, self.nb = make_grouped(names[0])
            kin = SH[names[0]][0]
            self.call = lambda x: self.obj(x)
            self.xs = [mx.array((rng.standard_normal((1, 8, 1, kin)) * .3).astype(np.float16)) for _ in range(8)]

    def go(self, i):
        return self.call(self.xs[i & 7])

    def warm(self, sec):
        i, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < sec:
            for _ in range(20):
                mx.eval(self.go(i)); i += 1


def decode_rate(K=5, cb="mul1", iters=4096, threads=1 << 18):
    """Pure decode ALU rate (weights/s): 32 independent states decoded per iteration and FMA'd into two accumulator chains,
    the run's words held in registers (w0 ^= i keeps the decodes in the loop). No loads inside the loop."""
    mid, srcs = S.gen_windows(K)
    body = [f"{{ half d = dec_{cb}({srcs[p]}); a{p & 1} = fma(float(d), xv, a{p & 1}); }}" for p in range(32)]
    src = ("uint tid = thread_position_in_grid.x;\n"
           + " ".join(f"uint w{i} = par[{i}];" for i in range(K)) + " uint wp = par[0];\n"
           + "float a0 = 0.0f, a1 = 0.0f, xv = float(tid & 7);\n"
           + f"for (uint i = 0; i < {iters}; ++i) {{ w0 ^= i; wp ^= (i >> 3);\n" + "\n".join(mid) + "\n" + "\n".join(body) + "\n}\n"
           + "out[tid] = a0 + a1;")
    kern = mx.fast.metal_kernel(header=S.dense_header(cb), name=f"dg_dec_roof_{cb}_k{K}", input_names=["par"], output_names=["out"], source=src)
    par = mx.array(rng.integers(0, 2**31 - 1, K).astype(np.int64).tolist(), dtype=mx.uint32)
    def go():
        return kern(inputs=[par], grid=(threads, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(threads,)], output_dtypes=[mx.float32])[0]
    for _ in range(2):
        mx.eval(go())
    t = min(timed(go, 3) for _ in range(3))
    return threads * 32 * iters / t


# ---------------------------------------------------------------- run
decs = [int(v) for v in a.dec.split(",") if v != ""]
DEC_RATE = {} if a.no_alu else {K: decode_rate(K) for K in (5, 6)}
if DEC_RATE:
    print("# decode-ALU roof: " + ", ".join(f"K{K} {v/1e12:.2f} Tw/s" for K, v in DEC_RATE.items()), flush=True)

rows = []
for label, kind, names, calls, served in LAYERS:
    arms, ts = {}, {d: [] for d in decs}
    for d in decs:
        S.DENSE_DEC = d
        arms[d] = Arm(kind, names)
    for d in decs:
        S.DENSE_DEC = d                       # read per call: the arm must see its own decode for the whole burst
        arms[d].warm(a.warm_s)
    for r in range(a.rounds):
        order = decs[r % len(decs):] + decs[:r % len(decs)]
        for d in order:
            S.DENSE_DEC = d
            ts[d].append(timed(lambda: arms[d].go(r), a.calls))
    K = SH[names[0]][2]
    alu = 0.0 if a.no_alu else (arms[decs[0]].nb / (K / 8.0) / DEC_RATE[K]) * 1e3
    rows.append(dict(label=label, kind=kind, names=names, calls=calls, served=served, nb=arms[decs[0]].nb, K=K, alu=alu,
                     ms={d: min(ts[d]) for d in decs}))
    del arms
    mx.clear_cache()

S.DENSE_DEC = 0
NAME = {0: "today (mul1)", 1: "DENSE_DEC=1 (r1)", 2: "DENSE_DEC=2 (fd)"}
w = max(28, max(len(r["label"]) for r in rows))


def table(dec):
    hdr = (f"{'kernel':{w}s} {'calls':>5s} {'MB':>7s} {'GB/s':>6s} {'ms/call':>8s} {'ms/step':>8s} {'mem@'+str(int(a.bw)):>9s} "
           f"{'ALU':>6s} {'excess':>7s} {'served':>7s}")
    print("\n" + hdr)
    print("-" * len(hdr))
    tot = dict(ms=0.0, mem=0.0, alu=0.0, excess=0.0, served=0.0)
    ex = []
    for r in rows:
        ms = r["ms"][dec]
        ms_step = ms * 1e3 * r["calls"]
        mem_step = r["nb"] / (a.bw * 1e9) * 1e3 * r["calls"]
        alu_step = r["alu"] * r["calls"]
        floor = max(mem_step, alu_step)
        exc = ms_step - floor
        gbs = r["nb"] / (ms * 1e9)
        print(f"{r['label']:{w}s} {r['calls']:5d} {r['nb']/1e6:7.2f} {gbs:6.1f} {ms*1e3:8.3f} {ms_step:8.3f} {mem_step:9.3f} "
              f"{alu_step:6.3f} {exc:7.3f} {r['served']:7.3f}")
        tot["ms"] += ms_step; tot["mem"] += mem_step; tot["alu"] += alu_step; tot["excess"] += exc; tot["served"] += r["served"]
        ex.append((exc, r["label"], ms_step, floor))
    print("-" * len(hdr))
    print(f"{'TOTAL':{w}s} {'':5s} {'':7s} {'':6s} {'':8s} {tot['ms']:8.3f} {tot['mem']:9.3f} {tot['alu']:6.3f} {tot['excess']:7.3f} {tot['served']:7.3f}")
    ex.sort(reverse=True)
    print("most ms above floor: " + ", ".join(f"{l} {e:+.3f} (of {m:.3f})" for e, l, m, f in ex[:3]))
    return tot


print(f"# dense decode GEMV bench, R1, synthetic trellis at the pack's shapes and bitrates; memory floor {a.bw:.0f} GB/s, "
      f"{TOTAL[0]/2**30:.2f} GiB of weights, arms interleaved per kernel, min of {a.rounds} rounds of {a.calls} calls")
tots = {}
for d in decs:
    print(f"\n===== {NAME.get(d, d)} =====")
    tots[d] = table(d)
print("\n===== per-step totals =====")
base = tots[decs[0]]["ms"]
for d in decs:
    t = tots[d]
    print(f"{NAME.get(d, d):22s} {t['ms']:7.3f} ms/step  {t['ms']/base:6.3f}x  (memory floor {t['mem']:7.3f} ms, excess {t['excess']:6.3f})")
print(f"served in-chain dense GEMV of the same names: {tots[decs[0]]['served']:.3f} ms/step (opprof8 plain, 40 layers)")
