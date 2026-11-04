#!/usr/bin/env python3
"""opalu: the prefill expert-GEMM decode ALU, measured alone against the matmul alone, and the lever candidates on the real layer shape.

Arms (one variable each, interleaved, first round dropped, median + spread):
  full      today's kernel (decode + matmul, bundled `mul1` decode)
  nodec     the matmul alone (the decode is replaced by a cheap ALU filler): the tensor-op floor
  dec1      the decode alone, mul1: this tool's own kernel (_DEC_SRC), the production 8-tile / 64-weight step over the real trellis words,
            every decoded value consumed by eight half accumulators, the production launch geometry and the per-expert row check
  dec2      the same loop with dec_mul1r2
  nomm      the production kernel with the matmul never run: NOT a decode measurement (with the matmul gone nothing reads the decoded tile,
            so the compiler drops the decode and the loads; this arm measures the zero/store epilogue only). Kept as a warning.
  r2        decode by dec_mul1r2 (pf_dec 2, the cheapest exact scalar form available today)
  pipe      opalu: decode of step k+1 into a second right-input tensor before the matmul of step k is issued
  pipe32    pipe with bks 32 (two 16-k tiles per step: the same register budget as one bks-64 tensor)
  ... plus d0 / nopf / k32 / b16 variants, see ARMCFG
Every arm is checked bit for bit against `full` on the same inputs and routing.

Shapes: the routed experts of DS4.1 (gate/up 5120x2304, down 2304x5120), a pool of `--pool` synthetic experts aliased up to `--e`
experts (the bank layout, the row blocks and the launch size of one production layer), `--m` rows per expert.

usage: tools/opalu_dec.py [--e 192] [--m 64] [--k 3] [--pool 12] [--reps 7] [--json F] [--arms ...]
bounds: --e x --m rows, pool x 11 MB of weights, ~1.5 GiB peak at the defaults (peak is printed).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "bench"))

import mlx.core as mx  # noqa: E402
from bench_moe import synth_stack  # noqa: E402
from kyojin_mlx import metal_nax as NX  # noqa: E402
from kyojin_mlx import metal_src as S  # noqa: E402


# --------------------------------------------------------------------------------------------------------------- decode alone
# `mode="nomm"` of the production kernel is not a decode measurement: the decoded tile is never read (the matmul is what reads it),
# so the compiler drops the decode and the loads and the arm measures the epilogue. This kernel is the lane's decode loop alone, with
# the production tile/word structure and launch geometry, and the values consumed by an XOR chain that ends in a conditional store.
_DEC_HDR = S.HEADER_MOE + S.HEADER_MOE_ROOF + """
#include <metal_stdlib>
using namespace metal;
"""
# metal_kernel takes the BODY: the inputs (tr, off, params), the output (out) and threadgroup_position_in_grid /
# thread_position_in_threadgroup are the generated signature's free variables. Eight independent half accumulators keep the decode off a
# single dependency chain and stop the compiler from dropping it: every decoded value is summed and the sum is compared, so nothing can be
# eliminated the way it is in the production kernel's `nomm` mode (where the decoded tile is never read).
_DEC_SRC = r"""
  const uint3 tg = threadgroup_position_in_grid;
  const uint tid = thread_position_in_threadgroup.x, lane = tid & 31u;
  const uint KT = params[0], NT = params[1], GL = params[2], RB = params[3];
  const uint z = tg.z;
  if (tg.y * RB >= rowoff[z + 1] - rowoff[z]) return;          // no such row block for this expert
  const device uint* tbase = tr + off[z % GL];
  const uint nt0 = (tg.x * 4u + (tid >> 5)) * 2u;
  if (nt0 >= NT) return;
  const uint G = lane >> 3, nb = (lane & 7u) + 8u * (z & 1u);
  uint B0 = KCONST32 * nb + KCONST + 8u * G * KCONST + 256u * KCONST - 16u; if (B0 >= 256u * KCONST) B0 -= 256u * KCONST;
  const uint a0 = B0 >> 5, sh0 = B0 & 31u;
  uint a1 = a0 + 1u; if (a1 >= 8u * KCONST) a1 -= 8u * KCONST;
  uint a2 = a1 + 1u; if (a2 >= 8u * KCONST) a2 -= 8u * KCONST;
  half h[8] = { half(0.f), half(0.f), half(0.f), half(0.f), half(0.f), half(0.f), half(0.f), half(0.f) };
  for (uint kt = 0; kt < KT; kt += 4u) {
    for (uint t = 0; t < 8u; ++t) {
      const device uint* tp = tbase + (ulong(kt + (t >> 1)) * NT + nt0 + (t & 1u)) * KCONST8;
      const uint w0 = tp[a0], w1 = tp[a1], w2 = tp[a2];
      const uint shi = (w0 << sh0) | ((w1 >> 1u) >> (31u - sh0));
      const uint slo = (w1 << sh0) | ((w2 >> 1u) >> (31u - sh0));
      const uint smid = insert_bits(slo >> 16, shi, 16u, 16u);
DECODE
    }
  }
  const half2 s2 = half2(h[0] + h[4], h[1] + h[5]) + half2(h[2] + h[6], h[3] + h[7]);
  if (as_type<uint>(s2) == 0xdeadbeefu) out[(tid + 128u * tg.y) & 0x3ffu] = 1u;
"""


def deconly_source(K: int, cb: str) -> str:
    """The decode loop of one lane of the production kernel (8 tiles x 8 weights per 4-k-tile step), values consumed by an XOR chain."""
    dec = []
    for i in range(8):
        c = 48 - i * K
        if c >= 32:
            e = f"extract_bits(shi, {c - 32}u, 16u)"
        elif c + 16 <= 32:
            e = f"extract_bits(slo, {c}u, 16u)"
        else:
            e = f"extract_bits(smid, {c - 16}u, 16u)"
        dec.append(f"      h[{i}] = h[{i}] + dec_{cb}({e});")
    return (_DEC_SRC.replace("KCONST8", f"{8 * K}u").replace("KCONST32", f"{32 * K}u").replace("KCONST", f"{K}")
            .replace("DECODE", "\n".join(dec)))

H, I = 5120, 2304


def virtual(st, V, pool):
    """A stack of V experts aliased onto the pool's real expert slabs (same words, same offsets layout)."""
    phys = np.arange(V) % pool
    off = np.array(st.offset)[:pool][phys]
    off = np.concatenate([off, off[-1:] + 1]).astype("<u4")
    return type(st)(st.words, mx.array(off), mx.array(np.array(st.k)[phys]),
                    mx.array(np.array(st.suh)[phys]), mx.array(np.array(st.svh)[phys]),
                    st.kt, st.nt, st.cb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e", type=int, default=192)
    ap.add_argument("--m", type=int, default=64, help="rows per expert")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--pool", type=int, default=12)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--burn", type=int, default=4)
    ap.add_argument("--arms", default="full,nodec,nomm,r2,pipe,pipe32")
    ap.add_argument("--skew", type=float, default=0.0, help="row-count skew: the hottest quarter of the experts takes this share of the pairs "
                    "(pfroof's production routing is 0.75 at an 8192-row pass)")
    ap.add_argument("--json")
    a = ap.parse_args()
    if not mx.metal.is_available():
        print("Metal not available")
        return 1
    mx.set_cache_limit(1 << 30)
    E, M, pool = a.e, a.m, min(a.pool, a.e)
    g, u, d = (synth_stack(H, I, [a.k] * pool, "mul1", s) for s in (1, 2, 3))
    mx.eval(g.words, u.words, d.words)
    g, u, d = (virtual(s, E, pool) for s in (g, u, d))
    rng = np.random.default_rng(0)
    eids = mx.array(np.arange(E, dtype=np.int32))
    nh = E // 4                                            # the hot quarter of the experts
    if a.skew:
        # every expert of the hot quarter takes `r` times the rows of a cold one, and the hot quarter takes `skew` of all pairs
        r = a.skew / (1.0 - a.skew) * (E - nh) / nh
        cold = E * M / (nh * r + (E - nh))
        counts = np.array([cold * r] * nh + [cold] * (E - nh))
        counts = np.maximum(np.round(counts).astype(int), 1)
        counts = np.round(counts * (E * M) / counts.sum()).astype(int)
    else:
        counts = np.full(E, M)
    n = int(counts.sum())
    A = mx.array(NX.permute_k(rng.standard_normal((2 * n, H)).astype(np.float16)))
    Ad = mx.array(NX.permute_k(rng.standard_normal((n, I)).astype(np.float16)))
    cl = np.concatenate([[0], np.cumsum(counts)]).astype(np.uint32)
    ro2 = mx.array(np.concatenate([cl[:-1], n + cl]))
    ro1 = mx.array(cl)
    mx.eval(A, Ad)
    fl_gu, fl_dn = 4.0 * n * H * I, 2.0 * n * H * I
    nb = int(sum(-(-c // M) for c in counts))
    print(f"E={E} M={M} K={a.k} pool={pool} skew={a.skew}  bytes/expert {(2 * H * I + I * H) * a.k / 8 / 1e6:.1f} MB  "
          f"weights {3 * pool * H * I * a.k / 8 / 1e9:.2f} GB  rows {n}  blocks {nb} "
          f"(min/mean/max rows {counts.min()}/{counts.mean():.0f}/{counts.max()})", flush=True)

    # arm table: cfg (bmu, nmb, bns, bks), kernel mode, decoder, pipe
    ARMCFG = {
        "full":    dict(cfg=(32, 2, 32, 64)),
        "nodec":   dict(cfg=(32, 2, 32, 64), mode="nodec"),
        "nomm":    dict(cfg=(32, 2, 32, 64), mode="nomm"),
        "r2":      dict(cfg=(32, 2, 32, 64), dec=2),
        "pipe":    dict(cfg=(32, 2, 32, 64), pipe=True),
        "pipe32":  dict(cfg=(32, 2, 32, 32), pipe=True),
        "b16":     dict(cfg=(32, 2, 16, 32)),
        "pipe16":  dict(cfg=(32, 2, 16, 32), pipe=True),
        "b16a":    dict(cfg=(16, 1, 16, 64)),
        "pipe16a": dict(cfg=(16, 1, 16, 64), pipe=True),
        "r2p16":   dict(cfg=(32, 2, 16, 32), dec=2, pipe=True),
        "r2b16":   dict(cfg=(32, 2, 16, 32), dec=2),
        "b16k":    dict(cfg=(32, 2, 32, 16)),
        # d0: decode into the plain half[] staging array and copy it into the right-input tensor afterwards (DIRECT_RB off): the copy is the
        # only loop-carried dependency on rB, so the compiler may hoist the next step's decode above the current matmul
        "d0":      dict(cfg=(32, 2, 32, 64), direct=False),
        "d0b16":   dict(cfg=(32, 2, 16, 32), direct=False),
        "d0r2":    dict(cfg=(32, 2, 32, 64), direct=False, dec=2),
        "d0k32":   dict(cfg=(32, 2, 32, 32), direct=False),
        # nopf: no word prefetch: one word set instead of two (24 registers back), loads issued just before the decode
        "nopf":    dict(cfg=(32, 2, 32, 64), prefetch=False),
        "nopfr2":  dict(cfg=(32, 2, 32, 64), prefetch=False, dec=2),
        # k32: 32-k steps: half the right-input registers (16 instead of 32 per lane), same 32-column matmul
        "k32":     dict(cfg=(32, 2, 32, 32)),
        "r2k32":   dict(cfg=(32, 2, 32, 32), dec=2),
        # dec1 / dec2: the decode loop alone (this tool's own kernel, see _DEC_SRC), mul1 and mul1r2, on the production launch geometry
        "dec1":    dict(deconly="mul1"),
        "dec2":    dict(deconly="mul1r2"),
        "pipe16k": dict(cfg=(32, 2, 32, 16), pipe=True),
    }
    ARMS = [x for x in a.arms.split(",") if x]
    for arm in ARMS:
        assert arm in ARMCFG, f"unknown arm {arm}"
    EXACT = ("full", "r2", "pipe", "pipe32", "pipe16", "pipe16a", "r2p16", "r2b16", "b16", "b16a", "b16k", "pipe16k", "d0", "d0b16", "d0r2", "d0k32", "nopf", "nopfr2", "k32", "r2k32")

    def make(arm):
        c = dict(ARMCFG[arm])
        if c.get("deconly"):
            return dict(deconly=c["deconly"], cfg=None, mode="deconly")
        return dict(stacks=(g, u), A=A, ro=ro2, rows=2 * n, fl=fl_gu, cfg=c["cfg"], mode=c.get("mode", "full"),
                    dec=c.get("dec", 0), pipe=c.get("pipe", False), direct=c.get("direct", True),
                    prefetch=c.get("prefetch", True))

    deconly = {}

    def run_deconly(cb):
        if cb not in deconly:
            deconly[cb] = mx.fast.metal_kernel(name=f"opalu_deconly_{cb}_{a.k}", header=_DEC_HDR, input_names=["tr", "off", "rowoff", "params"],
                                               output_names=["out"], source=deconly_source(a.k, cb))
        kern = deconly[cb]
        kt, nt_ = H // 16, I // 16
        params = mx.array([kt, nt_, E, 64], dtype=mx.uint32)
        gy = -(-int(counts.max()) // 64)
        out = kern(inputs=[g.words, g.offset, ro2, params],
                   grid=(-(-nt_ // 8) * 128, gy, 2 * E), threadgroup=(128, 1, 1),
                   output_shapes=[(1024,)], output_dtypes=[mx.uint32])[0]
        return out, float(fl_gu)

    def run(arm):
        c = make(arm)
        if c.get("deconly"):
            return run_deconly(c["deconly"])
        old = NX.PF_DEC, NX.PIPE, NX.DIRECT_RB, NX.PREFETCH
        NX.PF_DEC, NX.PIPE, NX.DIRECT_RB, NX.PREFETCH = c["dec"], c["pipe"], c["direct"], c["prefetch"]
        try:
            return NX.expert_gemm(c["stacks"], eids, c["ro"], c["A"], c["rows"], int(counts.max()), c["cfg"], True, c["mode"]), c["fl"]
        finally:
            NX.PF_DEC, NX.PIPE, NX.DIRECT_RB, NX.PREFETCH = old

    ref = None
    bits = {}
    for arm in ARMS:
        y = np.array(run(arm)[0])
        if ref is None:
            ref = y
            bits[arm] = None
            print(f"  {arm:8s} reference (sum {np.abs(y).sum():.6g})", flush=True)
            continue
        bits[arm] = bool(np.array_equal(y.view(np.uint32), ref.view(np.uint32)))
        tag = "" if arm in EXACT else "   (timing floor: different values by construction)"
        print(f"  {arm:8s} bit-identical to {ARMS[0]}: {bits[arm]}   (sum {np.abs(y).sum():.6g}){tag}", flush=True)
    for _ in range(a.burn):
        for arm in ARMS:
            mx.eval(run(arm)[0])
    order = np.random.default_rng(7)
    T = {arm: [] for arm in ARMS}
    for r in range(a.reps + 1):
        arms = ARMS if r == 0 else [ARMS[i] for i in order.permutation(len(ARMS))]
        for arm in arms:
            y, fl = run(arm)
            mx.synchronize()
            t0 = time.perf_counter()
            mx.eval(y)
            dt = time.perf_counter() - t0
            if r:
                T[arm].append(dt)
    res = {"geometry": dict(E=E, M=M, K=a.k, pool=pool, rows=n, skew=a.skew, blocks=nb,
                            counts=[int(c) for c in counts], arms={}), "bits": bits}
    base = np.array(T[ARMS[0]]) if ARMS[0] in T else None
    print(f"{'arm':10s} {'ms':>8s} {'TF':>6s} {'TFmin':>6s} {'ratio':>7s} {'spread':>14s}")
    for arm in ARMS:
        v = np.array(T[arm])
        med = float(np.median(v))
        ratio = float(np.median(v / base)) if base is not None else 1.0
        se = float((v / base).std(ddof=1) / np.sqrt(len(v))) if base is not None else 0.0
        fl = make(arm).get("fl") or fl_gu
        res["geometry"]["arms"][arm] = dict(ms=med, ms_min=float(v.min()), tf=fl / med / 1e12,
                                            tf_min=fl / v.min() / 1e12, ratio=ratio, ratio_se=se, spread=[v.min(), v.max()])
        print(f"{arm:10s} {med * 1e3:8.2f} {fl / med / 1e12:6.1f} {fl / v.min() / 1e12:6.1f} {ratio:7.3f} "
              f"{'%.3f-%.3f' % (v.min() / np.median(base), v.max() / np.median(base)):>14s}")
    print(f"peak {mx.get_peak_memory() / 2**30:.2f} GiB")
    NX.PF_DEC, NX.PIPE = 0, False
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
        print(f"json -> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
