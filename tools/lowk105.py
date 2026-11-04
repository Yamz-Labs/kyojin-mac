#!/usr/bin/env python3
"""Low-K trellis decode ALU ladder of the 1-row EXL3 GEMV partial (macds105): at K <= 4 the GEMV time stops falling with K (decode ALU-bound,
~260 G weights/s on the Air), so the routed experts (K2/K3) and small-K packs sit at 50-75 % of the byte roof.
Variants of the per-weight decode (dec_<cb>), same kernel otherwise (kernels.Mul1Gemv.partial), every variant in ONE process, interleaved per rep:
  mul1  : the engine's codebook (reference)
  v1    : bit-identical: the two 16-bit byte sums and the 0x6400 bias in one integer multiply-add, the high half read as a half (no shift / ushort adds)
  v2    : bit-identical: same with a shift-add instead of the multiply-add
  noaff : timing only: no affine (r * a + b) in the decode (what folding a and b out of the inner loop would save)
  nomul : timing only: no 32-bit hash multiply
  h16a  : bit-identical: the 32-bit hash product from 16-bit multiplies (low product, mulhi, high cross term), same byte sum
  h16b  : h16a with the byte sum in 16-bit lanes
  m24   : bit-identical: mul24 low product + an 8-bit top cross term
  f1    : bit-identical: the hash product in fp32 (exact two-product for the low and high 16-bit halves of the constant, float -> uint)
  f2    : bit-identical: byte halves of the state times the constant halves, all products exact in fp32
  f3    : f1 low half + a 16-bit multiply for the high half
  lut   : bit-identical: the hash product as the sum of two 256-entry threadgroup tables (low byte * C, high byte * C << 8)
  v1na/v1nb : timing only: v1 / v2 without the affine
  mulonly : timing only: the hash multiply and nothing else (the multiply floor)
  null  : timing only: the 16-bit window is the weight (load + window + FMA floor)
Slope timing: pool chains of n and 4n partials (distinct weights), (t(4n) - t(n)) / 3n, min and median over reps.
usage: lowk105.py [--K 2,3,4] [--shape 5120x2304] [--chain 16] [--reps 7]"""
import argparse, os, statistics, sys, time

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kyojin_mlx import kernels as KN, metal_src as S  # noqa: E402

S.HEADER += r"""
inline half dec_v1(uint s) { uint h = s * 0x83DCD12Du; uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu);
    half r = as_type<half2>(t * 0x10001u + 0x64000000u).y; return fma(r, as_type<half>(ushort(0x1eee)), as_type<half>(ushort(0xc931))); }
inline half dec_v2(uint s) { uint h = s * 0x83DCD12Du; uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu);
    half r = as_type<half2>(t + (t << 16) + 0x64000000u).y; return fma(r, as_type<half>(ushort(0x1eee)), as_type<half>(ushort(0xc931))); }
inline half dec_noaff(uint s) { uint h = s * 0x83DCD12Du; uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu);
    return as_type<half>(ushort(ushort(t) + ushort(t >> 16) + ushort(0x6400))); }
inline half dec_nomul(uint s) { uint h = s; uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu);
    half r = as_type<half>(ushort(ushort(t) + ushort(t >> 16) + ushort(0x6400))); return fma(r, as_type<half>(ushort(0x1eee)), as_type<half>(ushort(0xc931))); }
inline half aff1(half r) { return fma(r, as_type<half>(ushort(0x1eee)), as_type<half>(ushort(0xc931))); }
inline half tail1(uint h) { uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu); return aff1(as_type<half>(ushort(ushort(t) + ushort(t >> 16) + ushort(0x6400)))); }
inline half dec_h16a(uint s) { ushort a = ushort(s); ushort lo = a * ushort(0xD12Du); ushort hi = ushort(mulhi(a, ushort(0xD12Du)) + a * ushort(0x83DCu));
    return tail1((uint(hi) << 16) | uint(lo)); }
inline half dec_h16b(uint s) { ushort a = ushort(s); ushort lo = a * ushort(0xD12Du); ushort hi = ushort(mulhi(a, ushort(0xD12Du)) + a * ushort(0x83DCu));
    ushort t = ushort(lo & 0xffu) + ushort(lo >> 8) + ushort(hi & 0xffu) + ushort(hi >> 8) + ushort(0x6400); return aff1(as_type<half>(t)); }
inline half dec_m24(uint s) { uint h = mul24(s, 0xDCD12Du) + (uint(ushort(s) * ushort(0x83u)) << 24); return tail1(h); }
inline uint tp1(float fs, float c) { float p = fs * c; float e = fma(fs, c, -p); return uint(p) + uint(int(e)); }
inline half dec_f1(uint s) { float fs = float(s); return tail1(tp1(fs, 53549.0f) + (tp1(fs, 33756.0f) << 16)); }
inline half dec_f2(uint s) { float sl = float(s & 0xffu), sh = float(s >> 8);
    uint lo = uint(sl * 53549.0f) + (uint(sh * 53549.0f) << 8); uint hi = uint(sl * 33756.0f) + (uint(sh * 33756.0f) << 8); return tail1(lo + (hi << 16)); }
inline half dec_f3(uint s) { float fs = float(s); return tail1(tp1(fs, 53549.0f) + (uint(ushort(s) * ushort(0x83DCu)) << 16)); }
inline half dec_lutt(threadgroup const uint* L, uint s) { return tail1(L[s & 0xffu] + L[256u + (s >> 8)]); }
inline half dec_mulonly(uint s) { return as_type<half>(ushort((s * 0x83DCD12Du) >> 16)); }
inline half dec_v1na(uint s) { uint h = s * 0x83DCD12Du; uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu); return as_type<half2>(t * 0x10001u + 0x64000000u).y; }
inline half dec_v1nb(uint s) { uint h = s * 0x83DCD12Du; uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu); return as_type<half2>(t + (t << 16) + 0x64000000u).y; }
inline half dec_null(uint s) { return as_type<half>(ushort(s)); }
"""
CBS = ["mul1", "v1", "noaff", "v1na", "v1nb", "mulonly", "nomul", "null"]


def lut_source(K):
    src = S.gemv_source(K, 1, False, "lut").replace("dec_lut(", "dec_lutt(lA, ")
    init = ("threadgroup uint lA[512];\n    for (uint i = tid; i < 512u; i += 256u) lA[i] = (i < 256u ? i : (i - 256u) << 8) * 0x83DCD12Du;\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n    ")
    j = src.index("    const uint wprev")
    return src[:j] + "    " + init + src[j:]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--K", default="2,3,4")
    ap.add_argument("--shape", default="5120x2304")
    ap.add_argument("--cbs", default=",".join(CBS))
    ap.add_argument("--chain", type=int, default=16)
    ap.add_argument("--pool", type=int, default=16)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--bw", type=float, default=131.9)
    a = ap.parse_args()
    kin, nout = (int(v) for v in a.shape.split("x"))
    rng = np.random.default_rng(3)
    xh = [mx.array(rng.standard_normal((1, kin)).astype(np.float32)) for _ in range(8)]
    mx.eval(*xh)
    cbs = a.cbs.split(",")
    for K in [int(v) for v in a.K.split(",")]:
        g = {cb: [] for cb in cbs}
        for i in range(a.pool):
            tr = rng.integers(-32768, 32768, (kin // 16, nout // 16, 16 * K), dtype=np.int16)
            ones = np.ones(kin, np.float16), np.ones(nout, np.float16)
            base = KN.Mul1Gemv(mx.array(tr), *ones, cb="mul1")
            for cb in cbs:
                m = base if cb == "mul1" else KN.Mul1Gemv(mx.array(tr), *ones, cb="mul1" if cb == "lut" else cb)
                if cb == "lut":
                    m.kern = KN._kern(("lowk_lut", K), name=f"lowk_lut_k{K}", input_names=["xh", "trellis", "params"], output_names=["part"], source=lut_source(K))
                g[cb].append(m)
        ref = mx.concatenate([g["mul1"][0].partial(x) for x in xh]) if "mul1" in cbs else None
        for cb in cbs:
            if cb not in ("mul1", "noaff", "nomul", "null") and ref is not None:
                out = mx.concatenate([g[cb][0].partial(x) for x in xh])
                print(f"K{K} {cb}: bit-identical to mul1 = {bool(mx.array_equal(out, ref).item())}", flush=True)
        once = lambda cb, n: [g[cb][i % a.pool].partial(xh[i % len(xh)]) for i in range(n)]
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 1.5:
            mx.eval(once(cbs[0], 64))
        T = {(cb, n): [] for cb in cbs for n in (a.chain, 4 * a.chain)}
        for _ in range(a.reps):
            for v in T:
                o = once(*v); t0 = time.perf_counter(); mx.eval(o); T[v].append((time.perf_counter() - t0) * 1e6)
        mb = (kin * nout * K / 8) / 1e6
        for cb in cbs:
            b = (min(T[(cb, 4 * a.chain)]) - min(T[(cb, a.chain)])) / (3 * a.chain)
            md = (statistics.median(T[(cb, 4 * a.chain)]) - statistics.median(T[(cb, a.chain)])) / (3 * a.chain)
            print(f"K{K} {kin}x{nout} {cb:6s} {b:6.1f} us (med {md:6.1f})  {kin * nout / b / 1e3:5.0f} Gw/s  {100 * mb / a.bw * 1e3 / b:5.1f} % roof", flush=True)


if __name__ == "__main__":
    main()
