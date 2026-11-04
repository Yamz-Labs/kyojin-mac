"""Ablation/variant timing of the partial GEMV. python tools/exp.py VARIANT [K ...]"""
import sys, time, numpy as np, mlx.core as mx
from kyojin_mlx import kernels as KN, metal_src as S, ref_numpy as R

def timeit(fn, iters=300):
    for _ in range(10): mx.eval(fn())
    mx.synchronize(); t0 = time.perf_counter()
    outs = [fn() for _ in range(iters)]; mx.eval(outs); mx.synchronize()
    return (time.perf_counter() - t0) / iters

def run(variant, K, k=4096, n=12288):
    src = S.gemv_source(K, 1, True, "mul1")
    if variant == "nodec":      # skip hash: value from the state directly
        src = src.replace("dec_mul1(", "(half)(ushort)(")
    elif variant == "mulonly":   # keep imul, drop byte sum
        src = src.replace("dec_mul1(", "dec_mulonly(")
    elif variant == "nomul":     # keep byte sum on the state itself (no imul)
        src = src.replace("dec_mul1(", "dec_nomul(")
    elif variant in ("v1", "v2", "v3", "e1", "e2", "e3", "t1", "t2", "t3", "t4"):
        src = src.replace("dec_mul1(", f"dec_{variant}(")
    elif variant == "x1":     # exact: lanes -> half2 (or), sums and affine in the float pipe, one rounding to half
        src = src.replace("dec_mul1(", "dec_x1(")
    elif variant == "x2":     # same, no final rounding to half (not bit-exact)
        src = src.replace("half d = dec_mul1(", "float d = dec_x2(").replace("float(d)", "d")
    elif variant in ("f1", "f2", "f3"):
        src = src.replace("half d = dec_mul1(", "float d = dec_nf%s(" % variant[1]).replace("float(d)", "d")
    elif variant == "noload":   # words from registers, no trellis traffic (same ALU work)
        import re
        src = src.replace("tp += tstride;", "tp += 0;").replace("const device uint* tn = tp + tstride;", "const device uint* tn = tp;")
    elif variant == "lut":
        src = "#define dec_mul1(x) lut[(x)]\n" + src
    elif variant == "float":
        src = src.replace("half d = dec_mul1(", "float d = dec_mul1f(").replace("float(d)", "d")
    hdr = S.HEADER + """
inline half dec_mulonly(uint s){ uint h = s * 0x83DCD12Du; return as_type<half>(ushort(h >> 16)); }
inline half dec_nomul(uint s){ uint h = s | (s << 16); uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu);
  half r = as_type<half>(ushort(ushort(t) + ushort(t >> 16) + ushort(0x6400)));
  return fma(r, as_type<half>(ushort(0x1eee)), as_type<half>(ushort(0xc931))); }

inline half dec_x1(uint s){ uint h = s * 0x83DCD12Du; uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu);
  half2 v = as_type<half2>(t | 0x64006400u); float w = fma(float(v.x) + float(v.y), 0.00676727294921875f, -17.3125f); return half(w); }
inline float dec_x2(uint s){ uint h = s * 0x83DCD12Du; uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu);
  half2 v = as_type<half2>(t | 0x64006400u); return fma(float(v.x) + float(v.y), 0.00676727294921875f, -17.3125f); }
#define AFF(r) fma(r, as_type<half>(ushort(0x1eee)), as_type<half>(ushort(0xc931)))
inline half dec_v1(uint s){ uint h = s * 0x83DCD12Du; uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu);
  return AFF(as_type<half>(ushort(((t * 0x10001u) >> 16) + 0x6400u))); }
inline half dec_v2(uint s){ uint h = s * 0x83DCD12Du; uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu) + 0x6400u;
  return AFF(as_type<half>(ushort((t * 0x10001u) >> 16))); }
inline half dec_v3(uint s){ uint h = s * 0x83DCD12Du; uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu);
  return AFF(as_type<half>(ushort((t + (t >> 16) + 0x6400u)))); }
inline half dec_e1(uint s){ uint h = s * 0x83DCD12Du; uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu);
  return as_type<half>(ushort((t + (t >> 16) + 0x6400u))); }
inline half dec_e2(uint s){ uint h = s * 0x83DCD12Du; return AFF(as_type<half>(ushort((h >> 16) & 0x3ffu | 0x6400u))); }
inline half dec_e3(uint s){ uint h = s * 0x83DCD12Du; uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu);
  return as_type<half>(ushort(t)) ; }
inline float dec_nf1(uint s){ uint h = s * 0x83DCD12Du; uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu);
  return float((t * 0x10001u) >> 16); }
inline float dec_nf2(uint s){ uint h = s * 0x83DCD12Du; uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu);
  return float((t & 0xffffu) + (t >> 16)); }
inline float dec_nf3(uint s){ uint h = s * 0x83DCD12Du;
  return float(extract_bits(h,0u,8u) + extract_bits(h,8u,8u) + extract_bits(h,16u,8u) + extract_bits(h,24u,8u)); }
inline half dec_t1(uint s){ uint h = s * 0x83DCD12Du; uint t = (h & 0x00ff00ffu) + (((h >> 8) & 0x00ff00ffu) | 0x6400u);
  return AFF(as_type<half>(ushort(ushort(t) + ushort(t >> 16)))); }
inline half dec_t2(uint s){ uint h = s * 0x83DCD12Du; uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu);
  t += rotate(t, 16u); return AFF(as_type<half>(ushort(ushort(t) + ushort(0x6400)))); }
inline half dec_t3(uint s){ uint h = s * 0x83DCD12Du; uint t = (h & 0x00ff00ffu) + (((h >> 8) & 0x00ff00ffu) | 0x6400u);
  return AFF(as_type<half>(ushort(t + (t >> 16)))); }
inline half dec_t4(uint s){ uint h = s * 0x83DCD12Du; uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu);
  half2 v = as_type<half2>(t | 0x64006400u); return AFF(v.x) + v.y * as_type<half>(ushort(0x1eee)) ; }
"""
    t, su, sv = R.random_linear(k, n, K, seed=K)
    lin = KN.Mul1Gemv(t, su, sv, rows=1)
    lin.kern = mx.fast.metal_kernel(name=f"exp_{variant}_{K}", header=hdr, input_names=["xh","trellis","params"] + (["lut"] if variant == "lut" else []),
                                    output_names=["part"], source=src)
    x = mx.array(np.random.default_rng(0).standard_normal((1, k)).astype(np.float16))
    xh = KN.rot(x, lin.suh); mx.eval(xh)
    if variant == "lut":
        sidx = np.arange(65536, dtype=np.uint64)
        h = (sidx * 0x83DCD12D) & 0xffffffff
        nn = sum(((h >> (8*i)) & 255) for i in range(4))
        r = (0x6400 + nn).astype(np.uint16).view(np.float16)
        d = (r.astype(np.float32) * np.float32(np.uint16(0x1eee).view(np.float16)) + np.float32(np.uint16(0xc931).view(np.float16))).astype(np.float16)
        lutm = mx.array(d)
        lin.partial = lambda xh: lin.kern(inputs=[xh, lin.words, lin.params, lutm], template=[("BITS", lin.K), ("MR", 1)],
              grid=(lin.blocks * 256, lin.tgk, 1), threadgroup=(256, 1, 1), output_shapes=[(lin.tgk, 1, lin.n)], output_dtypes=[mx.float32])[0]
    dt = min(timeit(lambda: lin.partial(xh)) for _ in range(3))
    print(f"{variant:10s} K={K} {dt*1e6:7.1f} us {k*n/dt/1e9:6.1f} Gw/s {k*n*K/8/dt/1e9:6.1f} GB/s", flush=True)

if __name__ == "__main__":
    v = sys.argv[1]
    for K in map(int, sys.argv[2:] or ["2", "3"]): run(v, K)
