import statistics, sys, numpy as np, mlx.core as mx
from kyojin_mlx import kernels as KN, mma_gemv as MM, metal_src as S
from mma_bench import synth, SHAPES, timeit
S.HEADER += r"""
inline half dec_null(uint s) { return as_type<half>(ushort(s)); }
inline half dec_d1(uint s) { uint h = s * 0x83DCD12Du; return as_type<half>(ushort(h >> 8)); }
inline half dec_d1b(uint s) { uint h = s * 0x83DCD12Du; uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu); return as_type<half>(ushort(t)); }
inline half dec_d1c(uint s) { uint h = s * 0x83DCD12Du; uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu); uint u = t * 0x10001u + 0x64000000u; return as_type<half2>(u).y; }
inline half dec_c1(uint s) { uint h = s * 0x83DCD12Du; uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu); return as_type<half>(ushort(ushort(t) ^ ushort(t >> 16))); }
inline half dec_c2(uint s) { uint h = s * 0x83DCD12Du; uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu); return as_type<half>(ushort(t + (t >> 16))); }
inline half dec_c3(uint s) { uint h = s * 0x83DCD12Du; uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu); return as_type<half>(ushort(t)) + as_type<half>(ushort(t >> 16)); }
inline half dec_c4(uint s) { uint h = s * 0x83DCD12Du; uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu); half2 v = as_type<half2>(t); return v.x + v.y; }
inline half dec_v1(uint s) { uint h = s * 0x83DCD12Du; uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu); uint u = t * 0x10001u + 0x64000000u; half r = as_type<half2>(u).y; return fma(r, as_type<half>(ushort(0x1eee)), as_type<half>(ushort(0xc931))); }
"""
R = int(sys.argv[1]) if len(sys.argv) > 1 else 5
kin, nout, K = SHAPES["mlp_up"]; lin = synth(kin, nout, K); mx.eval(lin._words); g = lin._g[1]
rng = np.random.default_rng(5)
xs = [mx.array(rng.standard_normal((R, kin)).astype(np.float16)) for _ in range(12)]; mx.eval(*xs)
xh = [KN.rot(x, g.suh) for x in xs]; mx.eval(*xh)
def run(cb):
    src = MM.mma_gemv_source(K, cb)
    kern = KN._kern(("lad", cb, K, R), name=f"lad_{cb}", input_names=["xh", "trellis", "params"], output_names=["part"], source=src)
    f = lambda h: kern(inputs=[h, g.words, g.params], template=[("BITS", K), ("MR", R), ("WSP", 4)], grid=(g.blocks * 256, g.tgk, 1), threadgroup=(256, 1, 1), output_shapes=[(g.tgk, R, g.n)], output_dtypes=[mx.float32])[0]
    return timeit(f, xh, 7, 12)[1]
names = ["null", "d1", "d1b", "c1", "c2", "c3", "c4", "d1c", "mul1"]
T = {n: [] for n in names}
for rep in range(6):
    for n in names: T[n].append(run(n))
print(f"R={R} " + " ".join(f"{n} {min(t):.0f}" for n, t in T.items()), flush=True)
