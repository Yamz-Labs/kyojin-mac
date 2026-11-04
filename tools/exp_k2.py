"""K2 (softmax -> fp16 P) variants at NK=640, 512 queries x 64 heads. Bit-compare vs shipped. usage: exp_k2.py"""
import time, numpy as np, mlx.core as mx
from kyojin_mlx import attn_prefill as AP, metal_nax as NX
H, D, NK, TC = 64, 512, 640, 512
rng = np.random.default_rng(0)
ids = np.stack([rng.choice(6144, NK, replace=False) for _ in range(TC)]).astype(np.int32); ids[:, ::37] = -1
ids = mx.array(ids)
S = mx.array((rng.standard_normal((TC * H, NK)) * 3).astype(np.float32)); sk = mx.array(rng.standard_normal(H).astype(np.float32))
Sm = mx.where(mx.repeat(ids >= 0, H, axis=0), S, -mx.inf)
base = AP._kernels(H, D, NK, 0.1)[1]
fb = lambda: base(inputs=[S, ids, sk], grid=(TC * H * 32, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(TC * H, NK)], output_dtypes=[mx.float16])[0]
TAIL = r"""
    const float lse = m + metal::precise::log(sum), sk = sinks[h];
    const float norm = max(lse, sk) + metal::precise::log(1.0f + metal::precise::exp(-fabs(lse - sk)));
"""
# V1: masked keys are -inf in S already (no ids read), float4 loads/stores
V1 = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint row = threadgroup_position_in_grid.x * 4 + simdgroup_index_in_threadgroup;
    const uint h = row % HH;
    const device float4* sp = (const device float4*)(S + ulong(row) * NK);
    float4 v[NK / 128];
    float m = -INFINITY;
    for (uint i = 0; i < NK / 128; ++i) { v[i] = sp[lane + 32 * i]; m = max(m, max(max(v[i].x, v[i].y), max(v[i].z, v[i].w))); }
    m = simd_max(m);
    float sum = 0.0f;
    for (uint i = 0; i < NK / 128; ++i) sum += metal::precise::exp(v[i].x - m) + metal::precise::exp(v[i].y - m) + metal::precise::exp(v[i].z - m) + metal::precise::exp(v[i].w - m);
    sum = simd_sum(sum);
""" + TAIL + r"""
    device half4* pp = (device half4*)(P + ulong(row) * NK);
    for (uint i = 0; i < NK / 128; ++i) pp[lane + 32 * i] = half4(metal::precise::exp(v[i].x - norm), metal::precise::exp(v[i].y - norm), metal::precise::exp(v[i].z - norm), metal::precise::exp(v[i].w - norm));
"""
# V2: V1 order of summation differs -> keep the shipped order (lane + 32 i) but with the -inf trick and no ids
V2 = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint row = threadgroup_position_in_grid.x * 4 + simdgroup_index_in_threadgroup;
    const uint h = row % HH;
    const device float* sp = S + ulong(row) * NK;
    float v[NK / 32];
    float m = -INFINITY;
    for (uint i = 0; i < NK / 32; ++i) { v[i] = sp[lane + 32 * i]; m = max(m, v[i]); }
    m = simd_max(m);
    float sum = 0.0f;
    for (uint i = 0; i < NK / 32; ++i) sum += metal::precise::exp(v[i] - m);
    sum = simd_sum(sum);
""" + TAIL + r"""
    device half* pp = P + ulong(row) * NK;
    for (uint i = 0; i < NK / 32; ++i) pp[lane + 32 * i] = half(metal::precise::exp(v[i] - norm));
"""
V3 = r'''
    const uint lane = thread_index_in_simdgroup;
    const uint row = threadgroup_position_in_grid.x * 4 + simdgroup_index_in_threadgroup;
    const uint h = row % HH;
    const device float* sp = S + ulong(row) * NK;
    float v[NK / 32];
    float m = -INFINITY;
    for (uint i = 0; i < NK / 32; ++i) { v[i] = sp[lane + 32 * i]; m = max(m, v[i]); }
    m = simd_max(m);
    float sum = 0.0f;
    for (uint i = 0; i < NK / 32; ++i) { v[i] = metal::precise::exp(v[i] - m); sum += v[i]; }
    sum = simd_sum(sum);
''' + TAIL + r'''
    const float f = metal::precise::exp(m - norm);
    device half* pp = P + ulong(row) * NK;
    for (uint i = 0; i < NK / 32; ++i) pp[lane + 32 * i] = half(v[i] * f);
'''
V4 = V3.replace("metal::precise::exp(v[i] - m)", "metal::fast::exp(v[i] - m)").replace("metal::precise::exp(m - norm)", "metal::fast::exp(m - norm)")
def mk(src, tg=128):
    k = mx.fast.metal_kernel(name=f"exp_k2_{abs(hash(src))%10**8}", header=NX.HEADER_NAX, input_names=["S", "sinks"], output_names=["P"], source=src.replace("HH", str(H)).replace("NK", str(NK)))
    return lambda: k(inputs=[Sm, sk], grid=(TC * H * 32, 1, 1), threadgroup=(tg, 1, 1), output_shapes=[(TC * H, NK)], output_dtypes=[mx.float16])[0]
arms = {"base": fb, "V1_f4": mk(V1), "V2_noids": mk(V2), "V3_oneexp": mk(V3), "V4_fast": mk(V4)}
ref = fb(); mx.eval(ref)
ok = {}
for k, f in arms.items():
    o = f(); mx.eval(o); ok[k] = float(mx.max(mx.abs(o.astype(mx.float32) - ref.astype(mx.float32))).item())
best = {k: 1e9 for k in arms}
for r in range(8):
    time.sleep(1.0)
    for k in list(arms)[r % 5:] + list(arms)[:r % 5]:
        for _ in range(3):
            t0 = time.perf_counter(); mx.eval(arms[k]()); best[k] = min(best[k], time.perf_counter() - t0)
for k in arms: print(f"{k:10s} {best[k]*1e3:6.2f} ms  {(TC*H*NK*6)/best[k]/1e9:5.0f} GB/s  maxdiff {ok[k]:.2e}")
