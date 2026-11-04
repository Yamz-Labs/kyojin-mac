"""K1 with the keys loaded straight into the right-input cooperative tensor (no threadgroup staging, no barriers). Bit-compare vs shipped K1 + interleaved times."""
import time, numpy as np, mlx.core as mx
from kyojin_mlx import attn_prefill as AP, metal_nax as NX
H, D, NK, TC, T = 64, 512, 640, 512, 2048
rng = np.random.default_rng(0)
def rnd(*s, sc=0.5): return mx.array((rng.standard_normal(s) * sc).astype(np.float32)).astype(mx.float16)
q = rnd(H, T, D); tab = rnd(6144, D)
ids_np = np.stack([rng.choice(6144, NK, replace=False) for _ in range(T)]).astype(np.int32); ids_np[:, 5::37] = -1
ids = mx.array(ids_np)
R = r"""
    const uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    const uint t = threadgroup_position_in_grid.y;
    const uint nbase = threadgroup_position_in_grid.x * (NSGT * 16 * NTT) + sg * 16 * NTT;
    const int qrow = int(params[1]) * DD;
    constexpr auto desc = matmul2d_descriptor(32, 16 * NTT, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tA((device half*)(Q + ulong(params[0] + t) * DD), dextents<int32_t, 2>(DD, HH), array<int32_t, 2>{1, qrow});
    auto tA0 = tA.slice(0, 0);
    auto rB = op.get_right_input_cooperative_tensor<half, half, float>();
    constexpr auto RA = 0;
    auto cT0 = op.get_destination_cooperative_tensor<mpp::tensor_ops::__tensor_ops_detail::__remove_addrspace_t<decltype(tA0)>, mpp::tensor_ops::__tensor_ops_detail::__remove_addrspace_t<decltype(rB)>, float>();
    auto cT1 = op.get_destination_cooperative_tensor<mpp::tensor_ops::__tensor_ops_detail::__remove_addrspace_t<decltype(tA0)>, mpp::tensor_ops::__tensor_ops_detail::__remove_addrspace_t<decltype(rB)>, float>();
    for (uint16_t i = 0; i < cT0.get_capacity(); ++i) if (cT0.is_valid_element(i)) cT0[i] = 0.0f;
    for (uint16_t i = 0; i < cT1.get_capacity(); ++i) if (cT1.is_valid_element(i)) cT1[i] = 0.0f;
    const auto ix0 = rB.get_multidimensional_index(ushort(0));
    const uint G = uint(ix0[0]) >> 2, nb = uint(ix0[1]);
    int kid[2 * NTT];
    for (uint s = 0; s < 2 * NTT; ++s) { const uint n = nbase + nb + 16 * (s >> 1) + 8 * (s & 1); kid[s] = n < NK ? ids[ulong(t) * NK + n] : -1; }
    for (uint kt = 0; kt < DD; kt += 64) {
        for (uint g = 0; g < 8 * NTT; ++g) {
            const uint kk2 = g / (2 * NTT), s = g % (2 * NTT);
            const int id = kid[s];
            const uint2 v = id >= 0 ? *((const device uint2*)(tab + ulong(id) * DD + kt + 16 * kk2 + 4 * G)) : uint2(0);
            rB[4 * g + 0] = as_type<half2>(v.x).x; rB[4 * g + 1] = as_type<half2>(v.x).y; rB[4 * g + 2] = as_type<half2>(v.y).x; rB[4 * g + 3] = as_type<half2>(v.y).y;
        }
        auto sA = tA.slice(int(kt), 0); op.run(sA, rB, cT0);
        auto sB = tA.slice(int(kt), 32); op.run(sB, rB, cT1);
    }
    for (uint16_t i = 0; i < cT0.get_capacity(); ++i) if (cT0.is_valid_element(i)) {
        const auto ix = cT0.get_multidimensional_index(i);
        const uint n = nbase + uint(ix[0]);
        if (n < NK) S[(ulong(t) * HH + uint(ix[1])) * NK + n] = cT0[i] * SCALE;
    }
    for (uint16_t i = 0; i < cT1.get_capacity(); ++i) if (cT1.is_valid_element(i)) {
        const auto ix = cT1.get_multidimensional_index(i);
        const uint n = nbase + uint(ix[0]);
        if (n < NK) S[(ulong(t) * HH + 32 + uint(ix[1])) * NK + n] = cT1[i] * SCALE;
    }
"""

sub = lambda s, nt, nsg=4: s.replace("NSGT", str(nsg)).replace("NTT", str(nt)).replace("HH", str(H)).replace("DD", str(D)).replace("NK", str(NK)).replace("SCALE", repr(D ** -0.5) + "f")
a = 512; par = mx.array([a, T], dtype=mx.uint32); ia = ids[a:a + TC]
def mk(nt, tg=128):
    k = mx.fast.metal_kernel(name=f"exp_k1r_nt{nt}_{tg}", header=NX.HEADER_NAX, input_names=["Q", "tab", "ids", "params"], output_names=["S"], source=sub(R, nt, tg // 32))
    nk = 16 * nt * (tg // 32)
    return lambda: k(inputs=[q, tab, ia, par], grid=(-(-NK // nk) * tg, TC, 1), threadgroup=(tg, 1, 1), output_shapes=[(TC * H, NK)], output_dtypes=[mx.float32])[0]
k_old = AP._kernels(H, D, NK, D ** -0.5)[0]
arms = {"old": lambda: k_old(inputs=[q, tab, ia, par], grid=(5 * 128, TC, 1), threadgroup=(128, 1, 1), output_shapes=[(TC * H, NK)], output_dtypes=[mx.float32])[0]}
for nt in (1, 2, 3, 4): arms[f"nt{nt}"] = mk(nt)
arms["nt2_tg32"] = mk(2, 32); arms["nt2_tg64"] = mk(2, 64); arms["nt2_tg256"] = mk(2, 256); arms["nt1_tg64"] = mk(1, 64); arms["nt1_tg256"] = mk(1, 256)
valid = mx.repeat(ia >= 0, H, axis=0); ref = arms["old"](); mx.eval(ref)
for k, f in list(arms.items()):
    try: o = f(); mx.eval(o); print(k, "maxdiff", float(mx.max(mx.abs(mx.where(valid, o - ref, 0))).item()), flush=True)
    except Exception as e: print(k, "FAIL", str(e)[-200:]); del arms[k]
best = {k: 1e9 for k in arms}
for r in range(8):
    time.sleep(1.0)
    ks = list(arms); ks = ks[r % len(ks):] + ks[:r % len(ks)]
    for k in ks:
        for _ in range(3):
            t0 = time.perf_counter(); mx.eval(arms[k]()); best[k] = min(best[k], time.perf_counter() - t0)
fl = 2 * H * NK * D * TC
for k, v in best.items(): print(f"{k:10s} {v*1e3:.2f} ms {fl/v/1e12:.1f} TF")
