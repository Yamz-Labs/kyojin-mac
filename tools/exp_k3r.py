"""K3 (values) with V rows loaded straight into the right-input cooperative tensor (2-byte gathers). Bit-compare vs shipped K3, interleaved times."""
import time, numpy as np, mlx.core as mx
from kyojin_mlx import attn_prefill as AP, metal_nax as NX
H, D, NK, TC = 64, 512, 640, 512
rng = np.random.default_rng(0)
def rnd(*s, sc=0.5): return mx.array((rng.standard_normal(s) * sc).astype(np.float32)).astype(mx.float16)
tab = rnd(6144, D)
ids_np = np.stack([rng.choice(6144, NK, replace=False) for _ in range(TC)]).astype(np.int32); ids_np[:, 5::37] = -1
ids = mx.array(ids_np)
P = mx.softmax(mx.array(rng.standard_normal((TC * H, NK)).astype(np.float32) * 3), axis=-1).astype(mx.float16)
P = mx.where(mx.repeat(ids >= 0, H, axis=0), P, 0).astype(mx.float16)
R = r"""
    const uint sg = simdgroup_index_in_threadgroup, tid = thread_index_in_threadgroup;
    const uint t = threadgroup_position_in_grid.y, TC = params[0];
    const uint dbase = threadgroup_position_in_grid.x * 128 + sg * 32;
    threadgroup int sid[NK];
    for (uint j = tid; j < NK; j += 128) sid[j] = ids[ulong(t) * NK + j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr auto desc = matmul2d_descriptor(32, 32, 64, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tA((device half*)(P + ulong(t) * HH * NK), dextents<int32_t, 2>(NK, HH));
    auto tA0 = tA.slice(0, 0);
    auto rB = op.get_right_input_cooperative_tensor<half, half, float>();
    using RA = mpp::tensor_ops::__tensor_ops_detail::__remove_addrspace_t<decltype(tA0)>;
    using RB = mpp::tensor_ops::__tensor_ops_detail::__remove_addrspace_t<decltype(rB)>;
DECL
    const auto ix0 = rB.get_multidimensional_index(ushort(0));
    const uint G = uint(ix0[0]) >> 2, nb = uint(ix0[1]);
    for (uint kt = 0; kt < NK; kt += 64) {
        for (uint g = 0; g < 16; ++g) {
            const uint kb = kt + 16 * (g >> 2) + 4 * G, col = dbase + nb + 16 * ((g >> 1) & 1) + 8 * (g & 1);
            for (uint c = 0; c < 4; ++c) {
                const int id = sid[kb + c];
                rB[4 * g + c] = id >= 0 ? tab[ulong(id) * DD + col] : half(0);
            }
        }
RUN
    }
STORE
"""
def src(nmb=2):
    decl = "\n".join(f"    auto cT{m} = op.get_destination_cooperative_tensor<RA, RB, float>();\n    for (uint16_t i = 0; i < cT{m}.get_capacity(); ++i) if (cT{m}.is_valid_element(i)) cT{m}[i] = 0.0f;" for m in range(nmb))
    run = "\n".join(f"        {{ auto sA = tA.slice(int(kt), {32 * m}); op.run(sA, rB, cT{m}); }}" for m in range(nmb))
    store = "\n".join(f"    for (uint16_t i = 0; i < cT{m}.get_capacity(); ++i) if (cT{m}.is_valid_element(i)) {{ const auto ix = cT{m}.get_multidimensional_index(i);\n        O[(ulong({32 * m}u + uint(ix[1])) * TC + t) * DD + dbase + uint(ix[0])] = half(cT{m}[i]); }}" for m in range(nmb))
    return R.replace("DECL", decl).replace("RUN", run).replace("STORE", store).replace("HH", str(H)).replace("DD", str(D)).replace("NK", str(NK))
k_new = mx.fast.metal_kernel(name="exp_k3r", header=NX.HEADER_NAX, input_names=["P", "tab", "ids", "params"], output_names=["O"], source=src())
k_old = AP._kernels(H, D, NK, 0.1)[2]
par = mx.array([TC], dtype=mx.uint32)
mk = lambda k: (lambda: k(inputs=[P, tab, ids, par], grid=(4 * 128, TC, 1), threadgroup=(128, 1, 1), output_shapes=[(H, TC, D)], output_dtypes=[mx.float16])[0])
arms = {"old": mk(k_old), "new": mk(k_new)}
o, n = arms["old"](), arms["new"](); mx.eval(o, n)
print("maxdiff", float(mx.max(mx.abs(o.astype(mx.float32) - n.astype(mx.float32))).item()), "equal", bool(mx.array_equal(o, n).item()))
best = {k: 1e9 for k in arms}
for r in range(8):
    time.sleep(1.0)
    for k in (list(arms) if r % 2 == 0 else list(arms)[::-1]):
        for _ in range(3):
            t0 = time.perf_counter(); mx.eval(arms[k]()); best[k] = min(best[k], time.perf_counter() - t0)
fl = 2 * H * NK * D * TC
for k, v in best.items(): print(f"{k:5s} {v*1e3:.2f} ms {fl/v/1e12:.1f} TF")
