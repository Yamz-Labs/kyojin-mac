"""Probe the matmul2d right-input / destination cooperative layouts for small-M descriptors. usage: rows101_layout.py M,N,K [...]"""
import sys, numpy as np, mlx.core as mx
from kyojin_mlx import metal_nax as NX
SRC = r"""
#define RA mpp::tensor_ops::__tensor_ops_detail::__remove_addrspace_t
    const uint lane = thread_index_in_simdgroup;
    constexpr auto desc = matmul2d_descriptor(MM, NN, KK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tA((device half*)a, dextents<int32_t, 2>(KK, MM));
    auto rB = op.get_right_input_cooperative_tensor<half, half, float>();
    auto cT = op.get_destination_cooperative_tensor<RA<decltype(tA)>, RA<decltype(rB)>, float>();
    for (uint16_t i = 0; i < rB.get_capacity(); ++i) { const auto ix = rB.get_multidimensional_index(i);
        lb[(lane * 64 + i) * 2] = ix[0]; lb[(lane * 64 + i) * 2 + 1] = ix[1]; }
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) { const auto ix = cT.get_multidimensional_index(i);
        lc[(lane * 64 + i) * 3] = ix[0]; lc[(lane * 64 + i) * 3 + 1] = ix[1]; lc[(lane * 64 + i) * 3 + 2] = cT.is_valid_element(i); }
    cap[0] = rB.get_capacity(); cap[1] = cT.get_capacity();
"""
for spec in sys.argv[1:]:
    M, N, K = (int(v) for v in spec.split(","))
    k = mx.fast.metal_kernel(name=f"r101lay{M}_{N}_{K}", header=NX.HEADER_NAX, input_names=["a"], output_names=["lb", "lc", "cap"],
                             source=SRC.replace("MM", str(M)).replace("NN", str(N)).replace("KK", str(K)))
    try:
        lb, lc, cap = k(inputs=[mx.zeros((M, K), mx.float16)], grid=(32, 1, 1), threadgroup=(32, 1, 1),
                        output_shapes=[(32 * 64 * 2,), (32 * 64 * 3,), (2,)], output_dtypes=[mx.int32] * 3, init_value=-1)
        mx.eval(lb)
    except Exception as e:
        print(spec, "FAIL", str(e)[:300]); continue
    cb, cc = np.array(cap)
    lb = np.array(lb).reshape(32, 64, 2)[:, :cb]; lc = np.array(lc).reshape(32, 64, 3)[:, :cc]
    print(f"{spec}: rB cap {cb} cT cap {cc}")
    for ln in (0, 1, 2, 3, 4, 8, 16, 31):
        print(f"  lane {ln} rB " + " ".join(f"{a},{b}" for a, b in lb[ln]))
        print("         cT " + " ".join(f"{a},{b}" + ("" if v else "x") for a, b, v in lc[ln]))
