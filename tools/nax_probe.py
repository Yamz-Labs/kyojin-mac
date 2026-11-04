"""NAX matmul2d throughput probe (Air M5): time per op.run for different M tiles / A sources / TG counts; B is a register tensor (like the fused trellis kernel)."""
import argparse, time, statistics, numpy as np, mlx.core as mx
from kyojin_mlx import metal_nax as NX
ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=40); ap.add_argument("--tgs", default="72,144,288,576"); ap.add_argument("--bmus", default="16,32"); ap.add_argument("--nk", default="32:64"); a = ap.parse_args()
SRC = r"""
#define RA mpp::tensor_ops::__tensor_ops_detail::__remove_addrspace_t
    const uint sg = simdgroup_index_in_threadgroup;
    constexpr auto desc = matmul2d_descriptor(BMU, BNN, BKK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tA((device half*)A, dextents<int32_t, 2>(2560, 32));
    auto rB = op.get_right_input_cooperative_tensor<half, half, float>();
    auto cT0 = op.get_destination_cooperative_tensor<RA<decltype(tA)>, RA<decltype(rB)>, float>();
    for (uint16_t i = 0; i < cT0.get_capacity(); ++i) if (cT0.is_valid_element(i)) cT0[i] = 0.0f;
    for (uint16_t i = 0; i < rB.get_capacity(); ++i) rB[i] = half(float(i + sg) * 0.001f);
    for (uint kt = 0; kt < STEPS; ++kt) {
        auto sA = tA.slice(int(kt) * BKK, 0);
        op.run(sA, rB, cT0);
    }
    float s = 0;
    for (uint16_t i = 0; i < cT0.get_capacity(); ++i) if (cT0.is_valid_element(i)) s += cT0[i];
    Y[threadgroup_position_in_grid.x * 128 + thread_position_in_threadgroup.x] = s;
"""
A = mx.array(np.random.default_rng(1).standard_normal((32, 2560)).astype(np.float16)); mx.eval(A)
for bmu in [int(v) for v in a.bmus.split(",")]:
    k = mx.fast.metal_kernel(name=f"naxp_{bmu}_{a.steps}_{a.nk.replace(':','x')}", header=NX.HEADER_NAX, input_names=["A"], output_names=["Y"], source=SRC.replace("BMU", str(bmu)).replace("BNN", a.nk.split(":")[0]).replace("BKK", a.nk.split(":")[1]).replace("STEPS", str(a.steps)))
    for tgs in [int(v) for v in a.tgs.split(",")]:
        f = lambda: k(inputs=[A], grid=(tgs * 128, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(tgs * 128,)], output_dtypes=[mx.float32])[0]
        mx.eval(f()); ts = []
        for r in range(25):
            t0 = time.perf_counter(); mx.eval([f() for _ in range(8)]); ts.append((time.perf_counter() - t0) / 8 * 1e6)
        BN, BK = (int(v) for v in a.nk.split(':')); n_ops = tgs * 4 * a.steps
        print(f"bmu {bmu} tgs {tgs}: {min(ts):7.1f} us  ({n_ops} ops of {bmu}x32x64 -> {min(ts)*1e3/n_ops*10:.0f} ns/op per core-equivalent; {2*bmu*BN*BK*n_ops/min(ts)/1e6:.2f} TFLOPs)", flush=True)
