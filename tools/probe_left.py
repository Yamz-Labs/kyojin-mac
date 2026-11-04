"""Layout of the left-input cooperative tensor of matmul2d(M, N, K): (k, m) index of every element per lane. usage: probe_left.py M N K [trA]"""
import sys, numpy as np, mlx.core as mx
from kyojin_mlx import metal_nax as NX
M, N, K = (int(v) for v in sys.argv[1:4]); trA = sys.argv[4] if len(sys.argv) > 4 else "false"
src = f"""
    const uint lane = thread_index_in_simdgroup;
    constexpr auto desc = matmul2d_descriptor({M}, {N}, {K}, {trA}, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto lA = op.get_left_input_cooperative_tensor<half, half, float>();
    const uint cap = lA.get_capacity();
    for (uint16_t i = 0; i < cap; ++i) {{ const auto ix = lA.get_multidimensional_index(i); out[(lane * 256 + i) * 2] = int(ix[0]); out[(lane * 256 + i) * 2 + 1] = int(ix[1]); }}
    out[16384 + lane] = int(cap);
"""
k = mx.fast.metal_kernel(name="probe_left", header=NX.HEADER_NAX, input_names=["dummy"], output_names=["out"], source=src)
o = np.array(k(inputs=[mx.zeros(1)], grid=(32, 1, 1), threadgroup=(32, 1, 1), output_shapes=[(16384 + 32,)], output_dtypes=[mx.int32])[0])
cap = o[16384]; print("capacity", cap)
for lane in (0, 1, 2, 4, 8, 16, 31):
    pts = [(o[(lane * 256 + i) * 2], o[(lane * 256 + i) * 2 + 1]) for i in range(cap)]
    print("lane", lane, pts[:int(cap)])
