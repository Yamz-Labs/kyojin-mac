"""Raw NAX GEMM ceiling: Y[M,N] = A[M,K] @ W[N,K]^T, fp16 in, fp32 acc, both operands device tensors, per-simdgroup tiles."""
import sys, time, itertools, numpy as np, mlx.core as mx
from kyojin_mlx import metal_nax as NX
SRC = r"""
    const uint sg = simdgroup_index_in_threadgroup;
    const uint M = params[0], N = params[1], KD = params[2];
    const uint mg = threadgroup_position_in_grid.y * BMU;
    const uint n0 = (threadgroup_position_in_grid.x * NSG + sg) * BNS;
    if (mg >= M || n0 >= N) return;
    constexpr auto desc = matmul2d_descriptor(BMU, BNS, BKS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tA((device half*)A, dextents<int32_t, 2>(int(KD), int(M)));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tB((device half*)W, dextents<int32_t, 2>(int(KD), int(N)));
    auto sA0 = tA.slice(0, int(mg)); auto sB0 = tB.slice(0, int(n0));
    auto cT = op.get_destination_cooperative_tensor<decltype(sA0), decltype(sB0), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;
    for (uint kt = 0; kt < KD; kt += BKS) {
        auto sA = tA.slice(int(kt), int(mg)); auto sB = tB.slice(int(kt), int(n0));
        op.run(sA, sB, cT);
    }
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) { const auto ix = cT.get_multidimensional_index(i);
        const uint m = mg + uint(ix[1]), n = n0 + uint(ix[0]); if (m < M && n < N) Y[ulong(m) * N + n] = half(cT[i]); }
"""
M, N, KD = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
rng = np.random.default_rng(0)
A = mx.array(rng.standard_normal((M, KD)).astype(np.float16)); W = mx.array((rng.standard_normal((N, KD)) * 0.03).astype(np.float16))
mx.eval(A, W)
def tm(f, reps=6):
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); o = f(); mx.eval(o); best = min(best, time.perf_counter() - t0)
    return best, o
fl = 2 * M * N * KD
t, ref = tm(lambda: A @ W.T); print(f"mlx matmul {t*1e3:.2f} ms {fl/t/1e12:.1f} TF")
for bmu, bns, bks, nsg in itertools.product((32, 64), (32, 64), (16, 32, 64), (4,)):
    src = SRC.replace("BMU", str(bmu)).replace("BNS", str(bns)).replace("BKS", str(bks)).replace("NSG", str(nsg))
    try:
        k = mx.fast.metal_kernel(name=f"ng_{bmu}_{bns}_{bks}", header=NX.HEADER_NAX, input_names=["A", "W", "params"], output_names=["Y"], source=src)
        params = mx.array([M, N, KD], dtype=mx.uint32)
        f = lambda: k(inputs=[A, W, params], grid=(-(-N // (bns * nsg)) * 32 * nsg, -(-M // bmu), 1), threadgroup=(32 * nsg, 1, 1), output_shapes=[(M, N)], output_dtypes=[mx.float16])[0]
        mx.eval(f()); t, o = tm(f)
        rel = (mx.linalg.norm(o.astype(mx.float32) - ref.astype(mx.float32)) / mx.linalg.norm(ref.astype(mx.float32))).item()
        print(f"nax tile {bmu}x{bns} bks {bks}: {t*1e3:.2f} ms {fl/t/1e12:.1f} TF rel {rel:.1e}", flush=True)
    except Exception as e:
        print(f"nax tile {bmu}x{bns} bks {bks}: FAIL {str(e)[:150]}", flush=True)
