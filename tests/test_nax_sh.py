"""metal_nax_sh (decode once per threadgroup) == metal_nax.expert_gemm (decode per simdgroup), bit for bit."""
import numpy as np
import pytest
import mlx.core as mx
from kyojin_mlx import exl3_linear as EL, kernels as KN, metal_nax as NX, metal_nax_sh as SH

pytestmark = pytest.mark.skipif(not NX.nax_available(), reason="needs the M5 neural accelerators")


@pytest.mark.parametrize("K,cb", [(2, "mul1"), (3, "mul1"), (4, "mcg"), (6, "mul1")])
@pytest.mark.parametrize("rows", [17, 64, 200, 512])
def test_sh_matches_coop(K, cb, rows):
    rng = np.random.default_rng(K)
    kin, nout = 1024, 256
    tr = mx.array(rng.integers(-32768, 32768, (kin // 16, nout // 16, 16 * K), dtype=np.int16))
    f16 = lambda n: mx.array((rng.choice([-1.0, 1.0], n) * rng.uniform(0.5, 1.5, n)).astype(np.float16))
    lin = EL.Exl3Linear(tr, f16(kin), f16(nout), cb=cb)
    st = NX.SingleStack(lin._words, K, lin._kt, lin._nt, cb)
    xh = KN.rot((mx.random.normal((rows, kin)) * 0.5).astype(mx.float16), lin._suh, mx.float16, perm=True)
    a = NX.expert_gemm((st,), mx.array([0], dtype=mx.int32), mx.array([0, rows], dtype=mx.uint32), xh, rows, rows, (32, 2, 32, 64))
    for nsg, nq in ((4, 2), (8, 1)):
        b = SH.dense_sh(st, xh, rows, (64, 1), True, "", nq, False, nsg)
        assert mx.array_equal(a, b).item()
