"""G10d: the fused MLA decode glue of the DSA layers (kyojin_mlx.glm_mla) against the stock embed_q / SDPA / unembed_out on random data of the real shapes."""
import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
L = pytest.importorskip("mlx_vlm.models.glm5_next.language")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
os.environ["KYOJIN_GLM_MLA_FUSE"] = "1"
from kyojin_mlx import glm_mla as M    # noqa: E402
from mlx_vlm.models.mla import MultiLinear    # noqa: E402


def _ml(i, o, dt, seed):
    mx.random.seed(seed)
    m = MultiLinear(i, o, 64)
    m.weight = (mx.random.normal(m.weight.shape) / i ** 0.5).astype(dt)
    mx.eval(m.weight)
    return m


@pytest.mark.parametrize("dt", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("B,Lr", [(1, 1), (1, 3), (2, 2), (4, 4)])
def test_proj_matches_stock_and_rows_invariant(dt, B, Lr):
    eq, uo = _ml(256, 512, dt, 1), _ml(512, 256, dt, 2)
    mx.random.seed(3)
    for m, K in ((eq, 256), (uo, 512)):
        x = mx.random.normal((B, 64, Lr, K)).astype(dt)
        ref = (x.astype(mx.float32) @ m.weight.astype(mx.float32).swapaxes(-1, -2))
        got = M.proj(x, m.weight)
        mx.eval(ref, got)
        err = float(mx.max(mx.abs(got.astype(mx.float32) - ref)))
        assert err < (0.02 if dt == mx.float16 else 0.1) * float(mx.max(mx.abs(ref))), err
        for b in range(B):                                       # a row alone == the row in the batch, bit for bit
            for j in range(Lr):
                one = M.proj(x[b:b + 1, :, j:j + 1], m.weight)
                assert bool(mx.array_equal(one, got[b:b + 1, :, j:j + 1]))


@pytest.mark.parametrize("dt", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("B,NQ,K", [(1, 1, 2051), (1, 2, 2051), (2, 2, 2051), (3, 1, 2051), (2, 4, 700)])
def test_attend_matches_stock(dt, B, NQ, K):
    eq, uo = _ml(256, 512, dt, 1), _ml(512, 256, dt, 2)
    mx.random.seed(5)
    CAP, nt = 3000, 2900
    lat = (mx.random.normal((B, 1, CAP, 512)) * 0.5).astype(dt)
    q = mx.random.normal((B, 64, NQ, 256)).astype(dt)
    tk = []
    for b in range(B):
        for j in range(NQ):
            idx = np.random.RandomState(b * 10 + j).permutation(nt)[:K].astype(np.int32)
            idx[-3:] = -1 if j == 0 else idx[-3:]
            tk.append(idx)
    topk = mx.array(np.stack(tk).reshape(B, NQ, K))
    mx.eval(lat, q, topk)
    scale = 256 ** -0.5
    rows, vm = M.gather_pad(lat, topk.reshape(-1), B, NQ, K, nt)
    got = M.attend(M.proj(q, eq.weight), rows, vm, uo.weight, B, NQ, K, scale)
    outs = []
    for j in range(NQ):
        per = []
        for b in range(B):
            sel_i = topk[b, j]
            ok = (sel_i >= 0) & (sel_i < nt)
            sel = mx.where(ok[:, None], lat[b, 0][mx.clip(sel_i, 0, CAP - 1)], 0)[None, None]
            query = eq(q[b:b + 1, :, j:j + 1])
            per.append(L.scaled_dot_product_attention(query, sel, sel, cache=None, scale=scale, mask=ok[None, None, None, :]))
        outs.append(uo(mx.concatenate(per, axis=0)))
    ref = mx.concatenate(outs, axis=2).transpose(0, 2, 1, 3).reshape(B, NQ, -1)
    mx.eval(got, ref)
    r32 = ref.astype(mx.float32)
    err = float(mx.max(mx.abs(got.astype(mx.float32) - r32)))
    assert err < (0.01 if dt == mx.float16 else 0.02) * float(mx.max(mx.abs(r32))) + 1e-3, (err, float(mx.max(mx.abs(r32))))
    # row invariance: one (session, row) alone gives the same bits as inside the batch
    for b in range(B):
        for j in range(NQ):
            r1, v1 = M.gather_pad(lat[b:b + 1], topk[b:b + 1, j:j + 1].reshape(-1), 1, 1, K, nt)
            one = M.attend(M.proj(q[b:b + 1, :, j:j + 1], eq.weight), r1, v1, uo.weight, 1, 1, K, scale)
            assert bool(mx.array_equal(one[0, 0], got[b, j])), (b, j)
