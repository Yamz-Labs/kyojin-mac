# SPDX-License-Identifier: Apache-2.0
"""decfuse.add_rms_moe == foreign_moe._combine followed by decfuse.add_rms, bit for bit (fp16 stream, saturating)."""
import mlx.core as mx
import mlx.nn as nn
import pytest

from kyojin_mlx import decfuse as DF
from kyojin_mlx import foreign_moe as FM


@pytest.mark.parametrize("T,K,H", [(1, 8, 2048), (4, 8, 2048), (8, 4, 1024), (3, 2, 4096)])
def test_add_rms_moe_matches_combine_then_add_rms(T, K, H):
    mx.random.seed(T * 7 + K)
    x = (mx.random.normal((T, H)) * 3).astype(mx.float16)
    y = (mx.random.normal((T * K, H)) * 2).astype(mx.float16)
    scs = mx.softmax(mx.random.normal((T * K,)), axis=-1)
    sgs = mx.sigmoid(mx.random.normal((T,)))
    sh = (mx.random.normal((T, H))).astype(mx.float16)
    norm = nn.RMSNorm(H, eps=1e-6)
    norm.weight = (mx.random.normal((H,)) * 0.1 + 1).astype(mx.float16)
    p = FM.Pending(y, scs, sgs, sh, T, K, H, (T,))
    h1, n1 = DF.add_rms_moe(x, p, norm)
    h0, n0 = DF.add_rms(x, p.value(), norm)
    mx.eval(h1, n1, h0, n0)
    assert mx.array_equal(h1, h0).item() and mx.array_equal(n1, n0).item()


def test_saturates():
    T, K, H = 1, 2, 256
    x = mx.full((T, H), 60000, mx.float16)
    y = mx.full((T * K, H), 60000, mx.float16)
    scs = mx.ones((T * K,))
    sgs = mx.ones((T,))
    sh = mx.full((T, H), 60000, mx.float16)
    norm = nn.RMSNorm(H)
    norm.weight = mx.ones((H,), mx.float16)
    h, n = DF.add_rms_moe(x, FM.Pending(y, scs, sgs, sh, T, K, H, (T,)), norm)
    mx.eval(h, n)
    assert bool(mx.all(mx.isfinite(h))) and bool(mx.all(mx.isfinite(n)))


def test_gated_norm_scatter_equals_gather():
    import numpy as np

    from kyojin_mlx import gdn_pf
    HV, S = 32, 3
    mx.random.seed(5)
    y = mx.random.normal((1, S, HV, 128)).astype(mx.float16)
    z = mx.random.normal((1, S, HV, 128)).astype(mx.float16)
    w = (mx.random.normal((128,)) * 0.1 + 1).astype(mx.float16)
    perm = np.random.RandomState(1).permutation(HV * 128)
    inv = mx.array(np.argsort(perm).astype(np.uint32))
    ref = mx.take(gdn_pf.gated_norm(y, z, w, 1e-6).reshape(S, -1), mx.array(perm.astype(np.int32)), axis=1)
    got = gdn_pf.gated_norm(y, z, w, 1e-6, inv=inv).reshape(S, -1)
    mx.eval(ref, got)
    assert mx.array_equal(ref, got).item()


def test_proj_ba_matches_separate():
    import mlx.nn as nn

    from kyojin_mlx import gdn_pf

    class G(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj_b, self.in_proj_a = nn.Linear(2048, 32, bias=False), nn.Linear(2048, 32, bias=False)
    g = G()
    g.set_dtype(mx.float16)
    x = mx.random.normal((1, 2, 2048)).astype(mx.float16)
    rb, ra = g.in_proj_b(x), g.in_proj_a(x)
    b, a = gdn_pf.proj_ba(g, x)
    mx.eval(rb, ra, b, a)
    assert mx.array_equal(rb, b).item() and mx.array_equal(ra, a).item()
    assert mx.array_equal(g.in_proj_b(x), rb).item()
