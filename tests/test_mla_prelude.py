# SPDX-License-Identifier: Apache-2.0
"""mla_prelude: the one-launch row preparation (kv_a norm + k_pe rope + q split / rope + [latent | k_pe | pool value | pool gate] concat with the saturating cast) equals the stock
project() + concat + clamp / cast, fp32 and fp16 rows, 1..16 rows, any offset, with and without the pool rows. Metal only."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
import mlx.nn as nn  # noqa: E402
from kyojin_mlx import attn_fp16 as AF  # noqa: E402
from kyojin_mlx import foreign_skeleton_mla as MLA  # noqa: E402
from kyojin_mlx import mla_prelude as MP  # noqa: E402


def _model(rng, kp):
    kpool = dict(H=2, dx=16, kp=4, top_k=2048, select_tail=True, n_rot=8, base=10000.0) if kp else None
    m = MLA.MlaAttention(64, 4, 48, 128, 32, 16, 16, sinks=True, gate=True, kpool=kpool)
    flat = {}
    for k, a in nn.utils.tree_flatten(m.parameters()):
        s = rng.standard_normal(a.shape).astype(np.float32)
        flat[k] = mx.array(1.0 + 0.1 * s if ("norm" in k and a.ndim == 1) else 0.2 * s)
    m.update(nn.utils.tree_unflatten(list(flat.items())))
    return m


@pytest.mark.parametrize("kp", [False, True])
@pytest.mark.parametrize("buf_dt", [mx.float32, mx.float16])
@pytest.mark.parametrize("L", [1, 3, 6, 16])
@pytest.mark.parametrize("off", [0, 37, 2000])
def test_prelude_equals_stock(kp, buf_dt, L, off):
    rng = np.random.default_rng(3)
    m = _model(rng, kp)
    x = mx.array(rng.standard_normal((1, L, 64)).astype(np.float32))
    o = mx.array([off], dtype=mx.int32)
    qr, q_nope, q_pe, lat, k_pe = m.project(x, o.reshape(()), True, False)
    parts = [lat, k_pe]
    val = gate = None
    if kp:
        val, gate = m.kpool.rows(x)
        parts += [val[:, None], gate[:, None]]
    ref = mx.concatenate(parts, axis=-1)
    ref = AF.sat(ref) if buf_dt == mx.float16 else ref
    qq, kva = m.q_b_proj(qr), m.kv_a_proj(x)
    assert MP.eligible(L, kva, qq, buf_dt, m.kv, m.r, m.nope)
    row, qn, qp = MP.prelude(kva, qq, m.kv_a_norm.weight, m.kv_a_norm.eps, m.base, o, m.nh, m.kv, m.r, m.nope, buf_dt, val, gate)
    mx.eval(ref, q_nope, q_pe, row, qn, qp)
    assert row.shape == ref.shape and qn.shape == q_nope.shape and qp.shape == q_pe.shape
    tol = 3e-3 if buf_dt == mx.float16 else 2e-5
    for a, b, nm in ((row, ref, "row"), (qn, q_nope, "q_nope"), (qp, q_pe, "q_pe")):
        a, b = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
        mag = max(1.0, float(np.abs(b).max()))
        assert np.abs(a - b).max() <= tol * mag * (50 if nm == "q_pe" and off >= 2000 else 1), (nm, float(np.abs(a - b).max()), mag)
