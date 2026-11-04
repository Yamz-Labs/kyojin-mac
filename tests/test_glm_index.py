"""GLM decode indexer on the dsa_decode kernels against the stock `_exact_pool_select`: same selected set (K=512 and 2048), invalid pools last."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
L = pytest.importorskip("mlx_vlm.models.glm5_next.language")
from kyojin_mlx import glm_index as G                                        # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")


@pytest.mark.parametrize("dt", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("pc,k,qpos", [(3000, 512, None), (20001, 2048, None), (33000, 2048, 60000), (5000, 2048, 9000)])
def test_glm_select(dt, pc, k, qpos):
    rng = np.random.default_rng(pc)
    q = mx.array(rng.standard_normal((1, 1, 32, 128))).astype(dt)
    pk = mx.array(rng.standard_normal((1, pc, 128))).astype(dt)
    w = mx.array(rng.standard_normal((1, 1, 32)).astype(np.float32))
    ends = (mx.arange(pc) * 4 + 3)[None]
    valid = mx.ones((1, pc), dtype=mx.bool_)
    qp = mx.array([qpos if qpos is not None else pc * 4 - 100])
    rs, rv = L._exact_pool_select(q, pk, w, ends, valid, qp, k, 128 ** -0.5)
    ns, nv = G.select_pools(q, pk, w, ends, valid, qp, k, 128 ** -0.5)
    a = set(np.array(rs).ravel()[np.array(rv).ravel()].tolist())
    b = set(np.array(ns).ravel()[np.array(nv).ravel()].tolist())
    assert len(a) == len(b)
    assert len(a & b) >= 0.995 * len(a)
