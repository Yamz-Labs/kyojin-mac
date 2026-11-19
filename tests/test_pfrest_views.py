"""P3-PFREST: the reshape-view seams (KYOJIN_HC_MIX_VIEW, KYOJIN_DSA_INDEXER_PF_VIEW, KYOJIN_DSA_PF_VIEW) give the same arrays as the `a[0]` gathers they replace."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("needs the GPU", allow_module_level=True)

from kyojin_mlx import glm_dsa_fused as GF, glm_pf_idx as PI  # noqa: E402


def test_x0_is_a_gather_and_reshape_is_a_view():
    x = mx.zeros((1, 64, 4, 64), dtype=mx.float16)
    import io, tempfile, os
    def prims(y):
        f = tempfile.NamedTemporaryFile("w+", suffix=".dot", delete=False)
        mx.export_to_dot(f, y); f.close()
        t = open(f.name).read(); os.unlink(f.name)
        return t
    assert "Gather" in prims(x[0].reshape(64, 256)) and "Gather" not in prims(x.reshape(64, 256))


@pytest.mark.parametrize("T,pc,k", [(128, 700, 512), (1000, 2100, 512)])
def test_pf_idx_view_identical(T, pc, k, monkeypatch):
    monkeypatch.setattr(PI, "ENABLED", True)
    rng = np.random.default_rng(T + pc)
    q = mx.array(rng.standard_normal((1, T, 32, 128)).astype(np.float16))
    keys = mx.array(rng.standard_normal((1, pc, 128)).astype(np.float16))
    w = mx.array(np.abs(rng.standard_normal((1, T, 32))).astype(np.float32))
    pool_ends = (mx.arange(pc, dtype=mx.int32) * 4 + 3)[None]
    pool_valid = mx.ones((1, pc), dtype=mx.bool_)
    qpos = (pc * 4) + mx.arange(T)
    outs = []
    for v in (False, True):
        monkeypatch.setattr(PI, "VIEW", v)
        a, b = PI.select(q, keys, w, pool_ends, pool_valid, qpos, k, 0.0883883)
        mx.eval(a, b)
        outs.append((a, b))
    assert mx.array_equal(outs[0][0], outs[1][0]).item() and mx.array_equal(outs[0][1], outs[1][1]).item()


@pytest.mark.parametrize("T,N,W", [(256, 1024, 64), (300, 2000, 96)])
def test_dsa_fused_view_identical(T, N, W, monkeypatch):
    rng = np.random.default_rng(T)
    q = mx.array((rng.standard_normal((1, 64, T, 512)) * 0.1).astype(np.float16))
    lat = mx.array(rng.standard_normal((1, 1, N, 512)).astype(np.float16))
    ids = mx.array(rng.integers(0, N, (1, T, W)).astype(np.int32))
    monkeypatch.setattr(GF, "ENABLED", True)
    if not GF.supported(q, lat, ids):
        pytest.skip("fused DSA kernel unsupported here")
    outs = []
    for v in (False, True):
        monkeypatch.setattr(GF, "VIEW", v)
        o = GF.latent_attention(q, lat, ids, 0.07)
        mx.eval(o)
        outs.append(o)
    assert mx.array_equal(outs[0], outs[1]).item()
