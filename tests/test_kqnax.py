"""G5f: the tile-decode NAX schedule (foreign_kqnax) against the register-fed kqgmm on a small NVFP4 expert stack. Needs the GPU (skipped on CPU-only runs)."""
import os, sys
import numpy as np
import pytest
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

pytestmark = pytest.mark.skipif(mx.default_device() != mx.gpu or not mx.metal.is_available(), reason="needs the Metal GPU")


def test_nax_matches_kqgmm(monkeypatch):
    from g2b_bench import rand_pack
    import kyojin_mlx.foreign_linear as FL, kyojin_mlx.foreign_kqgmm as KQ, kyojin_mlx.foreign_kqnax as NX
    E, K, H, I, T = 12, 4, 1024, 512, 700
    gate, up, down = rand_pack(E, I, H, 1), rand_pack(E, I, H, 2), rand_pack(E, H, I, 3)
    glu = FL.ForeignSwitchGLU(gate, up, down, limit=10.0)
    rng = np.random.default_rng(0)
    x2 = mx.array((rng.standard_normal((T, H)) * 0.005).astype(np.float16))
    e = np.stack([rng.permutation(E)[:K] for _ in range(T)]).reshape(-1).astype(np.uint32)
    e[:200] = 3                                                   # one big run, an empty expert, ragged tiles
    e32 = mx.array(e)
    monkeypatch.setenv("KYOJIN_KQ_GMM_GLM", "1")
    ref = KQ.mlp(glu, x2, e32, K, 32)
    for mp in ("1", "0"):
        monkeypatch.setenv("KYOJIN_KQ_NAX_MAP", mp)
        y = NX.mlp(glu, x2, e32, K)
        r, n = np.array(ref).astype(np.float32), np.array(y).astype(np.float32)
        assert not np.isnan(n).any()
        assert np.sqrt(((r - n) ** 2).mean()) <= 2e-3 * np.sqrt((r ** 2).mean())
        assert np.abs(r - n).max() <= 2e-3 * np.abs(r).max() + 1e-3
