"""Expert GEMM at low M (<= 64 rows per expert): the register path (metal_nax.expert_gemm) is the default, the decode-once threadgroup kernel
(metal_nax_tg) is opt-in (KYOJIN_NAX_TG); both give the same bits. Needs a GPU with NAX (M5)."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

from kyojin_mlx import metal_nax as NX, metal_nax_tg as TG
if not NX.nax_available():
    pytest.skip("no neural accelerators on this GPU", allow_module_level=True)
from test_moe import make_stack, KS


@pytest.mark.parametrize("cb", ["mul1", "mcg"])
@pytest.mark.parametrize("M", [8, 33, 64])
def test_lowm_register_path_is_default_and_matches_decode_once(cb, M, monkeypatch):
    g, _ = make_stack(256, 384, KS, cb, 1)
    u, _ = make_stack(256, 384, KS, cb, 2)
    G = g.experts
    eids = mx.array(np.arange(G, dtype=np.int32))
    counts = np.full(G, M)
    counts[0] = max(1, M // 3)
    cl = np.concatenate([[0], np.cumsum(counts)]).astype(np.uint32)
    n = int(counts.sum())
    rowoff = mx.array(np.concatenate([cl[:-1], n + cl]).astype(np.uint32))
    A = mx.array(NX.permute_k(np.random.default_rng(M).standard_normal((2 * n, 256)).astype(np.float16)))
    ref = np.array(TG.expert_gemm_tg((g, u), eids, rowoff, A, 2 * n, M))
    monkeypatch.setattr(NX, "TG_MIN_ROWS", 0)
    monkeypatch.setattr(TG, "expert_gemm_tg", lambda *a, **k: pytest.fail("decode-once kernel taken by default"))
    y = np.array(NX.expert_gemm((g, u), eids, rowoff, A, 2 * n, M, (32, 2, 32, 64)))
    assert np.array_equal(y, ref), (cb, M)
