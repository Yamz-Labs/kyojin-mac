"""Single-copy + GPU-sort MoE prefill path (KYOJIN_PF_GLU=1, MoePrefill._chunk_glu) vs the production
fused path: bit-identical output (same arithmetic, same order; only the plan and the activation
copy change). Covers several (E, top-k, rows) including experts with 0 rows."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

from kyojin_mlx import moe as M
from kyojin_mlx import moe_prefill as MP
from test_moe import make_stack, E, TOPK, KS
from test_moe_prefill import route, xs


def _stacks(cb, seed0):
    g, _ = make_stack(256, 384, KS, cb, seed0)
    u, _ = make_stack(256, 384, KS, cb, seed0 + 1)
    d, _ = make_stack(384, 256, KS, cb, seed0 + 2)
    return g, u, d


def _ref_decode(dec, x, ids, w, step=4):
    return np.concatenate([np.array(dec(x[i:i + step], ids[i:i + step], w[i:i + step]))
                           for i in range(0, x.shape[0], step)])


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.parametrize("wpre", [False, True])
@pytest.mark.parametrize("rows", [9, 70])
def test_glu_matches_production(rows, wpre, monkeypatch):
    for cb in ("mul1", "mcg"):
        g, u, d = _stacks(cb, 1)
        dec = M.MoeGather(g, u, d, TOPK, wpre=wpre)
        ids, w = route(rows, 600 + rows)
        x = xs(rows, g.kin, 601)
        monkeypatch.setattr(MP, "GLU", True)
        got = np.array(MP.MoePrefill(dec, fused=True, fused_max=10 ** 9)(x, ids, w))
        monkeypatch.setattr(MP, "GLU", False)
        ref = np.array(MP.MoePrefill(dec, fused=True, fused_max=10 ** 9)(x, ids, w))
        bad = int((got.view(np.uint32) != ref.view(np.uint32)).sum())
        assert bad == 0, (cb, rows, wpre, bad, _rel(got, ref))


def test_glu_matches_decode_path(rows=45, topk=TOPK):
    g, u, d = _stacks("mul1", 2)
    dec = M.MoeGather(g, u, d, TOPK, wpre=True)
    ids, w = route(rows, 610)
    x = xs(rows, g.kin, 611)
    import kyojin_mlx.moe_prefill as MPm
    old = MPm.GLU
    MPm.GLU = True
    try:
        got = np.array(MP.MoePrefill(dec, fused=True, fused_max=10 ** 9)(x, ids, w))
    finally:
        MPm.GLU = old
    assert _rel(got, _ref_decode(dec, x, ids, w)) < 3e-3


@pytest.mark.parametrize("topk,ks", [(1, [4] * 12), (4, [3, 4, 5, 6, 4, 3, 5, 6, 4, 4, 3, 6]),
                                     (8, [2, 5, 3, 4, 2, 2, 3, 5, 4, 3, 2, 6])])
def test_glu_any_topk_any_k(topk, ks, monkeypatch):
    g, _ = make_stack(256, 384, ks, "mul1", 61)
    u, _ = make_stack(256, 384, ks, "mul1", 62)
    d, _ = make_stack(384, 256, ks, "mul1", 63)
    dec = M.MoeGather(g, u, d, topk)
    rng = np.random.default_rng(64)
    ids = mx.array(np.stack([rng.choice(len(ks), topk, replace=False) for _ in range(33)]).astype(np.int32))
    w = mx.array(rng.uniform(0.1, 1.0, (33, topk)).astype(np.float32))
    x = xs(33, 256, 65)
    monkeypatch.setattr(MP, "GLU", True)
    got = np.array(MP.MoePrefill(dec, fused=True, fused_max=10 ** 9)(x, ids, w))
    monkeypatch.setattr(MP, "GLU", False)
    ref = np.array(MP.MoePrefill(dec, fused=True, fused_max=10 ** 9)(x, ids, w))
    assert np.array_equal(got.view(np.uint32), ref.view(np.uint32)), (topk, ks)
