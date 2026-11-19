"""BP-TAIL: `KYOJIN_GLM_IDX_SCALE_FUSE` -- drop the no-op `.astype(mx.float32)` on `weights` in
`glm_dsa.call` (one row) and `glm_fuse_dsa.indexer_batch` (rows/batch) before multiplying by
`self.softmax_scale`. `_project_queries` (stock mlx_vlm) already returns `weights` as float32,
scaled by `n_heads**-0.5` -- the extra `.astype(mx.float32)` here was a no-op cast on an
already-float32 array (value-identical either way; dropping it removes one graph node per call,
found while tracing BP-IDX's per-layer DSA indexer launch count). Flag off by default.
"""
import json
import os

import pytest

mx = pytest.importorskip("mlx.core")
L = pytest.importorskip("mlx_vlm.models.glm5_next.language")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
os.environ["KYOJIN_GLM_DSA_FAST"] = "1"
os.environ["KYOJIN_GLM_FUSE_DSA"] = "1"
from kyojin_mlx import glm_dsa as G, glm_fuse_dsa as F    # noqa: E402

CFG = os.path.expanduser("/path/to/data")
DT = mx.bfloat16


def _cfg():
    from mlx_vlm.models.glm5_next.config import TextConfig
    if not os.path.exists(CFG):
        pytest.skip("no GLM config on this machine")
    return TextConfig.from_dict(json.load(open(CFG))["text_config"])


def _rand(m, seed):
    from mlx.utils import tree_flatten, tree_unflatten
    mx.random.seed(seed)
    new = []
    for k, v in tree_flatten(m.parameters()):
        if v.ndim >= 2:
            new.append((k, (mx.random.normal(v.shape) * (1.0 / v.shape[-1] ** 0.5)).astype(mx.float32 if "compress" in k else DT)))
        else:
            new.append((k, mx.zeros(v.shape, DT) if "bias" in k else mx.ones(v.shape, DT)))
    m.update(tree_unflatten(new))
    mx.eval(m.parameters())
    m.eval()
    return m


def _one_row_caches(cfg, n, seed=2):
    from mlx_vlm.models.cache import KVCache, PoolingCache
    mx.random.seed(seed)
    idx = KVCache()
    idx.update_and_fetch(mx.ones((1, 1, n, 1), dtype=mx.bool_), mx.zeros((1, 1, n, 0), dtype=mx.bool_))
    pc = PoolingCache(cfg.index_kpool)
    pc.pooled = mx.random.normal((1, n // 4, cfg.index_head_dim)).astype(DT)
    pc.buf_kv = mx.zeros((1, 4, cfg.index_head_dim), dtype=DT)
    pc.buf_gate = mx.zeros((1, 4, cfg.index_head_dim), dtype=DT)
    pc.remainder = 0
    return idx, pc


def test_flag_off_is_the_exact_pre_existing_branch():
    """The flag only selects between two literal source branches already in the diff; off runs the
    original (pre-BP-TAIL) line unconditionally -- checked here as a static-source invariant."""
    import inspect
    src_g = inspect.getsource(G.install)
    src_f = inspect.getsource(F.indexer_batch)
    assert "weights[0, 0].astype(mx.float32) * self.softmax_scale" in src_g
    assert "weights.reshape(B * Lr, 32).astype(mx.float32) * self.softmax_scale" in src_f


@pytest.mark.parametrize("fuse", [False, True])
def test_glm_dsa_one_row_bit_identical_scale_fuse(fuse):
    cfg = _cfg()
    ix = _rand(L.Glm5NextIndexer(cfg, 3), 0)
    assert G.install() and F.install()
    idx, pc = _one_row_caches(cfg, 4096)
    mx.random.seed(11)
    x = (mx.random.normal((1, 1, cfg.hidden_size)) * 0.5).astype(DT)
    qr = mx.random.normal((1, 1, cfg.q_lora_rank)).astype(DT)
    off = idx.offset
    G.IDX_SCALE_FUSE = False
    ref = L.Glm5NextIndexer.__call__(ix, x, qr, None, idx, pc, off)
    mx.eval(ref)
    idx2, pc2 = _one_row_caches(cfg, 4096)
    G.IDX_SCALE_FUSE = fuse
    got = L.Glm5NextIndexer.__call__(ix, x, qr, None, idx2, pc2, idx2.offset)
    mx.eval(got)
    G.IDX_SCALE_FUSE = False
    assert mx.array_equal(ref, got)


def _batched_batch(ctxs, seed0=7):
    from kyojin_mlx import rows as R
    from mlx_vlm.models.cache import CacheList, KVCache, PoolingCache
    cfg = _cfg()
    users = []
    for i, n in enumerate(ctxs):
        mx.random.seed(seed0 + i)
        lat = KVCache()
        idx = KVCache()
        done = 0
        while done < n:
            m = min(2048, n - done)
            lat.update_and_fetch(mx.random.normal((1, 1, m, 512)).astype(DT), mx.zeros((1, 1, m, 0), dtype=DT))
            idx.update_and_fetch(mx.ones((1, 1, m, 1), dtype=mx.bool_), mx.zeros((1, 1, m, 0), dtype=mx.bool_))
            done += m
        pc = PoolingCache(cfg.index_kpool)
        pc.pooled = mx.random.normal((1, n // 4, cfg.index_head_dim)).astype(DT)
        pc.buf_kv = mx.zeros((1, 4, cfg.index_head_dim), dtype=DT)
        pc.buf_gate = mx.zeros((1, 4, cfg.index_head_dim), dtype=DT)
        pc.remainder = 0
        users.append([CacheList(lat, idx, pc, KVCache())])
    return R.batch(users)[0]


@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("Lr", [1, 2])
def test_glm_fuse_dsa_batch_bit_identical_scale_fuse(fuse, Lr):
    cfg = _cfg()
    ix = _rand(L.Glm5NextIndexer(cfg, 3), 0)
    assert G.install() and F.install()
    ctxs = [4096, 4000]
    A = _batched_batch(ctxs)
    idxA, pcA = A.caches[1], A.caches[2]
    mx.random.seed(21)
    x = (mx.random.normal((len(ctxs), Lr, cfg.hidden_size)) * 0.5).astype(DT)
    qr = mx.random.normal((len(ctxs), Lr, cfg.q_lora_rank)).astype(DT)
    F.IDX_SCALE_FUSE = False
    ref = L.Glm5NextIndexer.__call__(ix, x, qr, None, idxA, pcA, A.caches[0].offset)
    mx.eval(ref)
    Bc = _batched_batch(ctxs)
    idxB, pcB = Bc.caches[1], Bc.caches[2]
    F.IDX_SCALE_FUSE = fuse
    got = L.Glm5NextIndexer.__call__(ix, x, qr, None, idxB, pcB, Bc.caches[0].offset)
    mx.eval(got)
    F.IDX_SCALE_FUSE = False
    assert mx.array_equal(ref, got)
