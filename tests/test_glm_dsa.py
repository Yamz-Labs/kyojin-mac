"""Fused GLM DSA decode (kyojin_mlx.glm_dsa): select + expand against dsa_decode.select, the scoring split against dsa_decode.scores, the one-launch gather
against the stock LatentTQCache.gather (and an exact fp64 inverse rotation), the whole indexer against the stock Glm5NextIndexer."""
import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
L = pytest.importorskip("mlx_vlm.models.glm5_next.language")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
os.environ["KYOJIN_GLM_DSA_FAST"] = "1"
from kyojin_mlx import glm_dsa as G, dsa_decode as D, latent_tq as LT, tq    # noqa: E402


@pytest.mark.parametrize("P", [513, 2048, 4099, 20000, 32768])
@pytest.mark.parametrize("mode", ["rand", "ties", "relu"])
def test_select_expand(P, mode):
    mx.random.seed(P)
    s = mx.random.normal((P,))
    if mode == "ties":
        s = mx.round(s * 4) / 4
    if mode == "relu":
        s = mx.maximum(s, 0.0)
    s = mx.concatenate([s, mx.zeros((-(-P // 64) * 64 - P,))])
    n = mx.array([P], dtype=mx.int32)
    for tail in (0, 1, 3):
        ref = G.expand(D.select(s, n, P, 512), 4, 4 * P, tail)
        got = G.select_expand(s, n, P, 512, 4, 4 * P, tail)
        assert mx.array_equal(ref, got)
    r = np.array(ref)
    assert (r[:2048].reshape(512, 4)[:, 0] % 4 == 0).all() and (np.diff(r[:2048:4]) > 0).all()
    assert r[2048:].tolist() == [4 * P + j if j < 3 else -1 for j in range(3)]


@pytest.mark.parametrize("P", [2048, 5000])
def test_scores_split(P):
    mx.random.seed(1)
    q = mx.random.normal((32, 128)).astype(mx.bfloat16)
    k = mx.random.normal((P, 128)).astype(mx.bfloat16)
    w = mx.abs(mx.random.normal((32,))) * 0.1
    n = mx.array([P], dtype=mx.int32)
    ref = D.scores(q, w, k, n, P, f32=True)
    got = G.scores(q, w, k, n, P)
    assert mx.array_equal(ref[:P], got[:P])


def _cache(n, seed=5):
    mx.random.seed(seed)
    c = LT.LatentTQCache(bits=6, sink=4, window=64)
    c.update_and_fetch(mx.random.normal((1, 1, n, 512)).astype(mx.bfloat16), mx.zeros((1, 1, n, 0), dtype=mx.bfloat16))
    return c


@pytest.mark.parametrize("dt", [mx.bfloat16, mx.float16])
def test_gather_rows(dt):
    n = 3000
    c = _cache(n)
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(n, 700, replace=False)).astype(np.int32)
    idx[:3] = [0, 1, 3]
    idx[-3:] = [n - 3, n - 2, n - 1]
    topk = np.concatenate([idx, [-1, n + 5, -1]]).astype(np.int32)
    rows, vm = G.gather_rows(c, mx.array(topk), dt)
    safe = np.clip(topk, 0, n - 1)
    ref = c.gather(mx.array(safe)[None, None, :, None], dt)
    valid = (topk >= 0) & (topk < n)
    assert np.array_equal(np.array(vm), valid)
    got = np.array(rows.astype(mx.float32))
    assert (got[~valid] == 0).all()
    r = np.array(ref.astype(mx.float32)).reshape(-1, 512)
    # exact fp64 reference for the pooled rows (the stock fp32 matmul rotates with ~4e-4 relative error, the butterfly with ~1e-7)
    pool = [i for i in range(len(topk)) if valid[i] and 4 <= topk[i] < n - 64]
    w = c.words[0, 0][mx.array(topk[pool])]
    s = c.scales[0, 0][mx.array(topk[pool])]
    y = tq.tq_dequant(w.reshape(-1, 48), s.reshape(-1), 6, 256, mx.float32)
    ex = (np.array(y).astype(np.float64) @ tq.rotation(256).astype(np.float64)).reshape(len(pool), 512)
    tol = 1e-2 if dt == mx.bfloat16 else 2e-3
    assert np.abs(got[pool] - ex).max() <= tol * np.abs(ex).max()
    assert np.abs(r[valid] - got[valid]).max() <= tol * np.abs(r[valid]).max()
    sink = [i for i in range(len(topk)) if valid[i] and (topk[i] < 4 or topk[i] >= n - 64)]
    assert np.array_equal(r[sink], got[sink])                      # exact fp16 / bf16 rows


def test_indexer_matches_stock():
    from mlx_vlm.models.glm5_next.config import TextConfig
    from mlx_vlm.models.cache import KVCache, PoolingCache
    import json
    p = os.path.expanduser("/path/to/data")
    if not os.path.exists(p):
        pytest.skip("no GLM config on this machine")
    cfg = TextConfig.from_dict(json.load(open(p))["text_config"])
    mx.random.seed(0)
    ix = L.Glm5NextIndexer(cfg, 3)
    from mlx.utils import tree_flatten, tree_unflatten
    new = []
    for k, v in tree_flatten(ix.parameters()):
        new.append((k, (mx.random.normal(v.shape) * (1.0 / v.shape[-1] ** 0.5)).astype(mx.float32 if "compress" in k else mx.bfloat16) if v.ndim >= 2 else mx.ones(v.shape, mx.bfloat16)))
    ix.update(tree_unflatten(new))
    mx.eval(ix.parameters())
    ix.eval()                                                  # a fresh module is in training mode and _fast_ok refuses it: the test used to compare stock with stock
    assert G.install()
    cls = L.Glm5NextIndexer
    stock = cls._kyojin_stock_call

    def caches(n):
        mx.random.seed(2)
        idx = KVCache()
        idx.update_and_fetch(mx.ones((1, 1, n, 1), dtype=mx.bool_), mx.zeros((1, 1, n, 0), dtype=mx.bool_))
        pc = PoolingCache(cfg.index_kpool)
        pc.pooled = mx.random.normal((1, n // 4, cfg.index_head_dim)).astype(mx.bfloat16)
        pc.buf_kv = mx.zeros((1, 4, cfg.index_head_dim), dtype=mx.bfloat16)
        pc.buf_gate = mx.zeros((1, 4, cfg.index_head_dim), dtype=mx.float32)
        pc.remainder = 0
        return idx, pc

    n = 4096
    a, b = caches(n), caches(n)
    G.hits_reset()
    for step in range(9):
        mx.random.seed(100 + step)
        x = (mx.random.normal((1, 1, cfg.hidden_size)) * 0.5).astype(mx.bfloat16)
        qr = (mx.random.normal((1, 1, cfg.q_lora_rank))).astype(mx.bfloat16)
        ta = stock(ix, x, qr, None, a[0], a[1], n + step)
        tb = cls.__call__(ix, x, qr, None, b[0], b[1], n + step)
        assert ta.shape == tb.shape == (1, 1, 2051)
        assert mx.array_equal(ta, tb), step
    # fewer pools than select_k, and a padded call: the stock code
    small = caches(1024)
    x = mx.zeros((1, 1, cfg.hidden_size), dtype=mx.bfloat16)
    out = cls.__call__(ix, x, mx.zeros((1, 1, cfg.q_lora_rank), dtype=mx.bfloat16), None, small[0], small[1], 1024)
    assert out.shape[-1] == 2051
    h = G.hits_reset()
    assert h.get("fast") == 9 and h.get("stock:few-pools") == 1, h              # the census the served-path proof reads


@pytest.mark.parametrize("dt", [mx.bfloat16, mx.float16])
def test_pool_compress_bit_exact(dt):
    from mlx_vlm.models.glm5_next.config import TextConfig
    import json
    p = os.path.expanduser("/path/to/data")
    if not os.path.exists(p):
        pytest.skip("no GLM config on this machine")
    ix = L.Glm5NextIndexer(TextConfig.from_dict(json.load(open(p))["text_config"]), 3)
    ape = (mx.random.normal((4, 128)) * 0.5).astype(mx.float32)
    ix.index_kpool_compress_ape = ape
    for seed in range(20):
        mx.random.seed(seed)
        k = (mx.random.normal((1, 4, 128)) * (0.3 + seed % 5)).astype(dt)
        g = (mx.random.normal((1, 4, 128)) * (0.5 + seed % 4)).astype(dt)
        assert mx.array_equal(ix._compress_pools(k, g), G.pool_compress(k, g, ape))


@pytest.mark.parametrize("dt", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("K,nvalid", [(2051, 2050), (700, 650), (96, 96)])
def test_attend(dt, K, nvalid):
    mx.random.seed(K)
    rows = (mx.random.normal((K, 512)) * 0.7).astype(dt)
    q = (mx.random.normal((1, 64, 1, 512)) * 0.5).astype(dt)
    vm = mx.arange(K) < nvalid
    KP = -(-K // 32) * 32
    rp = mx.concatenate([rows, mx.zeros((KP - K, 512), dtype=dt)])
    vp = mx.concatenate([vm, mx.zeros((KP - K,), dtype=mx.bool_)])
    ref = L.scaled_dot_product_attention(q, rows[None, None], rows[None, None], cache=None, scale=512 ** -0.5, mask=vm[None, None, None, :])
    got = G.attend(q, rp, vp, 512 ** -0.5)
    r, g = np.array(ref.astype(mx.float32)), np.array(got.astype(mx.float32))
    assert np.abs(r - g).max() <= (8e-3 if dt == mx.bfloat16 else 2e-3) * np.abs(r).max()
