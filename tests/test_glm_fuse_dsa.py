"""G10c: DSA decode of 2..4 rows (kyojin_mlx.glm_fuse_dsa): the row-axis kernels against the one-row kernels, the whole indexer against the stock Glm5NextIndexer,
the attention of the rows against the solo rows (bit for bit), and the late binding of rows sites."""
import json
import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
L = pytest.importorskip("mlx_vlm.models.glm5_next.language")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
os.environ["KYOJIN_GLM_DSA_FAST"] = "1"
os.environ["KYOJIN_GLM_FUSE_DSA"] = "1"
from kyojin_mlx import glm_dsa as G, glm_fuse_dsa as F, dsa_decode as D, latent_tq as LT    # noqa: E402

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


@pytest.mark.parametrize("P", [1100, 5000])
def test_rows_kernels_equal_one_row_kernels(P):
    mx.random.seed(P)
    Q = 4
    q = mx.random.normal((Q, 32, 128)).astype(DT)
    k = mx.random.normal((P, 128)).astype(DT)
    w = mx.abs(mx.random.normal((Q, 32))) * 0.1
    n = mx.array([P - 3, P - 2, P - 1, P], dtype=mx.int32)
    sc = F.scores_rows(q, w, k, n, P)
    for u in range(Q):
        one = G.scores(q[u], w[u], k, n[u:u + 1], P)
        nu = int(n[u].item())
        assert mx.array_equal(sc[u, :nu], one[:nu]), u
    meta = mx.array([v for u in range(Q) for v in (4 * int(n[u].item()), u, 0)], dtype=mx.int32)
    sel = F.select_expand_rows(sc, n, P, 512, 4, meta)
    for u in range(Q):
        nu = int(n[u].item())
        ref = G.select_expand(sc[u], n[u:u + 1], nu, 512, 4, 4 * nu, u)
        assert mx.array_equal(sel[u], ref), u


def _caches(cfg, n, seed=2):
    from mlx_vlm.models.cache import KVCache, PoolingCache
    mx.random.seed(seed)
    idx = KVCache()
    idx.update_and_fetch(mx.ones((1, 1, n, 1), dtype=mx.bool_), mx.zeros((1, 1, n, 0), dtype=mx.bool_))
    pc = PoolingCache(cfg.index_kpool)
    pc.pooled = mx.random.normal((1, n // 4, cfg.index_head_dim)).astype(DT)
    pc.buf_kv = mx.zeros((1, 4, cfg.index_head_dim), dtype=DT)
    pc.buf_gate = mx.zeros((1, 4, cfg.index_head_dim), dtype=mx.float32)
    pc.remainder = 0
    return idx, pc


@pytest.mark.parametrize("Lr", [2, 3, 4])
def test_indexer_rows_equal_stock(Lr):
    cfg = _cfg()
    ix = _rand(L.Glm5NextIndexer(cfg, 3), 0)
    assert G.install() and F.install()
    cls = L.Glm5NextIndexer
    n = 4096
    a, b = _caches(cfg, n), _caches(cfg, n)
    base = F.STATS["ix_rows"]
    for step in range(7):                                   # remainders 0..3 twice: windows complete in different rows
        mx.random.seed(50 + step)
        x = (mx.random.normal((1, Lr, cfg.hidden_size)) * 0.5).astype(DT)
        qr = mx.random.normal((1, Lr, cfg.q_lora_rank)).astype(DT)
        off = a[0].offset
        F.ENABLED = False
        ta = cls.__call__(ix, x, qr, None, a[0], a[1], off)
        F.ENABLED = True
        tb = cls.__call__(ix, x, qr, None, b[0], b[1], off)
        assert ta.shape == tb.shape == (1, Lr, 2051)
        assert mx.array_equal(ta, tb), (step, Lr)
        assert a[1].pooled.shape == b[1].pooled.shape and mx.array_equal(a[1].pooled, b[1].pooled) and a[1].remainder == b[1].remainder
    assert F.STATS["ix_rows"] > base
    F.ENABLED = True


def test_indexer_rows_few_pools_take_stock():
    cfg = _cfg()
    ix = _rand(L.Glm5NextIndexer(cfg, 3), 0)
    assert G.install() and F.install()
    a = _caches(cfg, 1024)
    x = (mx.random.normal((1, 2, cfg.hidden_size)) * 0.5).astype(DT)
    qr = mx.random.normal((1, 2, cfg.q_lora_rank)).astype(DT)
    s0 = F.STATS["ix_stock"]
    out = L.Glm5NextIndexer.__call__(ix, x, qr, None, a[0], a[1], 1024)
    assert out.shape[-1] == 2051 and F.STATS["ix_stock"] == s0 + 1


def _same(a, b):
    """Bit equality; with KYOJIN_GLM_MLA_FUSE (G10d: fp32 attention kernels instead of the stock SDPA fallback) the stock reference differs by rounding only."""
    from kyojin_mlx import glm_mla
    if not glm_mla.ENABLED:
        assert mx.array_equal(a, b)
        return
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    assert float(mx.max(mx.abs(a - b))) <= 0.02 * float(mx.max(mx.abs(b))) + 1e-3


def _tq_cache(n, seed=3):
    from mlx_vlm.models.cache import KVCache
    mx.random.seed(seed)
    lat = LT.LatentTQCache(bits=6, sink=4, window=256)
    done = 0
    while done < n:
        m = min(2048, n - done)
        lat.update_and_fetch(mx.random.normal((1, 1, m, 512)).astype(DT), mx.zeros((1, 1, m, 0), dtype=DT))
        done += m
    mx.eval(lat.words)
    return lat


@pytest.mark.parametrize("Lr", [2, 3, 4])
def test_attend_rows_bit_equal_to_solo_rows(Lr):
    cfg = _cfg()
    a = _rand(L.Glm5NextAttention(cfg, 3), 5)
    assert G.install() and F.install()
    LT.install_attend()
    lat = _tq_cache(4096)
    q = (mx.random.normal((1, 64, Lr, 256)) * 0.5).astype(DT)
    nl = mx.random.normal((1, 1, Lr, 512)).astype(DT)
    lat.update_and_fetch(nl, mx.zeros((1, 1, Lr, 0), dtype=DT))
    tk = mx.minimum(mx.sort(mx.random.randint(-1, 4096, (1, Lr, 2051)).astype(mx.int32), axis=-1), lat.offset - 1)
    base = F.STATS["att_rows"]
    o, t = a._attend(q, None, nl, tk, lat, None)
    ss = [a._attend(q[:, :, i:i + 1], None, nl[:, :, i:i + 1], tk[:, i:i + 1], lat, None)[0] for i in range(Lr)]
    assert F.STATS["att_rows"] == base + 1
    _same(o, mx.concatenate(ss, axis=1))
    o1, t1 = a._attend(q, None, nl, tk, lat, None, True)                    # last_only: the last row only
    _same(o1, ss[-1])
    assert t1.shape == (1, 1, 2051)


def test_attend_rows_match_stock_dense_path():
    cfg = _cfg()
    a = _rand(L.Glm5NextAttention(cfg, 3), 5)
    assert F.install()
    lat = _tq_cache(4096)
    Lr = 2
    q = (mx.random.normal((1, 64, Lr, 256)) * 0.5).astype(DT)
    nl = mx.random.normal((1, 1, Lr, 512)).astype(DT)
    full, _ = lat.update_and_fetch(nl, mx.zeros((1, 1, Lr, 0), dtype=DT))
    tk = mx.minimum(mx.sort(mx.random.randint(0, 4096, (1, Lr, 2051)).astype(mx.int32), axis=-1), lat.offset - 1)
    got, _ = F.attend_rows(a, lambda *x, **k: pytest.fail("fell back"), q, full, nl, tk, lat, None)
    F.ENABLED = False
    ref, _ = L.Glm5NextAttention._attend(a, q, full, nl, tk, lat, None)
    F.ENABLED = True
    g, r = np.array(got.astype(mx.float32)), np.array(ref.astype(mx.float32))
    assert np.abs(g - r).max() <= 0.03 * np.abs(r).max()                    # the fused gather rotates exactly, the stock matmul with 4.5e-4 relative error (glm_dsa)


def test_late_bind_reaches_overridden_instances():
    from kyojin_mlx import rows as R

    class Mod:
        def f(self, x):
            return ("stock", x)

    m = Mod()
    F.late_bind()
    R._override(m, "f", lambda self, orig, x: orig(self, x))
    Mod.f = lambda self, x: ("later", x)                                    # installed after the override (latent_tq.install_attend does this at the first cache)
    assert m.f(1) == ("later", 1)


@pytest.mark.parametrize("Lr", [2, 4])
def test_indexer_rows_under_speculation_commit_matches_stock(Lr):
    """A verify round: the pooling cache is speculating; rows path and stock leave the same pool state after commit (full and partial accept)."""
    cfg = _cfg()
    ix = _rand(L.Glm5NextIndexer(cfg, 3), 0)
    assert G.install() and F.install()
    cls = L.Glm5NextIndexer
    n = 4096
    for keep in (Lr, 1):
        a, b = _caches(cfg, n), _caches(cfg, n)
        for rep in range(3):
            mx.random.seed(70 + rep)
            x = (mx.random.normal((1, Lr, cfg.hidden_size)) * 0.5).astype(DT)
            qr = mx.random.normal((1, Lr, cfg.q_lora_rank)).astype(DT)
            outs = []
            for en, (idx, pc) in ((False, a), (True, b)):
                F.ENABLED = en
                gen = pc.start_speculation(Lr)
                off = idx.offset
                t = cls.__call__(ix, x, qr, None, idx, pc, off)
                mx.eval(t)
                outs.append(t)
                pc.commit_speculation([keep], gen)
                idx.trim(Lr - keep)
            assert mx.array_equal(outs[0], outs[1])
            assert a[1].remainder == b[1].remainder and a[1].pooled.shape == b[1].pooled.shape and mx.array_equal(a[1].pooled, b[1].pooled)
            assert mx.array_equal(a[1].buf_kv[:, :a[1].remainder], b[1].buf_kv[:, :b[1].remainder])
    F.ENABLED = True


# ---------------------------------------------------------------------------------------------------------- batch caches (the served step)
def _batched(ctxs, seed0=7, plain=False):
    from kyojin_mlx import rows as R
    from mlx_vlm.models.cache import CacheList, KVCache, PoolingCache
    cfg = _cfg()
    users = []
    for i, n in enumerate(ctxs):
        mx.random.seed(seed0 + i)
        from mlx_vlm.models.cache import KVCache as _KV
        lat = _KV() if plain else LT.LatentTQCache(bits=6, sink=4, window=256)
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


@pytest.mark.parametrize("ctxs", [[4096], [4096, 4092], [4096, 4000, 3600], [5000, 4096, 4040, 3996]])
@pytest.mark.parametrize("Lr", [1, 3])
@pytest.mark.parametrize("spec", [False, True])
def test_indexer_batch_equal_stock(ctxs, Lr, spec):
    """B sessions with left padding, L rows each, plain and under a speculative round: the same top-k as the stock indexer, the same pool state."""
    cfg = _cfg()
    ix = _rand(L.Glm5NextIndexer(cfg, 3), 0)
    assert G.install() and F.install()
    cls = L.Glm5NextIndexer
    A, Bc = _batched(ctxs), _batched(ctxs)
    assert type(A.caches[1]).__name__ == "BatchKVCache" and type(A.caches[2]).__name__ == "BatchPoolingCache"
    base = F.STATS["ix_batch"]
    for step in range(6):
        mx.random.seed(90 + step)
        x = (mx.random.normal((len(ctxs), Lr, cfg.hidden_size)) * 0.5).astype(DT)
        qr = mx.random.normal((len(ctxs), Lr, cfg.q_lora_rank)).astype(DT)
        outs = []
        for en, c in ((False, A), (True, Bc)):
            F.ENABLED = en
            idx, pc = c.caches[1], c.caches[2]
            gen = pc.start_speculation(Lr) if spec else None
            t = cls.__call__(ix, x, qr, None, idx, pc, c.caches[0].offset)
            mx.eval(t)
            outs.append(t)
            if spec:
                keep = Lr if step % 2 == 0 else 1
                pc.commit_speculation([keep] * len(ctxs), gen)
                idx.trim(Lr - keep)
        assert mx.array_equal(outs[0], outs[1]), (ctxs, Lr, spec, step)
        pa, pb = A.caches[2], Bc.caches[2]
        assert pa.remainder == pb.remainder and pa._pool_lengths == pb._pool_lengths and pa._processed == pb._processed
        assert mx.array_equal(pa.pooled, pb.pooled) and mx.array_equal(pa.buf_kv, pb.buf_kv)
    F.ENABLED = True
    assert F.STATS["ix_batch"] > base


def test_pool_compress_b_equals_one_window_kernel():
    mx.random.seed(1)
    keys = mx.random.normal((3, 4, 128)).astype(DT)
    gate = mx.random.normal((3, 4, 128)).astype(DT)
    ape = mx.random.normal((4, 128)).astype(mx.float32)
    got = F.pool_compress_b(keys, gate, ape)
    for b in range(3):
        assert mx.array_equal(got[b:b + 1], G.pool_compress(keys[b:b + 1], gate[b:b + 1], ape))


def test_dense_gather_matches_take_along_axis():
    mx.random.seed(2)
    buf = mx.random.normal((2, 1, 512, 512)).astype(DT)
    nt = 300
    topk = mx.random.randint(-1, 400, (2 * 2 * 37,)).astype(mx.int32)                 # 2 sessions x 2 rows x 37 slots
    rows, vm = F.dense_gather(buf, topk, 2 * 37, nt)
    for b in range(2):
        t = topk[b * 74:(b + 1) * 74]
        ok = (t >= 0) & (t < nt)
        ref = mx.take_along_axis(buf[b:b + 1, :, :nt], mx.clip(t, 0, nt - 1)[None, None, :, None], axis=2)[0, 0]
        got = rows[b * 74:(b + 1) * 74]
        assert mx.array_equal(vm[b * 74:(b + 1) * 74], ok)
        assert mx.array_equal(mx.where(ok[:, None], got, 0), mx.where(ok[:, None], ref, 0))
        assert not mx.any(mx.where(ok[:, None], 0, got) != 0).item()


@pytest.mark.parametrize("plain", [False, True])
@pytest.mark.parametrize("B,Lr", [(1, 1), (1, 3), (3, 1), (2, 2)])
def test_attend_batch_equals_per_user_stock(B, Lr, plain):
    """LatentBatch: one gather launch, the SDPA of the solo shape per (session, row): bit equal to the rows.py per-user SDPA of the stock code."""
    from kyojin_mlx import rows as R
    cfg = _cfg()
    a = _rand(L.Glm5NextAttention(cfg, 3), 5)
    assert F.install()
    c = _batched([4096 - 8 * i for i in range(B)], plain=plain)
    lat = c.caches[0]
    assert type(lat).__name__ == ("BatchKVCache" if plain else "LatentBatch")
    q = (mx.random.normal((B, 64, Lr, 256)) * 0.5).astype(DT)
    nl = mx.random.normal((B, 1, Lr, 512)).astype(DT)
    full, _ = lat.update_and_fetch(nl, mx.zeros((B, 1, Lr, 0), dtype=DT))
    tk = mx.minimum(mx.sort(mx.random.randint(-1, 4096, (B, Lr, 2051)).astype(mx.int32), axis=-1), lat._idx - 1)
    got, _ = F.attend_batch(a, lambda *x, **k: pytest.fail("fell back"), q, full, nl, tk, lat, None)
    sdpa = L.scaled_dot_product_attention
    per_row = []
    for i in range(Lr):                                           # the stock loop, with the per-user SDPA of the rows site
        sel = tk[:, i:i + 1]
        valid = (sel >= 0) & (sel < full.shape[2])
        selected = mx.take_along_axis(full, mx.clip(sel, 0, full.shape[2] - 1)[:, None, 0, :, None], axis=2)
        query = a.embed_q(q[:, :, i:i + 1])
        out = mx.concatenate([sdpa(query[b:b + 1], selected[b:b + 1], selected[b:b + 1], cache=lat, scale=a.scale, mask=valid[b:b + 1, None]) for b in range(B)], axis=0)
        per_row.append(a.unembed_out(out))
    ref = mx.concatenate(per_row, axis=2).transpose(0, 2, 1, 3).reshape(B, Lr, -1)
    _same(got, ref)


@pytest.mark.parametrize("Lr,every,fires", [(2, 1, True), (3, 2, True), (1, 1, False), (2, 0, False)])
def test_attend_batch_eval_flag_is_value_neutral(Lr, every, fires, monkeypatch):
    """DSA2: KYOJIN_GLM_FUSE_DSA_EVAL only moves the host wait (async_eval of the attention output on multi-row passes): same bits, and it fires only for L > 1 when on."""
    cfg = _cfg()
    a = _rand(L.Glm5NextAttention(cfg, 3), 5)
    assert F.install()
    B = 2
    outs = {}
    for ev in (0, every):
        monkeypatch.setattr(F, "EVAL", ev)
        monkeypatch.setattr(F, "_ORD", {})
        c = _batched([4096, 4040])
        lat = c.caches[0]
        mx.random.seed(11)
        q = (mx.random.normal((B, 64, Lr, 256)) * 0.5).astype(DT)
        nl = mx.random.normal((B, 1, Lr, 512)).astype(DT)
        full, _ = lat.update_and_fetch(nl, mx.zeros((B, 1, Lr, 0), dtype=DT))
        tk = mx.minimum(mx.sort(mx.random.randint(-1, 4096, (B, Lr, 2051)).astype(mx.int32), axis=-1), lat._idx - 1)
        got, _ = F.attend_batch(a, lambda *x, **k: pytest.fail("fell back"), q, full, nl, tk, lat, None)
        outs[ev] = got
        if ev and not fires:
            assert not F._ORD, "no ordinal is taken on a one-row step"
    _same(outs[0], outs[every])


def _state_equal(pa, pb):
    return (pa.remainder == pb.remainder and pa._pool_lengths == pb._pool_lengths and pa._processed == pb._processed
            and mx.array_equal(pa.pooled, pb.pooled) and mx.array_equal(pa.buf_kv, pb.buf_kv))


@pytest.mark.parametrize("ctx", [131068, 131080])
def test_indexer_batch_past_32768_pools_takes_stock_before_any_write(ctx):
    """select_expand_rows is bounded at 32768 pools (> ~131075 tokens): the fused batch indexer must take the stock path BEFORE any cache / pool write (it used to raise after them)."""
    cfg = _cfg()
    ix = _rand(L.Glm5NextIndexer(cfg, 3), 0)
    assert G.install() and F.install()
    cls = L.Glm5NextIndexer
    A, Bc = _batched([ctx], plain=True), _batched([ctx], plain=True)
    mx.random.seed(5)
    x = (mx.random.normal((1, 1, cfg.hidden_size)) * 0.5).astype(DT)
    qr = mx.random.normal((1, 1, cfg.q_lora_rank)).astype(DT)
    over = (ctx + 1) // cfg.index_kpool > F.MAX_POOLS
    base, w0 = dict(F.STATS), F.WHY.get("pools>32768", 0)
    F.ENABLED = False
    ta = cls.__call__(ix, x, qr, None, A.caches[1], A.caches[2], A.caches[0].offset)
    F.ENABLED = True
    tb = cls.__call__(ix, x, qr, None, Bc.caches[1], Bc.caches[2], Bc.caches[0].offset)
    mx.eval(ta, tb)
    assert mx.array_equal(ta, tb)
    assert _state_equal(A.caches[2], Bc.caches[2]) and A.caches[1]._idx == Bc.caches[1]._idx
    assert F.WHY.get("pools>32768", 0) == w0 + (1 if over else 0)
    assert F.STATS["ix_batch"] == base["ix_batch"] + (0 if over else 1)


@pytest.mark.parametrize("Lr", [2, 3])
def test_indexer_rows_past_32768_pools_takes_stock_before_any_write(Lr):
    cfg = _cfg()
    ix = _rand(L.Glm5NextIndexer(cfg, 3), 0)
    assert G.install() and F.install()
    n = 131080
    a, b = _caches(cfg, n), _caches(cfg, n)
    mx.random.seed(6)
    x = (mx.random.normal((1, Lr, cfg.hidden_size)) * 0.5).astype(DT)
    qr = mx.random.normal((1, Lr, cfg.q_lora_rank)).astype(DT)
    base, w0 = F.STATS["ix_rows"], F.WHY.get("pools>32768", 0)
    F.ENABLED = False
    ta = L.Glm5NextIndexer.__call__(ix, x, qr, None, a[0], a[1], n)
    F.ENABLED = True
    tb = L.Glm5NextIndexer.__call__(ix, x, qr, None, b[0], b[1], n)
    mx.eval(ta, tb)
    assert mx.array_equal(ta, tb)
    assert a[1].pooled.shape == b[1].pooled.shape and mx.array_equal(a[1].pooled, b[1].pooled) and a[1].remainder == b[1].remainder and a[0].offset == b[0].offset
    assert F.WHY.get("pools>32768", 0) == w0 + 1 and F.STATS["ix_rows"] == base


@pytest.mark.parametrize("ctxs,Lr,pattern", [
    ([4096, 4092, 4000], 3, [(3, 1, 2), (1, 3, 3), (2, 2, 1), (3, 3, 3), (1, 1, 2), (2, 3, 1), (3, 2, 2), (1, 2, 3)]),
    ([5000, 4096, 4040, 3996], 2, [(2, 1, 1, 2), (1, 2, 2, 1), (2, 2, 1, 1), (1, 1, 2, 2), (2, 1, 2, 1), (1, 2, 1, 2)]),
])
def test_indexer_batch_ragged_speculative_accepts_equal_stock(ctxs, Lr, pattern):
    """Sessions accepting DIFFERENT token counts per round: commit_speculation(keeps), the stock ragged tail (trim to max keep + prepare(right_padding) / finalize), then the next round:
    top-k and pool state bit-identical to the stock indexer."""
    cfg = _cfg()
    ix = _rand(L.Glm5NextIndexer(cfg, 3), 0)
    assert G.install() and F.install()
    cls = L.Glm5NextIndexer
    A, Bc = _batched(ctxs), _batched(ctxs)
    base = F.STATS["ix_batch"]
    for step, keeps in enumerate(pattern):
        mx.random.seed(190 + step)
        x = (mx.random.normal((len(ctxs), Lr, cfg.hidden_size)) * 0.5).astype(DT)
        qr = mx.random.normal((len(ctxs), Lr, cfg.q_lora_rank)).astype(DT)
        outs = []
        for en, c in ((False, A), (True, Bc)):
            F.ENABLED = en
            idx, pc = c.caches[1], c.caches[2]
            gen = pc.start_speculation(Lr)
            t = cls.__call__(ix, x, qr, None, idx, pc, c.caches[0].offset)
            mx.eval(t)
            outs.append(t)
            pc.commit_speculation(list(keeps), gen)
            mk = max(keeps)
            if Lr - mk:
                idx.trim(Lr - mk)
            if any(mk - k for k in keeps):
                idx.prepare(right_padding=[mk - k for k in keeps])
                idx.finalize()
        assert mx.array_equal(outs[0], outs[1]), (ctxs, step, keeps)
        pa, pb = A.caches[2], Bc.caches[2]
        assert pa.remainder == pb.remainder and pa._pool_lengths == pb._pool_lengths and pa._processed == pb._processed, (step, keeps)
        assert A.caches[1]._idx == Bc.caches[1]._idx
    F.ENABLED = True
    assert F.STATS["ix_batch"] - base == len(pattern)
