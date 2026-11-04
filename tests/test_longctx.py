"""Long-context exact levers (kyojin_mlx.longctx): every path equals the stock one bit for bit.

rope table (CPU ok): one power-of-two table sliced == the vendor table at each length (256 / power-of-two edges).
grow (CPU ok): the vendor cache _append with slack publishes the same arrays; capacity off == stock.
pftopk (Metal): tiled scores + per-tile exact top-k + stable merge == stock scores_mm + argsort chain, with heavy ties (zero head weights,
quantised scores), the causal -inf tail, a candidate mask, partial last tile; scores_mm on a key slice (n0) == the columns of the full one."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from kyojin_mlx import longctx as LC

try:
    from mlx_vlm.models.deepseek_v41 import language as L
except Exception:                                   # vendor package not on the path
    L = None

needs_vendor = pytest.mark.skipif(L is None, reason="needs vlm_pkg (mlx_vlm deepseek_v41) on PYTHONPATH")
needs_metal = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
YARN = (16.0, 32, 1, 65536)


@pytest.fixture
def rope_patch(monkeypatch):
    orig_i, orig_c = L._index_cos_sin, L._cached_index_cos_sin
    monkeypatch.setattr(LC, "ROPE", True)
    LC._tables.clear()
    if getattr(L, "_kyojin_longctx_rope", False):
        orig_i, orig_c = LC._orig["index"], LC._orig["cached"]
        L._kyojin_longctx_rope = False
    assert LC.install_rope(L) == 1
    yield orig_i, orig_c
    L._index_cos_sin, L._cached_index_cos_sin = orig_i, orig_c
    L._kyojin_longctx_rope = False


@needs_vendor
@pytest.mark.parametrize("theta", [10000.0, 160000.0])
def test_rope_table_bitwise(rope_patch, theta):
    orig_i, orig_c = rope_patch
    for n in (1, 7, 255, 256, 257, 511, 4095, 4096, 4097, 8192, 70001, 131072, 131073):   # growing: several rebuilds of the table
        for yarn in (YARN, (1.0, 32, 1, 0)):
            c0, s0 = orig_i(n, 64, theta, yarn)
            c1, s1 = L._index_cos_sin(n, 64, theta, yarn)
            assert c1.shape == c0.shape and np.array_equal(np.array(c0), np.array(c1)) and np.array_equal(np.array(s0), np.array(s1))
            cap = ((n + 255) // 256) * 256
            c2, _ = L._cached_index_cos_sin(cap, 64, theta, yarn)
            assert np.array_equal(np.array(orig_c(cap, 64, theta, yarn)[0]), np.array(c2))
    assert max(t[0] for t in LC._tables.values()) == 262144              # one table per key, power of two


def test_capacity(monkeypatch):
    from kyojin_mlx import depthflat as DF
    monkeypatch.setattr(DF, "ENABLED", False)                               # KYOJIN_DEPTHFLAT carries its own slack (tests/test_depthflat.py)
    monkeypatch.setattr(LC, "GROW", False)
    assert [LC.capacity(n, 256) for n in (1, 256, 257, 100000)] == [256, 256, 512, 100096]
    monkeypatch.setattr(LC, "GROW", True)
    for n in (1, 256, 257, 100000, 1 << 20):
        c = LC.capacity(n, 256)
        assert c % 256 == 0 and c >= n + (n >> 3)


class _C:
    step = 256


@needs_vendor
def test_append_grow_same_arrays(monkeypatch):
    rng = np.random.default_rng(1)
    chunks = [mx.array(rng.standard_normal((1, m, 8)).astype(np.float16)) for m in (300, 1, 1, 255, 600, 1, 2048, 3)]
    outs = []
    for grow in (False, True):
        monkeypatch.setattr(LC, "GROW", grow)
        arrays, buffers, self = [None], [None], _C()
        for v in chunks:
            out = (LC._append if grow else L.DeepseekV41Cache._append)(self, arrays, buffers, 0, v)
        outs.append((np.array(out), buffers[0].shape[1]))
    assert np.array_equal(outs[0][0], outs[1][0]) and outs[1][1] >= outs[0][1]


# ----------------------------------------------------------------------------------------------------------------- prefill top-k (Metal)
def _stock(q, keys, w, start, ratio, k, cand):
    from kyojin_mlx import ds_pf_idx as PI
    sc = PI.scores_mm(q, keys, w, start, ratio)
    if cand is not None:
        sc = mx.where(cand[0], sc, -mx.inf)
    return mx.sort(mx.argsort(-sc, axis=-1)[..., :k], axis=-1).astype(mx.int32)


def _inputs(T, Tc, start, ratio, kind, seed):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((1, T, 32, 128)).astype(np.float16)
    keys = (rng.standard_normal((1, Tc, 128)) * 0.5).astype(np.float16)
    w = (rng.standard_normal((1, T, 32)) * 0.05).astype(np.float16)
    if kind == "zero_w":                                   # every score exactly 0: ties over the whole past
        w[:] = 0
    elif kind == "quant":                                  # few distinct scores: big tie groups at the k-th value
        q = np.round(q).astype(np.float16); keys = np.round(keys * 2).astype(np.float16) / 2; w = np.where(w > 0, 0.0625, -0.0625).astype(np.float16)
        w[..., 1:] = 0
    return mx.array(q), mx.array(keys), mx.array(w)


@needs_metal
@pytest.mark.parametrize("kind", ["rand", "zero_w", "quant"])
@pytest.mark.parametrize("T,Tc,start,ratio,tile", [(64, 5000, 9936, 2, 1024), (40, 4096, 4056, 1, 2048), (33, 3000, 6000, 2, 512),
                                                   (16, 2600, 1, 1, 1024)])
def test_pf_topk_bitwise(kind, T, Tc, start, ratio, tile):
    q, keys, w = _inputs(T, Tc, start, ratio, kind, seed=T + Tc)
    k = 512
    ref = _stock(q, keys, w, start, ratio, k, None)
    got = LC.pf_topk(q, keys, w, start, ratio, k, None, tile=tile)
    assert got.shape == ref.shape and np.array_equal(np.array(ref), np.array(got)), kind


@needs_metal
@pytest.mark.parametrize("kind", ["rand", "zero_w"])
def test_pf_topk_candidates(kind):
    T, Tc, start, ratio = 48, 6000, 6000 - 48, 1
    q, keys, w = _inputs(T, Tc, start, ratio, kind, seed=5)
    rng = np.random.default_rng(9)
    cand = mx.array(rng.random((1, T, Tc)) < 0.07)         # ~420 candidates per row < k: masked -inf keys inside the causal range are selected too
    ref = _stock(q, keys, w, start, ratio, 512, cand)
    got = LC.pf_topk(q, keys, w, start, ratio, 512, cand, tile=1024)
    assert np.array_equal(np.array(ref), np.array(got))


@needs_metal
def test_scores_mm_slice_n0():
    from kyojin_mlx import ds_pf_idx as PI
    q, keys, w = _inputs(72, 3000, 2928, 1, "rand", seed=3)
    full = np.array(PI.scores_mm(q, keys, w, 2928, 1))
    for a, b in ((0, 1024), (1024, 2048), (2048, 3000), (777, 2999)):
        part = np.array(PI.scores_mm(q, keys[:, a:b], w, 2928, 1, n0=a))
        assert np.array_equal(full[:, a:b], part)


# ----------------------------------------------------------------------------------------------------------------- decode scores, register-blocked (Metal)
@needs_metal
@pytest.mark.parametrize("cand,rt,sess,kind", [(False, False, False, "rand"), (True, False, False, "rand"), (False, True, False, "rand"),
                                               (True, True, True, "rand"), (False, False, False, "quant"), (True, True, False, "neg")])
def test_iscore_kb_bitwise(monkeypatch, cand, rt, sess, kind):
    from kyojin_mlx import dsv4_core as C
    rng = np.random.default_rng(11)
    N, NH, DI, cap = 3, 32, 128, 1300
    q = rng.standard_normal((N, NH, DI)).astype(np.float16)
    keys = (rng.standard_normal((N, cap, DI)) * 0.5).astype(np.float16)
    w = (rng.standard_normal((N, NH)) * 0.1).astype(np.float16)
    if kind == "quant":
        q, keys = np.round(q).astype(np.float16), np.round(keys * 2).astype(np.float16) / 2
    elif kind == "neg":                                      # many relu zeros
        w = np.abs(w); q = -np.abs(q); keys = np.abs(keys)
    lens = mx.array([1, 777, 1290], dtype=mx.int32)
    cd = mx.array(rng.random((N, cap)) < 0.4) if cand else None
    se = mx.array([2, 0, 2], dtype=mx.uint32) if sess else None
    tm = cap if rt else 1290
    args = (mx.array(q), mx.array(keys), mx.array(w), lens, tm, cd, se, rt)
    monkeypatch.setattr(C, "ISCORE_KB", False)
    ref = np.array(C.index_scores(*args).astype(mx.float32))
    monkeypatch.setattr(C, "ISCORE_KB", True)
    monkeypatch.setattr(C, "ISCORE_KB_MIN", 0)
    got = np.array(C.index_scores(*args).astype(mx.float32))
    for u, n in enumerate([1, 777, 1290]):
        assert np.array_equal(ref[u, :n], got[u, :n], equal_nan=True), u
    if not rt:
        assert np.array_equal(ref, got)
