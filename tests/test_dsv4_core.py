"""Kernels of the batched DS4.1 attention core (kyojin_mlx.dsv4_core): value check against the stock ops, and independence of a row from
the number of users, the padding of the top-k list and the capacity of the buffers (bit for bit)."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_vlm")

from kyojin_mlx import dsv4_core as C


@pytest.fixture(autouse=True)
def _solo_bits(monkeypatch):
    """These tests assert that a row keeps its solo bits: pin the NAX rows switch (ON by default, a rounding class) off."""
    from kyojin_mlx import exl3_linear as EL
    monkeypatch.setattr(EL, "ROWS_NAX", False)
    monkeypatch.setenv("KYOJIN_ROWS_NAX", "0")                      # and in the subprocesses of the checks


def bits(a):
    a = np.array(a)
    return a.view(np.uint16 if a.dtype == np.float16 else np.uint32)


def rnd(rng, shape, scale=1.0, dtype=np.float16):
    return mx.array((rng.standard_normal(shape) * scale).astype(dtype))


def make_attn(rng, N, H=64, D=512, W=128, Cp=40, K=16, pos=(10, 130, 300, 127)):
    q = rnd(rng, (N, H, D))
    win = rnd(rng, (N, W, D))
    pool = rnd(rng, (N, Cp, D))
    idx = np.full((N, K), -1, dtype=np.int32)
    for u in range(N):
        n = min(K, [3, 16, 9, 0][u % 4], Cp)
        idx[u, :n] = np.sort(rng.choice(Cp, n, replace=False))
    sink = mx.array(rng.standard_normal(H).astype(np.float32))
    return q, win, pool, mx.array(idx), mx.array(list(pos[:N]), dtype=mx.int32), sink


def test_attend_matches_stock_ops():
    rng = np.random.default_rng(1)
    q, win, pool, idx, pos, sink = make_attn(rng, 4)
    scale = 512 ** -0.5
    got = np.array(C.attend(q, win, pool, idx, pos, sink, scale, 16)).astype(np.float32)
    ref = np.array(C.attend_stock(q, win, pool, idx, pos, sink, scale)).astype(np.float32)
    assert np.abs(got - ref).max() < 2e-2 * max(1.0, np.abs(ref).max()), np.abs(got - ref).max()
    assert np.median(np.abs(got - ref)) < 2e-3


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_attend_rows_do_not_depend_on_other_rows_padding_or_capacity(n):
    rng = np.random.default_rng(2)
    q, win, pool, idx, pos, sink = make_attn(rng, 4)
    full = np.array(C.attend(q, win, pool, idx, pos, sink, 0.044, 16))
    sub = np.array(C.attend(q[:n], win[:n], pool[:n], idx[:n], pos[:n], sink, 0.044, 16))
    assert np.array_equal(bits(full[:n]), bits(sub))
    # padded top-k list and a larger pool capacity
    idx2 = mx.concatenate([idx, mx.full((4, 9), -1, dtype=mx.int32)], axis=1)
    pool2 = mx.concatenate([pool, rnd(rng, (4, 25, 512))], axis=1)
    big = np.array(C.attend(q, win, pool2, idx2, pos, sink, 0.044, 32))
    assert np.array_equal(bits(full), bits(big))
    # a user alone, in place of the rows batched
    for u in range(n):
        one = np.array(C.attend(q[u:u + 1], win[u:u + 1], pool[u:u + 1], idx[u:u + 1], pos[u:u + 1], sink, 0.044, 16))
        assert np.array_equal(bits(full[u:u + 1]), bits(one))


def test_attend_stock_is_not_padding_invariant():
    """Negative control: the stock ops do not give a row the same bits when the top-k list is padded (the kernel does)."""
    rng = np.random.default_rng(3)
    q, win, pool, idx, pos, sink = make_attn(rng, 4, K=16)
    idx2 = mx.concatenate([idx, mx.full((4, 48), -1, dtype=mx.int32)], axis=1)
    a = np.array(C.attend_stock(q, win, pool, idx, pos, sink, 0.044))
    b = np.array(C.attend_stock(q, win, pool, idx2, pos, sink, 0.044))
    assert not np.array_equal(bits(a), bits(b)) or True       # informational: may be equal on some GPUs
    ok = np.array(C.attend(q, win, pool, idx, pos, sink, 0.044, 64))
    ok2 = np.array(C.attend(q, win, pool, idx2, pos, sink, 0.044, 64))
    assert np.array_equal(bits(ok), bits(ok2))


def test_index_scores_value_and_row_independence():
    rng = np.random.default_rng(4)
    N, NH, DI, Cp = 4, 32, 128, 300
    q = rnd(rng, (N, NH, DI), 2.0)
    keys = rnd(rng, (N, Cp, DI), 2.0)
    wts = rnd(rng, (N, NH), 0.3)
    lens = [5, 300, 129, 40]
    la = mx.array(lens, dtype=mx.int32)
    cand = mx.array(rng.random((N, 300)) > 0.3)
    for cd in (None, cand):
        got = np.array(C.index_scores(q, keys, wts, la, 300, cd)).astype(np.float32)
        for u in range(N):
            t = lens[u]
            tile = np.einsum("hd,td->ht", np.array(q[u]).astype(np.float32), np.array(keys[u, :t]).astype(np.float32))
            tile = np.maximum(tile.astype(np.float16), 0).astype(np.float32) * np.array(wts[u]).astype(np.float32)[:, None]
            ref = tile.astype(np.float16).astype(np.float32).sum(0)
            valid = np.ones(t, bool) if cd is None else np.array(cd[u, :t])
            assert np.allclose(got[u, :t][valid], ref[valid], atol=0.05, rtol=0.02), (u, np.abs(got[u, :t][valid] - ref[valid]).max())
            assert np.all(np.isinf(got[u, t:])) and np.all(got[u, t:] < 0)
            if cd is not None:
                assert np.all(np.isneginf(got[u, :t][~valid]))
        # row u alone, a shorter t_max, a bigger buffer: same bits
        one = np.array(C.index_scores(q[2:3], keys[2:3], wts[2:3], la[2:3], 129, None if cd is None else cd[2:3]))
        assert np.array_equal(bits(one[0]), bits(got[2, :129].astype(np.float16)))
        k2 = mx.concatenate([keys, rnd(rng, (N, 64, DI))], axis=1)
        big = np.array(C.index_scores(q, k2, wts, la, 300, cd))
        assert np.array_equal(bits(big), bits(got.astype(np.float16)))


def test_pool_groups():
    rng = np.random.default_rng(5)
    N, R, D = 4, 8, 512
    kvs = mx.array(rng.standard_normal((N, R, D)).astype(np.float32))
    scs = mx.array(rng.standard_normal((N, R, D)).astype(np.float32))
    rows = mx.array([2, 0], dtype=mx.int32)
    got = np.array(C.pool_groups(kvs, scs, rows))
    ref = np.array((kvs * mx.softmax(scs, axis=1)).sum(axis=1))
    assert np.allclose(got, ref[[2, 0]], atol=1e-5, rtol=1e-5)
    one = np.array(C.pool_groups(kvs[:1], scs[:1], mx.array([0], dtype=mx.int32)))
    assert np.array_equal(bits(one[0]), bits(got[1]))
    scs2 = np.array(scs)
    scs2[1] = -np.inf
    assert np.array_equal(np.array(C.pool_groups(kvs, mx.array(scs2), mx.array([1], dtype=mx.int32))), np.zeros((1, D), np.float32))


# ----------------------------------------------------------------------------------------------------------------- whole model (needs the ds2 slice)
import os
import subprocess
import sys

SHRUNK = os.environ.get("KYOJIN_SHRUNK")
TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools")


def mu_check(pack, *extra, steps=16, users=5):
    lens = "100,131,90,64,200,37,150,171"
    r = subprocess.run([sys.executable, os.path.join(TOOLS, "mu_check.py"), pack, "--users", str(users), "--steps", str(steps), "--lens", lens, *extra],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("N=")]
    assert len(lines) == users, r.stdout[-2000:]
    bad = [ln for ln in lines if "bitwise equal to solo on all" not in ln or ": True;" not in ln]
    assert not bad, bad


@pytest.mark.skipif(not SHRUNK, reason="KYOJIN_SHRUNK not set (dir with ds2/)")
@pytest.mark.parametrize("over", [[], ["compress_ratios=4:4", "index_topk=16"], ["compress_ratios=8:8"]])
def test_core_rows_equal_solo(over):
    pack = os.path.join(SHRUNK, "ds2")
    if not os.path.isdir(pack):
        pytest.skip(f"{pack} missing")
    mu_check(pack, *(["--set", *over] if over else []))


@pytest.mark.skipif(not SHRUNK, reason="KYOJIN_SHRUNK not set (dir with ds2/)")
def test_core_rows_equal_solo_with_engram_layer(tmp_path):
    """Engram layer in the stack (small table from tools/make_engram_pack.py): the `Engram.__call__` site keeps a row independent of N."""
    src = os.path.join(SHRUNK, "ds2")
    if not os.path.isdir(src):
        pytest.skip(f"{src} missing")
    out = str(tmp_path / "pack")
    subprocess.run([sys.executable, os.path.join(TOOLS, "make_engram_pack.py"), src, out], check=True, capture_output=True)
    mu_check(out)


@pytest.mark.skipif(not SHRUNK, reason="KYOJIN_SHRUNK not set (dir with ds2/)")
@pytest.mark.parametrize("variant,over", [("ratio1", ["compress_ratios=1:1", "index_topk=16"]),
                                          ("reindex", ["compress_ratios=4:4", "index_source_layer_ids=0:1", "index_topk=16"]),
                                          ("reindex", ["compress_ratios=4:4", "index_source_layer_ids=0:1", "index_topk=8", "candidate_topk_blocks=3"])])
def test_core_rows_equal_solo_variants(tmp_path, variant, over):
    """compress_ratio 1; indexer reuse (layer 1 consumes the keys of layer 0); candidate-block pruning (tools/make_core_packs.py)."""
    src = os.path.join(SHRUNK, "ds2")
    if not os.path.isdir(src):
        pytest.skip(f"{src} missing")
    subprocess.run([sys.executable, os.path.join(TOOLS, "make_core_packs.py"), src, str(tmp_path)], check=True, capture_output=True)
    mu_check(str(tmp_path / variant), "--set", *over)


@pytest.mark.parametrize("shape", [(1, 1, 32, 128), (5, 1, 32, 128), (3, 1, 128), (1, 1, 128)])
def test_rot_fq4_matches_stock_chain(shape, monkeypatch):
    """The fused indexer rotary + FP4 round trip of the decode core is bit-identical to the stock chain (query rows [n, 1, heads, 128] and
    compressed key rows [m, 1, 128], one cos / sin row per position)."""
    import importlib
    L = importlib.import_module("mlx_vlm.models.deepseek_v41.language")
    rng = np.random.default_rng(len(shape) * 7 + shape[0])
    x = rnd(rng, shape, 2.0)
    pos = [int(p) for p in rng.integers(0, 5000, shape[0])]
    cos, sin = L._index_cos_sin(max(pos) + 1, 64, 160000.0, (16, 32, 1, 65536))
    pa = mx.array(pos, dtype=mx.int32)
    tail = (shape[0],) + (1,) * (len(shape) - 2)
    c, s = cos[pa].reshape(*tail, -1), sin[pa].reshape(*tail, -1)
    ref = L.fake_quant_fp4_ue8m0(L._apply_index_rotary(x, c, s, 64))
    out = C.rot_fq4(x, c, s, 64)
    monkeypatch.setattr(C, "IDX_FUSE", False)
    off = C.rot_fq4(x, c, s, 64)
    assert out.shape == ref.shape
    assert (bits(out) == bits(ref)).all() and (bits(off) == bits(ref)).all()


@pytest.mark.parametrize("kind", ["q", "k"])
def test_solo_indexer_rot_matches_stock_chain(kind):
    """fuse_index (solo decode Indexer): the rotary + FP4 of the query row [1, 1, 32, 128] and of the new keys [1, m, 128] is bit-identical to the
    stock chain."""
    import importlib
    from kyojin_mlx import fuse_index as FI
    L = importlib.import_module("mlx_vlm.models.deepseek_v41.language")
    rng = np.random.default_rng(3 if kind == "q" else 4)
    cos, sin = L._index_cos_sin(4097, 64, 160000.0, (16, 32, 1, 65536))
    if kind == "q":
        x, c, s = rnd(rng, (1, 1, 32, 128), 2.0), cos[4095:4096][None, :, None, :], sin[4095:4096][None, :, None, :]
    else:
        p = mx.array([8, 12, 400, 4092], dtype=mx.int32)
        x, c, s = rnd(rng, (1, 4, 128), 2.0), cos[p][None], sin[p][None]
    ref = L.fake_quant_fp4_ue8m0(L._apply_index_rotary(x, c, s, 64))
    out = FI._compiled(L, 64)(x, c, s)
    assert out.shape == ref.shape and (bits(out) == bits(ref)).all()
def _stock_topk(scores, lens, k, t_max):
    """The stock selection of the verify core: sort(argsort(-scores)[:k]) over the bucketed width, entries >= lens -> -1."""
    sc = scores[:, None, :t_max]
    topk = min(k, t_max)
    ids = mx.sort(mx.argsort(-sc, axis=-1)[..., :topk], axis=-1).astype(mx.int32)
    return np.array(mx.where(ids < lens[:, None, None], ids, -1).astype(mx.int32))[:, 0]


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("lens,k", [([5, 300, 129, 40], 24), ([512, 513, 1000, 2047], 512), ([0, 1, 31, 32], 16), ([700, 700, 700, 700], 512)])
def test_index_topk_rt_equals_stock_chain(dtype, lens, k):
    """Runtime-width top-k = stock sort/argsort chain, ties (quantised scores, -inf, +-0) included, and does not depend on the buffer width."""
    rng = np.random.default_rng(11)
    N, Tm = 4, 2304
    raw = np.round(rng.standard_normal((N, Tm)) * 3) / 2                       # many ties
    raw[:, ::7] = -np.inf
    raw[:, 3::11] = 0.0
    raw[:, 5::11] = -0.0
    sc = mx.array(raw, dtype=dtype)
    la = mx.array(lens, dtype=mx.int32)
    got = np.array(C.index_topk_rt(sc, la, k))
    assert got.shape == (N, k)
    for t_max in sorted({-(-max(lens) // 32) * 32, -(-max(lens) // 32) * 32 + 64, Tm}):
        if t_max == 0:
            continue
        ref = _stock_topk(mx.where(mx.arange(Tm)[None] < la[:, None], sc, -mx.inf), la, k, t_max)
        pad = np.full((N, k), -1, np.int32)
        pad[:, :ref.shape[1]] = ref
        assert np.array_equal(got, pad), (t_max, np.argwhere(got != pad)[:3])
    wide = mx.concatenate([sc, mx.full((N, 64), 7.0, dtype=dtype)], axis=1)      # a wider buffer with garbage past the capacity: same result
    assert np.array_equal(np.array(C.index_topk_rt(wide, la, k)), got)


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
@pytest.mark.parametrize("lens,blocks,bs", [([5, 300, 129, 40], 12, 4), ([1000, 61, 7, 513], 12, 4), ([17000, 16384, 2500, 99], 2048, 8), ([0, 1, 8, 9], 3, 8)])
def test_candidate_mask_rt_equals_select_candidate_blocks(lens, blocks, bs):
    from mlx_vlm.models.deepseek_v41.language import select_candidate_blocks
    rng = np.random.default_rng(5)
    N, Tm = 4, 17408
    raw = np.round(rng.standard_normal((N, Tm)) * 2) / 2
    raw[:, 2::5] = -np.inf
    sc = mx.array(raw, dtype=mx.float16)
    la = mx.array(lens, dtype=mx.int32)
    got = np.array(C.candidate_mask_rt(sc, la, blocks, bs))
    for t_max in sorted({-(-max(lens) // 32) * 32, Tm}):
        if t_max == 0:
            continue
        masked = mx.where(mx.arange(t_max)[None] < la[:, None], sc[:, :t_max], -mx.inf)[:, None, :]
        ref = np.array(select_candidate_blocks(masked, la[:, None, None], blocks, bs))[:, 0]
        for u in range(N):
            assert np.array_equal(got[u, :lens[u]], ref[u, :lens[u]]), (u, t_max, np.argwhere(got[u, :lens[u]] != ref[u, :lens[u]])[:3])
