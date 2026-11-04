"""Decode-cost-flat levers (E8 packed index keys, E9 short-context register-blocked scores, E7 rope prefetch) on synthetic shapes: the new path against the
old one, bit for bit where the lever is exact.

E8  the fp4 codes of a KeySlots entry unpacked inside the score kernel == the fp16 / bf16 mirror rows (same fp4 grid the vendor rounds to), same candidates, and
    the fused indexer returns the same top-k either way (the packed kernel is the non-NAX reduction, so the fp16 dense arm may take the NAX kernel instead).
E9  the register-blocked decode score kernel runs from a short key count (ISCORE_KB_MIN = 0) and returns the scalar kernel's scores.
E7  the table grown ahead of use on a background thread serves the same rows as the vendor build, and the step that outgrows the table joins it.
"""
import time

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
L = pytest.importorskip("mlx_vlm.models.deepseek_v41.language")
from types import SimpleNamespace                                            # noqa: E402

from mlx.utils import tree_map                                               # noqa: E402
from mlx_vlm.models.deepseek_v41.config import ModelConfig                   # noqa: E402
from mlx_vlm.models.deepseek_v41.fakequant import fake_quant_fp4_ue8m0 as fq4   # noqa: E402

from kyojin_mlx import dsa_decode as D                                       # noqa: E402
from kyojin_mlx import dsv4_core as C                                        # noqa: E402
from kyojin_mlx import fp4pack                                               # noqa: E402
from kyojin_mlx import fuse_index as FI                                      # noqa: E402
from kyojin_mlx import longctx as LC                                         # noqa: E402
from kyojin_mlx.fp4pool import KeySlots                                      # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
TOPK = 64


# ----------------------------------------------------------------------------------------------------------------- E8
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("n,cand", [(7, False), (8, True), (321, False), (517, True), (1024, False), (3003, True)])
def test_scores_packed_bitwise(monkeypatch, n, cand, dtype):
    monkeypatch.setattr(D, "USE_NAX", False)                              # the packed kernel is the simdgroup path: compare like for like
    rng = np.random.default_rng(n)
    keys = fq4(mx.array((rng.standard_normal((n, 128)) * 0.5).astype(np.float32)).astype(dtype))   # exactly the vendor grid KeySlots stores
    codes, scales = fp4pack.pack(keys[None], 32, False)
    q = mx.array(rng.standard_normal((32, 128)).astype(np.float32)).astype(dtype)
    w = mx.array((rng.standard_normal(32) * 0.1).astype(np.float32)).astype(dtype)
    nn = mx.array([n], dtype=mx.int32)
    cd = mx.array(np.repeat(rng.random((-(-n // 8),)) < 0.6, 8)[:n]) if cand else None
    ref = np.array(D.scores(q, w, keys, nn, n, cd).astype(mx.float32))
    got = np.array(D.scores_packed(q, w, codes[0], scales[0], nn, n, cd).astype(mx.float32))
    assert np.array_equal(ref[:n], got[:n], equal_nan=True), (n, dtype)


def test_scores_packed_mask_skips_blocks(monkeypatch):
    """A masked block is -inf, as in the dense kernel (the consumer layers rely on the block test before the unpack)."""
    monkeypatch.setattr(D, "USE_NAX", False)
    n = 256
    rng = np.random.default_rng(3)
    keys = fq4(mx.array((rng.standard_normal((n, 128)) * 0.5).astype(np.float16)))
    codes, scales = fp4pack.pack(keys[None], 32, False)
    q = mx.array(rng.standard_normal((32, 128)).astype(np.float16))
    w = mx.array((rng.standard_normal(32) * 0.1).astype(np.float16))
    nn = mx.array([n], dtype=mx.int32)
    cd = mx.array((np.arange(n) // 8) % 2 == 0)
    got = np.array(D.scores_packed(q, w, codes[0], scales[0], nn, n, cd).astype(mx.float32))
    assert np.all(got[:n][~np.array(cd)] == -np.inf)
    assert np.all(np.isfinite(got[:n][np.array(cd)]))


def make_indexer(ratio, cand_layer, dtype, owns=True):
    cfg = ModelConfig()
    cfg.compress_ratios = [ratio, ratio]
    cfg.kv_source_layer_ids = [1] if owns else []
    cfg.candidate_source_layer_id = cand_layer
    cfg.index_topk = TOPK
    cfg.candidate_topk_blocks = 40
    cfg.candidate_block_size = 8
    cfg.hidden_size = 256
    cfg.q_lora_rank = 128
    cfg.index_n_heads = 32
    cfg.index_head_dim = 128
    cfg.qk_rope_head_dim = 64
    mx.random.seed(11)
    idx = L.Indexer(cfg, 1)
    idx.update(tree_map(lambda p: (p * 8).astype(dtype), idx.parameters()))
    mx.eval(idx.parameters())
    return idx, cfg


class FakeCache:
    """The cache slice the indexer touches: the packed key slot and the handoff."""

    def __init__(self, keys, cand):
        self.keys = KeySlots([None, keys])
        self.index_k = None
        self.candidates = cand

    def append_index_keys(self, i, v):
        return self.keys.append_rows(i, v)


@pytest.mark.parametrize("cand_layer", [-1, 0, 1], ids=["plain", "consumer", "source"])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16], ids=["fp16", "bf16"])
def test_indexer_packed_keys(monkeypatch, cand_layer, dtype):
    """PACKED_KEYS on vs off through the fused indexer: same top-k set / same candidate mask (the fp16 arm is the same kernel's values, so the ids agree)."""
    idx, cfg = make_indexer(1, cand_layer, dtype)
    fused = FI._impl_indexer(L.Indexer.__call__)
    rng = np.random.default_rng(7 + cand_layer)
    for start_pos in (9, 70, 601, 3001):
        n = start_pos + 1
        x = mx.array(rng.standard_normal((1, 1, cfg.hidden_size)).astype(np.float32)).astype(dtype)
        qr = mx.array(rng.standard_normal((1, 1, cfg.q_lora_rank)).astype(np.float32)).astype(dtype)
        keys = fq4(mx.array(rng.standard_normal((1, n + 11, cfg.index_head_dim)).astype(np.float32)).astype(dtype))
        cand = mx.array(np.repeat(rng.random((1, 1, -(-n // 8))) < 0.7, 8, axis=-1)[..., :n]) if cand_layer == 0 else None
        out = {}
        for packed in (False, True):
            monkeypatch.setattr(FI, "PACKED_KEYS", packed)
            cache = FakeCache(keys, cand)
            r = fused(idx, x, qr, None, start_pos, 0, cache)
            mx.eval(r)
            out[packed] = (np.array(r), None if cache.candidates is None else np.array(cache.candidates))
        a, b = out[False][0].reshape(-1), out[True][0].reshape(-1)
        assert out[False][0].shape == out[True][0].shape == (1, 1, min(TOPK, n))
        if dtype == mx.float16:
            assert np.array_equal(a, b), (cand_layer, start_pos)                 # the packed kernel reproduces the dense rows: same selection
        else:
            assert len(set(a.tolist()) & set(b.tolist())) >= 0.995 * len(a), (cand_layer, start_pos)   # fp16 dense scores may take the NAX kernel (another head-sum order)
        if cand_layer >= 0:
            assert np.array_equal(out[False][1], out[True][1]), (cand_layer, start_pos)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16], ids=["fp16", "bf16"])
def test_indexer_packed_keys_reads_codes(monkeypatch, dtype):
    """The packed arm really takes the packed branch for both key dtypes."""
    idx, cfg = make_indexer(1, 1, dtype)
    fused = FI._impl_indexer(L.Indexer.__call__)
    rng = np.random.default_rng(5)
    start_pos, n = 401, 402
    x = mx.array(rng.standard_normal((1, 1, cfg.hidden_size)).astype(np.float32)).astype(dtype)
    qr = mx.array(rng.standard_normal((1, 1, cfg.q_lora_rank)).astype(np.float32)).astype(dtype)
    keys = fq4(mx.array(rng.standard_normal((1, n + 11, cfg.index_head_dim)).astype(np.float32)).astype(dtype))
    cache = FakeCache(keys, None)
    entry = cache.keys.raw(1)
    assert entry is not None and entry.mirror is None and entry.n == n + 11
    monkeypatch.setattr(FI, "PACKED_KEYS", True)
    r = fused(idx, x, qr, None, start_pos, 0, cache)
    mx.eval(r)
    assert cache.keys.raw(1).mirror is not None                             # the handoff publish is the only dense read left (O(new rows))
    assert cache.index_k is not None and cache.index_k.shape[1] == n + 11


# ----------------------------------------------------------------------------------------------------------------- E9
@pytest.mark.parametrize("kind", ["rand", "quant"])
def test_iscore_kb_short_context(monkeypatch, kind):
    """ISCORE_KB_MIN = 0 (the lever default): the register-blocked kernel runs from a short key count and returns the scalar kernel's scores."""
    rng = np.random.default_rng(11)
    N, NH, DI, cap = 3, 32, 128, 900
    q = rng.standard_normal((N, NH, DI)).astype(np.float16)
    keys = (rng.standard_normal((N, cap, DI)) * 0.5).astype(np.float16)
    w = (rng.standard_normal((N, NH)) * 0.1).astype(np.float16)
    if kind == "quant":
        q, keys = np.round(q).astype(np.float16), np.round(keys * 2).astype(np.float16) / 2
    lens = mx.array([1, 333, 897], dtype=mx.int32)
    cd = mx.array(rng.random((N, cap)) < 0.4)
    args = (mx.array(q), mx.array(keys), mx.array(w), lens, 900, cd, None, False)
    monkeypatch.setattr(C, "ISCORE_KB", False)
    ref = np.array(C.index_scores(*args).astype(mx.float32))
    monkeypatch.setattr(C, "ISCORE_KB", True)
    assert C.ISCORE_KB_MIN == 0                                             # short contexts: no key-count gate by default
    hit = LC.HITS["iscore_kb"]
    got = np.array(C.index_scores(*args).astype(mx.float32))
    assert LC.HITS["iscore_kb"] == hit + 1
    for u, n in enumerate([1, 333, 897]):
        assert np.array_equal(ref[u, :n], got[u, :n], equal_nan=True), (u, kind)
    monkeypatch.setattr(C, "ISCORE_KB_MIN", 1000)                           # the gate still holds it above a measured context
    C.index_scores(*args)
    assert LC.HITS["iscore_kb"] == hit + 1


# ----------------------------------------------------------------------------------------------------------------- E7
def _vendor_build():
    return LC._orig.get("cached", L._cached_index_cos_sin)


def test_rope_table_ahead_of_use(monkeypatch):
    """ROPE_PF: the next power-of-two table is built on a background thread at 3/4 use; the step that outgrows the table joins it and slices the same rows."""
    orig = _vendor_build()
    monkeypatch.setitem(LC._orig, "cached", orig)
    monkeypatch.setattr(LC, "ROPE_PF", True)
    calls = []
    build = LC._build

    def counted(length, *a):
        calls.append(length)
        return build(length, *a)

    monkeypatch.setattr(LC, "_build", counted)
    LC._tables.clear()
    LC._pf.clear()
    theta, yarn = 160000.0, (16.0, 32, 1, 65536)
    try:
        key = (64, theta, yarn)
        cos, sin = LC.index_cos_sin(3072, 64, theta, yarn)                   # 3/4 of the first table (4096): the 8192 build starts
        assert LC._tables[key][0] == 4096
        assert key in LC._pf and LC._pf[key][0] == 8192
        for _ in range(4000):                                              # the thread is a numpy build of 8192 rows: short
            if LC._pf[key][2]:
                break
            time.sleep(0.001)
        assert LC._pf[key][2], "prefetch did not finish"
        c2, s2 = LC.index_cos_sin(5000, 64, theta, yarn)                     # outgrows 4096: joins the finished build, no second build
        assert LC._tables[key][0] == 8192 and key not in LC._pf
        assert sorted(calls) == [4096, 8192], calls
        oc, os_ = orig(5000, 64, theta, yarn)
        assert np.array_equal(np.array(c2), np.array(oc))
        assert np.array_equal(np.array(s2), np.array(os_))
        assert np.array_equal(np.array(cos), np.array(oc)[:3072])
        for n in (1, 4096, 4097, 9000):                                     # every slice still equals the vendor rows
            a, b = LC.index_cos_sin(n, 64, theta, yarn)
            oa, ob = orig(n, 64, theta, yarn)
            assert np.array_equal(np.array(a), np.array(oa)) and np.array_equal(np.array(b), np.array(ob))
    finally:
        for _, thread, _ in list(LC._pf.values()):
            thread.join()
        LC._pf.clear()
        LC._tables.clear()
