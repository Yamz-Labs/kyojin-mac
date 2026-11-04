"""idx_short on any box (no Metal): the identity selection against the stock chain, the candidate mask against a numpy model of the block level, the gate,
and the wiring of the flag. The kernel-level and logits-level checks need Metal: tools/idx_short_probe.py."""
import types

import numpy as np
import pytest

import mlx.core as mx

from kyojin_mlx import idxshort as S


def stock_idx(scores, lens, k):
    """The stock chain (docstring of dsv4_core.index_topk_rt): where(sort(argsort(-scores)[:, :k]) < lens, ., -1), with -1 filling up to k."""
    sc = mx.array(scores)
    ids = mx.sort(mx.argsort(-sc.astype(mx.float32), axis=-1)[:, :k], axis=-1).astype(mx.int32)
    out = mx.where(ids < mx.array(lens, dtype=mx.int32)[:, None], ids, -1)
    if out.shape[1] < k:
        out = mx.concatenate([out, mx.full((out.shape[0], k - out.shape[1]), -1, dtype=mx.int32)], axis=1)
    return np.array(out)


def rt_model(scores, lens, k):
    """Numpy model of the rt select (_SELECT, BLK 0): the k best t < lens, descending score, ties to the lower index, in increasing index order, -1 tail."""
    out = np.full((scores.shape[0], k), -1, dtype=np.int32)
    for u, L in enumerate(lens):
        s = scores[u, :L].astype(np.float32)
        order = np.argsort(-s, kind="stable")[:k]
        sel = np.sort(order)
        out[u, :len(sel)] = sel
    return out


def make_scores(rng, n, tm, kind):
    if kind == "rand":
        s = rng.standard_normal((n, tm)).astype(np.float16)
    elif kind == "ties":
        s = rng.integers(-2, 3, (n, tm)).astype(np.float16)
    elif kind == "zeros":
        s = np.zeros((n, tm), np.float16)
        s[:, ::3] = -0.0
    else:                                                      # -inf at random valid columns (a candidate consumer whose mask dropped keys)
        s = rng.standard_normal((n, tm)).astype(np.float16)
        s[rng.random((n, tm)) < 0.4] = -np.inf
    return s


@pytest.mark.parametrize("kind", ["rand", "ties", "zeros", "dropped"])
@pytest.mark.parametrize("k", [1, 8, 33, 64])
def test_identity_equals_stock_chain_and_kernel_model(kind, k):
    rng = np.random.default_rng(k * 31 + len(kind))
    tm = 3 * k + 7
    lens = [0, 1, k - 1, k, max(k // 2, 1), 2]
    lens = [min(int(v), k) for v in lens]                    # the gate: lens <= k
    scores = make_scores(rng, len(lens), tm, kind)
    for u, L in enumerate(lens):                             # columns past a row's lens are -inf (index_scores(rt=True))
        scores[u, L:] = -np.inf
    got = np.array(S.short_idx(mx.array(lens, dtype=mx.int32), k))
    assert got.shape == (len(lens), k) and got.dtype == np.int32
    assert (got == stock_idx(scores, lens, k)).all()
    assert (got == rt_model(scores, lens, k)).all()


def test_stock_width_is_min_topk_tmax():
    """The stock (non rt) path builds [N, min(index_topk, t_max)]: the short idx at that width equals its chain too."""
    rng = np.random.default_rng(5)
    k, tm = 16, 10
    lens = [0, 3, 10, 7]
    scores = make_scores(rng, len(lens), tm, "rand")
    for u, L in enumerate(lens):
        scores[u, L:] = -np.inf
    w = min(k, tm)
    got = np.array(S.short_idx(mx.array(lens, dtype=mx.int32), w))
    assert (got == stock_idx(scores, lens, w)[:, :w]).all()


def test_beyond_the_gate_a_real_selection_drops_keys():
    """lens = k + 1: the stock chain drops a key, so no identity row can equal it and the gate refuses the case."""
    scores = np.zeros((1, 20), np.float16)
    scores[0, :9] = np.arange(9, dtype=np.float16)[::-1]       # key 8 is the lowest of 9 valid keys
    scores[0, 9:] = -np.inf
    ref = stock_idx(scores, [9], 8)
    assert ref[0].tolist() == list(range(8))                   # key 8 dropped: the row has 8 slots for 9 keys
    assert not S.applies(9, 8, enabled=True)
    assert S.applies(8, 8, enabled=True) and S.applies(1, 8, enabled=True)


def block_mask_model(scores, lens, topk_blocks, bs):
    """Numpy model of the BLK kernel (level one of the two-level top-k): blocks of bs keys, the last block pinned, the others ranked by block max
    (descending, ties to the lower block), kept = ranked in among topk_blocks - 1 and finite; mask[t] for t < lens."""
    n, tc = scores.shape
    out = np.zeros((n, tc), bool)
    for u, L in enumerate(lens):
        nel = -(-L // bs)
        if nel == 0:
            continue
        bm = np.array([scores[u, b * bs:min(L, (b + 1) * bs)].astype(np.float32).max() for b in range(nel)])
        nd = nel - 1
        order = np.argsort(-bm[:nd], kind="stable")[:topk_blocks - 1]
        keep = np.zeros(nel, bool)
        keep[order] = bm[order] > -np.inf
        keep[nel - 1] = bm[nel - 1] > -np.inf
        for b in range(nel):
            out[u, b * bs:min(L, (b + 1) * bs)] = keep[b]
    return out


@pytest.mark.parametrize("bs,tb", [(8, 2048), (8, 64), (4, 12), (8, 3)])
def test_all_mask_equals_block_level_when_blocks_fit(bs, tb):
    rng = np.random.default_rng(bs * 100 + tb)
    k = 512
    tc = 1100
    lens = [1, 7, 8, 9, 64, min(k, bs * tb), min(k, 200)]
    lens = [L for L in lens if -(-L // bs) <= tb and L <= k]
    scores = rng.standard_normal((len(lens), tc)).astype(np.float16)
    for u, L in enumerate(lens):
        scores[u, L:] = -np.inf
    ref = block_mask_model(scores, lens, tb, bs)
    m = np.array(S.all_mask(len(lens), tc))[:, 0, :]
    for u, L in enumerate(lens):
        assert m[u, :L].all() and ref[u, :L].all()


def test_mask_not_all_true_when_blocks_do_not_fit():
    """ceil(lens / bs) > topk_blocks: the block level drops blocks, the gate must refuse the source layer."""
    rng = np.random.default_rng(3)
    bs, tb, L = 4, 3, 40
    scores = rng.standard_normal((1, 64)).astype(np.float16)
    scores[0, L:] = -np.inf
    ref = block_mask_model(scores, [L], tb, bs)
    assert not ref[0, :L].all()
    assert not S.applies(L, 512, enabled=True, is_source=True, topk_blocks=tb, block_size=bs)
    assert S.applies(L, 512, enabled=True, is_source=True, topk_blocks=10, block_size=bs)


def test_gate_table():
    A = lambda *a, **k: S.applies(*a, enabled=True, **k)
    assert not S.applies(10, 512, enabled=False)                          # flag off
    assert not A(10, 512, canon=True)                                     # explicit state / verify replay
    assert not A(0, 512) and not A(10, 0)
    assert A(512, 512) and not A(513, 512)                                # <= not <
    assert not A(10, 512, is_source=True, topk_blocks=2048, block_size=8, ratios_uniform=False)
    assert not A(10, 512, is_source=True, topk_blocks=1, block_size=8)
    assert A(512, 512, is_source=True, topk_blocks=64, block_size=8)      # exactly 64 blocks
    assert not A(513, 512, is_source=True, topk_blocks=64, block_size=8)
    assert not A(512, 512, is_source=True, topk_blocks=63, block_size=8)


REAL = types.SimpleNamespace(                      # the DS4.1 layout of tools/ds41_glue_bench.py / ULTRA.md: ratio 2 on layers 2-19, ratio 1 on 20-39
    compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0], index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36], candidate_source_layer_id=20)


def test_ratios_uniform():
    assert S.ratios_uniform(REAL)                                      # the consumers 24..36 share the source's ratio 1; layers 2, 8, 14 never read its mask
    cfg = types.SimpleNamespace(compress_ratios=[0, 0, 2, 2, 1, 1], index_source_layer_ids=[4, 5], candidate_source_layer_id=4)
    assert S.ratios_uniform(cfg)
    cfg.index_source_layer_ids = [4, 5, 3]
    cfg.compress_ratios = [0, 0, 2, 2, 1, 2]                           # a consumer on another ratio: its lens differ from the source's
    assert not S.ratios_uniform(cfg)
    cfg.candidate_source_layer_id = -1
    assert not S.ratios_uniform(cfg)
    assert not S.ratios_uniform(types.SimpleNamespace())


def _em(lens):
    return types.SimpleNamespace(lens=list(lens), tmax=max(lens), lens_arr=mx.array(list(lens), dtype=mx.int32))


def _idxr(source=False, cand=False, topk=512):
    d = dict(index_topk=topk, is_candidate_source=source)
    if cand:
        d.update(candidate_topk_blocks=2048, candidate_block_size=8)
    return types.SimpleNamespace(**d)


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(S, "ENABLED", True)


def test_decide_real_ds41_layers(on):
    """Every indexer layer of the real layout, a context of 300 / 600 / 1100 keys: which layers skip, and the outputs."""
    em = _em([300])
    plain = _idxr()                                                  # a layer that carries no candidate_* attribute at all
    src = _idxr(source=True, cand=True)
    for rt, width, tm in ((True, 512, 1024), (False, 300, 300)):      # stock path: min(K, t_max) with t_max = the bucketed count
        got = S.decide(plain, REAL, em, rt, tm, 1, False)
        assert got is not None and got[1] is None and tuple(got[0].shape) == (1, 1, width)
        assert np.array(got[0])[0, 0].tolist() == list(range(300)) + [-1] * (width - 300)
        got = S.decide(src, REAL, em, rt, tm, 1, False)           # layer 20: the source skips too
        assert got is not None and tuple(got[1].shape) == (1, 1, tm) and bool(np.array(got[1]).all())
    assert S.decide(plain, REAL, em, True, 1024, 1, True) is None    # explicit state / verify replay
    assert S.decide(plain, REAL, _em([513]), True, 1024, 1, False) is None
    assert S.decide(src, REAL, _em([513]), True, 1024, 1, False) is None
    assert S.decide(plain, REAL, _em([512, 100]), True, 1024, 2, False) is not None      # the max over the rows decides
    assert S.decide(plain, REAL, _em([512, 513]), True, 1024, 2, False) is None
    bad = types.SimpleNamespace(compress_ratios=[0, 0, 2, 1], index_source_layer_ids=[2, 3], candidate_source_layer_id=2)
    assert S.decide(src, bad, em, True, 1024, 1, False) is None      # a consumer on another ratio: the source must not skip
    assert S.decide(plain, bad, em, True, 1024, 1, False) is not None


def test_decide_off_by_default(monkeypatch):
    monkeypatch.setattr(S, "ENABLED", False)
    assert S.decide(_idxr(), REAL, _em([10]), True, 64, 1, False) is None


def test_idx_for_is_cached_per_pass_and_width():
    em = types.SimpleNamespace(lens_arr=mx.array([3, 9], dtype=mx.int32))
    a = S.idx_for(em, 12)
    assert S.idx_for(em, 12) is a and S.idx_for(em, 9) is not a
    assert np.array(a).tolist() == [[0, 1, 2] + [-1] * 9, list(range(9)) + [-1] * 3]
    assert S.all_mask(2, 16) is S.all_mask(2, 16) and tuple(S.all_mask(2, 16).shape) == (2, 1, 16)


def test_flag_registered_default_on_and_switchable():
    from kyojin_mlx import flags
    assert flags.FLAGS["idx_short"].default == "1" and flags.FLAGS["idx_short"].env == "KYOJIN_IDX_SHORT"
    old = S.ENABLED
    try:
        flags.set("idx_short", 1)
        assert S.ENABLED is True
        flags.set("idx_short", 0)
        assert S.ENABLED is False
    finally:
        S.ENABLED = old


def test_core_hook_is_wired_and_default_path_untouched():
    """dsv4_core imports the module and routes the skip only behind IDXS.ENABLED (no Metal needed to read the source)."""
    import inspect
    from kyojin_mlx import dsv4_core as C
    src = inspect.getsource(C.core)
    assert "IDXS.decide(idxr, cfg, em, rt, t_max, n, _CANON_POOLS) if idxr is not None and IDXS.ENABLED else None" in src and 'FD.hit("idx_short")' in src
    assert C.IDXS is S
