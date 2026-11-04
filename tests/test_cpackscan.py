"""cpackscan (KYOJIN_PF_IDX_CPACK): the packed-key bound scan of the plain prefill indexer layers, on the CPU (numpy scoring in the kernels' order, METAL_OPT False).

Order: the transient decode is the dense read bit for bit and builds no mirror; the grouped bound is sound, GROUP = 128 is the pf_bound bound, smaller groups are never looser;
the selected set equals the stock sort(argsort(-s)) on a case that prunes and on cases that fall back (isotropic keys, a missed first block); the wiring in
ds_pf_idx._impl_indexer on the mini DS4.1 model (index heads 32 x 128): logits flag 0 == flag 1, the path engaged, the plain layers' key mirrors never built.
Nothing here times anything: no claim about speed is made or implied."""
import os
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kyojin_mlx import cpackscan as CP      # noqa: E402
from kyojin_mlx import depthflat as DF      # noqa: E402
from kyojin_mlx import fp4pack              # noqa: E402
from kyojin_mlx import idx_topk as IT       # noqa: E402
from kyojin_mlx import pf_bound as PB       # noqa: E402
from kyojin_mlx import pf_flat as PF        # noqa: E402
from kyojin_mlx import pfroof2 as PR        # noqa: E402
from kyojin_mlx.fp4pool import KeySlots     # noqa: E402

NH, D = 32, 128
SB = 64


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setattr(PF, "METAL_OPT", False)
    monkeypatch.setattr(CP, "ENABLED", True)
    monkeypatch.setattr(CP, "MIN_KEYS", 0)
    monkeypatch.setattr(CP, "GROUP", 32)
    monkeypatch.setattr(PB, "SB", SB)
    monkeypatch.setattr(PB, "MAXPCT", 25)
    CP._state.clear()
    for k in CP.HITS:
        CP.HITS[k] = 0


def fp4(x):
    """Round an array to the vendor index-key grid (blocks of 32, power-of-two scale) and back to fp16."""
    xa = mx.array(np.asarray(x, dtype=np.float32))
    c, s = fp4pack.pack(xa, 32, False)
    return fp4pack.unpack(c, s, 32, False, mx.float16)


def rnd(T, Tc, seed=0, wscale=0.05):
    rng = np.random.default_rng(seed)
    q = fp4(rng.standard_normal((1, T, NH, D)) * 0.5)
    k = fp4(rng.standard_normal((1, Tc, D)))
    w = mx.array((rng.standard_normal((1, T, NH)) * wscale).astype(np.float16))
    return q, k, w


def clustered(T, Tc, seed=0, centres=48, noise=0.004, run=SB, noisy_dims=D):
    rng = np.random.default_rng(seed)
    q, _, w = rnd(T, Tc, seed)
    cen = rng.standard_normal((centres, D)).astype(np.float32)
    ids = rng.integers(0, centres, size=-(-Tc // run))
    nz = noise * rng.standard_normal((Tc, D)).astype(np.float32)
    nz[:, noisy_dims:] = 0.0                                                  # noise confined to the first `noisy_dims` dims (the rest of the vector is constant in a run)
    k = np.repeat(cen[ids], run, axis=0)[:Tc] + nz
    return q, fp4(k[None]), w


def scores(q, k, w, start, ratio, t0=0, n=None):
    n = q.shape[1] - t0 if n is None else n
    return PR.ref_scores(np.array(q[0, t0:t0 + n]), np.array(k[0]), np.array(w[0, t0:t0 + n]), start, ratio, t0)


def stock(q, k, w, start, ratio, topk):
    return lambda t0, n: IT.topk(mx.array(scores(q, k, w, start, ratio, t0, n)), topk)


def block_max(s, sb):
    T, Tc = s.shape
    nsb = -(-Tc // sb)
    sp = np.pad(s, [(0, 0), (0, nsb * sb - Tc)], constant_values=-np.inf)
    return sp.reshape(T, nsb, sb).max(-1)


def slots_with(keys):
    """A KeySlots holding `keys` [1, Tc, 128] (on the grid) as layer 0, in two appends like a two-chunk prefill."""
    s = KeySlots([None])
    h = keys.shape[1] // 2
    s.defer_dense = True
    s.append_rows(0, keys[:, :h])
    s.append_rows(0, keys[:, h:])
    s.defer_dense = False
    return s


# ----------------------------------------------------------------------------------------------------------------- storage
def test_decode_is_the_dense_read_and_builds_no_mirror():
    k = clustered(8, 1000)[1]
    s = slots_with(k)
    e = s.raw(0)
    assert e.mirror is None
    before = dict(DF.HITS)
    got = CP.decode(s, e, 1000, step=300)
    assert bool(mx.array_equal(got, k))
    assert e.mirror is None and DF.HITS["build"] == before["build"] and DF.HITS["tail"] == before["tail"]
    assert CP.HITS["decoded_rows"] == 1000


def test_defer_dense_keeps_the_append_from_building_the_mirror():
    k = clustered(8, 600)[1]
    s = KeySlots([None])
    s.append_rows(0, k[:, :300])
    assert s.append_rows(0, k[:, 300:400]) is not None                      # default: the dense layer comes back (and the mirror is built)
    assert s.raw(0).mirror is not None
    s2 = KeySlots([None])
    s2.defer_dense = True
    s2.append_rows(0, k[:, :300])
    s2.defer_dense = False
    with CP.deferred(type("C", (), {"keys": s2})(), True):
        assert s2.append_rows(0, k[:, 300:400]) is None
    assert s2.raw(0).mirror is None and s2.raw(0).n == 400
    assert bool(mx.array_equal(s2[0], k[:, :400]))                           # a later dense read still gives the rows (and builds the mirror then)
    assert not s2.defer_dense


def test_drop_mirror_frees_it_and_a_read_rebuilds_it():
    k = clustered(8, 700)[1]
    s = slots_with(k)
    _ = s[0]
    e = s.raw(0)
    assert CP.drop_mirror(e) > 0 and e.mirror is None and e.mn == 0
    assert bool(mx.array_equal(s[0], k))


# ----------------------------------------------------------------------------------------------------------------- bounds
def make_u(q, k, w, start, ratio, gsz, sb=SB):
    return np.array(CP.bounds(q, w, CP.summaries(k, sb, gsz), start, ratio, 0, q.shape[1]))


@pytest.mark.parametrize("T,Tc,ratio", [(24, 1000, 1), (16, 777, 2), (30, 2048, 4)])
@pytest.mark.parametrize("gsz", [128, 64, 32, 16])
def test_grouped_bound_is_never_below_a_score(T, Tc, ratio, gsz):
    q, k, w = rnd(T, Tc, seed=T, wscale=0.2)
    wn = np.array(w)
    wn[:, :, ::3] = 0.0
    wn[:, :, 1::3] = -np.abs(wn[:, :, 1::3])                                 # negative, zero and positive head weights
    w = mx.array(wn)
    start = ratio * (Tc - T)
    u = make_u(q, k, w, start, ratio, gsz)
    bm = block_max(scores(q, k, w, start, ratio), SB)
    fin = np.isfinite(bm)
    assert np.all(u[fin] >= bm[fin]), float((bm - u)[fin].max())
    assert np.all(np.isneginf(u[np.isneginf(bm)]))


def test_group_128_is_the_pf_bound_bound_and_smaller_groups_are_never_looser_than_it():
    q, k, w = clustered(20, 1500, seed=2, noise=0.3, noisy_dims=32)
    start = 1500 - 20
    ref = np.array(PB.bounds(q, w, PB.summaries(k, SB), start, 1, 0, 20))
    u128 = make_u(q, k, w, start, 1, 128)
    assert np.array_equal(u128, ref)
    fin = np.isfinite(u128)
    for g in (64, 32, 16):
        u = make_u(q, k, w, start, 1, g)
        assert np.all(u[fin] <= u128[fin] + 1e-3 * (1 + np.abs(u128[fin]))), g   # never looser than the whole-vector bound (fp32 slack only); not monotone between group sizes


def test_grouped_bound_prunes_at_least_as_much_on_loosely_clustered_keys():
    """The point of the groups: keys that are alike inside a group of dims but not over the whole vector. Survivor fractions only (no timing)."""
    T, Tc, topk = 24, 4096, 32
    q, k, w = clustered(T, Tc, seed=5, noise=0.1, centres=64, noisy_dims=32)
    start = Tc - T
    frac = {}
    for g in (128, 32):
        CP._state.clear()
        for h in CP.HITS:
            CP.HITS[h] = 0
        mp = pytest.MonkeyPatch()
        mp.setattr(CP, "GROUP", g)
        mp.setattr(PB, "MAXPCT", 100)
        try:
            got = CP.try_plain(q, k, w, start, 1, topk, T, key="g", stock=stock(q, k, w, start, 1, topk))
        finally:
            mp.undo()
        assert np.array_equal(np.array(got), np.array(stock(q, k, w, start, 1, topk)(0, T)))
        frac[g] = CP.survivor_fraction()
    print("survivor fraction by group size:", frac)
    assert frac[32] <= frac[128] + 1e-9
    assert frac[32] < frac[128], frac                                          # strictly tighter where the spread sits in one group of dims


# ----------------------------------------------------------------------------------------------------------------- selection
@pytest.mark.parametrize("T,Tc,topk,ratio", [(40, 3000, 32, 1), (40, 4100, 40, 2), (33, 5000, 24, 4)])
@pytest.mark.parametrize("gsz", [128, 32, 16])
def test_clustered_keys_prune_and_stay_exact(T, Tc, topk, ratio, gsz, monkeypatch):
    monkeypatch.setattr(CP, "GROUP", gsz)
    q, k, w = clustered(T, Tc, seed=T)
    start = ratio * (Tc - T)
    ref = np.array(stock(q, k, w, start, ratio, topk)(0, T))
    for qb in ("0", "11"):
        os.environ[IT.QBLOCK_ENV] = qb
        try:
            CP._state.clear()
            got = CP.try_plain(q, k, w, start, ratio, topk, T, key="L", stock=stock(q, k, w, start, ratio, topk))
        finally:
            os.environ.pop(IT.QBLOCK_ENV, None)
        assert got is not None and np.array_equal(np.array(got), ref)
    assert CP.HITS["fallback_blocks"] == 0
    assert CP.survivor_fraction() < 0.5                                       # the bound really pruned on keys that are clustered


def test_isotropic_keys_miss_on_the_first_block_and_leave_the_chunk_to_the_normal_chain():
    T, Tc, topk = 36, 4000, 32
    q, k, w = rnd(T, Tc)
    called = []
    got = CP.try_plain(q, k, w, Tc - T, 1, topk, T, key="L", stock=lambda t0, n: called.append(1))
    assert got is None and not called                                         # nothing computed twice: the caller runs its normal chain
    assert CP.HITS["missed_chunks"] == 1 and CP.HITS["fallback_blocks"] >= 1


def test_two_missed_chunks_in_a_row_skip_the_layer_for_a_while():
    T, Tc, topk = 36, 4000, 32
    q, k, w = rnd(T, Tc)
    for _ in range(2):
        assert CP.try_plain(q, k, w, Tc - T, 1, topk, T, key="L") is None
    assert CP._state["L"][1] == PB.SKIP_CHUNKS
    n = CP.HITS["chunks"]
    assert CP.try_plain(q, k, w, Tc - T, 1, topk, T, key="L") is None
    assert CP.HITS["chunks"] == n and CP.HITS["skipped_chunks"] == 1          # sat out, no work spent


def test_later_block_miss_runs_the_stock_block_and_stays_exact(monkeypatch):
    """The first query block is served by the bound, the second is forced to miss (survivors above the cap): the stock block runs for it, the answer is the stock answer."""
    T, Tc, topk = 44, 3600, 28
    q, k, w = clustered(T, Tc, seed=1)
    monkeypatch.setattr(PB, "MAXPCT", 100)
    real = CP._block
    calls = []

    def block(*a, **kw):
        calls.append(1)
        return real(*a, **kw) if len(calls) == 1 else None

    monkeypatch.setattr(CP, "_block", block)
    start = Tc - T
    ran = []
    st = stock(q, k, w, start, 1, topk)
    os.environ[IT.QBLOCK_ENV] = "20"
    try:
        got = CP.try_plain(q, k, w, start, 1, topk, T, key="M", stock=lambda t0, n: (ran.append((t0, n)), st(t0, n))[1])
    finally:
        os.environ.pop(IT.QBLOCK_ENV, None)
    ref = np.array(st(0, T))
    assert got is not None and np.array_equal(np.array(got), ref)
    assert len(calls) >= 2 and ran and ran[0][0] > 0 and CP.HITS["fallback_blocks"] == 0       # block 0 served by the bound, later blocks by the stock block


def test_ties_all_equal_scores_lowest_positions_win():
    T, Tc, topk = 24, 2048, 20
    k = mx.array(np.ones((1, Tc, D), dtype=np.float16) * 0.5)
    q = mx.array(np.ones((1, T, NH, D), dtype=np.float16) * 0.25)
    w = mx.array(np.ones((1, T, NH), dtype=np.float16) * 0.03125)
    ref = np.array(stock(q, k, w, Tc - T, 1, topk)(0, T))
    got = CP.try_plain(q, k, w, Tc - T, 1, topk, T, key="T", stock=stock(q, k, w, Tc - T, 1, topk))
    got = np.array(got) if got is not None else ref                           # a miss is also exact by construction (the caller runs the normal chain)
    assert np.array_equal(got, ref)


def test_causal_edge_rows_see_part_of_the_chunk(monkeypatch):
    T, Tc, topk, ratio = 48, 2600, 24, 2
    q, k, w = clustered(T, Tc, seed=8, noise=0.004, centres=30)
    monkeypatch.setattr(PB, "MAXPCT", 100)
    start = ratio * (Tc - T) + 1
    ref = np.array(stock(q, k, w, start, ratio, topk)(0, T))
    got = CP.try_plain(q, k, w, start, ratio, topk, T, key="E", stock=stock(q, k, w, start, ratio, topk))
    assert got is not None and np.array_equal(np.array(got), ref)


def test_group_size_falls_back_to_a_divisor():
    assert CP.group_size(128) == 32
    old = CP.GROUP
    try:
        CP.GROUP = 48
        assert CP.group_size(128) == 32
        CP.GROUP = 128
        assert CP.group_size(128) == 128
    finally:
        CP.GROUP = old


# ----------------------------------------------------------------------------------------------------------------- flags
def test_flag_table_row_default_zero_and_runtime_switch():
    from kyojin_mlx import flags as FL
    f = FL.FLAGS["pf_idx_cpack"]
    assert f.default == "0" and f.env == "KYOJIN_PF_IDX_CPACK" and f.runtime
    assert FL.FLAGS["pf_idx_cpack_min"].default == "131072" and FL.FLAGS["pf_idx_cpack_group"].default == "32"
    old = os.environ.get(f.env)
    try:
        FL.set("pf_idx_cpack", 1)
        assert CP.ENABLED is True
        FL.set("pf_idx_cpack", 0)
        assert CP.ENABLED is False
    finally:
        if old is None:
            os.environ.pop(f.env, None)
        else:
            os.environ[f.env] = old


# ----------------------------------------------------------------------------------------------------------------- wiring on the mini DS4.1 model
def _model():
    import mini_ds41 as M
    lm, cfg = M.make(seed=3, index_n_heads=32, index_head_dim=128, index_topk=16, candidate_topk_blocks=24, candidate_block_size=8, max_position_embeddings=1 << 15)
    from mlx.utils import tree_map
    for layer in lm.layers:                                                   # the indexers run in fp16 like the served model; the rest stays fp32 (CPU MoE gather)
        ix = getattr(layer.attn, "indexer", None)
        if ix is not None:
            ix.update(tree_map(lambda p: p.astype(mx.float16) if p.dtype == mx.float32 else p, ix.parameters()))
    mx.eval(lm.parameters())
    return lm, cfg


def _wire(monkeypatch, lm):
    """The prefill indexer hook on the CPU: the model stays fp32 (the CPU MoE gather is fp32 only), the hook sees fp16 x / qr / latent (as the served model gives it), the Metal
    rotary kernel is replaced by a deterministic fp4 round trip of the queries (both arms use the same one, so it cancels in the flag comparison)."""
    from kyojin_mlx import ds_pf_idx as PI
    mod = __import__(type(lm.layers[0].attn).__module__, fromlist=["Indexer"])
    patched = PI._impl_indexer(mod.Indexer.__call__)

    def call(self, x, qr, latent, start_pos, offset, cache):
        f = lambda a: None if a is None else a.astype(mx.float16)
        return patched(self, f(x), f(qr), f(latent), start_pos, offset, cache)

    monkeypatch.setattr(mod.Indexer, "__call__", call)
    monkeypatch.setattr(PI, "fq4_rot", lambda q, cos, sin, rd: fp4(np.array(q.astype(mx.float32))).reshape(q.shape))
    monkeypatch.setattr(PI, "ENABLED", True)
    monkeypatch.setattr(PF, "BMAX", True)
    monkeypatch.setattr(PF, "BMAX_MIN", 0)
    monkeypatch.setattr(PB, "MAXPCT", 100)                                    # random model keys prune nothing: let the bound path run to the end so the wiring is exercised
    return PI


def _prefill(lm, ids, chunk, PI):
    from kyojin_mlx import fp4pool
    cache = fp4pool.wrap_cache(lm.make_cache())
    out = None
    for i, s in enumerate(range(0, len(ids), chunk)):
        PI.ENABLED = i > 0                                                    # chunk 0 (no past, topk not reachable) runs the vendor chain; the hook serves the later ones
        out = lm(mx.array([ids[s:s + chunk]]), cache=cache).logits
        mx.eval(out)
    PI.ENABLED = True
    return cache[0], out[:, -1].astype(mx.float32)


def test_mini_model_logits_equal_flag_0_and_1_and_the_path_engaged(monkeypatch):
    try:
        lm, cfg = _model()
    except Exception as e:                                                    # the vendor package is not importable here
        pytest.skip(f"mini model unavailable: {e}")
    import mini_ds41 as M
    PI = _wire(monkeypatch, lm)
    ids = M.tokens(1536, seed=5)
    monkeypatch.setattr(CP, "ENABLED", False)
    c0, l0 = _prefill(lm, ids, 512, PI)
    mirror0 = CP.mirror_bytes(c0)
    monkeypatch.setattr(CP, "ENABLED", True)
    monkeypatch.setattr(CP, "MIN_KEYS", 256)
    CP._state.clear()
    for k in CP.HITS:
        CP.HITS[k] = 0
    c1, l1 = _prefill(lm, ids, 512, PI)
    assert CP.HITS["chunks"] + CP.HITS["missed_chunks"] > 0, f"the packed path never ran: {CP.HITS}"
    assert CP.HITS["mirror_avoided"] > 0 and CP.HITS["decoded_rows"] > 0
    assert bool(mx.array_equal(l0, l1)), "logits differ between flag 0 and flag 1"
    for i in range(len(c0.keys)):                                             # the stored keys are the same bits
        a, b = c0.keys.raw(i), c1.keys.raw(i)
        if a is not None:
            assert a.n == b.n and bool(mx.array_equal(a.codes[:, :a.n], b.codes[:, :b.n])) and bool(mx.array_equal(a.scales[:, :a.n], b.scales[:, :b.n]))
    assert CP.mirror_bytes(c1) < mirror0, (CP.mirror_bytes(c1), mirror0)
    assert c0.keys.raw(2).mirror is not None and c1.keys.raw(2).mirror is None      # layer 2 (plain, ratio 2): its mirror is gone; the candidate source (layer 4) keeps its own
    print("key mirror bytes after the prefill: flag 0 =", mirror0, " flag 1 =", CP.mirror_bytes(c1), CP.HITS)


def test_applies_before_the_decode_matches_applies_on_the_keys(monkeypatch):
    from kyojin_mlx import ds_pf_idx as PI
    q, k, w = rnd(8, 1500)
    ok = lambda keys, **kw: CP.applies(q, keys, w, **{"topk": 32, "start": 1400, "ratio": 1, **kw})
    for kw in ({}, {"topk": 2000}, {"start": 10}, {"ratio": 4}):
        assert ok(CP.shape_of(1500), **kw) == ok(k, **kw), kw
    assert ok(k)
    monkeypatch.setattr(CP, "ENABLED", False)
    assert not ok(CP.shape_of(1500))
    monkeypatch.setattr(CP, "ENABLED", True)
    monkeypatch.setattr(PI, "qmm_on", lambda: True)                           # different score numerics: the bound path stays out, nothing is decoded or dropped
    assert not ok(CP.shape_of(1500))
