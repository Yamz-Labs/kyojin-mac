"""KYOJIN_DEC_IDX_I8 (kyojin_mlx.decidx8): the int8 first pass of the decode index scores against the exact path it stands in for.

Pins (1) the block ranking + the exact rescore give the full-width path's top-k - plain rows, rows of several sessions, short rows, rows masked by
a candidate mask - for the margins 64 / 128 / 256, (2) a candidate-source layer rebuilds the same kept block mask and the same top-k, (3) the int8
codes (`Pack8`) hold what a fresh `pf_flat.i8pack` holds for the committed rows and are repacked for the rows an open pass may still change,
(4) the gate: fp16 32 x 128 on the runtime width at >= dec_idx_i8_min keys, a mask only at its own block size, nothing off the int8 grid, and the
exact path everywhere else, (5) the kernel source keeps the prefill int8 kernel's arithmetic with the decode row axis patched in, and flag 0 changes
nothing. One divergence is pinned too: a masked row with fewer kept keys than index_topk gets a -1 tail, where the full-width select fills up with
the (masked, -inf) keys below its count - `decfilter` and `idxblocks` return the same -1 tail.

Keys and queries are on the FP4 grid (mags x a power of two per 32 dims: the shape `rot_fq4` / `fake_quant_fp4_ue8m0` leaves), which is what makes
one int8 code per key lossless (`pf_flat.i8_on_grid`).
"""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_vlm.models.deepseek_v41.fakequant")

from kyojin_mlx import decidx8 as D8                                                # noqa: E402
from kyojin_mlx import dsa_decode as D                                              # noqa: E402
from kyojin_mlx import dsv4_core as C                                               # noqa: E402
from kyojin_mlx import flags as FL                                                  # noqa: E402
from kyojin_mlx import idxfast as IX                                                # noqa: E402
from kyojin_mlx import pf_flat as PF                                                # noqa: E402
from test_pf_flat import _fp4_like                                                  # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
NH, DI, BS, TOPK = 32, 128, 8, 32
CAP = 8192                                                                          # keys per session in every world (well under the 64K limit)
MARGINS = (64, 128, 256)


def world(rows=3, cap=CAP, S=2, lens=None, sess=None, seed=3):
    rng = np.random.default_rng(seed)
    q = mx.array(_fp4_like(rng, (rows, NH, DI)))
    w = mx.array((rng.standard_normal((rows, NH)) * 0.05).astype(np.float16))
    keys = mx.array(_fp4_like(rng, (S, cap, DI)))
    lens = [cap, cap - 40, cap // 2, cap // 3][:rows] if lens is None else list(lens)
    sess = [u % S for u in range(rows)] if sess is None else list(sess)
    a = (q, w, keys, mx.array(np.array(lens, dtype=np.int32)), mx.array(np.array(sess, dtype=np.int32)))
    mx.eval(*a)
    return a


def ours(q, w, keys, lens, sess, cand=None, kb=0, topk=TOPK, pack=None, committed=None, margin=None, t_max=None):
    """decidx8.select at the margin under test (a fresh pack unless one is given, committed = nothing packed yet)."""
    was, D8.MARGIN = D8.MARGIN, (D8.MARGIN if margin is None else margin)
    try:
        pk = pack if pack is not None else D8.Pack8(keys.shape[0])
        return D8.select(q, keys, w, lens, sess, keys.shape[1] if t_max is None else t_max, topk, pk,
                         [0] * keys.shape[0] if committed is None else committed, cand, kb=kb)
    finally:
        D8.MARGIN = was


def exact(q, w, keys, lens, sess, cand=None, topk=TOPK):
    return C.index_topk_rt(D.scores_rows(q, w, keys, lens, keys.shape[1], cand, sess), lens, topk)


def overlap(got, ref, k=TOPK):
    """Share of the k selected keys that agree, per row (the -1 tail of either side dropped)."""
    a, b = np.array(got), np.array(ref)
    return [len(set(a[u][a[u] >= 0].tolist()) & set(b[u][b[u] >= 0].tolist())) / k for u in range(a.shape[0])]


@pytest.fixture(autouse=True)
def small_gate(monkeypatch):
    monkeypatch.setattr(D8, "ENABLED", True)                                        # flag 0 is pinned in its own test
    monkeypatch.setattr(D8, "MIN_KEYS", 0)                                          # the worlds are 8K keys; the real gate is 262144
    monkeypatch.setattr(PF, "_I8_GRID", True)


# ----------------------------------------------------------------------------------------------------------------- selection vs the exact path
@pytest.mark.parametrize("margin", MARGINS)
@pytest.mark.parametrize("seed", [3, 4])
def test_plain_rows_are_the_exact_selection(margin, seed):
    """Unmasked rows of two sessions, three lengths: the top-k is the full-width path's, index for index (overlap 1.0)."""
    q, w, keys, lens, sess = world(seed=seed)
    ref = exact(q, w, keys, lens, sess)
    got, mask = ours(q, w, keys, lens, sess, margin=margin)
    mx.eval(got, ref)
    assert mask is None
    assert np.array_equal(np.array(got), np.array(ref))
    assert overlap(got, ref) == [1.0] * q.shape[0]


@pytest.mark.parametrize("margin", MARGINS)
def test_masked_rows_are_the_exact_selection(margin):
    """A consumer layer ranks only the blocks its source kept: the same top-k as the masked full-width path, for four rows of two sessions."""
    q, w, keys, lens, sess = world(rows=4, lens=[CAP, CAP - 40, 4096, 2048], seed=5)
    rng = np.random.default_rng(12)
    blocks = rng.random((4, CAP // BS)) < 0.25
    blocks[:, 0] = True
    cand = np.repeat(blocks, BS, axis=1)
    for u in range(4):                                                              # undefined by contract past the row's count: never read
        cand[u, int(np.array(lens)[u]):] = rng.random(CAP - int(np.array(lens)[u])) < 0.5
    cand = mx.array(cand)
    ref = exact(q, w, keys, lens, sess, cand)
    got, mask = ours(q, w, keys, lens, sess, cand, margin=margin)
    mx.eval(got, ref)
    assert mask is None
    assert np.array_equal(np.array(got), np.array(ref))
    assert overlap(got, ref) == [1.0] * q.shape[0]
    for u in range(4):                                                              # every selected key is a key the mask kept
        sel = np.array(got)[u]
        assert np.all(np.array(cand)[u][sel[sel >= 0]])


def test_candidate_source_is_the_exact_selection():
    """The source's top-k and its kept block mask are the mask `select_fused` builds from the exact scores (last visible block pinned, ties low)."""
    q, w, keys, lens, sess = world(seed=7)
    kb = 64
    ref_s = D.scores_rows(q, w, keys, lens, keys.shape[1], None, sess)
    ref, refm = IX.select_fused(ref_s, lens, TOPK, kb, BS)
    for margin in MARGINS:
        got, mask = ours(q, w, keys, lens, sess, kb=kb, margin=margin)
        mx.eval(got, mask, ref, refm)
        assert np.array_equal(np.array(got), np.array(ref))
        assert np.array_equal(np.array(mask), np.array(refm))
        assert overlap(got, ref) == [1.0] * q.shape[0]
        assert np.all(np.array(mask)[:, int(np.array(lens).max()):] == 0)            # nothing past the rows' counts


@pytest.mark.parametrize("lens", [[8, 0, 3, 5000], [1, 1, 1, CAP]])
def test_short_rows(lens):
    """A row shorter than one block, an empty row and a row of one key: the same answer as the full-width path (all its keys, then the -1 tail)."""
    q, w, keys, lens, sess = world(rows=4, lens=lens, seed=9)
    ref = exact(q, w, keys, lens, sess)
    got, _ = ours(q, w, keys, lens, sess)
    mx.eval(got, ref)
    assert np.array_equal(np.array(got), np.array(ref))


def test_a_width_off_the_block_and_tile_grid():
    """t_max neither a multiple of the 8-key block nor of the 32-key tile: the tail tile and the partial last block hold the full-width answer."""
    q, w, keys, lens, sess = world(rows=3, cap=8250, lens=[8203, 8000, 4003], seed=27)
    ref = C.index_topk_rt(D.scores_rows(q, w, keys, lens, 8203, None, sess), lens, TOPK)
    got, _ = ours(q, w, keys, lens, sess, t_max=8203)
    mx.eval(got, ref)
    assert np.array_equal(np.array(got), np.array(ref))


def test_masked_row_with_few_kept_keys_fills_with_minus_one():
    """Pinned divergence (decfilter / idxblocks return the same): the exact select fills the tail with the masked (-inf) keys below the row's count,
    this path leaves -1 there. The finite part is the exact path's."""
    q, w, keys, lens, sess = world(rows=2, lens=[CAP, 100], seed=11)
    rng = np.random.default_rng(2)
    blocks = rng.random((2, CAP // BS)) < 0.25
    blocks[:, 0] = True
    blocks[1, 1:13] = False                                                         # row 1 keeps ONE block: 8 keys < topk
    cand = mx.array(np.repeat(blocks, BS, axis=1))
    ref = np.array(exact(q, w, keys, lens, sess, cand))
    got = np.array(ours(q, w, keys, lens, sess, cand)[0])
    np.testing.assert_array_equal(got[0], ref[0])
    fin = got[1][got[1] >= 0]
    assert len(fin) == 8 and np.array_equal(np.sort(fin), ref[1][:8])                # the 8 kept keys, ascending
    assert np.all(got[1][8:] == -1)                                                  # where the exact path pads with masked indices
    assert np.all(np.array(cand)[1][ref[1][8:]] == 0)


# ----------------------------------------------------------------------------------------------------------------- the int8 codes
def test_pack_holds_what_a_fresh_pack_holds():
    """The cached codes of the committed rows are a fresh `pf_flat.i8pack` of them, bit for bit; an append packs only the new rows, and a wider
    width grows the buffer (a copy of the old one) without losing a row."""
    q, w, keys, lens, sess = world(rows=2, seed=13)
    pk = D8.Pack8(2)
    ki, ke = pk.get(keys, [CAP // 2, CAP // 2], CAP // 2)                           # the narrow call allocates, the wider one grows and copies
    mx.eval(ki, ke)
    c0, e0 = PF.i8pack(keys[0, :CAP // 2])
    mx.eval(c0, e0)
    np.testing.assert_array_equal(np.array(ki[0, :CAP // 2]), np.array(c0))
    ki, ke = pk.get(keys, [CAP // 2, CAP], CAP)
    mx.eval(ki, ke)
    c0, e0 = PF.i8pack(keys[0, :CAP // 2])
    c1, e1 = PF.i8pack(keys[1, :CAP])
    mx.eval(c0, e0, c1, e1)
    np.testing.assert_array_equal(np.array(ki[0, :CAP // 2]), np.array(c0))
    np.testing.assert_array_equal(np.array(ke[1, :CAP]), np.array(e1))
    assert pk.have == [CAP // 2, CAP] and D8.HITS["pack_rows"] >= CAP + CAP // 2
    before = D8.HITS["pack_open"]
    ki2, _ = pk.get(keys, [CAP // 2, CAP], CAP)
    mx.eval(ki2)
    assert np.array_equal(np.array(ki2[1]), np.array(c1))                            # a second call packs no committed row again
    assert D8.HITS["pack_open"] == before + CAP // 2                                 # only row 0's open rows (CAP - CAP // 2) are repacked


def test_pack_repacks_the_open_rows_and_resets_on_a_rollback():
    """The rows past the committed count may change (a rejected pass row is overwritten): they are repacked at every call, and a shorter
    committed count drops the cache instead of trusting it."""
    q, w, keys, lens, sess = world(rows=2, seed=17)
    pk = D8.Pack8(2)
    pk.get(keys, [CAP // 2, CAP // 2], CAP)
    mx.eval(pk.ki)
    changed = keys * 2.0                                                            # what a rolled-back pass leaves behind
    ki, _ = pk.get(changed, [CAP // 2, CAP // 2], CAP)
    mx.eval(ki)
    c, _ = PF.i8pack(changed[0, CAP // 2:CAP])
    mx.eval(c)
    np.testing.assert_array_equal(np.array(ki[0, CAP // 2:CAP]), np.array(c))
    c, _ = PF.i8pack(keys[0, :CAP // 2])                                            # the committed rows are kept as they were
    mx.eval(c)
    np.testing.assert_array_equal(np.array(ki[0, :CAP // 2]), np.array(c))
    pk.get(changed, [16, 16], CAP)
    assert pk.have == [16, 16]
    ki, _ = pk.get(changed, [16, 16], CAP)
    mx.eval(ki)
    c, _ = PF.i8pack(changed[1, 16:CAP])
    mx.eval(c)
    np.testing.assert_array_equal(np.array(ki[1, 16:CAP]), np.array(c))


def test_a_warm_cache_gives_the_same_answer_as_the_transient_one():
    """The cached path (two passes over the same keys, one with the committed counts) selects what the transient path selects."""
    q, w, keys, lens, sess = world(seed=19)
    ref = np.array(ours(q, w, keys, lens, sess)[0])
    pk = D8.Pack8(2)
    ours(q, w, keys, lens, sess, pack=pk, committed=[CAP - 64, CAP - 64])
    got = np.array(ours(q, w, keys, lens, sess, pack=pk, committed=[CAP - 64, CAP - 64])[0])
    np.testing.assert_array_equal(got, ref)


# ----------------------------------------------------------------------------------------------------------------- the gate
def idxr(src=False, cons=False, nh=NH, dim=DI, topk=TOPK, bs=BS, kb=64):
    from types import SimpleNamespace
    return SimpleNamespace(n_heads=nh, index_head_dim=dim, index_topk=topk, candidate_block_size=bs, candidate_topk_blocks=kb,
                           is_candidate_source=src, uses_candidates=cons)


def gate(q, w, keys, lens, sess, cand=None, t_max=None, **kw):
    return D8.applies(idxr(**kw), keys, cand, q, w, lens, sess, keys.shape[1] if t_max is None else t_max)


def test_applies_takes_the_supported_layer():
    q, w, keys, lens, sess = world(rows=2, seed=21)
    assert gate(q, w, keys, lens, sess)
    assert gate(q, w, keys, lens, sess, src=True, kb=64)


def test_applies_refuses_what_the_kernels_do_not_serve(monkeypatch):
    q, w, keys, lens, sess = world(rows=2, seed=23)
    cand = mx.zeros((2, CAP), dtype=mx.bool_)
    assert not gate(q, w, keys, lens, sess, nh=16)                                   # another head shape
    assert not gate(q, w, keys, lens, sess, dim=64)
    assert not gate(q.astype(mx.float32), w, keys, lens, sess)                       # another dtype than the keys
    assert not gate(q, w, keys.astype(mx.bfloat16), lens, sess)
    assert not gate(q, w, keys, lens, sess, cons=True, bs=16, cand=cand)             # a mask of another block size
    assert not gate(q, w, keys, lens, sess, cons=True, cand=cand[:, :CAP // 2])      # a mask narrower than t_max
    assert not gate(q, w, keys, lens, sess, cand=cand, src=True, kb=64)              # the source decides the mask, it has none
    assert not gate(q, w, keys, lens, sess, cand=cand, cons=True, topk=CAP + 8)      # topk wider than the keys
    assert not gate(q, w, keys[:, :, :64], lens, sess)                               # not the 128 channels
    assert not gate(q, w, keys[:, :CAP // 2], lens, sess, t_max=CAP)                 # t_max past the key buffer
    assert not gate(q, w, keys, lens, sess, src=True, kb=1)                          # the source keeps >= 2 blocks
    assert not gate(q, w, keys, lens, sess, t_max=16)                                # narrower than the kernel's 32-key tile
    monkeypatch.setattr(D8, "MIN_KEYS", CAP + 1)                                     # the width gate
    assert not gate(q, w, keys, lens, sess)
    monkeypatch.setattr(D8, "MIN_KEYS", 0)
    monkeypatch.setattr(PF, "_I8_GRID", False)                                       # keys off the int8 grid: the half first pass stays
    assert not gate(q, w, keys, lens, sess)
    monkeypatch.setattr(PF, "_I8_GRID", True)
    assert gate(q, w, keys, lens, sess, cand=cand, cons=True)
    assert gate(q, w, keys, lens, sess)


def test_applies_is_off_at_flag_zero(monkeypatch):
    q, w, keys, lens, sess = world(rows=2, seed=25)
    monkeypatch.setattr(D8, "ENABLED", False)
    assert not gate(q, w, keys, lens, sess)
    st = type("St", (), {"kcnt": [[CAP] * 2], "i8k": {}})()
    from types import SimpleNamespace
    em = SimpleNamespace(lens_arr=lens)
    assert D8.layer(st, 0, idxr(), keys, None, q, w, em, CAP, sess) is None
    monkeypatch.setattr(D8, "ENABLED", True)
    got = D8.layer(st, 0, idxr(), keys, None, q, w, em, CAP, sess)
    assert got is not None and got[1] is None and np.array_equal(np.array(got[0]), np.array(exact(q, w, keys, lens, sess)))
    assert set(st.i8k) == {0}                                                        # keyed by layer, 132 B per key held


# ----------------------------------------------------------------------------------------------------------------- kernel source and flag table
def test_kernel_source_is_the_prefill_kernel_with_the_row_axis():
    m = PF.i8_tile_map()
    src = D8.bmax_source(m)
    assert src.count("lens[t]") == (2 if m else 1) and "kof" in src and "n + n0 < clen" not in src
    assert src.count("-INFINITY") >= 2 and "S[ulong(t) * Tc + n]" not in src         # the block write, not the full score row
    plain = PF.i8_source()
    for line in ("threadgroup int8_t Qs[32 * 128];", "decltype(tQ), int32_t>()", "s = metal::max(s, simd_shuffle_xor(s, 4u));"):
        assert line in src and line in plain                                         # the prefill arithmetic, unchanged
    if m:
        assert "simd_shuffle_xor(a, ushort(" in src                                  # the register post is kept
        assert "float4(E[n.x + kof], E[n.y + kof], E[n.z + kof], E[n.w + kof])" in src


def test_flag_table():
    f = FL.FLAGS["dec_idx_i8"]
    assert f.default == "0" and f.kind == "bool" and f.attrs == (("decidx8", "ENABLED", FL._on),)
    assert FL.FLAGS["dec_idx_i8_margin"].kind == "int" and FL.FLAGS["dec_idx_i8_margin"].default == "128"
    assert FL.FLAGS["dec_idx_i8_margin"].attrs == (("decidx8", "MARGIN", int),)
    assert FL.FLAGS["dec_idx_i8_min"].kind == "int" and FL.FLAGS["dec_idx_i8_min"].default == "262144"


def test_core_takes_the_pass_after_the_two_exact_levers_and_before_the_fused_select():
    """Where the branch sits in `core`: after dec_filter / iscore_blocks (the exact ones), before idx_fuse, and its state dies at the unstack."""
    import inspect
    src = inspect.getsource(C.core)
    assert 'FD.hit("dec_idx_i8")' in src and "f8 = IX8.layer(" in src
    assert "IX8.ENABLED and rt and fres is None and fblk is None" in src
    assert src.index("elif fblk is not None") < src.index("elif f8 is not None") < src.index("elif rt and fuse")
    assert "cache.i8k = {}" in inspect.getsource(C.stack_rows) and '"i8k"' in inspect.getsource(C._rows_cache_class)
    before = D8.ENABLED
    try:
        FL.set("dec_idx_i8", "1")                                                    # switchable at call time, like the other levers
        assert D8.ENABLED is True
        FL.set("dec_idx_i8", "0")
        assert D8.ENABLED is False
    finally:
        FL.set("dec_idx_i8", "1" if before else "0")
