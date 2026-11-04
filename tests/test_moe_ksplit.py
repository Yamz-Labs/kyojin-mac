"""KYOJIN_MOE_KSPLIT (per-bitrate k-split of the per-pair gather GEMV) and KYOJIN_MOE_TGT (grid target of pick_tgk).
Everything here runs without Metal: the split plan, the generated source, the numpy emulation of the kernel (emulate.gemv_partial with `act`) and the
parameter arrays MoeGather builds. The Metal run (old vs new path, error against an f32 reference) is tools/moe_k_bench.py on the Mac."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from kyojin_mlx import emulate as E, flags as FL, metal_src as S, moe as M  # noqa: E402


@pytest.fixture(autouse=True)
def restore():
    t, k = M.TGT, M.KSPLIT
    yield
    M.TGT, M.KSPLIT = t, k


@pytest.mark.parametrize("kt", [144, 320, 64])
@pytest.mark.parametrize("tgk", [1, 2, 4, 6])
@pytest.mark.parametrize("mode", [1, 2])
def test_plan_follows_the_bitrate(kt, tgk, mode):
    grid, act = M.ksplit_plan((2, 3), kt, tgk, mode)
    assert len(act) == 2 and all(1 <= a <= max(1, kt // 4) for a in act)
    assert act[0] <= act[1] <= grid                                   # the higher bitrate never splits less, and fits the grid
    if mode == 1:
        assert grid == tgk and act[1] == min(tgk, max(1, kt // 4))
    else:
        assert act[0] == min(tgk, max(1, kt // 4))
        assert grid == act[1]


def test_plan_single_bitrate_is_the_stock_split():
    for mode in (1, 2):
        assert M.ksplit_plan((3,), 320, 4, mode) == (4, (4,))


def test_plan_equalises_bytes_per_threadgroup():
    """tiles per threadgroup x K: the stock split gives K=3 1.5x the bytes of K=2, mode 1 / 2 bring them close."""
    kt, tgk = 320, 4
    def per_tg(k, a):
        return k * -(-kt // (a * 4))
    stock = per_tg(3, tgk) / per_tg(2, tgk)
    for mode in (1, 2):
        _, (a2, a3) = M.ksplit_plan((2, 3), kt, tgk, mode)
        assert abs(per_tg(3, a3) / per_tg(2, a2) - 1) < abs(stock - 1)


def test_source_without_the_lever_is_the_stock_text():
    s0 = S.gather_gemv_source((2, 3), 2, "mul1", True)
    assert s0 == S.gather_gemv_source((2, 3), 2, "mul1", True, False)
    assert "TGA" not in s0 and "const uint S = TGK * WS;" in s0


@pytest.mark.parametrize("ks", [(3,), (2, 3), (2, 3, 4)])
@pytest.mark.parametrize("pf", [False, True])
def test_source_with_the_lever(ks, pf):
    s = S.gather_gemv_source(ks, 2, "mul1r1", pf, True)
    assert s.count("const uint TGA = ") == 1 and "const uint S = TGA * WS;" in s and "const uint S = TGK * WS;" not in s
    assert s.index("TGA") > s.index("const uint K = uint(kk[e]);")                      # K is known where the split is chosen
    assert "if (kblock >= TGA)" in s and "return;" in s
    sel = S.ksplit_sel(ks)
    assert sel.count("params[") == len(ks) and f"params[{4 + len(ks) - 1}]" in sel
    assert sel.count("(") == sel.count(")")
    # everything else is the stock kernel: dropping the inserted block gives it back
    a = S.gather_gemv_source(ks, 2, "mul1r1", pf)
    b = s.replace(s[s.index("    const uint TGA"):s.index("    const uint S = TGA * WS;\n")], "").replace("TGA * WS", "TGK * WS")
    assert a == b


def test_sel_picks_the_right_slot():
    sel = S.ksplit_sel((2, 3, 4))
    assert sel == "(K == 2u ? params[4] : (K == 3u ? params[5] : params[6]))"


@pytest.mark.parametrize("K", [2, 3])
def test_emulated_split_sums_the_same_dot(K):
    """A bitrate-specific active count gives the same dot product as the uniform split (f32 reordering only), and the slices beyond it stay zero."""
    rng = np.random.default_rng(K)
    kt, nt = 16, 8
    words = rng.integers(0, 2 ** 32, (kt * nt, 8 * K), dtype=np.uint64).astype(np.uint32)
    xh = rng.standard_normal((1, kt * 16)).astype(np.float32)
    full = E.gemv_partial(words, K, kt, nt, xh, 4)
    for act in (1, 2, 3):
        part = E.gemv_partial(words, K, kt, nt, xh, 4, act=act)
        assert not part[act:].any()
        ref, got = full.sum(0).astype(np.float64), part.sum(0).astype(np.float64)
        assert np.abs(got - ref).max() <= 2e-5 * np.abs(ref).max()


def _stack(kt, nt, ks):
    ne = len(ks)
    return M.Stack(mx.zeros((8,), mx.uint32), mx.zeros((ne + 1,), mx.uint32), mx.array(np.array(ks, np.uint8)),
                   mx.zeros((ne, kt * 16), mx.float16), mx.zeros((ne, nt * 16), mx.float16), kt, nt, "mul1")


def _gather(ks=(2, 3, 2, 3)):
    g, u, d = _stack(320, 144, list(ks)), _stack(320, 144, list(ks)), _stack(144, 320, list(ks))
    return M.MoeGather(g, u, d, 6)


def test_params_stock_have_four_entries_and_flags_cache_apart():
    m = _gather()
    M.TGT, M.KSPLIT = 0, 0
    p0 = m._params(1)
    assert int(p0["gemv_gu"].size) == 4 and int(p0["gemv_dn"].size) == 4
    M.KSPLIT = 1
    p1 = m._params(1)
    assert p1 is not p0 and int(p1["gemv_gu"].size) == 4 + 2 and int(p1["gemv_dn"].size) == 4 + 2
    assert p1["tgu"] == p0["tgu"] and np.array(p1["gemv_gu"]).tolist()[:4] == np.array(p0["gemv_gu"]).tolist()
    M.KSPLIT = 2
    p2 = m._params(1)
    assert p2["tgu"] >= p0["tgu"] and p2["tgu"] == max(np.array(p2["gemv_gu"]).tolist()[4:])
    M.KSPLIT = 0
    assert m._params(1) is p0


def test_params_ksplit_only_for_the_gather_path():
    m = _gather()
    M.KSPLIT = 1
    assert int(m._params(2)["gemv_gu"].size) == 4                       # rows >= 2 and grouped (default): the grouped GEMV keeps its split


def test_pick_tgk_target():
    M.TGT = 0
    base = M.pick_tgk(320, 18, 12)
    assert base == 4
    M.TGT = 100
    assert M.pick_tgk(320, 18, 12) == 1
    M.TGT = 4000
    assert M.pick_tgk(320, 18, 12) > base
    assert M.pick_tgk(320, 18, 12, target=480) == base                    # an explicit target still wins


def test_flags_are_registered_and_aligned():
    for n, attr in (("moe_tgt", "TGT"), ("moe_ksplit", "KSPLIT")):
        f = FL.FLAGS[n]
        assert f.default == "0" and f.kind == "int" and f.runtime
        assert f.attrs == (("moe", attr, int),)
    import os, re
    src = open(M.__file__).read()
    assert re.search(r'KYOJIN_MOE_TGT", "0"', src) and re.search(r'KYOJIN_MOE_KSPLIT", "0"', src)
