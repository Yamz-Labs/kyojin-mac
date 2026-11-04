"""cpftrellis: two forms of the routed prefill expert GEMM (metal_nax.coop_source), both default 0.

pf_unroll2 (KYOJIN_NAX_UNROLL2): the k loop runs two steps per iteration with two alternating trellis word sets instead of copying one set into the next.
pf_bmu16 (KYOJIN_PF_BMU16): pick_cfg's row block is cut into 16-row matmul runs ((32, n, ..) -> (16, 2n, ..)).

Everything here but the last test runs on CPU (it checks generated source text, the control flow, the host plan and a count model). The last test needs Metal
and is skipped without it; the timings and the md5 against the stock kernel are tools/cpftrellis_gemm.py, run on the Mac.
"""
import os
import re
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from kyojin_mlx import flags as FL, metal_nax as NX  # noqa: E402

CFG = (32, 2, 32, 64)


def test_flags_default_off_and_wired():
    for name, env, attr in (("pf_unroll2", "KYOJIN_NAX_UNROLL2", "UNROLL2"), ("pf_bmu16", "KYOJIN_PF_BMU16", "BMU16")):
        f = FL.FLAGS[name]
        assert f.default == "0" and f.env == env and f.runtime
        assert f.attrs[0][0] == "metal_nax" and f.attrs[0][1] == attr
        assert getattr(NX, attr) is False
    levers = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "cloud", "LEVERS.md")).read()
    assert "| `KYOJIN_NAX_UNROLL2` | pf_unroll2 |" in levers and "| `KYOJIN_PF_BMU16` | pf_bmu16 |" in levers


@pytest.mark.parametrize("Ks", [(3,), (2, 3, 5), (6,)])
@pytest.mark.parametrize("cfg", [(32, 2, 32, 64), (16, 1, 32, 64), (32, 1, 32, 64), (32, 2, 16, 32), (16, 4, 32, 64)])
def test_flag_off_is_the_stock_source(Ks, cfg):
    """unroll off (the default) adds nothing: no wa / wb sets, the stock copy loop, and the same text whether the argument is omitted or False."""
    off = NX.coop_source(Ks, "mul1r2", *cfg)
    assert off == NX.coop_source(Ks, "mul1r2", *cfg, unroll=False)
    assert "wb[" not in off and "cw[" in off and "nw_[" in off


@pytest.mark.parametrize("Ks", [(3,), (2, 3, 5), (4, 6)])
@pytest.mark.parametrize("cfg", [(32, 2, 32, 64), (16, 4, 32, 64), (32, 1, 32, 64), (32, 2, 32, 32), (32, 2, 16, 64)])
def test_unroll_source_has_the_same_work(Ks, cfg):
    bmu, nmb, bns, bks = cfg
    off = NX.coop_source(Ks, "mul1r2", *cfg)
    on = NX.coop_source(Ks, "mul1r2", *cfg, unroll=True)
    assert on.count("{") == on.count("}") and off.count("{") == off.count("}")
    assert "cw[" not in on and "nw_[" not in on                              # the copy and its two arrays are gone
    # the stock body decodes nsub tiles once per step, the unrolled one twice (steps kt and kt + nk); the matmul runs likewise
    assert on.count("extract_bits(") == 2 * off.count("extract_bits(")
    assert on.count("op.run(") == 2 * off.count("op.run(") == 2 * nmb * len(Ks)
    assert on.count("insert_bits(") == 2 * off.count("insert_bits(")
    # loads: the unrolled body has three load sites (steps KT0, kt + nk, kt + 2 nk) where the stock one has two (KT0, kt + nk): same words per site
    assert on.count("= tp[a0_") * 2 == off.count("= tp[a0_") * 3
    # everything outside the k loop (prologue, zeroing, epilogue store) is the stock text
    head = lambda s: s[:s.index("uint B0") if "uint B0" in s else 0]
    assert head(on) == head(off)
    mark = "const auto ix = cT0.get_multidimensional_index(i); const uint m ="
    assert on[on.rindex(mark):] == off[off.rindex(mark):]                    # the epilogue store is the stock text


def _control(KT0, KT1, nk, unroll):
    """The (event, step) sequence of the generated k loop, written from the generated text's own guards: ('L', s) load of step s, ('D', s) decode, ('M', s) matmul."""
    ev = []
    if not unroll:                                              # stock: prefetch step KT0; per step: copy, load next if in range, decode, matmul
        if KT0 < KT1:
            ev.append(("L", KT0))
        for kt in range(KT0, KT1, nk):
            if kt + nk < KT1:
                ev.append(("L", kt + nk))
            ev += [("D", kt), ("M", kt)]
        return ev
    if KT0 < KT1:
        ev.append(("L", KT0))
    for kt in range(KT0, KT1, 2 * nk):
        two = kt + nk < KT1
        if two:
            ev.append(("L", kt + nk))
        ev += [("D", kt), ("M", kt)]
        if two:
            if kt + 2 * nk < KT1:
                ev.append(("L", kt + 2 * nk))
            ev += [("D", kt + nk), ("M", kt + nk)]
    return ev


@pytest.mark.parametrize("nk", [1, 2, 4])
def test_unrolled_loop_runs_every_step_once_in_order(nk):
    """For every k-tile count (even and odd number of steps, empty ks slices): each step is loaded before its decode, decoded and multiplied exactly once,
    steps in increasing order, no load past KT1 -- the same event sequence as the stock loop up to the position of the loads."""
    src = NX.coop_source((3,), "mul1r2", 32, 2, 32, 16 * nk, unroll=True)
    assert f"kt += {2 * nk}u" in src and f"const bool two = kt + {nk}u < KT1;" in src and f"if (kt + {2 * nk}u < KT1)" in src   # the guards modelled in _control
    for KT0 in (0, nk, 3 * nk):
        for steps in range(0, 14):
            KT1 = KT0 + steps * nk
            a, b = _control(KT0, KT1, nk, False), _control(KT0, KT1, nk, True)
            for ev in (a, b):
                dec = [s for e, s in ev if e == "D"]
                mm = [s for e, s in ev if e == "M"]
                assert dec == mm == list(range(KT0, KT1, nk))
                loads = [s for e, s in ev if e == "L"]
                assert sorted(loads) == list(range(KT0, KT1, nk)) and all(s < KT1 for s in loads)
                for s in dec:                                   # loaded before decoded
                    assert ev.index(("L", s)) < ev.index(("D", s))
            assert [e for e in a if e[0] != "L"] == [e for e in b if e[0] != "L"]


def _runs(M, bmu, nmb):
    """(row block mg, sub-block mb) matmul runs the kernel issues for an expert with M rows, from the generated guards: the block exists while mg < M, sub-block 0 always
    runs, sub-block mb > 0 runs when mg + bmu * mb < M."""
    return [(mg, mb) for mg in range(0, M, bmu * nmb) for mb in range(nmb) if mb == 0 or mg + bmu * mb < M]


@pytest.mark.parametrize("M", list(range(1, 200)))
def test_bmu16_covers_every_row_once_and_pads_less(M):
    for bmu, nmb in ((32, 2), (32, 1), (32, 4)):
        a = _runs(M, bmu, nmb)
        b = _runs(M, bmu // 2, nmb * 2)
        for runs, u in ((a, bmu), (b, bmu // 2)):
            # rows of a run: [row0, row0 + u) with row0 = mg + u * mb
            blocks = bmu * nmb
            cover = np.zeros(M + 2 * blocks, int)
            for mg, mb in runs:
                cover[mg + u * mb: mg + u * (mb + 1)] += 1
            assert (cover[:M] == 1).all() and (cover[M:] <= 1).all()          # every real row computed once; padding rows at most once
        padded = lambda runs, u: len(runs) * u
        assert padded(b, bmu // 2) <= padded(a, bmu)
        assert len({mg for mg, _ in a}) == len({mg for mg, _ in b})          # same row blocks: the decode count per expert is unchanged
        assert -(-M // (bmu * nmb)) == len({mg for mg, _ in b})              # and the grid height (rows per block unchanged)


def test_pick_cfg_flag_off_unchanged_and_on_splits_only_the_plain_pair(monkeypatch):
    rng = np.random.default_rng(3)
    for _ in range(30):
        counts = rng.integers(0, 300, size=int(rng.integers(1, 200)))
        monkeypatch.setattr(NX, "BMU16", False)
        base = NX.pick_cfg(counts, True)
        assert base == NX.pick_cfg(counts) and base[0] in (16, 32)
        monkeypatch.setattr(NX, "BMU16", True)
        assert NX.pick_cfg(counts) == base                                    # callers without the plain-pair allowance (dense, row map, epilogue) are untouched
        on = NX.pick_cfg(counts, True)
        if base[0] == 32:
            assert on == (16, 2 * base[1], base[2], base[3])
            assert on[0] * on[1] == base[0] * base[1]                        # same rows per block => same grid, same decodes per expert
            assert on[0] * on[1] * on[2] == base[0] * base[1] * base[2]      # same accumulator elements per simdgroup (registers)
        else:
            assert on == base


def test_bmu16_source_builds_and_keeps_the_register_budget():
    on = NX.coop_source((3,), "mul1r2", 16, 4, 32, 64)
    assert on.count("op.run(") == 4 and on.count("cT3") >= 2 and on.count("{") == on.count("}")
    assert 4 * 16 * 32 == 2 * 32 * 32                                         # fp32 accumulator elements per simdgroup, 4 x (16x32) = 2 x (32x32)


def test_model_padded_rows_on_skewed_routing():
    """MODEL (counts, not time): 384 experts, the hot quarter takes 75 % of 4096 x 6 pairs. Padded matmul rows of the 32-row sub-block form against the 16-row form."""
    rng = np.random.default_rng(1)
    E, P = 384, 4096 * 6
    pr = np.r_[np.full(96, .75 / 96), np.full(288, .25 / 288)]
    gain = []
    for _ in range(20):
        c = rng.multinomial(P, pr)
        a = sum(len(_runs(int(x), 32, 2)) * 32 for x in c if x)
        b = sum(len(_runs(int(x), 16, 4)) * 16 for x in c if x)
        assert b <= a and b >= P
        gain.append(1 - b / a)
    assert 0.03 < np.mean(gain) < 0.09, np.mean(gain)                         # ~5 % fewer padded rows (matmul only; the decode per block is unchanged)


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal not available")
@pytest.mark.parametrize("K,rows,H", [(2, 40, 320), (3, 40, 320), (3, 128, 512), (5, 71, 320), (3, 9, 320), (6, 100, 192)])
def test_unroll_is_bit_identical_and_bmu16_close(K, rows, H):
    """Needs the Mac. unroll2: the same bits as the stock kernel (odd and even step counts). bmu16: allclose here; the md5 verdict is the tool's."""
    from bench_moe import synth_stack
    E, I = 3, 384
    g = synth_stack(H, I, [K] * E, "mul1", 1)
    rng = np.random.default_rng(0)
    eids = mx.array(np.arange(E, dtype=np.int32))
    A = mx.array(NX.permute_k(rng.standard_normal((E * rows, H)).astype(np.float16)))
    ro = mx.array(np.arange(E + 1, dtype=np.uint32) * rows)

    def run(cfg, unroll):
        old = NX.UNROLL2
        NX.UNROLL2 = unroll
        try:
            return np.array(NX.expert_gemm((g,), eids, ro, A, E * rows, rows, cfg))
        finally:
            NX.UNROLL2 = old
    ref = run(CFG, False)
    assert np.array_equal(run(CFG, True).view(np.uint16), ref.view(np.uint16))
    assert np.allclose(run((16, 4, 32, 64), False).astype(np.float32), ref.astype(np.float32), rtol=2e-2, atol=2e-2)
