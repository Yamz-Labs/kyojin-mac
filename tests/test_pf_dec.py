"""pf_dec (KYOJIN_PF_DEC): the fused NAX prefill kernels decode an exact mul1 pack with dec_mul1r1 / dec_mul1r2. No Metal needed: the decode forms are
checked over all 65536 trellis states against the stock form, and the generated sources and kernel keys are checked for the swap. Timing: tools/bench_pf_dec.py."""
import numpy as np
import pytest

pytest.importorskip("mlx.core")

from kyojin_mlx import emulate as EM, flags as FL, metal_nax as NX, metal_nax_gu as GU, metal_src as S

ALL = np.arange(65536, dtype=np.uint32)


@pytest.mark.parametrize("level", [1, 2])
def test_roof_decode_equals_stock_decode_on_every_state(level):
    assert np.array_equal(EM.mul1r_values(ALL, level).view(np.uint16), EM.mul1_values(ALL).view(np.uint16))


def test_header_defines_the_roof_decoders():
    assert "dec_mul1r1" in NX.HEADER_NAX and "dec_mul1r2" in NX.HEADER_NAX and S.HEADER_MOE_ROOF in NX.HEADER_NAX
    assert NX.HEADER_NAX.index("dec_mul1r1") > NX.HEADER_NAX.index("#pragma METAL fp contract(off)")


@pytest.mark.parametrize("lvl,cb,want", [(0, "mul1", "mul1"), (1, "mul1", "mul1r1"), (2, "mul1", "mul1r2"), (3, "mul1", "mul1"), (2, "mcg", "mcg"), (1, "3inst", "3inst")])
def test_dec_cb(monkeypatch, lvl, cb, want):
    monkeypatch.setattr(NX, "PF_DEC", lvl)
    assert NX.dec_cb(cb) == want


@pytest.mark.parametrize("lvl", [1, 2])
def test_every_nax_source_uses_the_swapped_decoder(monkeypatch, lvl):
    monkeypatch.setattr(NX, "PF_DEC", lvl)
    cb = NX.dec_cb("mul1")
    name = f"dec_mul1r{lvl}("
    srcs = [NX.coop_source((2, 3), cb, 32, 2, 32, 64, True, "full", 2, True, "", False, False),
            NX.coop_source((2, 3), cb, 32, 2, 32, 32, True, "full", 1, True, "", False, True),
            NX.coop_rm_source((2, 3), cb, 32, 1, 32, 64, True, 2, True, 8),
            GU.gu_source(3, 3, cb, 32, 1, 16, 64, 7.0, True, True, 1.0, 0.0, True, True)]
    for src in srcs:
        assert name in src and "dec_mul1(" not in src


def test_default_is_r2_and_off_keeps_the_source(monkeypatch):
    assert NX.PF_DEC == 2 and FL.FLAGS["pf_dec"].default == "2" and NX.dec_cb("mul1") == "mul1r2"
    monkeypatch.setattr(NX, "PF_DEC", 0)
    assert NX.dec_cb("mul1") == "mul1"
    src = NX.coop_source((3,), NX.dec_cb("mul1"), 32, 2, 32, 64, True, "full", 1, True, "", False, False)
    assert "dec_mul1(" in src and "dec_mul1r" not in src


# ---------------------------------------------------------------------------------------------------------------- pf_blk128
def test_blk128_off_never_picks_it(monkeypatch):
    monkeypatch.setattr(NX, "BLK128", False)
    assert NX.pick_cfg([256] * 384, True)[1] != 4
    assert NX.pick_cfg([256] * 384, True) == NX.pick_cfg([256] * 384)


def test_blk128_needs_both_the_flag_and_the_opt_in(monkeypatch):
    monkeypatch.setattr(NX, "BLK128", True)
    assert NX.pick_cfg([256] * 384) == NX.pick_cfg([256] * 384, False)         # non-MoE callers (dense rows) never see it
    assert NX.pick_cfg([256] * 384, True) == (32, 4, 16, 64)


@pytest.mark.parametrize("rows,want128", [(8, False), (16, False), (32, False), (64, False), (96, False), (128, True), (256, True), (384, True)])
def test_blk128_choice_follows_the_cost_model(monkeypatch, rows, want128):
    monkeypatch.setattr(NX, "BLK128", True)
    assert (NX.pick_cfg([rows] * 64, True) == (32, 4, 16, 64)) == want128


def test_blk128_cost_table_is_the_linear_extrapolation():
    t = NX.block_costs(True)
    assert t[128] == pytest.approx(t[64] + 2 * (t[64] - t[32])) and set(NX.block_costs(False)) == {16, 32, 64}


def test_blk128_source_is_generated_for_the_gemm():
    src = NX.coop_source((2, 3), "mul1", 32, 4, 16, 64, True, "full", 2, True, "", False, False)
    assert src.count("op.run(") == 4 * 2 and "cT3" in src and "mg + 96u < M" in src
