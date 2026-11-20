"""Flag KYOJIN_MOE_PARTIALK (CPU only): routing of routed-expert stacks at k % 512 != 0, source edits, flags-off identity."""
import re
import pytest
from kyojin_mlx import foreign_metal as FM, foreign_kqgmm as KQ


def test_iq4nl_partial_source_is_stock_plus_four_edits():
    a, b = FM.gemv_source("iq4_nl", 2, True), FM.gemv_source("iq4_nl", 2, True, False, False, True)
    assert a != b and FM.gemv_source("iq4_nl", 2, True, False, False, False) == a
    assert "(uint(KIN) + 511u) / 512u" in b and b.count("min(ul, UPR - 1u)") == 1 and b.count("UPR - 1u);") >= 2 and "uint4(0u)" in b


def test_kqgmm_kinds_off_by_default(monkeypatch):
    class D:
        E, k, rowgs = 16, 640, False
    monkeypatch.delenv("KYOJIN_MOE_PARTIALK", raising=False)
    for f in KQ.PK_KINDS:
        d = D(); d.fmt = f
        assert not KQ.ok(d)
    monkeypatch.setenv("KYOJIN_MOE_PARTIALK", "1")
    for f in KQ.PK_KINDS:
        d = D(); d.fmt = f
        assert KQ.ok(d)


def test_kqnl_context_is_scoped():
    assert "iq4_nl" not in FM._FMMA_PAIR
    with FM.kqnl(True):
        assert FM._FMMA_PAIR["iq4_nl"] == "lut"
    assert "iq4_nl" not in FM._FMMA_PAIR
