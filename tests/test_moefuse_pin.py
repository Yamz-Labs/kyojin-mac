"""moefuse's inner union gather keeps the k-split pin of the gather it wraps (rows.enable sets pin_rows = 1 so a row's partial-sum order does not depend on how many rows share the step)."""
import types

import pytest

pytest.importorskip("mlx.core")

from kyojin_mlx import moe_fused as MF      # noqa: E402


class Rec:
    def __init__(self, g, u, d, topk, **kw):
        self.kw = kw


@pytest.mark.parametrize("pin", [None, 1])
def test_inner_gather_inherits_pin_rows(monkeypatch, pin):
    monkeypatch.setattr(MF.M, "MoeGather", Rec)
    m = types.SimpleNamespace(g=1, u=2, d=3, topk=6, limit=0, wpre=False, fq=False, tgk_gu=None, tgk_dn=None, alpha=1.0, beta=0.0, pin_rows=pin)
    inner = MF.MoeFused(m)._u
    assert inner.kw["pin_rows"] == pin and inner.kw["union"] is True and inner.kw["grouped"] is False
