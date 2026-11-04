# SPDX-License-Identifier: Apache-2.0
"""KYOJIN_MOE_GPF (flag moe_gpf): the grouped routed-MoE GEMV loads its trellis words one or two k tiles ahead of the tile it decodes.
Only the load order changes, so the partials and the layer output must be bit-identical to the stock kernel (pf = 0) for every row count,
routing and bitrate class. Synthetic ragged stacks (tests/test_moe.py helpers), mixed K, both codebooks. Needs a Mac (Metal)."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

from kyojin_mlx import metal_src as S, moe as M
from test_moe import KS, TOPK, make_stack, _x
from test_moe_union import route, bits


@pytest.fixture(scope="module", params=["mul1", "mcg"])
def layer(request):
    cb = request.param
    kin, ninter = 256, 384
    g, _ = make_stack(kin, ninter, KS, cb, 1)
    u, _ = make_stack(kin, ninter, KS, cb, 2)
    d, _ = make_stack(ninter, kin, KS, cb, 3)
    return cb, g, u, d


@pytest.mark.parametrize("rows", [2, 3, 5, 6, 8])
@pytest.mark.parametrize("mode", ["identical", "permuted", "partial", "random"])
def test_gpf_partials_are_bit_identical(layer, rows, mode, monkeypatch):
    """One layer through the grouped path with pf 0 / 1 / 2: same partials, same output, bit for bit."""
    cb, g, u, d = layer
    ids, w = route(rows, mode, 7 * rows + len(mode))
    x = _x(rows, g.kin, 11)
    outs, parts = {}, {}
    for pf in (0, 1, 2):
        monkeypatch.setattr(S, "GPF", pf)
        m = M.MoeGather(g, u, d, TOPK, wpre=False, tgk_gu=2, tgk_dn=2, grouped=True, hotk=0)
        outs[pf] = np.array(m(x, ids, w))
        fl = ids.reshape(-1).astype(mx.int32)
        P = m._params(rows)
        plan = m.plan(fl, rows)
        xh = m.rot_gu(x, fl)
        parts[pf] = np.array(m.ggemv((m.g, m.u), xh, fl, plan, P["ggemv_gu"], P["tgu"], m.ks_gu))
    assert np.array_equal(outs[0], outs[1]) and np.array_equal(outs[0], outs[2])
    assert np.array_equal(parts[0], parts[1]) and np.array_equal(parts[0], parts[2])
