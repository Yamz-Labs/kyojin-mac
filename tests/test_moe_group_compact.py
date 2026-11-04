"""Grouped MoE, packed slot layout (KYOJIN_MOEGRP): the slots of all member counts share one range of NP slots and each slot's body follows its
group's member count. Every per-pair partial and the layer output must stay bit-identical to the per-class layout (and to the plain gather path)
for any row count and routing, including the 5..7 member groups that the per-class layout pads to the 8-row pass. Synthetic ragged stacks
(tests/test_moe.py helpers), mixed K, both codebooks. Needs a Mac (Metal)."""
import contextlib

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

from kyojin_mlx import metal_src as S, moe as M
from test_moe import E, KS, TOPK, make_stack, _x
from test_moe_union import route, bits


@pytest.fixture(scope="module", params=["mul1", "mcg"])
def layer(request):
    cb = request.param
    kin, ninter = 256, 384
    g, _ = make_stack(kin, ninter, KS, cb, 1)
    u, _ = make_stack(kin, ninter, KS, cb, 2)
    d, _ = make_stack(ninter, kin, KS, cb, 3)
    return g, u, d


@pytest.fixture(scope="module")
def grp():
    """The module literal flags.py switches (KYOJIN_MOEGRP), restored after the module."""
    old = M.MOEGRP
    yield
    M.MOEGRP = old


@contextlib.contextmanager
def compact(on: bool):
    """Switch the lever around every call that reads it (the GEMV kernels, the plan and the grid sizes are keyed on it)."""
    old = M.MOEGRP
    M.MOEGRP = on
    try:
        yield
    finally:
        M.MOEGRP = old


def hot(rows, seed):
    """Every row holds expert 0 (a group of `rows` members): the counts 5..7 the per-class layout pads."""
    rng = np.random.default_rng(seed)
    ids = np.stack([np.concatenate([[0], rng.choice(np.arange(1, E), TOPK - 1, replace=False)]) for _ in range(rows)])
    w = rng.uniform(0.1, 1.0, (rows, TOPK)).astype(np.float32)
    return mx.array(ids.astype(np.int32)), mx.array((w / w.sum(-1, keepdims=True) * 1.5).astype(np.float32))


def moe(g, u, d, wpre, grouped, on):
    with compact(on):
        return M.MoeGather(g, u, d, TOPK, wpre=wpre, tgk_gu=2, tgk_dn=2, grouped=grouped, hotk=0)


@pytest.mark.parametrize("rows", [2, 3, 4, 5, 6, 7, 8, 9, 12])
@pytest.mark.parametrize("mode", ["identical", "permuted", "partial", "random", "hot"])
@pytest.mark.parametrize("wpre", [False, True])
def test_compact_equals_per_class(layer, grp, rows, mode, wpre):
    g, u, d = layer
    old = moe(g, u, d, wpre, True, False)
    new = moe(g, u, d, wpre, True, True)
    ids, w = hot(rows, rows) if mode == "hot" else route(rows, mode, 10 * rows + len(mode))
    x = _x(rows, g.kin, 7)
    with compact(False):
        y0 = bits(old(x, ids, w))
    with compact(True):
        y1 = bits(new(x, ids, w))
    assert np.array_equal(y0, y1)
    fl = ids.reshape(-1).astype(mx.int32)
    xh = old.rot_gu(x, fl)                                     # the rot does not depend on the lever
    P0, P1 = old._params(rows), new._params(rows)
    with compact(False):
        p0 = old.ggemv((old.g, old.u), xh, fl, old.plan(fl, rows), P0["ggemv_gu"], P0["tgu"], old.ks_gu)
    with compact(True):
        p1 = new.ggemv((new.g, new.u), xh, fl, new.plan(fl, rows), P1["ggemv_gu"], P1["tgu"], new.ks_gu)
    assert np.array_equal(bits(p0), bits(p1))


@pytest.mark.parametrize("rows", [2, 5, 8])
@pytest.mark.parametrize("mode", ["partial", "random", "hot"])
def test_compact_matches_the_gather_path(layer, grp, rows, mode):
    """The packed grouped path against the plain per-pair gather path (rows > 4 included: a row keeps its solo bits)."""
    g, u, d = layer
    plain = moe(g, u, d, True, False, False)
    new = moe(g, u, d, True, True, True)
    ids, w = hot(rows, rows) if mode == "hot" else route(rows, mode, 3 * rows)
    x = _x(rows, g.kin, 5)
    with compact(False):
        y0 = bits(plain(x, ids, w))
    with compact(True):
        y1 = bits(new(x, ids, w))
    assert np.array_equal(y0, y1)


@pytest.mark.parametrize("rows", [2, 4, 5, 6, 7, 8])
def test_compact_plan_slots(layer, grp, rows):
    """One packed range: slot s < #groups holds a leader, every group holds its members in ascending order, cnt_ is the member count."""
    g, u, d = layer
    new = moe(g, u, d, False, True, True)
    for mode in ("partial", "random", "hot"):
        ids, _ = hot(rows, rows) if mode == "hot" else route(rows, mode, 11 * rows)
        fl = ids.reshape(-1).astype(mx.int32)
        with compact(True):
            lead, cnt, mem = (np.array(a) for a in new.plan(fl, rows))
        ids_h, npairs = np.array(fl), rows * TOPK
        assert lead.shape == (npairs,) == S.plan_caps(npairs, 0, True)
        groups = [q for q in range(npairs) if list(ids_h[:q]).count(ids_h[q]) == 0]
        assert list(lead[:len(groups)]) == groups and set(lead[len(groups):].tolist()) == {0xFFFFFFFF}
        for p in groups:
            members = [q for q in range(npairs) if ids_h[q] == ids_h[p]]
            assert cnt[p] == len(members) and list(mem[p * rows:p * rows + len(members)]) == members


def test_compact_hot_groups_keep_the_class_layout(layer, grp):
    """hotk (the NAX path of multi-user passes) is not packed: the plan keeps the per-class ranges."""
    g, u, d = layer
    with compact(True):
        hotk = M.MoeGather(g, u, d, TOPK, grouped=True, hotk=4)
        ids, _ = route(8, "partial", 7)
        lead = np.array(hotk.plan(ids.reshape(-1).astype(mx.int32), 8)[0])
    assert lead.shape == (sum(S.plan_caps(8 * TOPK, 4)),)
