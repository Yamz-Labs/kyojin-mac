"""Grouped MoE decode (every distinct expert decoded once for all rows routed to it): per-pair partials and the layer output must be
bit-identical to the plain gather path for any row count and routing (no sharing, partial, identical, one hot expert), and the k = 1
groups must run the one-row arithmetic. Synthetic ragged stacks (tests/test_moe.py helpers), mixed K, both codebooks. Needs a Mac (Metal)."""
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


def hot(rows, seed):
    """Every row holds expert 0 (k = rows); the rest random."""
    rng = np.random.default_rng(seed)
    ids = np.stack([np.concatenate([[0], rng.choice(np.arange(1, E), TOPK - 1, replace=False)]) for _ in range(rows)])
    w = rng.uniform(0.1, 1.0, (rows, TOPK)).astype(np.float32)
    return mx.array(ids.astype(np.int32)), mx.array((w / w.sum(-1, keepdims=True) * 1.5).astype(np.float32))


@pytest.mark.parametrize("rows", [2, 3, 4, 5, 8, 9, 12, 16, 32, 42])
@pytest.mark.parametrize("mode", ["identical", "permuted", "partial", "random", "hot"])
@pytest.mark.parametrize("wpre", [False, True])
def test_grouped_equals_gather(layer, rows, mode, wpre):
    g, u, d = layer
    plain = M.MoeGather(g, u, d, TOPK, wpre=wpre, tgk_gu=2, tgk_dn=2, grouped=False, hotk=0)
    grp = M.MoeGather(g, u, d, TOPK, wpre=wpre, tgk_gu=2, tgk_dn=2, grouped=True, hotk=0)
    ids, w = hot(rows, rows) if mode == "hot" else route(rows, mode, 10 * rows + len(mode))
    x = _x(rows, g.kin, 7)
    assert np.array_equal(bits(plain(x, ids, w)), bits(grp(x, ids, w)))
    fl = ids.reshape(-1).astype(mx.int32)
    xh = plain.rot_gu(x, fl)
    P = grp._params(rows)
    plan = grp.plan(fl, rows)
    assert np.array_equal(bits(plain.gemv_gu(xh, fl, rows)), bits(grp.ggemv((grp.g, grp.u), xh, fl, plan, P["ggemv_gu"], P["tgu"], grp.ks_gu)))


@pytest.mark.parametrize("mode", ["identical", "partial", "random", "hot"])
def test_plan_is_a_grouping_by_expert(mode):
    rows = 12
    ids, _ = hot(rows, 3) if mode == "hot" else route(rows, mode, 5)
    fl = ids.reshape(-1).astype(mx.int32)
    g, _ = make_stack(256, 384, KS, "mul1", 1)
    u, _ = make_stack(256, 384, KS, "mul1", 2)
    d, _ = make_stack(384, 256, KS, "mul1", 3)
    mo = M.MoeGather(g, u, d, TOPK, grouped=True, hotk=0)
    lead, cnt, mem = (np.array(a) for a in mo.plan(fl, rows))
    ids_h, npairs = np.array(fl), rows * TOPK
    caps = S.plan_caps(npairs)
    seen = set()
    for slot, p in enumerate(lead):
        cls = max(c for c in range(len(caps)) if slot >= sum(caps[:c]))
        if p == 0xffffffff:
            continue
        members = [q for q in range(npairs) if ids_h[q] == ids_h[p]]
        assert p == members[0] and cnt[p] == len(members) and list(mem[p * rows:p * rows + len(members)]) == members
        assert cls == max(c for c, (lo, _) in enumerate(S.GROUP_CLASSES) if len(members) >= lo), (slot, members)
        assert ids_h[p] not in seen
        seen.add(int(ids_h[p]))
    assert seen == set(int(v) for v in ids_h)


def test_pinned_rows_do_not_depend_on_the_batch(layer):
    g, u, d = layer
    grp = M.MoeGather(g, u, d, TOPK, grouped=True, pin_rows=1, hotk=0)
    ids, w = route(6, "partial", 99)
    x = _x(6, g.kin, 3)
    y = bits(grp(x, ids, w))
    for r in range(6):
        assert np.array_equal(y[r], bits(grp(x[r:r + 1], ids[r:r + 1], w[r:r + 1]))[0]), r


def test_rows_ugrouped_switch(layer, monkeypatch):
    """KYOJIN_ROWS_UGROUPED = N: a pass of 2..N rows takes the per-pair split-K gather GEMV (the 1-row kernel) and keeps the grouped path's bits;
    N below the row count keeps the grouped GEMV. rows = 1 is untouched."""
    g, u, d = layer
    mo = M.MoeGather(g, u, d, TOPK, grouped=True, tgk_gu=2, tgk_dn=2, pin_rows=1, hotk=0)
    ids, w = route(3, "partial", 123)
    x = _x(3, g.kin, 5)
    monkeypatch.setattr(M, "UGROUPED", 0)
    ref = bits(mo(x, ids, w))
    calls = {"grouped": 0, "gather": 0}
    og, ov = M.MoeGather._call_grouped, M.MoeGather.gemv

    def cg(self, *a, **k):
        calls["grouped"] += 1
        return og(self, *a, **k)

    def gv(self, *a, **k):
        calls["gather"] += 1
        return ov(self, *a, **k)

    monkeypatch.setattr(M.MoeGather, "_call_grouped", cg)
    monkeypatch.setattr(M.MoeGather, "gemv", gv)
    monkeypatch.setattr(M, "UGROUPED", 3)
    got = bits(mo(x, ids, w))
    assert (calls["grouped"], calls["gather"]) == (0, 2), calls       # gate + up, down
    assert np.array_equal(got, ref)
    calls.update(grouped=0, gather=0)
    monkeypatch.setattr(M, "UGROUPED", 2)                            # 3 rows > 2: the grouped path stays
    assert np.array_equal(bits(mo(x, ids, w)), ref)
    assert (calls["grouped"], calls["gather"]) == (1, 0), calls
    calls.update(grouped=0, gather=0)
    monkeypatch.setattr(M, "UGROUPED", 4)                            # one row: the per-pair path it always had, no grouped call either way
    assert np.array_equal(bits(mo(x[0:1], ids[0:1], w[0:1]))[0], ref[0])
    assert calls["grouped"] == 0
