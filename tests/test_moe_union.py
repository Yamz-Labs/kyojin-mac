"""Union MoE decode (pairs of different users routed to the same expert share one trellis decode): per-pair results must be
bit-identical to the plain gather path, and with a pinned k-split a user's row must not depend on the other users in the step.
Synthetic ragged stacks (tests/test_moe.py helpers), mixed K, both codebooks. Needs a Mac (Metal)."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

from kyojin_mlx import moe as M
from test_moe import E, KS, TOPK, make_stack, _x


@pytest.fixture(scope="module", params=["mul1", "mcg"])
def layer(request):
    cb = request.param
    kin, ninter = 256, 384
    g, _ = make_stack(kin, ninter, KS, cb, 1)
    u, _ = make_stack(kin, ninter, KS, cb, 2)
    d, _ = make_stack(ninter, kin, KS, cb, 3)
    return g, u, d


def route(rows, mode, seed):
    rng = np.random.default_rng(seed)
    base = rng.choice(E, TOPK, replace=False)
    ids = []
    for r in range(rows):
        if mode == "identical":
            ids.append(base.copy())
        elif mode == "permuted":
            ids.append(rng.permutation(base))
        elif mode == "partial":                                      # half of the base experts + fresh ones
            keep = rng.permutation(base)[:TOPK // 2]
            rest = [e for e in rng.permutation(E) if e not in keep][:TOPK - len(keep)]
            ids.append(rng.permutation(np.concatenate([keep, rest])))
        else:
            ids.append(rng.choice(E, TOPK, replace=False))
    w = rng.uniform(0.1, 1.0, (rows, TOPK)).astype(np.float32)
    return mx.array(np.stack(ids).astype(np.int32)), mx.array((w / w.sum(-1, keepdims=True) * 1.5).astype(np.float32))


def bits(a):
    return np.array(a).view(np.uint32)


@pytest.mark.parametrize("rows", [2, 3, 4, 5, 8, 9])      # 5-8: eight-member variant, 9: falls back to the plain path
@pytest.mark.parametrize("mode", ["identical", "permuted", "partial", "random"])
@pytest.mark.parametrize("wpre", [False, True])
def test_union_equals_gather(layer, rows, mode, wpre):
    g, u, d = layer
    plain = M.MoeGather(g, u, d, TOPK, wpre=wpre, tgk_gu=2, tgk_dn=2)
    uni = M.MoeGather(g, u, d, TOPK, wpre=wpre, tgk_gu=2, tgk_dn=2, union=True)
    ids, w = route(rows, mode, 10 * rows + len(mode))
    x = _x(rows, g.kin, 7)
    assert np.array_equal(bits(plain(x, ids, w)), bits(uni(x, ids, w)))
    # stage level: the partials of every (pair, projection)
    fl = ids.reshape(-1).astype(mx.int32)
    xh = plain.rot_gu(x, fl)
    assert np.array_equal(bits(plain.gemv_gu(xh, fl, rows)), bits(uni.gemv_gu(xh, fl, rows)))


@pytest.mark.parametrize("mode", ["identical", "partial", "random"])
def test_pinned_rows_do_not_depend_on_the_batch(layer, mode):
    g, u, d = layer
    uni = M.MoeGather(g, u, d, TOPK, union=True, pin_rows=1)
    ids, w = route(4, mode, 99)
    x = _x(4, g.kin, 3)
    y = bits(uni(x, ids, w))
    for r in range(4):
        assert np.array_equal(y[r], bits(uni(x[r:r + 1], ids[r:r + 1], w[r:r + 1]))[0]), r
    y2 = bits(uni(x[:2], ids[:2], w[:2]))
    assert np.array_equal(y2, y[:2])


def test_union_kernel_is_the_one_running(layer):
    g, u, d = layer
    uni = M.MoeGather(g, u, d, TOPK, union=True, pin_rows=1)
    ids, w = route(3, "identical", 5)
    uni(_x(3, g.kin, 1), ids, w)
    assert any(k[0] == "ug" and k[-1] == 4 for k in M._kc), list(M._kc)
    uni(_x(1, g.kin, 1), ids[:1], w[:1])                      # one row: plain gather kernel, nothing to share
