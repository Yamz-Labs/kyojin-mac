"""port_sumoe: the routed + shared launches choose the k-split like the routed-only gather (routed top-k pairs, pinned to one row), so the plain-step row and
the verify row of the same token sum their partials in the same order, as the plain gather path does."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from kyojin_mlx import moe as M, moe_sumoe as SU          # noqa: E402

E, KT, NT = 4, 320, 144                                  # DS4.1 gate / up: kin 5120, nout 2304 (18 column blocks)


def stack(kt, nt):
    return M.Stack(mx.zeros((E, 8), dtype=mx.uint32), mx.zeros((E,), dtype=mx.uint32), mx.array([3, 2, 3, 2], dtype=mx.int32),
                   mx.zeros((E, kt * 16), dtype=mx.float16), mx.zeros((E, nt * 16), dtype=mx.float16), kt, nt)


def build():
    g, u, d = stack(KT, NT), stack(KT, NT), stack(NT, KT)
    routed = M.MoeGather(g, u, d, 6, pin_rows=1)
    shared = SU.MoeSharedFused(g, u, d, 6)
    return routed, shared


def split(m, rows):
    p = m._params(rows)
    return p["tgu"], p["tdn"]


def test_shared_split_equals_the_routed_one_row_split_at_every_row_count():
    routed, shared = build()
    want = split(routed, 1)
    assert want[0] > 1                                    # the 12 vs 14 pair counts do pick different splits at these dims, so the test can fail
    shared.pin_rows = 1
    for rows in range(1, 7):
        assert split(shared, rows) == want


def test_unpinned_shared_split_follows_the_routed_pair_count():
    routed, shared = build()
    routed.pin_rows = None
    for rows in (1, 2, 4):
        assert split(shared, rows) == split(routed, rows)
