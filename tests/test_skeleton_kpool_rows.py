# SPDX-License-Identifier: Apache-2.0
"""N3 kpool selection with per-row cache offsets (foreign_skeleton_kpool) vs a per-row loop reference (numpy pools + scalar-offset rope, one
token at a time), at 1 row and 4 rows of different offsets, prefill chunk and decode, with and without the forced tail group. CPU, no model."""
import numpy as np
import mlx.core as mx
import pytest

from kyojin_mlx import foreign_skeleton_kpool as KP

D, NROT, H, KPOOL, TOPK, BASE, EPS = 16, 8, 3, 4, 12, 10000.0, 1e-6


def _rope1(x, pos):
    """x [n, d] numpy -> tail rotated at scalar position pos (mx.fast.rope, traditional = rope NORM), front untouched."""
    t = mx.fast.rope(mx.array(x[None, None, :, D - NROT:]), NROT, traditional=True, base=BASE, scale=1.0, offset=int(pos))
    return np.concatenate([x[:, :D - NROT], np.array(t[0, 0])], axis=-1)


def _ref_row(iq, w, val, gate, ape, nw, off, select_tail):
    """One row, solo: iq [H, L, d], w [L, H], val / gate [n, d] for positions 0..n-1 (n = off + L) -> keep [L, n] bool."""
    L, n = iq.shape[1], val.shape[0]
    pools = []
    for j in range(n // KPOOL):
        g = gate[j * KPOOL:(j + 1) * KPOOL] + ape
        p = np.exp(g - g.max(0)); p /= p.sum(0)
        pk = (p * val[j * KPOOL:(j + 1) * KPOOL]).sum(0)
        pk = pk / np.sqrt((pk * pk).mean() + EPS) * nw
        pools.append(_rope1(pk[None].astype(np.float32), j * KPOOL)[0])
    keep = np.zeros((L, n), bool)
    n_top = TOPK // KPOOL - int(select_tail)
    for i in range(L):
        p = off + i
        q = np.stack([_rope1(iq[h, i:i + 1], p)[0] for h in range(H)])
        nvis = p // KPOOL if select_tail else (p + 1) // KPOOL
        sc = [sum(max(float(q[h] @ pools[j]), 0.0) * w[i, h] for h in range(H)) for j in range(nvis)]
        for j in np.argsort(-np.array(sc), kind="stable")[:n_top]:
            keep[i, j * KPOOL:(j + 1) * KPOOL] = True
        g0 = (p // KPOOL) * KPOOL if select_tail else p - (p + 1) % KPOOL + 1
        keep[i, g0:p + 1] = True
    return keep


@pytest.mark.parametrize("offs", [[13], [0, 5, 13, 30], [30, 2, 17, 9]])
@pytest.mark.parametrize("L", [1, 3])
@pytest.mark.parametrize("select_tail", [True, False])
def test_kpool_rows_vs_loop(offs, L, select_tail):
    rng = np.random.default_rng(len(offs) * 10 + L)
    B = len(offs)
    S = max(offs) + L
    iq = rng.standard_normal((B, H, L, D)).astype(np.float32)
    w = rng.standard_normal((B, L, H)).astype(np.float32)
    val = rng.standard_normal((B, S, D)).astype(np.float32)
    gate = rng.standard_normal((B, S, D)).astype(np.float32)
    ape = rng.standard_normal((KPOOL, D)).astype(np.float32)
    nw = (1.0 + 0.1 * rng.standard_normal(D)).astype(np.float32)
    ks = KP.KpoolSelect(mx.array(ape), mx.array(nw), EPS, KPOOL, TOPK, select_tail, NROT, BASE, True)
    off = mx.array(offs) if B > 1 else offs[0]
    keep = np.array(ks.keep(mx.array(iq), mx.array(w), mx.array(val), mx.array(gate), off))
    assert keep.shape == (B, 1, L, S)
    for b, o in enumerate(offs):
        pad = S - o - L
        ref = _ref_row(iq[b], w[b], val[b, pad:], gate[b, pad:], ape, nw, o, select_tail)
        assert not keep[b, 0, :, :pad].any(), "padding slots must stay hidden"
        np.testing.assert_array_equal(keep[b, 0, :, pad:], ref, err_msg=f"row {b} offset {o}")
        assert ref.sum(1).max() <= TOPK + (0 if select_tail else KPOOL - 1)     # n_sel slots (M:5544, 5713)


def test_kpool_selection_drops_pools():
    """The case is not trivial: at offset 30 the visible pools outnumber the selection, so some keys are dropped."""
    offs, L = [30], 1
    rng = np.random.default_rng(0)
    S = 31
    a = [rng.standard_normal(s).astype(np.float32) for s in [(1, H, L, D), (1, L, H), (1, S, D), (1, S, D), (KPOOL, D)]]
    ks = KP.KpoolSelect(mx.array(a[4]), mx.ones((D,)), EPS, KPOOL, TOPK, True, NROT, BASE, True)
    keep = np.array(ks.keep(*[mx.array(x) for x in a[:4]], offs[0]))
    assert 0 < keep.sum() < S


def test_block_keep_rows_vs_solo():
    """The skeleton's block-sparse indexer (minimax form) at 4 rows of different offsets = each row's solo call (the scalar-offset blocker is gone)."""
    from types import SimpleNamespace
    from kyojin_mlx.foreign_skeleton import Attention
    sp = SimpleNamespace(indexer_block=4, indexer_local=1, indexer_top_k=3)
    me = SimpleNamespace(sp=sp, nh=2)
    rng = np.random.default_rng(3)
    offs, L, Hd, dx = [0, 6, 21, 13], 2, 1, 8
    S = max(offs) + L
    iq = rng.standard_normal((4, Hd, L, dx)).astype(np.float32)
    ik = rng.standard_normal((4, S, dx)).astype(np.float32)
    keep = np.array(Attention._block_keep(me, mx.array(iq), mx.array(ik), mx.array(offs)))
    for b, o in enumerate(offs):
        pad = S - o - L
        solo = np.array(Attention._block_keep(me, mx.array(iq[b:b + 1]), mx.array(ik[b:b + 1, pad:]), o))
        assert not keep[b, :, :, :pad].any()
        np.testing.assert_array_equal(keep[b, :, :, pad:], solo[0], err_msg=f"row {b}")
