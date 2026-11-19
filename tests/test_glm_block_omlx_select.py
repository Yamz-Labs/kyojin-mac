"""Cross-check of the G8b DFlash2 candidate selector (kyojin_mlx.glm_block.DFlash2.select) against oMLX's selector (omlx/speculative/dflash/selection.py, select_sampled,
greedy path), transcribed in numpy: top-k candidates per draft row, unary + predecessor x projected hidden x successor edge, argmax, the chosen token is the next predecessor.
Adapted from oMLX (https://github.com/jundot/omlx), Apache-2.0, omlx/speculative/dflash/selection.py (the reference loop below)."""
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from kyojin_mlx import glm_block as GB


def omlx_greedy(logits, hidden_proj, pcb, scb, anchor, k):
    """select_sampled with every sampler None: candidates = top_k of the row's logits; scores = unary + sum(pcb[pred] * proj[pos] * scb[cand]); argmax; pred = pick."""
    pred, out = anchor, []
    for pos in range(logits.shape[0]):
        cand = np.argsort(-logits[pos])[:k]
        unary = logits[pos][cand]
        edges = (pcb[pred][None] * hidden_proj[pos][None] * scb[cand]).sum(-1)
        pick = int(cand[np.argmax(unary + edges)])
        out.append(pick)
        pred = pick
    return out


@pytest.mark.parametrize("seed", range(5))
def test_select_matches_omlx_greedy_path(seed):
    rng = np.random.default_rng(seed)
    nd, V, H, rank, k = 5, 257, 16, 8, 16
    logits = rng.normal(size=(nd, V)).astype(np.float32) * 3
    hid = rng.normal(size=(nd, H)).astype(np.float32)
    w = rng.normal(size=(rank, H)).astype(np.float32)
    pcb = rng.normal(size=(V, rank)).astype(np.float32)
    scb = rng.normal(size=(V, rank)).astype(np.float32)
    anchor = int(rng.integers(V))
    me = SimpleNamespace(cfg=SimpleNamespace(sel_k=k), sel_h=lambda h: h @ mx.array(w).T, pred_cb=mx.array(pcb), succ_cb=mx.array(scb))
    ours = GB.DFlash2.select(me, mx.array(hid), mx.array(logits), anchor, nd)
    ref = omlx_greedy(logits, hid @ w.T, pcb, scb, anchor, k)
    assert [int(t) for t in ours.tolist()] == ref
