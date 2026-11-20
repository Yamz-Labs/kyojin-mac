"""tools/archkit_logit_check.py near-tie awareness on a synthetic tiny MoE (CPU)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
mx.set_default_device(mx.cpu)
import mlx.nn as nn  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import archkit_logit_check as L  # noqa: E402


class Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.gate = nn.Linear(4, 6, bias=False)
        self.gate.weight = mx.array(w)
        self.up_proj = nn.Linear(4, 6, bias=False)         # name ends in _proj and has 6 outputs: must NOT be tapped

    def __call__(self, x):
        return x + self.gate(x).sum(-1, keepdims=True) * 0.0


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        w = np.zeros((6, 4), np.float32)
        w[:4] = np.eye(4)
        self.layers = [Block(w), Block(w * 1.0)]

    def __call__(self, x):
        for b in self.layers:
            x = b(x)
        return x


X = np.array([[[0.5, 0.4, 0.3, 0.2], [0.5, 0.2, 0.2 + 1e-5, 0.0], [0.9, 0.1, 0.05, 0.0]]], np.float32)


def test_tap_records_the_smallest_kth_gap_per_position():
    m = Tiny()
    tap, found = L.tap_routers(m, 6)
    assert [f[0] for f in found] == ["layers.0.gate", "layers.1.gate"] and [f[1] for f in found] == [0, 1]
    m(mx.array(X))
    g = L.position_gaps(tap, found, 2)
    # logits per position = x padded with 2 zeros; k = 2: gap between the 2nd and the 3rd largest
    want = []
    for row in X[0]:
        lg = sorted(list(row) + [0.0, 0.0], reverse=True)
        want.append(lg[1] - lg[2])
    assert np.allclose(g, want, atol=1e-7) and g[1] < 2e-5 and g[0] > 0.09
    assert L.first_tie(g, 1e-3) == 1 and L.first_tie(g, 1e-7) is None and L.first_tie(None, 1e-3) is None
    # per-layer k: layer 0 uses 1 expert (gap 1st-2nd), layer 1 uses 2: the minimum over layers is taken
    g2 = L.position_gaps(tap, found, [1, 2])
    assert np.allclose(g2, np.minimum([0.0 + (0.5 - 0.4), 0.5 - 0.2, 0.9 - 0.1], g), atol=1e-7)
    tap.reset()
    assert all(not v for v in tap.layers.values())


def test_two_comparisons_hide_a_tie_but_not_a_real_error():
    rng = np.random.default_rng(0)
    V, n = 16, 8
    ref = np.log(rng.dirichlet(np.ones(V) * 0.3, n)).astype(np.float32)
    valid = np.ones((n, V), bool)
    ours = ref.copy()
    ours[3:] = np.roll(ours[3:], 1, axis=1)                  # from position 3 on: a flipped expert changed everything (top-1 differs)
    ft = 3                                                   # near-tie at position 3
    allp, clean = L.Stats(), L.Stats()
    for j in range(n):
        allp.add(ours[j], ref[j], valid[j])
        if L.kept_before_tie(j, ft):
            clean.add(ours[j], ref[j], valid[j])
    assert allp.n == 8 and clean.n == 3 and clean.top1 == 3 and clean.dmax == 0.0 and allp.top1 < 8
    # without a tie the second line equals the first: a real error is not hidden
    real = ref.copy()
    real[5] = np.roll(real[5], 1)
    a2, c2 = L.Stats(), L.Stats()
    for j in range(n):
        a2.add(real[j], ref[j], valid[j])
        if L.kept_before_tie(j, None):
            c2.add(real[j], ref[j], valid[j])
    assert a2.n == c2.n == 8 and a2.top1 == c2.top1 < 8 and a2.dmax == c2.dmax > 0
    assert "top-1 agreement 3/3" in clean.line() and "nothing to compare" in L.Stats().line()


def test_no_router_found_is_reported_not_guessed():
    m = Tiny()
    tap, found = L.tap_routers(m, 7)                         # no linear has 7 outputs
    assert found == [] and L.position_gaps(tap, found, 2) is None
