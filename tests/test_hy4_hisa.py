# SPDX-License-Identifier: Apache-2.0
"""KYOJIN_HY4_HISA: the hierarchical candidate-restricted prompt indexer (kyojin_mlx/kpool_hisa.py). CPU only.
 * every block kept == the exact selection (random data, both head-weight signs, row counts that are not a tile multiple, with / without the select tail);
 * a small block budget still finds planted high-score pools (the Quest bound cannot dilute a lone needle);
 * the model path (pools_fixed_top -> gather attend) with every block kept equals the concatenating reference."""
import os
import sys

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kyojin_mlx import kpool_hisa as HI  # noqa: E402
from kyojin_mlx import foreign_skeleton_mla as MLA  # noqa: E402
from test_hy4_fixed_kv import _ref, _fixed, _diff  # noqa: E402
from test_hy4_attn_slice import _kp_model, _chunks  # noqa: E402


def _exact(iq, w, pk, off, kp, n_top, select_tail):
    """numpy: the exact top-n_top pool set of every row (sets), the production score relu(iq . pk) * w summed over heads."""
    H, L, d = iq.shape
    J = pk.shape[0]
    sc = (np.maximum(np.einsum("hld,jd->hlj", iq, pk), 0.0) * w.T[:, :, None]).sum(0)
    out = []
    for r in range(L):
        q = off + r
        nv = min(q // kp if select_tail else (q + 1) // kp, J)
        s = sc[r, :nv]
        o = np.argsort(-s, kind="stable")[:n_top]
        out.append(set(o.tolist()))
    return out


def _run(iq, w, pk, off, kp, n_top, select_tail, prm):
    J = pk.shape[0]
    t, ok = HI.top(mx.array(iq)[None], mx.array(w)[None], mx.array(pk)[None], mx.ones((1, J), dtype=mx.bool_), mx.array([off], dtype=mx.int32), kp, n_top, select_tail, prm)
    mx.eval(t, ok)
    t, ok = np.array(t)[0], np.array(ok)[0]
    return [set(t[r][ok[r]].tolist()) for r in range(t.shape[0])], t, ok


@pytest.mark.parametrize("select_tail", [True, False])
@pytest.mark.parametrize("L,off", [(40, 100), (33, 0), (64, 200)])
def test_all_blocks_kept_equals_exact(L, off, select_tail):
    rng = np.random.default_rng(3)
    H, d, kp, J, n_top, bp = 4, 16, 4, 96, 10, 4
    iq = rng.standard_normal((H, L, d)).astype(np.float32)
    w = rng.standard_normal((L, H)).astype(np.float32)                          # both signs
    pk = rng.standard_normal((J, d)).astype(np.float32)
    got, t, ok = _run(iq, w, pk, off, kp, n_top, select_tail, dict(bp=bp, m=J // bp, tile=16, first=2, last=2))
    ref = _exact(iq, w, pk, off, kp, n_top, select_tail)
    assert got == ref
    assert (np.diff(np.where(ok, t, 10 ** 6), axis=-1) >= 0).all()             # ascending, empty slots last


def test_small_budget_finds_planted_pools():
    rng = np.random.default_rng(5)
    H, d, kp, J, n_top, bp, T, L = 4, 32, 4, 512, 6, 8, 16, 64
    off = J * kp - L                                                             # the chunk is the end of the context: every pool but the tail is visible
    u = rng.standard_normal((L // T, d)).astype(np.float32)
    iq = np.stack([np.repeat(u, T, axis=0) + 0.2 * rng.standard_normal((L, d)).astype(np.float32) for _ in range(H)])
    w = np.abs(rng.standard_normal((L, H))).astype(np.float32) + 0.5
    pk = 0.3 * rng.standard_normal((J, d)).astype(np.float32)
    need = {}
    for ti in range(L // T):
        ids = rng.choice(np.arange(8, J - 40), 3, replace=False)
        need[ti] = set(ids.tolist())
        for i in ids:
            pk[i] = 4.0 * u[ti] / np.linalg.norm(u[ti])
    got, _, _ = _run(iq, w, pk, off, kp, n_top, True, dict(bp=bp, m=8, tile=T, first=2, last=2))   # 4 forced + 4 free blocks for 3 needles
    for r in range(L):
        assert need[r // T] <= got[r], (r, need[r // T], got[r])


def test_model_path_equals_reference(monkeypatch):
    """pools_fixed_top -> HISA (every block kept) -> gather slots -> attend, prefill chunks + decode rows, equals the concatenating reference."""
    rng = np.random.default_rng(11)
    d = 64
    m = _kp_model(rng, d, 4, 32, 24, 8, 16, 2, 16, top_k=32)
    m.swa, m.window = False, 0
    xs = _chunks(rng, d, 150, 6)
    xs = [xs[0][:, :90], xs[0][:, 90:]] + xs[1:]
    ref = _ref(m, xs, 0)
    monkeypatch.setenv("KYOJIN_HY4_FIXED_GATHER", "1")
    monkeypatch.setenv("KYOJIN_HY4_FIXED_GATHER_DEC", "1")
    monkeypatch.setenv("KYOJIN_HY4_HISA", "1")
    calls = []

    def wanted(L, B, J, n_top):
        if L <= 16 or B != 1 or J % 2:
            return None
        calls.append((L, J))
        return dict(bp=2, m=J // 2, tile=8, first=2, last=2)
    monkeypatch.setattr(HI, "wanted", wanted)
    new = _fixed(m, xs, 64, False)
    assert calls, "the hierarchical path did not run on the prompt chunks"
    assert _diff(ref, new) <= 1e-5
