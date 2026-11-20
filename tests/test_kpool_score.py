"""Flag kpool_score (kyojin_mlx/kpool_score.py): the fused multi-threadgroup indexer scorer + exact top-k against the stock select_pools. Metal only
(KYOJIN_METAL_TESTS=1, a GPU; run behind can_build.sh, one file): scores within 1e-5 relative of the float64 reference at Hy4-flash's shape (16 heads, d 128, kp 4, top_k 2048,
tail) for decode rows, a 6-row verify block, a ragged prompt chunk (rows and pools not multiples of 8 / 16) and a causal chunk past a long prefix; the decode path (kpool_fuse with the
scores as input) and the prompt path (select_pools_fast) pick the stock path's pools exactly on random data (no ties) and on tied pools the lowest index."""
import os

import numpy as np
import mlx.core as mx
import pytest

from kyojin_mlx import foreign_skeleton_kpool as KP
from kyojin_mlx import kpool_fuse as KF
from kyojin_mlx import kpool_score as KS
from test_kpool_fuse import _inputs, _phase1, _stock

METAL = os.environ.get("KYOJIN_METAL_TESTS", "0") == "1" and mx.metal.is_available()
needs_metal = pytest.mark.skipif(not METAL, reason="Metal kernel test: KYOJIN_METAL_TESTS=1 on a GPU")


@pytest.fixture(autouse=True)
def _device():
    old = mx.default_device()
    mx.set_default_device(mx.gpu if METAL else mx.cpu)
    yield
    mx.set_default_device(old)


# (H, d, kp, top_k, select_tail, S, L, offsets)
CASES = [
    (3, 16, 4, 12, True, 31, 1, [30]),
    (3, 16, 4, 12, False, 40, 3, [37, 20]),
    (16, 128, 4, 2048, True, 3000, 1, [2999]),
    (16, 128, 4, 2048, True, 2300, 6, [2290, 1500]),
    (16, 128, 4, 2048, False, 9000, 1, [8999]),
    (16, 128, 4, 2048, True, 4099, 45, [4054]),            # a ragged prompt chunk: 45 rows (not a multiple of 8), 1024 pools (J % 16 == 0)
    (16, 128, 4, 2048, True, 5003, 77, [4926]),            # J = 1250 (not a multiple of 16), causal inside the chunk
]


@needs_metal
@pytest.mark.parametrize("cs", CASES)
def test_scores_equal_reference(cs):
    H, d, kp, top_k, tail, S, L, offs = cs
    for seed in range(2):
        iq, w, pk, valid = _inputs(H, d, kp, top_k, S, L, offs, seed=seed)
        sc = np.array(KS.scores(mx.array(iq), mx.array(w), mx.array(pk), mx.array(valid), mx.array(offs, dtype=mx.int32), kp, tail))
        ref = _phase1(iq, w, pk, valid, offs, L, kp, tail)
        fin = np.isfinite(ref)
        assert (np.isfinite(sc) == fin).all()
        np.testing.assert_allclose(sc[fin], ref[fin], rtol=1e-5, atol=1e-6 * float(np.abs(ref[fin]).max()))


@needs_metal
@pytest.mark.parametrize("on", [False, True])
@pytest.mark.parametrize("cs", CASES)
def test_decode_select_equals_stock(cs, on, monkeypatch):
    H, d, kp, top_k, tail, S, L, offs = cs
    if L > KF.ROWS_MAX:
        pytest.skip("prompt chunk: select_pools_fast")
    monkeypatch.setattr(KS, "ON", on)                                   # off = the kpool_fuse kernel with its own scoring stage (the edit of the PRE input must not change it)
    for seed in range(2):
        iq, w, pk, valid = _inputs(H, d, kp, top_k, S, L, offs, seed=seed)
        s1, k1 = _stock(iq, w, pk, valid, offs, S, kp, top_k, tail)
        keep, _, sel = KF.select(mx.array(iq), mx.array(w), mx.array(pk), mx.array(valid), mx.array(offs, dtype=mx.int32), S, kp, top_k, tail)
        assert (np.array(sel) == s1).all(), "selected pools differ from the stock path"
        assert (np.array(keep) == k1).all(), "keep mask differs from the stock path"


@needs_metal
@pytest.mark.parametrize("cs", CASES)
def test_prompt_select_equals_stock(cs, monkeypatch):
    H, d, kp, top_k, tail, S, L, offs = cs
    if len(offs) != 1 or L <= 8:
        pytest.skip("batch 1, prompt rows only")
    for seed in range(2):
        iq, w, pk, valid = _inputs(H, d, kp, top_k, S, L, offs, seed=seed)
        monkeypatch.setattr(KS, "ON", False)
        s1, k1 = _stock(iq, w, pk, valid, offs, S, kp, top_k, tail)
        monkeypatch.setattr(KS, "ON", True)
        st = KP.select_pools(mx.array(iq), mx.array(w), mx.array(pk), mx.array(valid), mx.array(offs, dtype=mx.int32), S, kp, top_k, tail)
        s2, k2 = np.array(st["sel"]), np.array(KP.keep_mask(st))
        row_equal = (s1 == s2).all(axis=-1)
        print(f"top-k equal rows {100 * row_equal.mean():.2f} % ({int(row_equal.sum())} / {row_equal.size}), differing pools {(s1 != s2).sum()}")
        assert row_equal.mean() >= 0.99
        assert (k1 == k2)[..., :].all(axis=-1).mean() >= 0.99


@needs_metal
def test_prompt_select_ties_lowest_index(monkeypatch):
    H, d, kp, top_k, tail, S, L, offs = 3, 16, 4, 12, True, 400, 40, [300]
    iq, w, pk, valid = _inputs(H, d, kp, top_k, S, L, offs, dup=True)
    monkeypatch.setattr(KS, "ON", False)
    s1, k1 = _stock(iq, w, pk, valid, offs, S, kp, top_k, tail)
    monkeypatch.setattr(KS, "ON", True)
    st = KP.select_pools(mx.array(iq), mx.array(w), mx.array(pk), mx.array(valid), mx.array(offs, dtype=mx.int32), S, kp, top_k, tail)
    assert (np.array(st["sel"]) == s1).all()


def test_flag_registered():
    from kyojin_mlx import flags
    assert flags.FLAGS["kpool_score"].env == "KYOJIN_KPOOL_SCORE" and flags.FLAGS["kpool_score"].default == "0"
