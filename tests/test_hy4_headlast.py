"""Hy4 prefill, head on the last row (CPU only): inside headlast.last_row() a chunk above LAST_ROW_MIN rows returns [1, 1, vocab] that equals the
last row of the full logits bit for bit; outside the scope (and at or below LAST_ROW_MIN rows) every row is returned."""
import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)

from test_skeleton_hy4_e2e import built, _ids  # noqa: E402,F401
from kyojin_mlx import headlast  # noqa: E402


def test_last_row_scope_equals_last_row_of_all_rows(built, monkeypatch):
    _, _, _, model, _ = built
    ids = mx.array([_ids()])
    full = np.array(model(ids))
    assert full.shape[1] == ids.shape[1] and full.shape[1] > 4
    monkeypatch.setattr(headlast, "LAST_ROW_MIN", 4)
    with headlast.last_row():
        one = np.array(model(ids))
    assert one.shape == (1, 1, full.shape[2])
    assert np.array_equal(one[0, 0], full[0, -1])
    with headlast.last_row():                                              # at or below the limit: every row
        short = np.array(model(ids[:, :4]))
    assert short.shape[1] == 4
    assert np.array_equal(np.array(model(ids)), full)                    # outside the scope again: every row


def test_flag_off_keeps_every_row(built, monkeypatch):
    _, _, _, model, _ = built
    ids = mx.array([_ids()])
    monkeypatch.setattr(headlast, "LAST_ROW_MIN", 4)
    monkeypatch.setattr(headlast, "ENABLED", False)
    with headlast.last_row():
        assert model(ids).shape[1] == ids.shape[1]
