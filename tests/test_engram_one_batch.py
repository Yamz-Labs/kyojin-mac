"""Flag engram_one_batch on a table on disk (no model): the one-batch read of a decode-sized request returns the bytes of the per-tag read."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from kyojin_mlx import engram_rows as ER
import test_engram_rows as TE


@pytest.fixture
def tables(tmp_path):
    sp, rws = TE.spec_with_rows()
    TE.make_shards(tmp_path, sp, rws)
    t = ER.EngramTables(str(tmp_path), sp)
    yield t, rws
    t.close()


@pytest.mark.parametrize("cold", [False, True])
@pytest.mark.parametrize("n", [1, 7, 48])
def test_one_batch_same_bytes(tables, monkeypatch, cold, n):
    t, rws = tables
    rng = np.random.default_rng(n)
    for pos in range(2):
        rows = np.unique(rng.integers(0, rws[pos], n))
        monkeypatch.setattr(ER, "ONE_BATCH", False)
        w0, s0 = t.read_rows(pos, rows)
        monkeypatch.setattr(ER, "ONE_BATCH", True)
        if cold:
            monkeypatch.setattr(t, "COLD_S", -1.0)      # every pread counts as cold: the pool path
        w1, s1 = t.read_rows(pos, rows)
        assert np.array_equal(w0, w1) and np.array_equal(s0, s1)
