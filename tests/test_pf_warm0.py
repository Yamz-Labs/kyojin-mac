"""Flag pf_warm0 and the content counters on a table on disk (no model, no GPU)."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from kyojin_mlx import engram_mlx as EM, engram_rows as ER, flags
import test_engram_rows as TE


@pytest.fixture
def tables(tmp_path):
    sp, rws = TE.spec_with_rows()
    TE.make_shards(tmp_path, sp, rws)
    t = ER.EngramTables(str(tmp_path), sp)
    t.ahead = EM.EngramAhead(t, TE.token_map())
    yield t
    t.close()


def test_flag_registered_default_off():
    f = flags.FLAGS["pf_warm0"]
    assert f.env == "KYOJIN_PF_WARM0" and f.default == "0"


def test_warm_first_reads_the_head_rows(tables):
    ah = tables.ahead
    ids = list(np.random.default_rng(1).integers(0, 2000, 300))
    before = dict(ER.STATS)
    assert ah.warm_first(ids, 0, 300)
    tables._pre_warm.submit(lambda: None).result(10)                    # the single worker runs the jobs in order
    assert ah.warmed_first == 1 and ah.warmed == 1
    assert ER.STATS["calls"] - before["calls"] == len(tables.spec.layer_ids)
    assert ER.STATS["rows"] > before["rows"]


def test_warm_first_skips_small_and_mapless(tables):
    assert not tables.ahead.warm_first(list(range(10)), 0, 10)
    tables.ahead.token_map = None
    assert not tables.ahead.warm_first(list(range(300)), 0, 300)


def test_job_counts_distinct_rows(tables):
    ah = tables.ahead
    rep = np.tile(np.arange(5), 40)                                      # 200 positions, 5 distinct values
    hp = np.broadcast_to(rep[None, :, None], (1, 200, 24)).copy()
    _, uniq_a, _, _, _ = ah._job(0, hp)
    assert ah.rows_total == hp.size and ah.rows_uniq == uniq_a.size == 5
    rnd = np.random.default_rng(0).integers(0, int(tables.spec.num_embeddings[0]), (1, 200, 24))
    ah._job(0, rnd)
    assert ah.rows_uniq - 5 == np.unique(rnd).size > 1000
