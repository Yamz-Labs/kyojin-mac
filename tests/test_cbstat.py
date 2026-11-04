"""tools/cbstat.py and tools/cbill_bench.py without Metal: counter diff, row format, the bench's size cap and its chain construction."""
import os
import sys

import pytest

mx = pytest.importorskip("mlx.core")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import cbill_bench as B          # noqa: E402
import cbstat as C               # noqa: E402


def test_diff_and_row():
    a = {"t": 1.0, "act": 1 << 30, "cache": 0, "commits": 10, "gpu_us": 1000.0, "rs": [1, 5, 0, 0, 2, 3]}
    b = {"t": 1.021, "act": (1 << 30) + (4 << 20), "cache": 8 << 20, "commits": 62, "gpu_us": 21000.0, "rs": [4, 25, 0, 0, 3, 3]}
    d = C.Counters.diff(a, b)
    assert d["commits"] == 52 and abs(d["gpu_ms"] - 20.0) < 1e-9 and abs(d["wall_ms"] - 21.0) < 1e-6
    assert (d["rs_add"], d["rs_n"], d["rs_commit"], d["rs_req"]) == (3, 20, 1, 0)
    assert d["act_mb"] == 4.0 and d["cache_mb"] == 8.0
    assert "commits=52" in C.row(d)


def test_diff_without_metal_counters():
    d = C.Counters.diff({"t": 0.0, "act": 0, "cache": 0}, {"t": 0.001, "act": 0, "cache": 0})
    assert "commits" not in d and "n/a" in C.row(d)


def test_bench_geometry_and_cap():
    per = B.layer_bytes()
    assert 90 << 20 < per < 100 << 20
    assert (3.5 * 2 ** 30) // per >= 36                  # the cap leaves room for a whole trunk's dense layers minus a few
    with pytest.raises(SystemExit):
        B.Chain(40, False, 1.0)                           # over the cap: refuses before allocating


def test_chain_builds_with_own_buffers_per_layer():
    ch = B.Chain(2, False, 3.5)
    a, b = ch.L[0]["qb"]._words, ch.L[1]["qb"]._words
    assert a is not b and a.shape == b.shape
