"""The per-round depth-policy trace (flag round_log): off by default, bounded, and the tok/s identity the A/B prices a plan with."""
import json
import os

import pytest

from kyojin_mlx.serve import roundlog as R


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    monkeypatch.setenv("KYOJIN_ROUND_LOG", "0")
    monkeypatch.setenv("KYOJIN_ROUND_LOG_PATH", str(tmp_path / "r.jsonl"))
    monkeypatch.setenv("KYOJIN_ROUND_LOG_MAX", "4")
    R.reset_sink()
    yield
    R.reset_sink()


def test_off_by_default_writes_nothing(tmp_path):
    stats = {}
    from collections import Counter
    st = Counter()
    assert R.enabled() is False
    assert R.record({"k": "r", "tok": 3, "ms": 10.0}, st) is False
    assert not (tmp_path / "r.jsonl").exists()
    assert st["rlog_rounds"] == 0


def test_bounded_sink_and_stats_totals(tmp_path):
    import collections
    os.environ["KYOJIN_ROUND_LOG"] = "1"
    st = collections.Counter()
    for i in range(10):
        assert R.record({"k": "r" if i % 2 == 0 else "p", "tok": 2, "ms": 45.0, "draft": 1, "acc": i % 2, "r": i}, st) is (i < 4)
    lines = (tmp_path / "r.jsonl").read_text().splitlines()
    assert len(lines) == 5                                             # 4 records + the end marker, never more
    assert json.loads(lines[-1]) == {"end": 4}
    # the live totals are the caller's, not the file's: they keep counting after the bound closed the sink (an A/B prices a run of
    # requests that is longer than one file), while the file itself stops at the bound
    assert st["rlog_rounds"] == 10 and st["rlog_tok"] == 20 and st["rlog_ms"] == pytest.approx(450.0)
    assert st["rlog_draft"] == 10 and st["rlog_acc"] == 5
    assert st["rlog_plain"] == 5


def test_median_lower():
    assert R.median([]) == 0.0
    assert R.median([9.0, 5.0, 7.0]) == 7.0
    assert R.median([9.0, 5.0, 7.0, 1.0]) == 5.0


def test_engine_traces_every_round_of_a_reply(monkeypatch, tmp_path):
    """The engine hook: one line per generated token group, sum(tok) = the reply's tokens, the record carries the plan and the round ms."""
    monkeypatch.setenv("KYOJIN_ROUND_LOG", "1")
    monkeypatch.setenv("KYOJIN_ROUND_LOG_MAX", "5000")
    from test_serve_spec import FakeSpec, prompts, run
    reqs = {"a": ([1, 2, 3], dict(max_new=60))}
    got, st, be = run(reqs, FakeSpec(err=0), sessions=1, mtp_depth=3, prefix_bytes=0)
    recs = [json.loads(x) for x in (tmp_path / "r.jsonl").read_text().splitlines()]
    recs = [r for r in recs if "end" not in r]
    assert recs and all(r["k"] in ("r", "p") for r in recs)
    assert sum(r["tok"] for r in recs) == len(got["a"][0]) - 1          # every token after the prefill sample is in one record
    assert st["rlog_rounds"] == len(recs) and st["rlog_tok"] == sum(r["tok"] for r in recs)
    first = recs[0]
    assert {"pl", "rows", "draft", "cut", "acc", "tok", "ms", "ct", "pm", "cm", "dec"} <= set(first)
    assert first["dec"] in ("cap", "plain", "probe")
    assert all(r["ph"] == first["ph"] for r in recs)                    # one prompt tag for the reply
    assert [r["r"] for r in recs] == sorted(r["r"] for r in recs)       # round index increases


def test_engine_leaves_no_file_when_off(monkeypatch, tmp_path):
    from test_serve_spec import FakeSpec, run
    run({"a": ([1, 2, 3], dict(max_new=20))}, FakeSpec(err=0), sessions=1, mtp_depth=3, prefix_bytes=0)
    assert not (tmp_path / "r.jsonl").exists()
