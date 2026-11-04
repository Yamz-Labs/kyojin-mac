"""Oracle-draft probe (flag oracle_draft): the recorded continuation is the draft list, the verify and the accept loop are the served ones,
and a stale record (another prompt) drafts nothing. On the fake backend, whose next token is a hash of the whole history."""
import hashlib
import json

import numpy as np
import pytest

from test_serve_spec import FakeSpec, run


def _ph(ids):
    return hashlib.md5(np.asarray(ids, dtype=np.int32).tobytes()).hexdigest()[:8]


def _one(tmp_path, monkeypatch, reqs, **kw):
    from kyojin_mlx.serve import roundlog as R
    R.reset_sink()
    f = tmp_path / "r.jsonl"
    if f.exists():
        f.unlink()
    kw.setdefault("mtp_depth", 3)
    monkeypatch.setenv("KYOJIN_ROUND_LOG", "1")
    monkeypatch.setenv("KYOJIN_ROUND_LOG_PATH", str(tmp_path / "r.jsonl"))
    monkeypatch.setenv("KYOJIN_ROUND_LOG_MAX", "5000")
    return run(reqs, FakeSpec(err=0), sessions=1, prefix_bytes=0, **kw)


def _rounds(tmp_path):
    from kyojin_mlx.serve import roundlog as R
    R.reset_sink()
    recs = [json.loads(x) for x in (tmp_path / "r.jsonl").read_text().splitlines()]
    return [r for r in recs if "end" not in r and r.get("k") == "r"]


def test_oracle_drafts_are_the_reply(tmp_path, monkeypatch):
    ids = [1, 2, 3, 4, 5, 6, 7, 8]
    got, st, be = _one(tmp_path, monkeypatch, {"a": (ids, dict(max_new=24))})
    ref = got["a"][0]
    assert len(ref) > 8
    (tmp_path / "oracle.json").write_text(json.dumps({"ph": _ph(ids), "ids": ref}))
    monkeypatch.setenv("KYOJIN_ORACLE_IDS", str(tmp_path / "oracle.json"))
    monkeypatch.setenv("KYOJIN_ORACLE_DRAFT", "1")
    monkeypatch.setenv("KYOJIN_ORACLE_ROWS", "4")
    got, st, be = _one(tmp_path, monkeypatch, {"a": (ids, dict(max_new=24))})
    assert got["a"][0] == ref                                    # the verify's tokens, not the record's: the text does not move
    assert st["spec_oracle_rounds"] > 0 and st["spec_oracle_miss"] == 0
    assert st["spec_oracle_accepted"] == st["spec_oracle_drafted"]   # every recorded draft was accepted
    assert not be.depth_asked                                    # no draft-head call: the record stood in for it
    recs = _rounds(tmp_path)
    assert recs and all(r["draft"] == r["acc"] and r["draft"] == 3 for r in recs[:-1])


def test_stale_record_drafts_nothing(tmp_path, monkeypatch):
    ids = [1, 2, 3, 4, 5, 6, 7, 8]
    (tmp_path / "oracle.json").write_text(json.dumps({"ph": "deadbeef", "ids": [9, 9, 9]}))
    monkeypatch.setenv("KYOJIN_ORACLE_IDS", str(tmp_path / "oracle.json"))
    monkeypatch.setenv("KYOJIN_ORACLE_DRAFT", "1")
    monkeypatch.setenv("KYOJIN_ORACLE_ROWS", "4")
    got, st, be = _one(tmp_path, monkeypatch, {"a": (ids, dict(max_new=16))})
    assert st["spec_oracle_rounds"] == 0 and st["spec_oracle_miss"] > 0
    assert got["a"][0]                                          # the plain step still answers


def test_align_resyncs_after_a_divergence():
    import os
    os.environ.setdefault("KYOJIN_ORACLE_ROWS", "4")
    from kyojin_mlx.serve.engine import Engine
    from test_serve_engine import Fake
    eng = Engine(FakeSpec(err=0), sessions=1, ctx=4096, chunk=4, block=8, mtp_depth=2)

    class S:
        out = [61, 62, 63, 64, 65]
    s = S()
    assert eng._oracle_align(s, [1, 2, 61, 62, 63, 64, 65, 66, 67]) == 7      # the tail is found, not the prefix
    assert eng._oracle_align(s, [61, 62, 63, 64, 65, 66]) == 5               # still following the record
    assert eng._oracle_align(s, [7, 8, 9]) is None
