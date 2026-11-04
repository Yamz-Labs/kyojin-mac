"""The pure parts of tools/flatdec3_ab.py (no lane, no model): the document, the depth plan, the statistics and the checkpoint lookup."""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("flatdec3_ab", os.path.join(HERE, "tools", "flatdec3_ab.py"))
AB = importlib.util.module_from_spec(spec)
sys.modules["flatdec3_ab"] = AB
spec.loader.exec_module(AB)


def fake_tok(text):
    """One id per word: the tool only needs a tokenizer callable."""
    return [abs(hash(w)) % 100000 for w in text.split()]


def test_document_is_deterministic_and_sized():
    a, b, c = AB.doc_ids(5000, 11031, fake_tok), AB.doc_ids(5000, 11031, fake_tok), AB.doc_ids(5000, 11032, fake_tok)
    assert a == b and len(a) == 5000
    assert a != c                                                        # a different seed is a different document
    assert AB.digest(a, 1024) == AB.digest(b, 1024) and AB.digest(a, 1024) != AB.digest(a, 2048)


def test_depth_plan_snaps_to_the_checkpoint_grid_and_the_context():
    plan = AB.depth_plan([131072, 524288, 1040384, 2000000], 8192, 1048576, 200)
    assert plan == [("128K", 131072), ("512K", 524288), ("1M", 1040384)]   # 2M clamps onto 1M: measured once
    assert AB.depth_plan([1024, 2048], 8192, 1048576, 200) == []          # below one grid step there is no depth to measure
    for _, d in plan:
        assert d % 8192 == 0 and d + 200 <= 1048576


def test_statistics():
    assert AB.med([3, 1, 2]) == 2 and AB.med([]) != AB.med([])           # NaN on an empty arm
    assert AB.spread([10.0, 12.0]) == 18.2 and AB.spread([5.0]) is None      # (max - min) / median
    assert AB.se([10.0, 12.0]) == 1.0                                    # stdev / sqrt(n)


def test_checkpoint_lookup(tmp_path):
    ids = AB.doc_ids(4096, 7, fake_tok)
    ns = tmp_path / "ns0"
    ns.mkdir()
    p = ns / f"{4096:09d}-{AB.digest(ids, 4096)}.kv.safetensors"
    p.write_bytes(b"")
    assert AB.ckpt_file(str(tmp_path), ids, 4096) == str(p)              # the lane's digest, under its namespace
    assert AB.ckpt_file(str(tmp_path), ids, 2048) is None
    assert AB.ckpt_file("", ids, 4096) is None and AB.ckpt_file(str(tmp_path / "nope"), ids, 4096) is None
