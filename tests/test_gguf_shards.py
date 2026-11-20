"""Split GGUF shard sets (llama.cpp gguf-split layout) read natively by foreign_gguf.GgufFile: tiny synthetic files, CPU only."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
mx.set_default_device(mx.cpu)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_gguf_arch as T  # noqa: E402
from kyojin_mlx import foreign_gguf as GF, foreign_synth as SY, foreign_archkit_cpu as KC  # noqa: E402


def _single(tmp_path):
    """Tiny qwen35 file with a few Q4_0 tensors (as-stored bytes) among F32 ones."""
    p = tmp_path / "one.gguf"
    g = T._write(p, T._hf_qwen35())
    n = [k for k, a in g.items() if k.endswith("ffn_down.weight") and a.shape[1] % 32 == 0]
    T._write(p, T._hf_qwen35(), qtype={k: "Q4_0" for k in n})
    return p


def _tensors(p):
    g = GF.GgufFile(p)
    out = [(n, g.tensors[n].ne, g.tensors[n].tname, np.array(g.raw(n))) for n in g.tensors]
    kv = dict(g.kv)
    g.close()
    return kv, out


def _hand_set(d, kv, tl, parts=3, empty_first=True, **over):
    """Shard set written by hand: shard 1 carries the metadata and (like gguf-split) may hold no tensor."""
    d.mkdir(exist_ok=True)
    groups = [[]] if empty_first else []
    k = -(-len(tl) // parts)
    rest = [tl[i:i + k] for i in range(0, len(tl), k)]
    groups += rest
    n = len(groups)
    for i, part in enumerate(groups):
        m = {**kv, "split.no": i, "split.count": n, "split.tensors.count": len(tl), **over} if i == 0 else {"general.architecture": kv["general.architecture"], "split.no": i, "split.count": n, "split.tensors.count": len(tl)}
        SY.write_gguf(d / f"m-{i + 1:05d}-of-{n:05d}.gguf", m, part)
    return d / f"m-00001-of-{n:05d}.gguf", n


def _same(a, b):
    assert set(a.tensors) == set(b.tensors) and a.kv["general.architecture"] == b.kv["general.architecture"]
    for n in a.tensors:
        x, y = a.raw(n), b.raw(n)
        assert x.shape == y.shape and x.dtype == y.dtype and x.tobytes() == y.tobytes(), n
        assert not x.flags.writeable and not y.flags.writeable and x.base is not None          # zero-copy mmap view in both
        assert a.tensors[n].tname == b.tensors[n].tname and a.tensors[n].ne == b.tensors[n].ne


def test_hand_built_set_equals_single_file(tmp_path):
    p = _single(tmp_path)
    kv, tl = _tensors(p)
    first, n = _hand_set(tmp_path / "s", kv, tl)
    assert n == 4
    s, one = GF.GgufFile(GF.find_gguf(first.parent)), GF.GgufFile(p)
    assert GF.find_gguf(first.parent) == str(first)
    assert len(s.tensors) == len(tl) and s.kv["general.architecture"] == "qwen35" and any(t.tname == "Q4_0" for t in s.tensors.values())
    _same(s, one)
    assert np.array_equal(s.tensor_f32(tl[0][0]), one.tensor_f32(tl[0][0]))
    s.close()
    s2 = GF.GgufFile(first)          # the first-shard path works too
    _same(s2, one)


def test_llama_gguf_split_binary_set_equals_single_file(tmp_path):
    exe = shutil.which("llama-gguf-split") or "/opt/homebrew/bin/llama-gguf-split"
    if not os.path.exists(exe):
        pytest.skip("llama-gguf-split not installed")
    p = _single(tmp_path)
    d = tmp_path / "bin"
    d.mkdir()
    r = subprocess.run([exe, "--split", "--split-max-tensors", "7", str(p), str(d / "m")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-300:]
    shards = sorted(os.listdir(d))
    assert len(shards) > 1 and shards[0].endswith("-of-%05d.gguf" % len(shards))
    s, one = GF.GgufFile(GF.find_gguf(d)), GF.GgufFile(p)
    _same(s, one)
    assert int(s.kv["split.count"]) == len(shards)


def test_cpu_loader_logits_identical_from_either_form(tmp_path):
    p = _single(tmp_path)
    kv, tl = _tensors(p)
    first, _ = _hand_set(tmp_path / "s", kv, tl)
    ids = mx.array([[3, 7, 1, 30, 22, 5, 9, 2]])
    outs = []
    for path in (str(p), str(first.parent), str(first)):
        m, _ = KC.load_cpu(path)
        outs.append(np.array(m(ids).astype(mx.float32)))
    assert np.array_equal(outs[0], outs[1]) and np.array_equal(outs[0], outs[2]) and np.isfinite(outs[0]).all()


def test_missing_shard_is_refused(tmp_path):
    kv, tl = _tensors(_single(tmp_path))
    first, n = _hand_set(tmp_path / "s", kv, tl)
    os.remove(first.parent / f"m-00003-of-{n:05d}.gguf")
    with pytest.raises(GF.GgufError, match="shard 3 of 4 is missing"):
        GF.GgufFile(first)
    with pytest.raises(GF.GgufError, match="shard 3 of 4 is missing"):
        GF.GgufFile(GF.find_gguf(first.parent))
    os.remove(first)
    with pytest.raises(GF.GgufError, match="first shard"):
        GF.find_gguf(first.parent)


def test_extra_shard_is_refused(tmp_path):
    kv, tl = _tensors(_single(tmp_path))
    first, n = _hand_set(tmp_path / "s", kv, tl)
    shutil.copy(first.parent / f"m-00002-of-{n:05d}.gguf", first.parent / f"m-00005-of-{n:05d}.gguf")
    with pytest.raises(GF.GgufError, match="extra shard"):
        GF.GgufFile(first)
    os.remove(first.parent / f"m-00005-of-{n:05d}.gguf")
    shutil.copy(first.parent / f"m-00002-of-{n:05d}.gguf", first.parent / "m-00002-of-00009.gguf")
    with pytest.raises(GF.GgufError, match="extra shard"):
        GF.GgufFile(first)


def test_duplicate_tensor_and_wrong_total_are_refused(tmp_path):
    kv, tl = _tensors(_single(tmp_path))
    first, n = _hand_set(tmp_path / "s", kv, tl)
    # a tensor of shard 2 repeated in shard 3
    two = GF.GgufFile(first.parent / f"m-00002-of-{n:05d}.gguf", _shard=True)
    name = next(iter(two.tensors))
    dup = next(t for t in tl if t[0] == name)
    two.close()
    three = first.parent / f"m-00003-of-{n:05d}.gguf"
    g3 = GF.GgufFile(three, _shard=True)
    tl3 = [(k, t.ne, t.tname, np.array(g3.raw(k))) for k, t in g3.tensors.items()]
    g3.close()
    SY.write_gguf(three, {"general.architecture": "qwen35", "split.no": 2, "split.count": n, "split.tensors.count": len(tl)}, tl3 + [dup])
    with pytest.raises(GF.GgufError, match="duplicate tensor"):
        GF.GgufFile(first)
    d2 = tmp_path / "t"
    first2, _ = _hand_set(d2, kv, tl, **{"split.tensors.count": len(tl) + 1})
    with pytest.raises(GF.GgufError, match="split.tensors.count"):
        GF.GgufFile(first2)


def test_non_first_shard_and_unrelated_files_are_refused(tmp_path):
    kv, tl = _tensors(_single(tmp_path))
    first, n = _hand_set(tmp_path / "s", kv, tl)
    with pytest.raises(GF.GgufError, match="FIRST shard"):
        GF.GgufFile(first.parent / f"m-00002-of-{n:05d}.gguf")
    d = tmp_path / "u"
    d.mkdir()
    shutil.copy(tmp_path / "one.gguf", d / "a.gguf")
    shutil.copy(tmp_path / "one.gguf", d / "b.gguf")
    with pytest.raises(GF.GgufError, match="not one split shard set"):
        GF.find_gguf(d)


def test_single_file_unchanged(tmp_path):
    p = _single(tmp_path)
    g = GF.GgufFile(p)
    assert g._owner == {} and g._members == [] and GF.find_gguf(p) == str(p)
    d = tmp_path / "one_dir"
    d.mkdir()
    shutil.copy(p, d / "only.gguf")
    assert GF.find_gguf(d) == str(d / "only.gguf")
