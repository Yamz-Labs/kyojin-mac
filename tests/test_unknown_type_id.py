"""A ggml type id this reader has no table row for (the census: id 42, Qwen3.8-Flash-Next UD-IQ3_XXS routed experts) fails PER TENSOR, naming the id and the tensor:
the file opens, the other tensors read, tools/day0_inspect reports TYPE NEEDED (never SHARDS INVALID). Id 42 is GGML_TYPE_Q2_0 upstream (ggml.h, llama.cpp master of 16/11)."""
from __future__ import annotations

import os
import shutil
import struct
import sys

import numpy as np
import pytest

from kyojin_mlx import foreign_gguf as GF, foreign_synth as SY

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import day0_inspect as D  # noqa: E402

KV = {"general.architecture": "qwen4exp", "qwen4exp.block_count": 1, "qwen4exp.expert_count": 4, "qwen4exp.expert_used_count": 2}
EXPS = "blk.0.ffn_down_exps.weight"


def _write(p, tid, shape=(64, 1), kv=None, extra=()):
    SY.write_gguf(p, kv or KV, [(EXPS, shape, "Q4_0", np.zeros(int(np.prod(shape)) // 32 * 18, np.uint8)), ("token_embd.weight", (32, 2), "F32", np.zeros((2, 32), np.float32)), *extra])
    data = bytearray(p.read_bytes())
    i = data.index(struct.pack("<I", 2) + struct.pack("<Q", 0))                    # type id of the first tensor (offset 0)
    data[i:i + 4] = struct.pack("<I", tid)
    p.write_bytes(bytes(data))


def test_upstream_only_id_fails_per_tensor(tmp_path):
    p = tmp_path / "q2_0.gguf"
    _write(p, 42)
    g = GF.GgufFile(p)                                                          # opens
    ti = g.tensors[EXPS]
    assert not ti.known and ti.tname == "ID42" and ti.nbytes == 18               # Q2_0: 64 weights in 2 + 16 bytes
    assert g.tensor_f32("token_embd.weight").shape == (2, 32)                   # the other tensors read
    with pytest.raises(GF.GgufTypeError) as e:
        g.raw(EXPS)
    msg = str(e.value)
    assert "tensor " + EXPS in msg and "type id 42" in msg and "Q2_0" in msg
    with pytest.raises(GF.GgufTypeError, match="type id 42"):
        g.tensor_f32(EXPS)


def test_id_unknown_everywhere(tmp_path):
    p = tmp_path / "x.gguf"
    _write(p, 777)
    ti = GF.GgufFile(p).tensors[EXPS]
    assert not ti.known and ti.nbytes == 0
    with pytest.raises(GF.GgufTypeError, match=r"type id 777.*not in upstream"):
        GF.GgufFile(p).raw(EXPS)


def test_known_types_unchanged():
    assert all(GF.TensorInfo("t", (256,), i, 0).known for i in GF.TYPES)
    assert not set(GF.UPSTREAM_ONLY) & set(GF.TYPES)
    assert GF.UPSTREAM_ONLY[42] == ("Q2_0", 64, 18)


def test_tool_says_type_needed_not_shards_invalid(tmp_path):
    p = tmp_path / "q2_0.gguf"
    _write(p, 42)
    r = D.inspect(str(p), use_census=False)
    assert r["verdict"] == "TYPE NEEDED" and r["unknown_type_ids"][0]["id"] == 42 and r["unknown_type_ids"][0]["tensors"] == 1
    assert "unknown weight type id 42 (1 tensors, " in r["why"] and "Q2_0" in r["why"] and EXPS in r["why"]
    assert "TYPE NEEDED" in D.render(r) and "SHARDS INVALID" not in D.render(r)


def test_tool_on_a_complete_shard_set_with_an_unknown_type(tmp_path):
    """The census case: the old reader refused the whole set and the tool printed SHARDS INVALID."""
    d = tmp_path / "sh"
    d.mkdir()
    tl = [("token_embd.weight", (32, 2), "F32", np.zeros((2, 32), np.float32)), ("blk.0.attn_q.weight", (32, 2), "F32", np.zeros((2, 32), np.float32))]
    for i in range(2):
        SY.write_gguf(d / f"m-0000{i + 1}-of-00002.gguf", {**KV, "split.no": i, "split.count": 2, "split.tensors.count": 3},
                      [tl[i]] if i == 0 else [tl[1], (EXPS, (64, 1), "Q4_0", np.zeros(36, np.uint8))])
    f = d / "m-00002-of-00002.gguf"
    data = bytearray(f.read_bytes())
    j = data.index(struct.pack("<I", 2) + struct.pack("<Q", 256))                   # the Q4_0 tensor follows the 256-byte F32 one
    data[j:j + 4] = struct.pack("<I", 42)
    f.write_bytes(bytes(data))
    r = D.inspect(str(d), use_census=False)
    assert r["verdict"] == "TYPE NEEDED" and "unknown weight type id 42" in r["why"], r["why"]
