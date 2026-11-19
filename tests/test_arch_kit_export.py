"""Neutral export (tools/archkit_export.py): plain JSON, and the transform rows (named operations, numpy reference interpreter) equal to foreign_xform on every tensor name."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import archkit_export as E  # noqa: E402
import test_gguf_arch as T  # noqa: E402
from kyojin_mlx import foreign_archkit as K, foreign_xform as XF, foreign_xform_rows as XR  # noqa: E402


def test_export_is_plain_json_with_every_architecture():
    d = E.export()
    assert json.loads(json.dumps(d)) == d and d["schema"] == E.SCHEMA
    assert len(d["architectures"]) == 169
    a = d["architectures"]["gemma2"]
    assert a["vetted"]["cfg"]["query_pre_attn_scalar"].startswith("K(") and a["transform_set"] == "gemma3"
    assert d["architectures"]["qwen35"]["config_rows"]["fields"][0] == ["hidden_size", "K('embedding_length')"]
    assert d["architectures"]["olmo2"]["vetted"] is None and any(t["key"] == "ATTN_Q_NORM" for t in d["architectures"]["olmo2"]["gguf_tensors"])
    for arch, v in K.VETTED.items():
        if v.get("xform"):
            assert d["architectures"][arch]["transform_set"] in d["transforms"], arch


def kit_pipeline(set_name, name, kv, w, n_first_axis):
    """What foreign_archkit_cpu.load_cpu does to a decoded tensor."""
    xf = XF.plan(name, set_name, kv, n_first_axis)
    if xf is None:
        return w
    if xf.cols is not None:
        w = w[:, np.argsort(xf.cols, kind="stable")]
    w = w[xf.rows] if xf.rows is not None else w
    return XF.POST[xf.post](w) if xf.post else w


def _check(set_name, arch, kv, tensors):
    n = 0
    for name, w in tensors.items():
        want = kit_pipeline(set_name, name, kv, w.copy(), w.shape[0])
        got = E.apply(set_name, arch, name, kv, w.copy())
        assert got.shape == want.shape and got.dtype == want.dtype, name
        np.testing.assert_array_equal(got, want, err_msg=name)
        n += int(not np.array_equal(want, w) or want.shape != w.shape)
    return n


@pytest.mark.parametrize("tiled", [True, False])
def test_qwen35_rows_equal_foreign_xform(tiled):
    rng = np.random.default_rng(0)
    hf = T._hf_qwen35(mtp=True)
    kv = T._kv35(**({} if tiled else {"qwen35.ssm.group_count": 6}))
    g = T._convert_qwen35(hf, dict(T.C, nk=2 if tiled else 6), T.C["L"]) if tiled else None
    if g is None:                                                  # equal head counts: any shapes with the right row counts do (no reorder happens)
        g = T._convert_qwen35(hf, dict(T.C), T.C["L"])
    tensors = {n: (-np.abs(rng.normal(size=a.shape)) - 0.1).astype(np.float32) if n.endswith("ssm_a") else rng.normal(size=a.shape).astype(np.float32) for n, a in g.items()}
    changed = _check("qwen35", "qwen35", kv, tensors)
    assert changed > (20 if tiled else 10)                         # norms, A_log, conv1d always; the V-head gathers when tiled
    if tiled:
        names = {n.split(".", 2)[2] for n in tensors if n.startswith("blk.")}
        assert {"attn_qkv.weight", "ssm_out.weight", "ssm_a", "ssm_conv1d.weight"} <= names


def test_qwen35moe_rows_equal_foreign_xform():
    rng = np.random.default_rng(1)
    g = T._convert_qwen35(T._hf_qwen35(), dict(T.C), T.C["L"])
    kv = {k.replace("qwen35.", "qwen35moe."): v for k, v in T._kv35().items()}
    tensors = {n: (-np.abs(rng.normal(size=a.shape)) - 0.1).astype(np.float32) if n.endswith("ssm_a") else rng.normal(size=a.shape).astype(np.float32) for n, a in g.items()}
    tensors["blk.0.ffn_gate_inp_shexp.weight"] = rng.normal(size=(T.C["H"],)).astype(np.float32)        # stored as a vector
    assert _check("qwen35moe", "qwen35moe", kv, tensors) > 20
    assert E.apply("qwen35moe", "qwen35moe", "blk.0.ffn_gate_inp_shexp.weight", kv, tensors["blk.0.ffn_gate_inp_shexp.weight"]).shape == (1, T.C["H"])


@pytest.mark.parametrize("set_name,arch", [("llama", "llama"), ("mistral3", "mistral3"), ("granitemoe", "granitemoe")])
def test_rope_permute_rows_equal_foreign_xform(set_name, arch):
    rng = np.random.default_rng(2)
    kv = {f"{arch}.attention.head_count": 8, f"{arch}.attention.head_count_kv": 2}
    t = {"blk.0.attn_q.weight": rng.normal(size=(8 * 16, 32)), "blk.0.attn_k.weight": rng.normal(size=(2 * 16, 32)), "blk.1.attn_q.bias": rng.normal(size=(8 * 16,)),
         "blk.0.attn_v.weight": rng.normal(size=(2 * 16, 32)), "blk.0.ffn_up.weight": rng.normal(size=(64, 32))}
    tensors = {k: v.astype(np.float32) for k, v in t.items()}
    assert _check(set_name, arch, kv, tensors) == 3


@pytest.mark.parametrize("set_name", ["gemma3", "gemma2"])
def test_norm_shift_rows_equal_foreign_xform(set_name):
    rng = np.random.default_rng(3)
    t = {"blk.0.attn_norm.weight": rng.normal(size=(32,)), "output_norm.weight": rng.normal(size=(32,)), "blk.0.attn_q.weight": rng.normal(size=(32, 32)), "blk.0.ffn_norm.weight": rng.normal(size=(32,))}
    assert _check(set_name, set_name, {}, {k: v.astype(np.float32) for k, v in t.items()}) == 3


def test_qwen2moe_shared_gate_vector_to_row():
    w = np.arange(8, dtype=np.float32)
    assert E.apply("qwen2moe", "qwen2moe", "blk.3.ffn_gate_inp_shexp.weight", {}, w).shape == (1, 8)
    assert kit_pipeline("qwen2moe", "blk.3.ffn_gate_inp_shexp.weight", {}, w, 8).shape == (1, 8)
