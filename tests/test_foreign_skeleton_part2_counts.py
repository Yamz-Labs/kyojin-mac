"""Item 5: kv_counts takes a numpy array like a list / tuple; in the skeleton MoE path a per-layer expert_count array is checked against each layer's router rows."""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
mx.set_default_device(mx.cpu)

from kyojin_mlx import foreign_gguf as GF, foreign_skeleton as SK, foreign_skeleton_moe as SM  # noqa: E402

_spec = importlib.util.spec_from_file_location("tp2", os.path.join(os.path.dirname(__file__), "test_foreign_skeleton_part2.py"))
P2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P2)


@pytest.mark.parametrize("v", [[4, 6, 6], (4, 6, 6), np.array([4, 6, 6]), np.array([4, 6, 6], dtype=np.uint32)])
def test_kv_counts_takes_list_tuple_and_numpy(v):
    assert GF.kv_counts(v) == (4, 6, [4, 6, 6])
    with pytest.raises(GF.GgufError):
        GF.kv_count_uniform(v)
    assert GF.kv_count_uniform(np.array([6, 6])) == 6
    assert GF.kv_counts(np.int64(7)) == (7, 7, None) and GF.kv_counts(None) == (0, 0, None) and GF.kv_counts(np.array([], dtype=np.int64)) == (0, 0, None)


def _detect(tmp_path, counts_kv, exp_counts):
    rng = np.random.default_rng(0)
    path0, row, T, o = P2.F.make(tmp_path, rng)
    path = P2.moeify(tmp_path, path0, T, rng, exp_counts=exp_counts)
    g = GF.GgufFile(path)
    kv = dict(g.kv)
    kv["synth.expert_count"] = counts_kv
    return SM.detect("synth", kv, {k: ti.ne for k, ti in g.tensors.items()}, row)


def test_per_layer_expert_count_array_is_checked_against_the_router_rows(tmp_path):
    rows = [6, 5, 6, 4]
    det = _detect(tmp_path, np.array(rows, dtype=np.uint32), rows)                    # numpy array that matches: placed, per-layer counts follow the rows
    assert det is not None
    for kind in (list, tuple, np.array):
        with pytest.raises(SK.SkeletonError, match=r"layer 1: expert_count 6 but the router has 5 rows"):
            _detect(tmp_path, kind([6, 6, 6, 4]), rows)
    with pytest.raises(SK.SkeletonError, match="expert_count has 2 entries, layer 2"):
        _detect(tmp_path, np.array([6, 5]), rows)
    longer = _detect(tmp_path, np.array(rows + [8, 8]), rows)                           # entries past the decoded layers (MTP blocks) are not checked
    assert longer is not None
    with pytest.raises(SK.SkeletonError, match="router has 6 rows"):                    # a scalar is the largest count: a router wider than it is refused
        _detect(tmp_path, 5, rows)
    assert _detect(tmp_path, 6, rows) is not None


def test_per_layer_counts_load_through_the_loader(tmp_path):
    rng = np.random.default_rng(1)
    path0, row, T, o = P2.F.make(tmp_path, rng)
    path = P2.moeify(tmp_path, path0, T, rng, exp_counts=[6, 5, 6, 4])
    model, _ = SK.load_cpu(path, row=row)
    assert [l.mlp.moe_spec.n_expert for l in model.layers] == [6, 5, 6, 4]


def test_moe_every_n_layers_key_is_accepted_and_the_router_tensors_decide(tmp_path):
    rng = np.random.default_rng(2)
    path0, row, T, o = P2.F.make(tmp_path, rng)
    path = P2.moeify(tmp_path, path0, T, rng, extra_kv={"synth.moe_every_n_layers": 1})
    model, _ = SK.load_cpu(path, row=row)
    assert all(hasattr(l.mlp, "moe_spec") for l in model.layers)


def test_kv_counts_one_element_array_is_a_one_element_list():
    for v in (np.array([6]), np.array([6], dtype=np.uint32)):
        assert GF.kv_counts(v) == GF.kv_counts([6]) == (6, 6, [6]) and GF.kv_count_uniform(v) == 6
