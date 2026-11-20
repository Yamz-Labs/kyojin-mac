"""Hy4 spec layout through kyojin_mlx.ablit.load: D = 3072, 42 layers, scalar mu per layer, direction "r" [hidden] or, with
r_per_layer, "r" [n_layers, hidden] (must NOT be read as a 42-row subspace). Projection = the hook formula y - w ((r.y) - mu) r (CPU stream)."""
import json
import os
import sys

import mlx.core as mx
import numpy as np
import pytest
from safetensors.numpy import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kyojin_mlx import ablit as AB  # noqa: E402

NL, D = 42, 3072


@pytest.fixture(autouse=True)
def _cpu():
    dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(dev)


def _spec(tmp_path, per_layer, seed=1):
    rng = np.random.default_rng(seed)
    r = rng.standard_normal((NL, D) if per_layer else (D,)).astype(np.float32)
    r /= np.linalg.norm(r, axis=-1, keepdims=True)
    aw = [0.0] * 5 + [float(x) for x in rng.uniform(0.2, 3.0, NL - 5)]
    spec = {"hidden": D, "n_layers": NL, "model": "Hy4-flash", "attn_w": aw, "mlp_w": [0.5 * x for x in aw],
            "attn_mu": [float(x) for x in rng.normal(0, 0.1, NL)], "mlp_mu": [float(x) for x in rng.normal(0, 0.1, NL)]}
    if per_layer:
        spec["r_per_layer"] = True
    p = str(tmp_path / "edit_spec.json")
    json.dump(spec, open(p, "w"))
    save_file({"r": r}, str(tmp_path / "edit_spec.safetensors"))
    return p, spec, r


def test_global_r_loads(tmp_path):
    p, spec, r = _spec(tmp_path, False)
    s = AB.load(p)
    assert s.R.shape == (1, D) and s.n_layers == NL and np.allclose(s.R[0], r, atol=1e-6) and len(s.attn_mu) == NL


def test_per_layer_r_is_one_direction_per_layer(tmp_path):
    p, spec, r = _spec(tmp_path, True)
    s = AB.load(p)
    assert s.R.shape == (NL, 1, D) and np.allclose(s.R[:, 0, :], r, atol=1e-6)
    bad = dict(spec, n_layers=NL + 1, attn_w=spec["attn_w"] + [0.0], mlp_w=spec["mlp_w"] + [0.0])
    json.dump(bad, open(p, "w"))
    with pytest.raises(ValueError):
        AB.load(p)


@pytest.mark.parametrize("per_layer", [False, True])
def test_projection_matches_formula_and_zero_weight_is_identity(tmp_path, per_layer):
    p, spec, r = _spec(tmp_path, per_layer)
    s = AB.load(p)
    rng = np.random.default_rng(9)
    y = rng.standard_normal((6, D)).astype(np.float32)
    for L in (7, 30):
        R = mx.array(s.R[L] if s.R.ndim == 3 else s.R)
        rl = (r[L] if per_layer else r).astype(np.float64)
        for w, mu in ((s.attn_w[L], spec["attn_mu"][L]), (s.mlp_w[L], spec["mlp_mu"][L])):
            out = np.array(AB.project(mx.array(y), w, R, mx.array([mu], dtype=mx.float32)))
            y64 = y.astype(np.float64)
            ref = y64 - w * ((y64 @ rl) - mu)[:, None] * rl[None]
            assert np.abs(out - ref).max() < 2e-4
    z = mx.array(y)
    assert np.array_equal(np.array(AB.attn_out(type("B", (), {"__dict__": {}})(), z)), y)    # unarmed block: input returned as is


def test_spec_directory_is_accepted(tmp_path):
    p, spec, r = _spec(tmp_path, True)
    a, b = AB.load(p), AB.load(str(tmp_path))
    assert np.array_equal(a.R, b.R) and a.attn_w == b.attn_w
    assert AB.signature(p) == AB.signature(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        AB.load(str(tmp_path / "missing"))
