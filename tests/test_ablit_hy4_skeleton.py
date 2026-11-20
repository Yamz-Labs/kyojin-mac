"""Decensor hook on the Hy4 generic skeleton (CPU, synthetic 2-layer Hy4 GGUF of test_skeleton_hy4_e2e): KYOJIN_ABLIT_SPEC arms Model.layers.
  - spec with all weights 0 (and no spec): the same logits bit for bit (array_equal); unset env arms nothing
  - armed blocks run the unfused path; a non-zero spec changes the logits and equals the formula y - w ((r.y) - mu) r applied after the post norms
  - hidden mismatch is refused; a per-layer r spec (r_per_layer) arms one direction per layer"""
import json
import os
import sys

import numpy as np
import pytest
from safetensors.numpy import save_file

sys.path.insert(0, os.path.dirname(__file__))
import test_skeleton_hy4_e2e as T  # noqa: E402  (sets the CPU device)
import mlx.core as mx  # noqa: E402
from kyojin_mlx import ablit as AB  # noqa: E402
from kyojin_mlx import foreign_skeleton as SK  # noqa: E402
from kyojin_mlx import foreign_skeleton_rows as R  # noqa: E402

if os.environ.get("KYOJIN_METAL_TESTS") == "1":                  # Metal stage-0: same tests on the GPU stream (the e2e module pins the CPU)
    mx.set_default_device(mx.gpu)
D = T.D


@pytest.fixture(scope="module")
def gguf(tmp_path_factory):
    rng = np.random.default_rng(7)
    t = T._model_tensors(rng)
    table = rng.integers(0, 256, (sum(T.PRIMES), T.HD)).astype(np.uint8)
    table[(table & 127) == 127] = 0x38
    p = str(tmp_path_factory.mktemp("hy4abl") / "hy4_2l.gguf")
    T._write(p, t, table)
    return p


def _spec(tmp_path, aw, mw, per_layer=False, hidden=D, seed=2):
    rng = np.random.default_rng(seed)
    r = rng.standard_normal((2, hidden) if per_layer else (hidden,)).astype(np.float32)
    r /= np.linalg.norm(r, axis=-1, keepdims=True)
    spec = {"hidden": hidden, "n_layers": 2, "attn_w": aw, "mlp_w": mw, "attn_mu": [0.1, -0.2], "mlp_mu": [0.05, 0.3]}
    if per_layer:
        spec["r_per_layer"] = True
    p = str(tmp_path / "edit_spec.json")
    json.dump(spec, open(p, "w"))
    save_file({"r": r}, str(tmp_path / "edit_spec.safetensors"))
    return p, r


def _logits(gguf, env=None, monkeypatch=None):
    if env is None:
        monkeypatch.delenv(AB.ENV, raising=False)
    else:
        monkeypatch.setenv(AB.ENV, env)
    model, _ = SK.load_cpu(gguf, row=R.ROWS[T.A])
    return model, np.array(model(mx.array([T._ids()]))[0])


def test_zero_weight_spec_is_bit_identical(gguf, tmp_path, monkeypatch):
    m0, base = _logits(gguf, None, monkeypatch)
    assert all("_ablit" not in b.__dict__ for b in m0.layers)
    p, _ = _spec(tmp_path, [0.0, 0.0], [0.0, 0.0])
    m1, z = _logits(gguf, p, monkeypatch)
    assert np.array_equal(z, base) and all("_ablit" not in b.__dict__ for b in m1.layers)


def test_armed_changes_output_and_matches_formula(gguf, tmp_path, monkeypatch):
    p, r = _spec(tmp_path, [0.0, 1.3], [0.7, 0.0])
    m, got = _logits(gguf, p, monkeypatch)
    assert "_ablit" in m.layers[0].__dict__ and "_ablit" in m.layers[1].__dict__
    _, base = _logits(gguf, None, monkeypatch)
    assert not np.array_equal(got, base)
    # reference: no spec, the same edit applied by hand after the post norms of layer 0 mlp and layer 1 attention
    m2, _ = _logits(gguf, None, monkeypatch)
    rr = r.astype(np.float64)

    def edit(norm, w, mu):
        def f(x):
            y = np.array(norm(x)).astype(np.float64)
            return mx.array((y - w * ((y @ rr) - mu)[..., None] * rr).astype(np.float32))
        return f
    m2.layers[0].__dict__["post_mlp_layernorm"] = edit(m2.layers[0].post_mlp_layernorm, 0.7, 0.05)
    m2.layers[1].__dict__["post_self_attn_layernorm"] = edit(m2.layers[1].post_self_attn_layernorm, 1.3, -0.2)
    want = np.array(m2(mx.array([T._ids()]))[0])
    assert np.abs(got - want).max() / np.abs(want).max() < 1e-4


def test_per_layer_r_and_refusals(gguf, tmp_path, monkeypatch):
    p, r = _spec(tmp_path, [0.5, 0.5], [0.5, 0.5], per_layer=True)
    m, got = _logits(gguf, p, monkeypatch)
    assert m.layers[0].__dict__["_ablit"].R.shape == (1, D) and not np.allclose(np.array(m.layers[0].__dict__["_ablit"].R), np.array(m.layers[1].__dict__["_ablit"].R))
    assert np.isfinite(got).all()
    bad, _ = _spec(tmp_path, [0.5, 0.5], [0.5, 0.5], hidden=D + 1)
    monkeypatch.setenv(AB.ENV, bad)
    with pytest.raises(ValueError):
        SK.load_cpu(gguf, row=R.ROWS[T.A])


def test_armed_with_zero_weights_is_bit_identical(gguf, tmp_path, monkeypatch):
    """a block that is armed (unfused path, hook calls in place) with w = 0 returns its sub-layer outputs untouched"""
    from types import SimpleNamespace
    _, base = _logits(gguf, None, monkeypatch)
    m, _ = _logits(gguf, None, monkeypatch)
    for L, blk in enumerate(m.layers):
        blk.__dict__["_ablit"] = SimpleNamespace(L=L, wa=0.0, wm=0.0, R=mx.zeros((1, D)), mua=None, mum=None)
    assert np.array_equal(np.array(m(mx.array([T._ids()]))[0]), base)
