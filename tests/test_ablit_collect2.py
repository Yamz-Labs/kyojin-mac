"""tools/ablit_collect2.py on a tiny random DS4.1 stack (stock mlx-vlm modules, fp32, CPU): the fit recovers a planted
direction and picks the layer that separates, the output spec has orthonormal rows (Gram-Schmidt against the current
r), `ablit.load` accepts it and the K=2 projection removes both rows. All strings here are synthetic; the tool never
prints sample / prompt / system text."""
import json
import os
import sys

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
pytest.importorskip("mlx_vlm.models.deepseek_v41")
import mini_ds41 as M  # noqa: E402
import ablit_collect2 as AC  # noqa: E402
from kyojin_mlx import ablit as AB  # noqa: E402

LANG = pytest.importorskip("mlx_vlm.models.deepseek_v41.language")
STOCK_CALL = LANG.DeepseekV41Block.__dict__["__call__"]
N = M.CFG["num_hidden_layers"]
H = M.CFG["hidden_size"]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(AB.ENV, raising=False)
    dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(dev)
    LANG.DeepseekV41Block.__call__ = STOCK_CALL
    AB._installed = False


def current_spec(tmp_path, seed=3):
    """A K=1 spec in the shipped layout ("r" [hidden], nonzero attn_w / mlp_w)."""
    from safetensors.numpy import save_file
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(H).astype(np.float32)
    r /= np.linalg.norm(r)
    json.dump({"hidden": H, "n_layers": N, "attn_w": [0.5] * N, "mlp_w": [0.4] * N},
              open(tmp_path / "cur.json", "w"))
    save_file({"r": r}, str(tmp_path / "cur.safetensors"))
    return str(tmp_path / "cur.json")


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def test_fit_recovers_planted_direction_and_layer():
    rng = np.random.default_rng(7)
    u = unit(rng.standard_normal(H))
    n = 20
    base = rng.standard_normal((n, H))
    planted = base + 0.2 * rng.standard_normal((n, H)) + 10.0 * u[None, :]
    delta, d = AC.fit_direction(planted[: n // 2], base[: n // 2])   # paired base rows
    assert d > 3.0
    assert float(unit(delta) @ u) > 0.99

    ids = list(range(n))
    hidden = {2: {i: base[i] for i in ids},                       # no separation
              5: {i: (planted[i] if i < n // 2 else base[i - n // 2]) for i in ids}}   # planted at layer 5
    layer, delta5, d5, stats = AC.choose_layer(hidden, ids[: n // 2], ids[n // 2:])
    assert layer == 5 and d5 > 3.0 and float(unit(delta5) @ u) > 0.99
    assert {s["layer"] for s in stats} == {2, 5}


def test_gram_schmidt_orthonormal_against_current_r():
    rng = np.random.default_rng(11)
    r = np.zeros(H)
    r[0] = 1.0
    u = unit(rng.standard_normal(H))
    u[0] = 0.0
    u = unit(u)
    d = AC.gram_schmidt(u * 4.0 + r * 2.0, r[None])
    assert abs(float(d @ r)) < 1e-6
    assert float(d @ u) > 0.9999
    assert abs(float(np.linalg.norm(d)) - 1.0) < 1e-6


def test_collect_writes_k2_spec_and_ablit_loads_it(tmp_path):
    cur = current_spec(tmp_path)
    lm, _ = M.make()
    prompts = [{"text": "alpha"}, {"text": "beta"}, {"text": "gamma"}]
    samples = [{"i": i, "prompt-index": i % 3, "reasoning": f"reasoning {i} synthetic", "content": "answer"}
               for i in range(6)]
    labels = ([0, 1, 2], [3, 4, 5], [])
    vocab = M.CFG["vocab_size"]
    encode = lambda text: [3 + (ord(c) % (vocab - 3)) for c in text]  # noqa: E731
    logs = []
    out = tmp_path / "out"
    summary = AC.collect(lm, encode, samples, prompts, "synthetic system prompt", labels, cur, str(out),
                         layers=(2, 5), chunk=16, scale=0.5, log=logs.append)
    assert summary["used"] == 6 and summary["layer"] in (2, 5)
    assert summary["R"] == [2, H]

    spec_path = str(out / "edit_spec.json")
    spec = json.load(open(spec_path))
    assert spec["attn_w2"] == [0.5 * w for w in spec["attn_w"]]
    assert spec["mlp_w2"] == [0.5 * w for w in spec["mlp_w"]]
    assert len(spec["attn_w"]) == N and len(spec["attn_w2"]) == N

    loaded = AB.load(spec_path)
    assert loaded.R.shape == (2, H)
    R = np.array(loaded.R)
    assert np.allclose(R @ R.T, np.eye(2), atol=1e-5)              # orthonormal rows

    rng = np.random.default_rng(5)
    y = mx.array(rng.standard_normal((4, H)).astype(np.float32))
    proj = AB.project(y, 1.0, mx.array(R))
    mx.eval(proj)
    p = np.array(proj)
    assert np.abs(p @ R.T).max() < 1e-4 * np.abs(np.array(y)).max()   # both rows removed

    for msg in logs:                                               # never the text
        assert "alpha" not in msg and "synthetic system" not in msg and "reasoning 0" not in msg


def test_labels_derive_comply_when_absent(tmp_path):
    p = tmp_path / "labels.json"
    json.dump({"decline": [1, 2], "partial": [5]}, open(p, "w"))
    decline, comply, excluded = AC.load_labels(str(p), [1, 2, 3, 4, 5, 6])
    assert decline == [1, 2] and excluded == [5]
    assert comply == [3, 4, 6]
