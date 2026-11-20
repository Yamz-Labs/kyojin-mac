"""Item 6: a cache test for MoE layers that also carry the sliding window and the attention gate (token-by-token decode equals the full forward)."""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
mx.set_default_device(mx.cpu)

_spec = importlib.util.spec_from_file_location("tp2", os.path.join(os.path.dirname(__file__), "test_foreign_skeleton_part2.py"))
P2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P2)
F, IDS, TOL, run = P2.F, P2.IDS, P2.TOL, P2.run


@pytest.mark.parametrize("gate", ["head", "element"])
def test_moe_window_gate_cache(tmp_path, gate):
    fkw = dict(window=3, pattern=[1, 0, 1, 1], gate=gate, gate_bias=True, rope_on="all")
    model, got, want, diff = run(tmp_path, seed=5, fkw=fkw, mkw=dict(shared=True))
    print("moe + window + %s gate max abs diff vs numpy %.3g" % (gate, diff))
    assert diff < TOL * max(1.0, np.abs(want).max())
    cd = F.cache_consistent(model, got)
    print("moe + window + %s gate token-by-token vs full forward %.3g" % (gate, cd))
    assert cd < 1e-4
    assert model.layers[0].self_attn.swa and model.layers[0].self_attn.gate == gate
    # batch cache with the window: a second prompt of the same length gives the same rows
    from mlx_lm.models.cache import make_prompt_cache
    c = make_prompt_cache(model)
    a = np.array(model(mx.array(IDS[None, :7]), cache=c))[0]
    assert np.abs(a - got[:7]).max() < 1e-4
