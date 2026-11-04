"""DS4.1 with the fp8 fake quant ON and the NAX MoE prefill: repeated identical forwards (prefill, then one decode step with the cache) must be bit-identical
in every recorded tensor (per-layer attention, FFN, layer output, logits), in one process. Needs the shrunk ds2 pack: KYOJIN_SHRUNK=<dir holding ds2/>
(default models/ds2 of the repo or ~/kyojin-mlx) and mlx-vlm."""
import os
import sys

import numpy as np
import pytest

SHRUNK = os.environ.get("KYOJIN_SHRUNK") or next((d for d in (os.path.join(os.path.dirname(__file__), "..", "models"), os.path.expanduser("~/kyojin-mlx/models"))
                                                    if os.path.isdir(os.path.join(d, "ds2"))), None)


@pytest.mark.skipif(not SHRUNK, reason="no shrunk ds2 pack")
@pytest.mark.parametrize("rows", [5, 200])
def test_fq_on_forward_is_bit_identical(rows, monkeypatch):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_vlm")
    monkeypatch.setenv("KYOJIN_PF_NAX", "1")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    import layer_gate as LG
    from kyojin_mlx import patch as P
    pack = os.path.join(SHRUNK, "ds2")
    model, spec, cfg = P.build_model(pack)
    P.patch_model(model, pack, spec, cfg)
    toks = np.random.default_rng(rows).integers(100, cfg["text_config"]["vocab_size"] - 100, rows).tolist()
    runs = [LG.forward(model, spec.name, toks) for _ in range(4)]
    for r in runs[1:]:
        bad = [k for k in runs[0] if not k.endswith(".router") and not np.array_equal(runs[0][k], r[k])]
        assert not bad, f"non-deterministic: {bad[:6]}"
