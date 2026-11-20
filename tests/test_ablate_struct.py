"""ablate_struct (CPU only): the structural KYOJIN_ABLATE classes on the synthetic 2-layer Hy4-shaped model of test_skeleton_hy4_e2e: every class changes the decode
output (the stand-in really replaces the op), keeps the logits shape, leaves prefill alone, and switching the flag off restores the exact base logits."""
import numpy as np
import pytest

import mlx.core as mx

mx.set_default_device(mx.cpu)

from mlx_lm.models.cache import KVCache  # noqa: E402

from kyojin_mlx import ablate as AB  # noqa: E402
from kyojin_mlx import ablate_struct as AS  # noqa: E402
from test_skeleton_hy4_e2e import built, _ids  # noqa: E402,F401

CLASSES = ("attn", "attn_proj", "attn_core", "attn_out", "idx", "mlp", "router", "moe_routed", "moe_shared", "hc", "norm", "engram", "head")


def _decode(model):
    ids = _ids()
    cache = [KVCache() for _ in model.layers]
    model(mx.array([ids[:9]]), cache=cache)
    out = model(mx.array([[ids[9]]]), cache=cache)
    mx.eval(out)
    return np.array(out)


@pytest.fixture(autouse=True)
def _off(monkeypatch):
    monkeypatch.setattr(AS, "MAX_ROWS", 4)                               # the toy prompts (9 and 13 rows) count as prefill, one token as decode
    AB.ACTIVE = AB.parse("")
    yield
    AB.ACTIVE = AB.parse("")


def test_every_class_changes_the_decode_step_and_off_restores(built):
    model = built[3]
    base = _decode(model)
    again = _decode(model)
    assert np.array_equal(base, again)
    AB.ACTIVE = AB.parse(",".join(CLASSES))                              # installs the wrappers; this first ablated decode runs real and records the output shapes
    _decode(model)
    for c in CLASSES:
        AB.ACTIVE = AB.parse(c)
        AB.HITS.clear()
        got = _decode(model)
        assert got.shape == base.shape, c
        assert AB.HITS[c] > 0, f"{c}: stand-in never ran"
        assert not np.array_equal(got, base), f"{c}: output unchanged"
        AB.ACTIVE = AB.parse("")
        assert np.array_equal(_decode(model), base), f"{c}: flag off did not restore"


def test_prefill_rows_are_never_ablated(built):
    model = built[3]
    ids = _ids()
    base = np.array(model(mx.array([ids])))
    AB.ACTIVE = AB.parse(",".join(CLASSES))
    assert np.array_equal(np.array(model(mx.array([ids]))), base)
    assert AS.DECODE[0] is False
