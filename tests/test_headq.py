# SPDX-License-Identifier: Apache-2.0
"""Flag head_q: the trunk output head through a packed 8-bit copy (CPU, tiny shapes)."""
import mlx.core as mx
import mlx.nn as nn
import pytest

from kyojin_mlx import flags as F
from kyojin_mlx import headq as HQ


@pytest.fixture(autouse=True)
def _cpu_and_restore():
    dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    old = HQ.BITS
    yield
    HQ.BITS = old
    HQ._PACKS.clear()
    mx.set_default_device(dev)


def _lm(v=256, d=128):
    mx.random.seed(3)
    lm = nn.Linear(d, v, bias=False)
    lm.weight = mx.random.normal((v, d)) * 0.05
    mx.eval(lm.weight)
    return lm


def test_off_and_shape_gates():
    lm = _lm()
    h = mx.random.normal((1, 4, 128))
    HQ.BITS = 0
    assert HQ.project(lm, h) is None
    HQ.BITS = 8
    assert HQ.project(lm, mx.random.normal((1, HQ.MAX_ROWS + 1, 128))) is None      # prompt chunks keep the dense head
    assert HQ.project(nn.Linear(100, 8, bias=False), mx.zeros((1, 1, 100))) is None   # D % 64 != 0
    assert HQ.packed(mx.zeros((4, 64), dtype=mx.int32), 8) is None


def test_q8_close_to_dense_and_one_copy():
    lm = _lm()
    h = mx.random.normal((1, 6, 128))
    HQ.BITS = 8
    y = HQ.project(lm, h)
    ref = lm(h)
    assert y.shape == ref.shape
    rel = float(mx.abs(y - ref).max() / mx.abs(ref).max())
    assert rel < 1e-2
    assert (mx.argmax(y, -1) == mx.argmax(ref, -1)).all()
    a, b = HQ.packed(lm.weight, 8), HQ.packed(lm.weight, 8)
    assert a[0] is b[0]                                                             # one packed copy per (weight, bits)
    assert len(HQ._PACKS) == 1


def test_flag_registered_and_runtime():
    assert F.FLAGS["head_q"].runtime
    F.set_many({"head_q": 8})
    try:
        assert HQ.BITS == 8
    finally:
        F.set_many({"head_q": 0})
    assert HQ.BITS == 0
