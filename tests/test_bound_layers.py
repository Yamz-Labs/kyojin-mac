"""bound_layers: a prompt chunk evaluates every `EVERY` layers (output + cache state), decode rows are left alone, and the values do not move."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from mlx import nn                                                           # noqa: E402

from kyojin_mlx import bound_layers as BL                                    # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")


def make_model(n_layers=8):
    class Blk(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = mx.ones((4,))

        def __call__(self, h, pre_mix, image_mask, cache):
            return h * 2.0 + pre_mix, pre_mix

    class LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [Blk() for _ in range(n_layers)]

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = LM()

    return M(), Blk


class Cache:
    def __init__(self, n=4):
        self.live = mx.ones((n,))

    @property
    def state(self):
        return [self.live]


def run(lm, rows, cache):
    h = mx.full((1, rows, 2, 4), 0.5)
    pre = mx.zeros((1, rows, 2, 4))
    for layer in lm.layers:
        h, pre = layer(h, pre, None, cache)
    return h


def test_bounds_once_every_layers_and_keeps_the_values():
    m, Blk = make_model(8)
    lm = m.language_model
    cache = Cache()
    ref = np.array(run(lm, 128, cache))                       # unpatched
    BL.reset()
    assert BL.install(m) == 8
    assert BL.install(m) == 0                                 # idempotent
    assert np.array_equal(np.array(run(lm, 128, cache)), ref)
    assert BL.stats()["bounds"] == 2                          # 8 layers, every 4
    BL.reset()
    assert np.array_equal(np.array(run(lm, 128, cache)), ref)
    assert BL.stats()["bounds"] == 2


def test_decode_rows_do_not_bound_and_reset_the_phase(monkeypatch):
    monkeypatch.setattr(BL, "DEC_EVERY", 0)
    m, Blk = make_model(8)
    BL.reset()
    BL.install(m)
    cache = Cache()
    run(m.language_model, 1, cache)                           # a decode step
    assert BL.stats() == {"calls": 0, "bounds": 0, "submits": 0}
    run(m.language_model, 128, cache)
    assert BL.stats()["bounds"] == 2


def test_decode_submits_every_n_layers_and_keeps_the_values(monkeypatch):
    m, Blk = make_model(8)
    lm = m.language_model
    ref = np.array(run(lm, 1, Cache()))                       # unpatched
    BL.reset()
    BL.install(m)
    monkeypatch.setattr(BL, "DEC_EVERY", 2)
    assert np.array_equal(np.array(run(lm, 1, Cache())), ref)
    assert BL.stats()["submits"] == 4 and BL.stats()["bounds"] == 0
    monkeypatch.setattr(BL, "DEC_EVERY", 0)
    BL.reset()
    run(lm, 1, Cache())
    assert BL.stats()["submits"] == 0


def test_every_layer_count_is_honoured(monkeypatch):
    monkeypatch.setattr(BL, "EVERY", 3)
    m, Blk = make_model(9)
    BL.reset()
    BL.install(m)
    run(m.language_model, 128, Cache())
    assert BL.stats()["bounds"] == 3


def test_the_cache_state_is_evaluated_with_the_layer_output(monkeypatch):
    m, Blk = make_model(4)
    BL.reset()
    BL.install(m)
    seen = []
    real = mx.eval

    def spy(*a):
        seen.append([x.shape for x in a])
        return real(*a)
    monkeypatch.setattr(BL.mx, "eval", spy)
    cache = Cache(7)
    run(m.language_model, 128, cache)
    assert len(seen) == 1
    assert (1, 128, 2, 4) in seen[0] and (7,) in seen[0]


def test_a_model_without_layers_is_a_no_op():
    class M(nn.Module):
        pass
    BL.reset()
    assert BL.install(M()) == 0
