"""Capability gate that used to be a class name (compressor_pf.install): it must refuse a module it was not written for.
CPU only (tiny modules)."""
import os
import sys

import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))


@pytest.fixture(autouse=True)
def cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


def _compressor(extra=None, ratio=4):
    pytest.importorskip("mlx_vlm.models.deepseek_v41")
    import mini_ds41 as M
    import mlx_vlm.models.deepseek_v41.language as L
    from mlx_vlm.models.deepseek_v41.config import ModelConfig
    cfg = ModelConfig.from_dict({**M.CFG, "compress_ratios": [0, 0, ratio, ratio, 1, 1, 1, 1]})

    class Compressor(L.Compressor):                     # same class name as the vendor one, as a day-one module of another model would have
        def __init__(self, *a):
            super().__init__(*a)
            for k, v in (extra or {}).items():
                setattr(self, k, v)
    return Compressor(cfg, 2)


class _Holder(nn.Module):
    def __init__(self, comp):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.comp = comp


def test_compressor_hook_arms_the_vendor_layout():
    from kyojin_mlx import compressor_pf as PF
    assert PF.install(_Holder(_compressor())) == 1


@pytest.mark.parametrize("extra", [{"ape": mx.zeros((4, 64))}, {"overlap": True}, {"rotate": True}])
def test_compressor_hook_refuses_other_math(extra):
    from kyojin_mlx import compressor_pf as PF
    h = _Holder(_compressor(extra))
    assert PF.install(h) == 0
    assert not PF.pool_ok(h.language_model.comp)


def test_compressor_hook_refuses_missing_parts():
    from kyojin_mlx import compressor_pf as PF
    c = _compressor()
    del c["wgate"]
    assert not PF.pool_ok(c)
    assert not PF.pool_ok(_compressor(ratio=1))
