"""Flag f4copy: the resident fp16 copy of a 4-bit stored format serves the same product as the stored kernels."""
import mlx.core as mx
import numpy as np
import pytest

from kyojin_mlx import foreign_gguf as GF
from kyojin_mlx import foreign_linear as FL
from kyojin_mlx import foreign_synth as SY

N, K = 96, 1024


def _pack(t):
    raw = SY.random_blocks(t, N, K, np.random.default_rng(3))
    return GF.to_pack(t, raw, K)


@pytest.mark.parametrize("t", ["Q4_0", "Q4_K", "IQ4_NL"])
@pytest.mark.parametrize("R", [8, 16, 24])
def test_copy_route_matches_stored(t, R, monkeypatch):
    p = _pack(t)
    x = mx.array(np.random.default_rng(4).standard_normal((R, K)).astype(np.float16))
    monkeypatch.setenv("KYOJIN_F4COPY", "0")
    off = FL.DevPack(p)
    assert off.w16 is None
    ref = np.array(off.apply(x).astype(mx.float32))
    monkeypatch.setenv("KYOJIN_F4COPY", "1")
    on = FL.DevPack(p)
    assert on.w16 is not None and on.w16.dtype == mx.float16
    got = np.array(on.apply(x).astype(mx.float32))
    assert np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-6) < 5e-3


def test_no_copy_for_expert_stacks(monkeypatch):
    p = _pack("Q4_0")
    monkeypatch.setenv("KYOJIN_F4COPY", "1")
    assert FL.DevPack([p, p]).w16 is None
