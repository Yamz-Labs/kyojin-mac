"""Activation kind (SiTU) in the K-quant gather_glu (down projection of a routed expert, the gated product staged inside the launch): against the elementwise SiTU kernel followed by the
plain gather, and against float64 math on the stored blocks' decode. Metal only; small shapes."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from test_foreign_moe import E, H, I, RNG, _packs  # noqa: E402
from kyojin_mlx import foreign_linear as FL, foreign_metal as FM, foreign_moe as FMO  # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="no Metal GPU")
SITU = ("situ", 4.0, 25.0)


def test_helper_text():
    assert FM.glu_helper(None) == FM.GLU_HELPER
    assert FM.glu_helper(SITU) != FM.GLU_HELPER and "tanh" in FM.glu_helper(SITU)
    with pytest.raises(AssertionError):
        FM.glu_helper(("gelu", 1.0, 1.0))


def _inputs(S=6, seed=2):
    rng = RNG(seed)
    a = mx.array((rng.standard_normal((S, I)) * 8).astype(np.float16))       # |gate| up to ~30: both tanh clamps and the sigmoid are in range
    b = mx.array((rng.standard_normal((S, I)) * 20).astype(np.float16))
    ids = mx.array(np.array([3, 3, 0, E - 1, 7, 12][:S], np.uint32))
    return a, b, ids, mx.arange(S).astype(mx.uint32)


@pytest.mark.parametrize("t", ["Q5_K", "Q6_K", "IQ4_XS", "Q3_K", "IQ2_S"])
def test_situ_down_matches_elementwise_then_down(t, monkeypatch):
    monkeypatch.setenv("KYOJIN_KSTORED", "2")
    d = FL.DevPack(_packs(t, H, I, 11))
    assert d.glu_ok()
    a, b, ids, ar = _inputs()
    ref = d.gather(FMO._silu_mul(a, b, SITU), ar, ids)                       # the elementwise SiTU kernel, then the plain gather
    y = d.gather_glu(a, b, ar, ids, SITU)
    mx.eval(ref, y)
    ref, y = np.array(ref, np.float64), np.array(y, np.float64)
    assert np.isfinite(y).all()
    err = np.abs(y - ref).max() / np.abs(ref).max()
    print(f"REL_ERR {t}: gather_glu(situ) vs elementwise + gather {err:.2e}, bit equal {np.array_equal(ref, y)}")
    assert err <= 1e-3                                                        # same expression; the two kernels may differ by an ulp of the exp / tanh in f32 -> one fp16 step of h
    plain = np.array(d.gather_glu(a, b, ar, ids), np.float64)                 # silu staging unchanged and different from SiTU
    assert np.abs(plain - y).max() > 0.02 * np.abs(y).max()


def test_situ_down_vs_float64(monkeypatch):
    monkeypatch.setenv("KYOJIN_KSTORED", "2")
    d = FL.DevPack(_packs("Q5_K", H, I, 11))
    a, b, ids, ar = _inputs()
    an, bn = np.array(a, np.float64), np.array(b, np.float64)
    h = (4.0 * np.tanh(an / 4.0) / (1.0 + np.exp(-an)) * (25.0 * np.tanh(bn / 25.0))).astype(np.float16)
    ref = np.array(d.gather(mx.array(h), ar, ids), np.float64)                # the down projection of the float64 activation rounded to fp16, same down kernel
    y = np.array(d.gather_glu(a, b, ar, ids, SITU), np.float64)
    err = np.abs(y - ref).max() / np.abs(ref).max()
    print(f"REL_ERR Q5_K: gather_glu(situ) vs float64 activation + gather {err:.2e}")
    assert err <= 2e-3


def test_block_takes_the_kernel(monkeypatch):
    """foreign_moe's decode block hands the kind to gather_glu (it used to fall to the elementwise kernel)."""
    from test_foreign_moe import _block
    monkeypatch.setenv("KYOJIN_KSTORED", "2")
    blk = _block(("Q4_K", "Q4_K", "Q5_K"), True, True)
    blk.switch_mlp.set_act_kind(SITU)
    x = mx.array(RNG(5).standard_normal((1, 2, H)).astype(np.float16))
    ref = np.array(blk(x), np.float32)
    FMO.install(type("M", (), {"named_modules": lambda s: [("b", blk)]})(), log=lambda *_: None)
    calls = []
    real = FL.DevPack.gather_glu
    monkeypatch.setattr(FL.DevPack, "gather_glu", lambda self, a, b, xr, ei, act=None: (calls.append(act), real(self, a, b, xr, ei, act))[1])
    y = np.array(blk(x), np.float32)
    assert calls == [SITU]
    assert np.isfinite(y).all()
    err = np.abs(y - ref).max() / np.abs(ref).max()
    print(f"REL_ERR block: fused (gather_glu situ) vs the block's stock call {err:.2e}")
    assert err <= 3e-3
