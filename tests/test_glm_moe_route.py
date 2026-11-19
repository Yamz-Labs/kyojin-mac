"""Fused decode MoE block with a generic route description (glm_moe.Route): several routers, a per-expert output scale, an expert width different from the router input width
(a projection before the experts and a norm + projection after), a SiTU activation kind. Metal only; 12 random NVFP4 experts of 1024 x 1024.
References: select / combine against numpy; the block against a float64 numpy forward and against the block's own stock-ops call (the route off)."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn  # noqa: E402
from kyojin_mlx import foreign_linear as FL, glm_moe as GM  # noqa: E402
from test_foreign_act import E, KD, SITU, _experts, _ref_mlp  # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="no Metal GPU")

HR, NR, K, EPS = 2048, 2, 4, 1e-5            # router input width, routers, top-k, norm eps; the experts and the latent are KD = 1024 wide


def _sig(v):
    return 1.0 / (1.0 + np.exp(-v))


def test_sources_unchanged_for_one_router():
    assert GM.select_source(1, False) == GM._SELECT
    assert GM.combine_source(True) == GM._COMBINE
    assert GM.select_source(2, False) != GM._SELECT and GM.select_source(1, True) != GM._SELECT and GM.combine_source(False) != GM._COMBINE


@pytest.mark.parametrize("routers", [1, 2, 3])
@pytest.mark.parametrize("vec", [False, True])
def test_select_routers_and_scale(routers, vec):
    T, Ee, Kk = 3, 288, 8
    rng = np.random.default_rng(10 + routers)
    lg = rng.normal(size=(T, routers * Ee)).astype(np.float32) * 2
    bias = (rng.normal(size=(Ee,)) * 0.1).astype(np.float32)
    sc = rng.uniform(0.5, 2.0, Ee).astype(np.float32)
    inds, wts = GM.select(mx.array(lg), mx.array(bias), T, Ee, Kk, True, mx.array(sc) if vec else 2.5, routers)
    inds, wts = np.asarray(inds).reshape(T, Kk), np.asarray(wts).reshape(T, Kk)
    s = sum(_sig(lg[:, r * Ee:(r + 1) * Ee].astype(np.float64)) for r in range(routers))
    for t in range(T):
        ref = np.argsort(-(s[t] + bias), kind="stable")[:Kk]
        assert list(inds[t]) == list(ref)                                  # same set, same slot order (value desc, lowest index first)
        tot = s[t, ref].sum() + 1e-20
        exp = s[t, ref] / tot * (sc[ref] if vec else 2.5)
        assert np.abs(wts[t] - exp).max() < 1e-5 * exp.max()


def test_combine_f32_weights():
    rng = np.random.default_rng(5)
    T, Kk, H = 3, 8, 256
    y = rng.normal(size=(T * Kk, H)).astype(np.float16)
    w = rng.uniform(0.0, 1.0, size=(T * Kk,)).astype(np.float32)
    out = np.asarray(GM.combine(mx.array(y), mx.array(w), None, T, Kk, H, False)).astype(np.float64)
    ref = (y.astype(np.float64).reshape(T, Kk, H) * w.astype(np.float64).reshape(T, Kk, 1)).sum(1)
    assert np.abs(out - ref).max() <= 1e-3 * np.abs(ref).max()             # one fp16 rounding at the end, no fp16 product rounding


class _Blk(nn.Module):
    """Latent routed block written the way a model file would: stock-ops forward in __call__, the generic description in moe_route()."""

    def __init__(self, seed=0):
        super().__init__()
        r = np.random.default_rng(seed)
        g, u, d = _experts(seed)
        self.switch_mlp = FL.ForeignSwitchGLU(g, u, d, act_kind=SITU)
        self.rw = mx.array(r.normal(0, 1 / np.sqrt(HR), (NR * E, HR)).astype(np.float32))
        self.rb = mx.array(r.normal(0, 0.05, E).astype(np.float32))
        self.rs = mx.array(r.uniform(0.5, 2.0, E).astype(np.float32))
        self.pre_w = mx.array(r.normal(0, 1 / np.sqrt(HR), (KD, HR)).astype(np.float16))
        self.post_w = mx.array(r.normal(0, 1 / np.sqrt(KD), (HR, KD)).astype(np.float16))
        self.nw = mx.array(r.uniform(0.5, 1.5, KD).astype(np.float16))
        self.sh_w = mx.array(r.normal(0, 1 / np.sqrt(HR), (HR, HR)).astype(np.float16))

    def shared(self, x):
        return x @ self.sh_w.T

    def moe_route(self):
        return GM.Route(self.rw, K, self.rb, True, self.rs, routers=NR, pre=lambda v: v @ self.pre_w.T, wround=False, shared=self.shared, width=KD,
                        post=lambda a: mx.fast.rms_norm(a, self.nw, EPS) @ self.post_w.T)

    def __call__(self, x, *a, **k):                                        # the stock-ops forward
        l = x.astype(mx.float32) @ self.rw.T
        s = mx.sigmoid(l[..., :E]) + mx.sigmoid(l[..., E:])
        ids = mx.argsort(-(s + self.rb), axis=-1)[..., :K]
        w = mx.take_along_axis(s, ids, axis=-1)
        w = w / mx.maximum(w.sum(-1, keepdims=True), 1e-20) * self.rs[ids]
        y = self.switch_mlp(x @ self.pre_w.T, ids)
        acc = (y.astype(mx.float32) * w[..., None]).sum(-2).astype(x.dtype)
        return mx.fast.rms_norm(acc, self.nw, EPS) @ self.post_w.T + self.shared(x)


def _ref_block(b, x):
    xs = np.array(x).astype(np.float64)
    lg = xs @ np.array(b.rw).astype(np.float64).T
    s = _sig(lg[:, :E]) + _sig(lg[:, E:])
    ids = np.argsort(-(s + np.array(b.rb)), axis=-1, kind="stable")[:, :K]
    w = np.take_along_axis(s, ids, axis=-1)
    w = w / np.maximum(w.sum(-1, keepdims=True), 1e-20) * np.array(b.rs).astype(np.float64)[ids]
    lat = (xs.astype(np.float16).astype(np.float64) @ np.array(b.pre_w).astype(np.float64).T).astype(np.float16)
    sm = b.switch_mlp
    y = _ref_mlp(sm.g, sm.u, sm.d, mx.array(lat), mx.array(ids.reshape(-1).astype(np.uint32)), K).reshape(len(xs), K, KD)
    acc = (y * w[..., None]).sum(1).astype(np.float16).astype(np.float64)
    nrm = (acc / np.sqrt((acc ** 2).mean(-1, keepdims=True) + EPS) * np.array(b.nw).astype(np.float64)).astype(np.float16).astype(np.float64)
    return nrm @ np.array(b.post_w).astype(np.float64).T + xs.astype(np.float16).astype(np.float64) @ np.array(b.sh_w).astype(np.float64).T, ids


def _worst(got, ref):
    got = np.array(got).astype(np.float64).reshape(ref.shape)
    assert np.isfinite(got).all()
    return np.abs(got - ref).max() / np.abs(ref).max()


@pytest.mark.parametrize("moe_e", ["0", "1"])
@pytest.mark.parametrize("T", [1, 3])
def test_latent_block(monkeypatch, moe_e, T):
    monkeypatch.setenv("KYOJIN_GLM_MOE", "1")
    monkeypatch.setenv("KYOJIN_GLM_MOE_E", moe_e)
    monkeypatch.setenv("KYOJIN_FOREIGN_G2A", "1")
    b = _Blk()
    x = mx.array(np.random.default_rng(3).normal(0, 1, (T, HR)).astype(np.float16))
    stock = np.array(b(x).astype(mx.float32))
    ref, ids = _ref_block(b, x)
    holder = nn.Module()
    holder.blk = b
    assert GM.install(holder, log=lambda *_: None) == 1
    assert GM._why(b, x.reshape(1, T, HR)) is None
    from kyojin_mlx import fired as FD
    n0 = FD.STATS.get("glm_moe.fire", 0), FD.STATS.get("glm_moe_latent.fire", 0), FD.STATS.get("glm_moe_post.fire", 0)
    got = np.array(b(x.reshape(1, T, HR)).astype(mx.float32)).reshape(T, HR)
    assert tuple(FD.STATS.get(k + ".fire", 0) - v for k, v in zip(("glm_moe", "glm_moe_latent", "glm_moe_post"), n0)) == (1, 1, 1)      # the fused block served the call
    e_stock, e_fused = _worst(stock, ref), _worst(got, ref)
    print(f"REL_ERR moe_e={moe_e} T={T}: stock-ops vs f64 {e_stock:.2e}, fused vs f64 {e_fused:.2e}, fused vs stock {np.abs(got - stock).max() / np.abs(ref).max():.2e}")
    assert e_fused <= 5e-3
    assert np.abs(got - stock).max() <= 5e-3 * np.abs(ref).max()


def test_refusals(monkeypatch):
    monkeypatch.setenv("KYOJIN_GLM_MOE", "1")
    b = _Blk()
    x = mx.zeros((1, 1, HR), mx.float16)
    assert GM._why(b, x) is None
    r = b.moe_route()
    assert not r.plain and r.E == E and r.routers == NR
    assert GM._pf_why(b, mx.zeros((1, 40, HR), mx.float16)) in ("flag_off", "route")
    assert GM._why(b, mx.zeros((1, 1, HR), mx.float32)).startswith("x_dtype")
