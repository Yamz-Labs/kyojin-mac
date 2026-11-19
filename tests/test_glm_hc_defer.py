"""G10d: mHC glue with ONE launch per branch on the dependent chain (kyojin_mlx.glm_hc_defer) vs the vendor chain and vs glm_hc (G3b)."""
import os
os.environ["KYOJIN_GLM_HC_FUSE"] = "1"
os.environ["KYOJIN_GLM_HC_DEFER"] = "1"
import numpy as np
import pytest
import mlx.core as mx
from kyojin_mlx import glm_hc as G, glm_hc_defer as D
from test_glm_hc import mk

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")


def chain_defer(x0, hcs, H):
    """The layer sequence of glm_hc_defer.layer_call over branches: P (plain, then deferred) -> branch -> P ... -> lazy E."""
    xo, mixes, _ = D.pre(x0, hcs[0][0], hcs[0][1].weight, 1e-5)
    pc = D.post_comb(mixes, hcs[0][0])
    h, o_prev = x0, None
    for i, (hc, nrm, W) in enumerate(hcs):
        if i > 0:
            xo, mixes, h_new = D.pre(h, hc, nrm.weight, 1e-5, deferred=(o_prev, pc))
            pc = D.post_comb(mixes, hc)
            h = h_new
        o_prev = (xo @ W.T) * 0.7
    x, _ = G.expand(o_prev, h, pc, None)
    return x


class _Layer:
    is_linear = True

    def self_attn(self, xo, mask, cache):
        return (xo @ self._Wa.T) * 0.7

    def mlp(self, xo):
        return (xo @ self._Wf.T) * 0.7


def chain_layer_call(x0, hcs):
    """Branch pairs (attn_hc, ffn_hc) grouped 2 at a time into decoder layers, driven through D.layer_call + the SLOT /
    _g3b_next wiring exactly as glm_hc.install() sets it up on real decoder layers (regression for the nxt.attn_hc bug:
    _g3b_next is already layers[i + 1].attn_hc, not the layer object -- caught by this test, missed by chain_defer())."""
    assert len(hcs) % 2 == 0
    layers = []
    for i in range(0, len(hcs), 2):
        (ha, na, Wa), (hf, nf, Wf) = hcs[i], hcs[i + 1]
        ly = _Layer()
        ly.attn_hc, ly.ffn_hc, ly._Wa, ly._Wf = ha, hf, Wa, Wf
        ly.input_layernorm, ly.post_attention_layernorm = na, nf
        layers.append(ly)
    for i, ly in enumerate(layers):
        ly.__dict__["_g3b_next"] = layers[i + 1].attn_hc if i + 1 < len(layers) else None
    x = x0
    D.SLOT.x = D.SLOT.y = D.SLOT.h = D.SLOT.pc = D.SLOT.a = None
    for ly in layers:
        x, _ = D.layer_call(ly, x, None, None, None)
    return x


def chain_g3b(x0, hcs):
    x = x0
    part = G.partials(x, hcs[0][0].fn)
    for i, (hc, nrm, W) in enumerate(hcs):
        xo, pc = G.pre(x, part, hc, nrm.weight, 1e-5)
        o = (xo @ W.T) * 0.7
        nf = hcs[i + 1][0].fn if i + 1 < len(hcs) else None
        x, part = G.expand(o, x, pc, nf)
    return x


@pytest.mark.parametrize("fn16", [False, True])
@pytest.mark.parametrize("dt,tol", [(mx.float16, 0.05), (mx.bfloat16, 0.2)])
@pytest.mark.parametrize("T", [1, 2, 4, 8])
def test_chain_matches_stock_and_g3b(dt, tol, T, fn16):
    mx.random.seed(0)
    H, nbr = 4096, 8
    hcs = [mk(H, dt, 10 + i) for i in range(nbr)]
    if fn16:
        for hc, _, _ in hcs:
            hc.fn = hc.fn.astype(mx.float16)
    x0 = (mx.random.normal((T, 1, 4, H)) * 1.5).astype(dt)
    ref = x0
    for hc, nrm, W in hcs:
        ref = hc.apply_branch(ref, nrm, lambda z, W=W: (z @ W.T) * 0.7)
    got, g3b = chain_defer(x0, hcs, H), chain_g3b(x0, hcs)
    mx.eval(ref, got, g3b)
    r, a, b = (np.asarray(v.astype(mx.float32)) for v in (ref, got, g3b))
    assert np.abs(r - a).max() < tol and np.abs(r - a).mean() < tol / 4
    assert np.abs(a - b).max() < tol                # same arithmetic but the mix dot products' summation order (an 8-branch random chain amplifies ulp flips)


@pytest.mark.parametrize("dt,tol", [(mx.float16, 0.05), (mx.bfloat16, 0.2)])
def test_layer_call_g3b_next_wiring_matches_chain_defer(dt, tol):
    """Drives D.layer_call across 4 decoder layers (8 branches) with the real _g3b_next wiring: must match chain_defer /
    chain_g3b, which call D.pre / D.post_comb directly and so cannot see a bug in layer_call's own SLOT handoff."""
    mx.random.seed(3)
    H, nbr = 4096, 8
    hcs = [mk(H, dt, 20 + i) for i in range(nbr)]
    x0 = (mx.random.normal((3, 1, 4, H)) * 1.5).astype(dt)
    got, ref_defer, ref_g3b = chain_layer_call(x0, hcs), chain_defer(x0, hcs, H), chain_g3b(x0, hcs)
    mx.eval(got, ref_defer, ref_g3b)
    a, b, c = (np.asarray(v.astype(mx.float32)) for v in (got, ref_defer, ref_g3b))
    assert np.array_equal(a, b)                      # same code path (D.pre/D.post_comb), must be bitwise identical
    assert np.abs(a - c).max() < tol


def test_expand_inside_P_equals_stock_expand_bitwise():
    """h recomputed by the deferred P == the A kernel's expand of the same inputs (the streams every later branch reads)."""
    mx.random.seed(1)
    H, T, dt = 4096, 3, mx.float16
    hc, nrm, _ = mk(H, dt, 3)
    hc.fn = hc.fn.astype(mx.float16)
    x = (mx.random.normal((T, 1, 4, H)) * 1.5).astype(dt)
    y = (mx.random.normal((T, 1, H))).astype(dt)
    pc = mx.concatenate([mx.random.uniform(0.2, 2, (T, 4)), mx.random.uniform(0, 0.6, (T, 16))], axis=1).astype(mx.float32)
    ref, _ = G.expand(y, x, pc, None)
    _, _, h = D.pre(x, hc, nrm.weight, 1e-5, deferred=(y, pc))
    mx.eval(ref, h)
    assert bool(mx.array_equal(ref, h))


@pytest.mark.parametrize("dt", [mx.float16, mx.bfloat16])
def test_one_branch_equals_g3b_up_to_ulp_flips(dt):
    mx.random.seed(0)
    H = 4096
    hcs = [mk(H, dt, 10)]
    hcs[0][0].fn = hcs[0][0].fn.astype(mx.float16)
    x0 = (mx.random.normal((2, 1, 4, H)) * 1.5).astype(dt)
    a, b = chain_defer(x0, hcs, H), chain_g3b(x0, hcs)
    mx.eval(a, b)
    assert float(mx.mean((a != b).astype(mx.float32))) < 0.01
    assert float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)))) <= (2 ** -9 if dt == mx.float16 else 2 ** -6)


@pytest.mark.parametrize("dt", [mx.float16, mx.bfloat16])
def test_p_and_q_rows_are_invariant(dt):
    """A token alone == the token among T (P plain / deferred, Q), bit for bit."""
    mx.random.seed(2)
    H, T = 4096, 5
    hc, nrm, _ = mk(H, dt, 3)
    hc.fn = hc.fn.astype(mx.float16)
    x = (mx.random.normal((T, 1, 4, H)) * 1.5).astype(dt)
    y = mx.random.normal((T, 1, H)).astype(dt)
    pc = mx.concatenate([mx.random.uniform(0.2, 2, (T, 4)), mx.random.uniform(0, 0.6, (T, 16))], axis=1).astype(mx.float32)
    full = D.pre(x, hc, nrm.weight, 1e-5)[:2] + D.pre(x, hc, nrm.weight, 1e-5, deferred=(y, pc)) + (D.post_comb(D.pre(x, hc, nrm.weight, 1e-5)[1], hc),)
    for t in range(T):
        one = D.pre(x[t:t + 1], hc, nrm.weight, 1e-5)[:2] + D.pre(x[t:t + 1], hc, nrm.weight, 1e-5, deferred=(y[t:t + 1], pc[t:t + 1])) \
            + (D.post_comb(D.pre(x[t:t + 1], hc, nrm.weight, 1e-5)[1], hc),)
        for u, v in zip(one, full):
            assert bool(mx.array_equal(u[0], v[t]))
