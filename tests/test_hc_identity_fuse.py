# SPDX-License-Identifier: Apache-2.0
"""KYOJIN_HC_ID_FUSE: the identity-form hyper-connection seams (foreign_skeleton_hc.HcStream.fused) compute the stock function.

CPU part (always runs, mx.cpu, tiny arrays):
  * the kernel arithmetic, written out in numpy fp32 in the kernel's order (_np_pre / _np_expand), equals the stock op sequence (split + collapse + Norm,
    Norm + expand) within fp32 reordering of the norm sum;
  * the wiring: with the three kernels replaced by those numpy mirrors, HcStream.fused(h, core, pre_norm, post_norm) == HcStream(h, post(core(pre(.))))
    for the rms / mul_first / layer-norm (not folded) norms, hc 4, f32, and it makes exactly 3 kernel calls per sub-layer;
  * the stock launch count of one sub-layer (graph nodes from mx.export_to_dot), printed with -s.
Metal part (KYOJIN_METAL_TESTS=1 on a GPU, run by the chain cell c79): the real kernels against the stock ops (array_equal reported, tolerance asserted).
"""
import os
import re

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from kyojin_mlx import foreign_skeleton as FS
from kyojin_mlx import foreign_skeleton_hc as HC
from kyojin_mlx import fuse as F
from kyojin_mlx import gluefuse as GF

METAL = os.environ.get("KYOJIN_METAL_TESTS", "0") == "1" and mx.metal.is_available()
HCN = 4


def _sig(z):
    z = np.float32(z)
    e = np.float32(1.0) / (np.float32(1.0) + np.exp(-np.abs(z)).astype(np.float32))
    return np.where(z < 0, np.float32(1.0) - e, e).astype(np.float32)


def _rows_rms(v, eps):
    """the kernel's reduction: per thread 4 contiguous values of every 4096 in order (fp32), then a sum of the partials (pairwise order differs
    from the simd tree: fp32 reordering only)."""
    v = v.astype(np.float32)
    ss = np.float32((v.astype(np.float64) ** 2).sum())                # reference sum; the kernel tree is within fp32 reordering of it
    return np.float32(1.0) / np.sqrt(ss / np.float32(v.shape[-1]) + np.float32(eps), dtype=np.float32)


def _np_pre(x, m, scale, base, eps, mag, post_eps, nw=None, neps=0.0):
    x, m, scale, base = (np.asarray(a, np.float32) for a in (x, m, scale, base))
    R, H, D = x.shape
    y, post = np.zeros((R, D), np.float32), np.zeros((R, H), np.float32)
    for b in range(R):
        pre = _sig(m[b, :H] * scale[0] + base[:H]) + np.float32(eps)
        p = np.float32(mag) * _sig(m[b, H:] * scale[1] + base[H:])
        post[b] = p + np.float32(eps) if post_eps else p
        c = np.zeros(D, np.float32)
        for j in range(H):
            c = (c + (pre[j] * x[b, j]).astype(np.float32)).astype(np.float32)
        if nw is not None:
            inv = _rows_rms(c, neps)
            c = (np.asarray(nw, np.float32) * (c * inv).astype(np.float32)).astype(np.float32)
        y[b] = c
    return mx.array(y), mx.array(post)


def _np_expand(y, h, post, nw=None, neps=0.0, mul_first=False):
    y, h, post = (np.asarray(a, np.float32) for a in (y, h, post))
    out = np.zeros_like(h)
    for b in range(h.shape[0]):
        v = y[b]
        if nw is not None:
            w = np.asarray(nw, np.float32)
            if mul_first:
                v = (v * w).astype(np.float32)
                v = (v * _rows_rms(v, neps)).astype(np.float32)
            else:
                v = (w * (v * _rows_rms(v, neps)).astype(np.float32)).astype(np.float32)
        out[b] = (h[b] + (post[b][:, None] * v[None, :]).astype(np.float32)).astype(np.float32)
    return mx.array(out)


def _stream(rng, D, scale3=False):
    fn = mx.array(0.02 * rng.standard_normal((2 * HCN, HCN * D)).astype(np.float32))
    base = mx.array(0.5 * rng.standard_normal(2 * HCN).astype(np.float32))
    scale = mx.array((1.0 + 0.1 * rng.standard_normal(3 if scale3 else 2)).astype(np.float32))
    return HC.HcStream(fn, base, scale, HCN, 1e-6, 2.0, 1e-6, 0, True)


def _norm(rng, D, kind="rms", mul_first=False, bias=False):
    n = FS.Norm(D, kind, 1e-5, weight=True, bias=bias, mul_first=mul_first)
    n.weight = mx.array((1.0 + 0.1 * rng.standard_normal(D)).astype(np.float32))
    if bias:
        n.bias = mx.array(0.1 * rng.standard_normal(D).astype(np.float32))
    return n


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(1.0, np.abs(b).max()))


# ------------------------------------------------------------------------------------------------------------------------ CPU
@pytest.fixture
def cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


@pytest.mark.parametrize("post_eps", [True, False])
@pytest.mark.parametrize("D", [256, 4100])
def test_np_mirror_pre_equals_stock(cpu, post_eps, D):
    rng = np.random.default_rng(D)
    st, pn = _stream(rng, D), _norm(rng, D)
    st.post_eps = post_eps
    h = mx.array(rng.standard_normal((1, 2, HCN, D)).astype(np.float32))
    m = HC.mixes(h, st.fn, st.norm_eps)
    pre, post, _ = HC.split(m, st.scale, st.base, HCN, st.eps, st.magnitude, 0, post_eps)
    ref_x = pn(HC.collapse(h, pre))
    y, p = _np_pre(h.reshape(2, HCN, D), m.reshape(2, 2 * HCN), st.scale, st.base, st.eps, st.magnitude, post_eps, pn.weight, pn.eps)
    print(f"pre D {D} post_eps {post_eps}: rel x {_rel(y, ref_x.reshape(2, D)):.2e} post {_rel(p, post.reshape(2, HCN)):.2e}")
    assert _rel(p, post.reshape(2, HCN)) <= 2e-7
    assert _rel(y, ref_x.reshape(2, D)) <= 2e-6


@pytest.mark.parametrize("mul_first", [False, True])
def test_np_mirror_expand_equals_stock(cpu, mul_first):
    rng = np.random.default_rng(7)
    D = 512
    pn = _norm(rng, D, mul_first=mul_first)
    h = mx.array(rng.standard_normal((1, 1, HCN, D)).astype(np.float32))
    y = mx.array(rng.standard_normal((1, 1, D)).astype(np.float32))
    post = mx.array(rng.uniform(0, 2, (1, 1, HCN)).astype(np.float32))
    ref = HC.expand(pn(y), h, post, None)
    out = _np_expand(y.reshape(1, D), h.reshape(1, HCN, D), post.reshape(1, HCN), pn.weight, pn.eps, mul_first)
    print(f"expand mul_first {mul_first}: rel {_rel(out, ref.reshape(1, HCN, D)):.2e}")
    assert _rel(out, ref.reshape(1, HCN, D)) <= 2e-6


@pytest.mark.parametrize("norms", ["rms", "mul_first", "layer"])
def test_wiring_fused_equals_stock(cpu, monkeypatch, norms):
    """HcStream.fused with the kernels replaced by their numpy mirrors == the stock HcStream call; 3 kernel calls; a layer norm is not folded."""
    rng = np.random.default_rng(11)
    D = 1024
    st = _stream(rng, D)
    pre_n = _norm(rng, D, "layer" if norms == "layer" else "rms", bias=norms == "layer")
    post_n = _norm(rng, D, mul_first=norms == "mul_first")
    lin = mx.array(0.03 * rng.standard_normal((D, D)).astype(np.float32))
    core = lambda x: x @ lin                                                  # noqa: E731
    calls = []

    def mixvec(x, fn, eps):
        calls.append("mix")
        return mx.fast.rms_norm(x.astype(mx.float32), None, eps) @ fn.T

    def pre(x, m, scale, base, eps, mag, post_eps, nw=None, neps=0.0):
        calls.append("pre" + ("+norm" if nw is not None else ""))
        return _np_pre(x, m, scale, base, eps, mag, post_eps, nw, neps)

    def expand(y, h, post, nw=None, neps=0.0, mul_first=False):
        calls.append("expand" + ("+norm" if nw is not None else ""))
        return _np_expand(y, h, post, nw, neps, mul_first)

    monkeypatch.setattr(GF, "mixvec", mixvec)
    monkeypatch.setattr(F, "mhc_pre_identity", pre)
    monkeypatch.setattr(F, "mhc_expand_identity", expand)
    h = mx.array(rng.standard_normal((1, 1, HCN, D)).astype(np.float32))
    ref = st(h, lambda x: post_n(core(pre_n(x))))
    out = st.fused(h, core, pre_n, post_n)
    print(f"wiring {norms}: calls {calls} rel {_rel(out, ref):.2e}")
    assert calls == ["mix", "pre" if norms == "layer" else "pre+norm", "expand+norm"]
    assert out.shape == ref.shape and out.dtype == ref.dtype
    assert _rel(out, ref) <= 5e-6


def test_fusable_gate(cpu, monkeypatch):
    rng = np.random.default_rng(3)
    st = _stream(rng, 256)
    h = mx.zeros((1, 1, HCN, 256))
    monkeypatch.setattr(HC, "IDFUSE", True)
    assert not st.fusable(h)                                                  # CPU device: no Metal kernels
    monkeypatch.setattr(HC, "IDFUSE", False)
    assert not st.fusable(h)
    st24 = HC.HcStream(mx.zeros(((2 + HCN) * HCN, HCN * 256)), mx.zeros(24), mx.zeros(3), HCN, 1e-6, 2.0, 1e-6, 20, False)
    monkeypatch.setattr(HC, "IDFUSE", True)
    monkeypatch.setattr(HC.mx, "default_device", lambda: mx.gpu)
    assert st.fusable(h) and not st24.fusable(h)                              # the Sinkhorn form stays stock
    assert not st.fusable(mx.zeros((1, HC.MAX_ROWS + 1, HCN, 256)))           # prompt chunks stay stock


def test_flag_declared():
    from kyojin_mlx import flags as FL
    f = FL.FLAGS["hc_id_fuse"]
    assert f.env == "KYOJIN_HC_ID_FUSE" and f.default == "0" and f.attrs[0][:2] == ("foreign_skeleton_hc", "IDFUSE")


def test_stock_launch_count(cpu, tmp_path):
    """Graph nodes of one stock identity-form sub-layer (the HC glue + the two norms, the core is one matmul): printed, the fused path is 3 + core."""
    rng = np.random.default_rng(5)
    D = 256
    st, pn, qn = _stream(rng, D), _norm(rng, D), _norm(rng, D)
    lin = mx.array(rng.standard_normal((D, D)).astype(np.float32))
    h = mx.array(rng.standard_normal((1, 1, HCN, D)).astype(np.float32))
    mx.eval(h, lin, st.fn, st.base, st.scale, pn.weight, qn.weight)
    out = st(h, lambda x: qn(pn(x) @ lin))
    p = tmp_path / "g.dot"
    mx.export_to_dot(str(p), out)
    ops = [n for n in re.findall(r'label\s*=\s*"([A-Za-z_]+)', p.read_text()) if n[0].isupper()]
    print(f"stock sub-layer graph: {len(ops)} primitive nodes (incl. 1 matmul core): {sorted(set(ops))}")
    assert len(ops) >= 10


# ------------------------------------------------------------------------------------------------------------------------ Metal (chain only)
metal = pytest.mark.skipif(not METAL, reason="Metal kernels: KYOJIN_METAL_TESTS=1 on a GPU (chain cell c79)")
DTOL = {mx.float32: 1e-6, mx.float16: 2e-3, mx.bfloat16: 1.6e-2}


@metal
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16])
@pytest.mark.parametrize("D,R", [(1024, 1), (4096, 1), (4096, 3), (6144, 2)])
@pytest.mark.parametrize("norm", [False, True])
def test_metal_pre_identity(dtype, D, R, norm):
    rng = np.random.default_rng(D + R)
    st, pn = _stream(rng, D), _norm(rng, D)
    pn.weight = pn.weight.astype(dtype)
    h = mx.array(rng.standard_normal((1, R, HCN, D)).astype(np.float32)).astype(dtype)
    m = HC.mixes(h, st.fn, st.norm_eps)
    pre, post, _ = HC.split(m, st.scale, st.base, HCN, st.eps, st.magnitude, 0, True)
    ref = HC.collapse(h, pre)
    ref = pn(ref) if norm else ref
    y, p = F.mhc_pre_identity(h.reshape(R, HCN, D), m.reshape(R, 2 * HCN), st.scale, st.base, st.eps, st.magnitude, True,
                              *((pn.weight, pn.eps) if norm else ()))
    ry, rp = _rel(y.astype(mx.float32), ref.reshape(R, D).astype(mx.float32)), _rel(p, post.reshape(R, HCN))
    print(f"METAL pre {dtype} D {D} R {R} norm {norm}: equal x {bool(mx.array_equal(y, ref.reshape(R, D)))} post "
          f"{bool(mx.array_equal(p, post.reshape(R, HCN)))} rel x {ry:.2e} post {rp:.2e}")
    assert y.dtype == dtype and rp <= 1e-6 and ry <= DTOL[dtype]


@metal
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16])
@pytest.mark.parametrize("D,R", [(1024, 1), (4096, 1), (6144, 3)])
@pytest.mark.parametrize("mode", ["none", "rms", "mul_first"])
def test_metal_expand_identity(dtype, D, R, mode):
    rng = np.random.default_rng(D * 3 + R)
    pn = _norm(rng, D, mul_first=mode == "mul_first")
    pn.weight = pn.weight.astype(dtype)
    h = mx.array(rng.standard_normal((1, R, HCN, D)).astype(np.float32)).astype(dtype)
    y = mx.array(rng.standard_normal((1, R, D)).astype(np.float32)).astype(dtype)
    post = mx.array(rng.uniform(0, 2, (1, R, HCN)).astype(np.float32))
    ref = HC.expand(pn(y) if mode != "none" else y, h, post, None)
    out = F.mhc_expand_identity(y.reshape(R, D), h.reshape(R, HCN, D), post.reshape(R, HCN),
                                *((pn.weight, pn.eps, mode == "mul_first") if mode != "none" else ()))
    r = _rel(out.astype(mx.float32), ref.reshape(R, HCN, D).astype(mx.float32))
    print(f"METAL expand {dtype} D {D} R {R} {mode}: equal {bool(mx.array_equal(out, ref.reshape(R, HCN, D)))} rel {r:.2e}")
    assert out.dtype == dtype and r <= DTOL[dtype]


@metal
@pytest.mark.parametrize("D", [1024, 4096, 6144])
def test_metal_mixvec_n8(D):
    rng = np.random.default_rng(D)
    st = _stream(rng, D)
    h = mx.array(rng.standard_normal((2, HCN * D)).astype(np.float32))
    # exact reference: float64 on the CPU. The stock `rms_norm(h) @ fn.T` of mlx is NOT a usable oracle here: its fp32 GPU matmul (N = 8) is only
    # good to ~1e-3 relative (measured 7e-4 .. 1.2e-3 at K 4096 .. 24576, with an exactly normalised input), while the kernel is at ~1e-7 of the exact value.
    H, FN = np.asarray(h, np.float64), np.asarray(st.fn, np.float64)
    ref = (H / np.sqrt((H * H).mean(1, keepdims=True) + st.norm_eps)) @ FN.T
    m = GF.mixvec(h, st.fn, st.norm_eps)
    assert m is not None
    r = _rel(m, ref)
    stock = _rel(mx.fast.rms_norm(h, None, st.norm_eps) @ st.fn.T, ref)
    print(f"METAL mixvec N 8 K {HCN * D}: rel vs exact fp64 {r:.2e} (stock mlx matmul path: {stock:.2e})")
    assert r <= 1e-5


@metal
@pytest.mark.parametrize("norms", ["rms", "mul_first", "layer"])
def test_metal_fused_sublayer(monkeypatch, norms):
    rng = np.random.default_rng(17)
    D = 4096
    st = _stream(rng, D)
    pre_n = _norm(rng, D, "layer" if norms == "layer" else "rms", bias=norms == "layer")
    post_n = _norm(rng, D, mul_first=norms == "mul_first")
    lin = mx.array(0.02 * rng.standard_normal((D, D)).astype(np.float32))
    core = lambda x: x @ lin                                                  # noqa: E731
    h = mx.array(rng.standard_normal((1, 1, HCN, D)).astype(np.float32))
    ref = st(h, lambda x: post_n(core(pre_n(x))))
    monkeypatch.setattr(HC, "IDFUSE", True)
    assert st.fusable(h)
    n0 = HC.STATS["fused"]
    out = st.fused(h, core, pre_n, post_n)
    r = _rel(out, ref)
    print(f"METAL fused sub-layer {norms}: equal {bool(mx.array_equal(out, ref))} rel {r:.2e} stats {HC.STATS}")
    assert HC.STATS["fused"] == n0 + 1 and r <= 1e-5
