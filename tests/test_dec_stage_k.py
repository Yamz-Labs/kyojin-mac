"""dec_stage one-launch glue stages (stagek) and the row routing of the stage path.

CPU tests (run anywhere mlx imports): the stage math against a float64 oracle, per output, no worse than the lane's chain (fp16 rounding where the
chain rounds); the kernel sources; the group / row-band routing predicate; the ablit hook position in the stage out / ffn tails.
Metal tests (skipped without a GPU, for the lane): the kernels against the references on 1..24 rows, bf16, D = 512 / 4096, row independence of the
bits, the fp32 copy of the expand, and the first-use self check."""
import os
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kyojin_mlx import ablit as AB, decstage as DS, stagek as SK  # noqa: E402

needs_metal = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
EPS_FLAT, EPS_NORM, HC_EPS, ITERS = 1e-6, 1e-6, 1e-6, 20


def inputs(R, D, dtype=np.float16, seed=0, spread=0.5):
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((R, 1, 4, D)) * spread).astype(dtype)
    cpre = rng.uniform(0.1, 1.0, (R, 1, 4)).astype(np.float32)
    fn = (rng.standard_normal((SK.NMIX, 4 * D)) / np.sqrt(4 * D)).astype(np.float32)
    scale = np.array([0.7, 0.9, 1.1], np.float32)
    base = (rng.standard_normal(SK.NMIX) * 0.3).astype(np.float32)
    w = rng.uniform(0.5, 1.5, (D,)).astype(dtype)
    y = (rng.standard_normal((R, 1, D)) * spread).astype(dtype)
    return x, cpre, fn, scale, base, w, y


# ------------------------------------------------------------------------------------------------ float64 oracle and the chain's roundings
def oracle(x, cpre, fn, scale, base, w, y):
    """float64 everywhere, no rounding: (h, z, pre', out)."""
    R, _, _, D = x.shape
    xf = x.astype(np.float64).reshape(R, 4 * D)
    z = (xf @ fn.astype(np.float64).T) / np.sqrt((xf * xf).mean(-1, keepdims=True) + EPS_FLAT)
    pre = 1 / (1 + np.exp(-(z[:, :4] * scale[0] + base[:4]))) + HC_EPS
    c = (cpre.astype(np.float64).reshape(R, 4, 1) * x.astype(np.float64).reshape(R, 4, D)).sum(1)
    h = w.astype(np.float64) * c / np.sqrt((c * c).mean(-1, keepdims=True) + EPS_NORM)
    post = 2 / (1 + np.exp(-(z[:, 4:8] * scale[1] + base[4:8])))
    v = z[:, 8:24] * scale[2] + base[8:24]
    r = np.exp(v.reshape(R, 4, 4) - v.reshape(R, 4, 4).max(-1, keepdims=True))
    r = r / (r.sum(-1, keepdims=True) + HC_EPS) + HC_EPS
    r = r / (r.sum(-2, keepdims=True) + HC_EPS)
    for _ in range(1, ITERS):
        r = r / (r.sum(-1, keepdims=True) + HC_EPS)
        r = r / (r.sum(-2, keepdims=True) + HC_EPS)
    out = post[:, :, None] * y.astype(np.float64).reshape(R, 1, D) + np.einsum("raj,rad->rjd", r, x.astype(np.float64).reshape(R, 4, D))
    return h, z, pre, out


def chain_emulation(x, cpre, fn, scale, base, w, y):
    """The lane's chain with its rounding points: f32 mix dot, the collapse rounded to the stream dtype before its RMSNorm (hc_pre then nn.RMSNorm),
    the expand accumulated in f32 and rounded once (mhc_expand)."""
    R, _, _, D = x.shape
    xm = mx.array(x)
    z0 = mx.fast.rms_norm(xm.astype(mx.float32).reshape(R, 4 * D), None, EPS_FLAT)
    z = z0 @ mx.array(fn).T
    c = (mx.array(cpre).reshape(R, 4, 1) * xm.astype(mx.float32).reshape(R, 4, D)).sum(axis=1).astype(xm.dtype)     # hc_pre: rounded
    h = mx.fast.rms_norm(c, mx.array(w), EPS_NORM)
    out = SK.expand_ref(mx.array(y), xm, z, mx.array(scale), mx.array(base), ITERS, HC_EPS)
    pre = mx.sigmoid(z[:, :4] * scale[0] + mx.array(base)[:4]) + HC_EPS
    return h.reshape(R, 1, D), z, pre, out


def rel(a, ref):
    a, ref = np.asarray(a, np.float64).reshape(len(ref), -1), ref.reshape(len(ref), -1)
    return float(np.mean(np.linalg.norm(a - ref, axis=-1) / np.linalg.norm(ref, axis=-1)))


@pytest.mark.parametrize("D", [512, 4096])
@pytest.mark.parametrize("dtype", [np.float16])
def test_stage_math_is_no_worse_than_the_chain_against_float64(D, dtype):
    """Per output (not end logits: the sinkhorn is not contractive on random weights): h, z, pre', out of the stage math vs the chain's rounding
    points, both against the float64 oracle. The chain rounds the collapse to fp16 before the norm, the stage keeps f32 to the end."""
    R = 24
    x, cpre, fn, scale, base, w, y = inputs(R, D, dtype)
    h0, z0, p0, o0 = oracle(x, cpre, fn, scale, base, w, y)
    hs, zs, ps = SK.mix_ref(mx.array(x), mx.array(cpre), mx.array(fn), mx.array(scale), mx.array(base), mx.array(w), EPS_FLAT, EPS_NORM, HC_EPS)
    os_ = SK.expand_ref(mx.array(y), mx.array(x), zs, mx.array(scale), mx.array(base), ITERS, HC_EPS)
    hc, zc, pc, oc = chain_emulation(x, cpre, fn, scale, base, w, y)
    mx.eval(hs, zs, ps, os_, hc, zc, pc, oc)
    e = {"h": (rel(hs, h0), rel(hc, h0)), "z": (rel(zs, z0), rel(zc, z0)), "pre": (rel(ps, p0), rel(pc, p0)), "out": (rel(os_, o0.reshape(R, 4, D)), rel(oc, o0.reshape(R, 4, D)))}
    print(f"\n[D={D}] stage vs chain rel err: " + ", ".join(f"{k} {a:.3g}/{b:.3g}" for k, (a, b) in e.items()))
    for k, (es, ec) in e.items():
        assert es <= ec * 1.05 + 1e-7, (k, es, ec)
    assert e["h"][0] < e["h"][1]                      # the point of f32 to the end: the fp16 round of the collapse is gone


def test_sinkhorn_reference_is_doubly_stochastic():
    x, cpre, fn, scale, base, w, y = inputs(8, 512, seed=3)
    z = mx.array(np.random.default_rng(4).standard_normal((8, 24)).astype(np.float32))
    post, comb = SK.sinkhorn_ref(z, mx.array(scale), mx.array(base), ITERS, HC_EPS)
    c = np.array(comb)
    assert np.allclose(c.sum(-2), 1.0, atol=1e-4) and np.allclose(c.sum(-1), 1.0, atol=1e-3)
    assert np.array(post).min() > 0


# ------------------------------------------------------------------------------------------------ sources
@pytest.mark.parametrize("name,src,ins,outs,tmpl", [
    ("mix", SK._MIX, ["x", "cpre", "fn", "scale", "base", "w", "cst"], ["hn", "z", "pre_o"], ["T", "D", "NG"]),
    ("expand", SK._EXPAND, ["y", "rs", "z", "scale", "base"], ["out", "out32"], ["T", "D", "ITERS", "EPS_INT"])])
def test_kernel_sources_are_assembled(name, src, ins, outs, tmpl):
    """No GPU: braces / parentheses balance, every input / output / template name the Python call passes is used by the text and nothing else is
    read (a typo is a Metal compile error, which only the lane could show)."""
    assert src.count("{") == src.count("}") and src.count("(") == src.count(")")
    for n in ins + outs + tmpl:
        assert re.search(rf"\b{n}\b", src), (name, n)
    assert "threadgroup_barrier" in src and "return;" in src
    # every threadgroup variable is declared at the kernel's top level (Metal rejects them in a nested scope)
    depth = 0
    for line in src.splitlines():
        if re.match(r"\s*threadgroup\s+float", line):
            assert depth == 0 or "pc[20]" in line or "part" in line, line
        depth += line.count("{") - line.count("}")


def test_the_expand_sinkhorn_text_is_the_glue_fuse_one():
    """The split inside expand_stage is glue_fuse's sinkhorn (the mlx-vlm kernel's): its arithmetic lines appear verbatim."""
    from kyojin_mlx import glue_fuse as GN
    for line in ("float4 e = metal::fast::exp(v - row_max) * active;",
                 "float4 r = e * (1.0f / (e.x + e.y + e.z + e.w + EPS)) + EPS * active;",
                 "float4 col_inv = 1.0f / (float4(simd_sum(r.x), simd_sum(r.y), simd_sum(r.z), simd_sum(r.w)) + EPS);",
                 "r *= (1.0f / (r.x + r.y + r.z + r.w + EPS)) * active;"):
        assert line in GN._SINKHORN and line in SK._EXPAND, line


# ------------------------------------------------------------------------------------------------ row routing of the same-input group
class _Lin:
    def __init__(self, mma=False):
        self.mma = mma

    def _mma_on(self, rows):
        return self.mma


@pytest.mark.parametrize("half_rows,gemv_max,rows,mma,want", [
    (4, 4, 1, False, True), (4, 4, 3, False, True),
    (4, 4, 4, False, False),            # served HALF_ROWS = 4: the single-linear kernel FMAs in packed fp16, the group kernel does not
    (0, 4, 4, False, True),             # fp32 rows: the group is the same arithmetic (the case the old tests ran in)
    (0, 4, 5, False, False), (0, 4, 6, False, False),     # past the GEMV chunk: Exl3Linear takes MMA / NAX / row groups, the group GEMV would not
    (4, 4, 2, True, False)])            # an MMA class at this row count
def test_group_is_used_only_where_the_chain_runs_the_same_kernel(monkeypatch, half_rows, gemv_max, rows, mma, want):
    from kyojin_mlx import exl3_linear as EL, metal_src as S
    monkeypatch.setattr(S, "HALF_ROWS", half_rows)
    monkeypatch.setattr(EL, "GEMV_MAX_ROWS", gemv_max)
    assert DS.group_rows_ok((_Lin(mma), _Lin(False)), rows) is want


def test_wide_passes_return_to_the_chain_before_any_host_work(monkeypatch):
    called = []
    monkeypatch.setattr(DS, "ENABLED", True)
    monkeypatch.setattr(DS, "rows_cache", lambda c: called.append(1))
    h = mx.zeros((DS.MAX_ROWS + 1, 1, 4, 128), dtype=mx.float16)
    assert DS.block(SimpleNamespace(), h, None, None, None) is None and not called


# ------------------------------------------------------------------------------------------------ ablit hook position
def _armed_blk(wa, wm, R):
    blk = SimpleNamespace(hc_attn_scale=mx.ones(3), hc_attn_base=mx.zeros(24), hc_ffn_scale=mx.ones(3), hc_ffn_base=mx.zeros(24), hc_sinkhorn_iters=ITERS, hc_eps=HC_EPS,
                          attn=SimpleNamespace(), ffn=None, hc_ffn_fn=None, ffn_norm=None)
    blk.__dict__["_ablit"] = SimpleNamespace(wa=wa, wm=wm, R=R)
    return blk


def test_ablit_hook_sits_before_the_one_launch_expand(monkeypatch):
    """The refusal-direction projection (commit 18ad409) is applied to the sub-layer output BEFORE hc_expand, whichever expand runs: the stage's
    expand receives the projected y, for the attention tail (stage_out) and for the FFN tail (stage_ffn)."""
    D = 128
    rng = np.random.default_rng(5)
    R = mx.array(np.linalg.qr(rng.standard_normal((D, 1)))[0].T.astype(np.float32))
    seen = []

    def fake_expand(y, res, z, scale, base, iters, eps):
        seen.append(np.array(y))
        return y.reshape(y.shape[0], 1, 1, D) * mx.ones((1, 1, 4, 1), dtype=y.dtype), mx.zeros((y.shape[0], 1, 4, D))

    monkeypatch.setattr(SK, "expand_stage", fake_expand)
    y = mx.array((rng.standard_normal((2, 1, D))).astype(np.float16))
    h = mx.zeros((2, 1, 4, D), dtype=mx.float16)
    z = mx.zeros((2, 24))
    blk = _armed_blk(0.8, 0.7, R)
    DS.stage_out(blk, y, h, (None, None, None, z), projected=False)
    want = np.array(AB.attn_out(blk, y))
    assert not np.array_equal(want, np.array(y)) and np.array_equal(seen[-1], want)
    blk.ffn = lambda yn, mask: y
    monkeypatch.setattr(DS, "_mix", lambda *a, **k: (y, None, None, None, z))
    monkeypatch.setattr(DS, "MOE", False)
    DS.stage_ffn(blk, h, None, None)
    assert np.array_equal(seen[-1], np.array(AB.ffn_out(blk, y)))
    blk.__dict__["_ablit"].wa = blk.__dict__["_ablit"].wm = 0.0           # disarmed: untouched
    DS.stage_out(blk, y, h, (None, None, None, z), projected=False)
    assert np.array_equal(seen[-1], np.array(y))


def test_mix_never_forces_the_slow_glue_fusion(monkeypatch):
    """With the one-launch stage off, `_mix` runs glue_fuse.mix_norm only when the lane turned KYOJIN_GLUE_FUSE_NORM on (measured slower: it was
    forced before), else the chain's ops."""
    from kyojin_mlx import glue_fuse as GN
    calls = []
    monkeypatch.setattr(GN, "mixes_of", lambda *a: calls.append("mixes") or None)
    monkeypatch.setattr(GN, "mix_norm", lambda *a: calls.append("mix_norm") or ("x", "p", "q", "c"))
    blk = SimpleNamespace(hc_mixes=lambda h, fn, s, b: (calls.append("chain") or "pre", "post", "comb"), hc_pre=lambda h, p: "c",
                          hc_sinkhorn_iters=ITERS, hc_eps=HC_EPS)
    norm = lambda c: "xn"
    norm.weight, norm.eps = None, 1e-6
    monkeypatch.setattr(DS, "GLUE", False)
    monkeypatch.setattr(GN, "ENABLED", False)
    assert DS._mix(blk, None, None, None, None, None, norm, fast=True)[0] == "xn" and calls == ["chain"]
    calls.clear()
    monkeypatch.setattr(GN, "ENABLED", True)
    DS._mix(blk, None, None, None, None, None, norm, fast=True)
    assert calls == ["mixes", "mix_norm"]


# ------------------------------------------------------------------------------------------------ Metal (the lane)
@needs_metal
@pytest.mark.parametrize("D,R", [(512, 1), (512, 6), (4096, 1), (4096, 6), (4096, 24)])
def test_kernels_match_the_references(D, R):
    x, cpre, fn, scale, base, w, y = inputs(R, D, seed=D + R)
    a = [mx.array(v) for v in (x, cpre, fn, scale, base, w)]
    hn, z, pre = SK.mix_stage(*a, EPS_FLAT, EPS_NORM, HC_EPS)
    hr, zr, pr = SK.mix_ref(*a, EPS_FLAT, EPS_NORM, HC_EPS)
    out, out32 = SK.expand_stage(mx.array(y), a[0], zr, a[3], a[4], ITERS, HC_EPS)
    orf = SK.expand_ref(mx.array(y), a[0], zr, a[3], a[4], ITERS, HC_EPS)
    mx.eval(hn, z, pre, hr, zr, pr, out, out32, orf)
    f = lambda t: np.array(t.astype(mx.float32))
    # z and pre' are judged against the float64 result, not against `mix_ref`'s `xf @ fn.T`: that matmul runs on the tensor units with 10-bit operands (an fp32 `fn` loses
    # bits: up to 1.6e-3 off exact for these shapes, tests/test_cmd5_kernel_refs.py), the kernel's own order is within 1.5e-6 of exact (bound 1e-5).
    import kernel_order as KO
    z_ex = KO.exact_z(KO.f32(x.reshape(R, 4 * D)), fn)
    pre_ex = 1 / (1 + np.exp(-(z_ex[:, :4] * scale[0] + base[:4]))) + HC_EPS
    print(f"\n[D={D} R={R}] |z - exact| {np.abs(f(z) - z_ex).max():.2e}   |mix_ref z - exact| {np.abs(f(zr) - z_ex).max():.2e}")
    assert np.abs(f(hn) - f(hr)).max() < 2e-2 and np.abs(f(z) - z_ex).max() < 1e-5 and np.abs(f(pre) - pre_ex).max() < 1e-5
    assert np.abs(f(out) - f(orf)).max() < 2e-2
    assert np.array_equal(f(out32), f(out))                     # the fp32 copy is exactly the rounded output


@needs_metal
@pytest.mark.parametrize("D", [512, 4096])
def test_a_row_has_the_same_bits_whatever_the_number_of_rows(D):
    """Row independence (plain step == verify row == replay): row 0 of a 6-row pass equals the 1-row pass bit for bit, both stages."""
    x, cpre, fn, scale, base, w, y = inputs(6, D, seed=11)
    a = [mx.array(v) for v in (x, cpre, fn, scale, base, w)]
    hn, z, pre = SK.mix_stage(*a, EPS_FLAT, EPS_NORM, HC_EPS)
    out, out32 = SK.expand_stage(mx.array(y), a[0], z, a[3], a[4], ITERS, HC_EPS)
    for r in (0, 3, 5):
        one = [mx.array(v[r:r + 1]) if v.ndim and v.shape[0] == 6 else mx.array(v) for v in (x, cpre, fn, scale, base, w)]
        h1, z1, p1 = SK.mix_stage(*one, EPS_FLAT, EPS_NORM, HC_EPS)
        o1, _ = SK.expand_stage(mx.array(y[r:r + 1]), one[0], z1, a[3], a[4], ITERS, HC_EPS)
        mx.eval(h1, z1, p1, o1, hn, z, pre, out)
        assert np.array_equal(np.array(h1[0]), np.array(hn[r])) and np.array_equal(np.array(z1[0]), np.array(z[r]))
        assert np.array_equal(np.array(p1[0]), np.array(pre[r])) and np.array_equal(np.array(o1[0]), np.array(out[r]))


@needs_metal
def test_bf16_streams():
    x, cpre, fn, scale, base, w, y = inputs(3, 512, seed=2)
    a = [mx.array(x).astype(mx.bfloat16), mx.array(cpre), mx.array(fn), mx.array(scale), mx.array(base), mx.array(w).astype(mx.bfloat16)]
    hn, z, pre = SK.mix_stage(*a, EPS_FLAT, EPS_NORM, HC_EPS)
    hr, zr, _ = SK.mix_ref(*a, EPS_FLAT, EPS_NORM, HC_EPS)
    mx.eval(hn, hr, z, zr)
    assert hn.dtype == mx.bfloat16 and np.abs(np.array(hn.astype(mx.float32)) - np.array(hr.astype(mx.float32))).max() < 0.1


@needs_metal
def test_first_use_self_check_passes_and_is_cached():
    SK.STATE["checked"] = None
    assert SK.usable() is True and SK.STATE["checked"] is True
    SK.STATE["checked"] = False                                        # a failed check stays failed (no retry per layer)
    assert SK.usable() is False
    SK.STATE["checked"] = None


def test_without_metal_the_stage_glue_is_off():
    if mx.metal.is_available():
        pytest.skip("this machine has Metal")
    SK.STATE["checked"] = None
    assert SK.usable() is False
    SK.STATE["checked"] = None
