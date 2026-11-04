"""cmd5: kvchain / ropefuse / rope_fma, what is provable without a GPU.

* the stock op's fma form (read out of the shipped AIR, tests/data excerpt) is form 1, and `rope_fma=1` writes exactly that dataflow;
* the fp32 fma emulation is a true single rounding;
* the one place `contract(off)` could not matter in the kv rms (the sum of squares of fp16 values) is exact, so it cannot move the norm;
* `rope_fma=3` (auto) picks the form that mx.fast.rope is measured to use (CPU stream here; the GPU answer comes from tools/kvchain_bits.py);
* the pieces of tools/kvchain_bits.py that need no Metal (bit comparison report, stage kernel surgery)."""
import os
import sys
from fractions import Fraction

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "tools"))

from kyojin_mlx import dsv4_core as C, flags as FL  # noqa: E402
from kyojin_mlx.ref_numpy import fma32, rope_pairs_f32, rope_pairs_ref  # noqa: E402

import air_rope_form as AIR  # noqa: E402
import kvchain_bits as KB  # noqa: E402


def _exact_round32(ex: Fraction) -> np.float32:
    f = np.float32(float(ex))
    cands = [f, np.nextafter(f, np.float32(-np.inf)), np.nextafter(f, np.float32(np.inf))]
    return min(cands, key=lambda v: (abs(Fraction(float(v)) - ex), abs(float(v))))


def test_fma32_is_a_single_rounding():
    rng = np.random.default_rng(0)
    n = 4000
    a = (rng.standard_normal(n) * np.exp(rng.uniform(-8, 8, n))).astype(np.float32)
    b = (rng.standard_normal(n) * np.exp(rng.uniform(-8, 8, n))).astype(np.float32)
    c = (rng.standard_normal(n) * np.exp(rng.uniform(-30, 30, n))).astype(np.float32)
    got = fma32(a, b, c)
    for i in range(n):
        assert got[i] == _exact_round32(Fraction(float(a[i])) * Fraction(float(b[i])) + Fraction(float(c[i]))), i


def test_fma32_differs_from_two_roundings_often_enough_to_matter():
    rng = np.random.default_rng(1)
    a, b, c = (rng.standard_normal(100000).astype(np.float32) for _ in range(3))
    assert 0.05 < float((fma32(a, b, c) != a * b + c).mean()) < 0.5


def test_fp16_squares_are_exact_in_fp32_so_the_norm_sum_ignores_contraction():
    """acc += f * f (two roundings, contract off) == fma(f, f, acc) (the stock rms_norm AIR) because f * f is exact: 11 + 11 bits < 24, exponent in range."""
    h = np.arange(65536, dtype=np.uint16).view(np.float16)
    h = h[np.isfinite(h)].astype(np.float32)
    assert np.array_equal(h * h, (h.astype(np.float64) ** 2).astype(np.float32))
    assert np.array_equal((h.astype(np.float64) ** 2), (h * h).astype(np.float64))                 # and no rounding happened in the fp32 product
    acc = np.random.default_rng(2).standard_normal(h.size).astype(np.float32) * 50
    assert np.array_equal(acc + h * h, fma32(h, h, acc))


def test_form1_is_the_literal_dataflow_of_the_shipped_op():
    """AIR: r1 = fmuladd(x1, cos, (-sin) * x2), r2 = fmuladd(x1, sin, cos * x2); one rounding of the fused product, one of the other product."""
    rng = np.random.default_rng(3)
    x1, x2, c, s = (rng.standard_normal(5000).astype(np.float32) for _ in range(4))
    r1, r2 = rope_pairs_f32(x1, x2, c, s, 1)
    for i in range(0, 5000, 7):
        p = np.float32(-s[i]) * x2[i]
        assert r1[i] == _exact_round32(Fraction(float(x1[i])) * Fraction(float(c[i])) + Fraction(float(p)))
        q = c[i] * x2[i]
        assert r2[i] == _exact_round32(Fraction(float(x1[i])) * Fraction(float(s[i])) + Fraction(float(q)))
    assert not np.array_equal(r1, x1 * c - x2 * s) and not np.array_equal(r1, rope_pairs_f32(x1, x2, c, s, 2)[0])


def test_the_shipped_air_excerpt_is_form_1():
    ll = open(os.path.join(ROOT, "tests", "data", "air_rope_single_float16.ll")).read()
    fm = AIR.roles(ll)
    assert len(fm) == 4 and AIR.form_of(fm) == "form1", fm
    assert "fma(xa, cos, -(sin*xb))" in fm and "fma(xa, sin, (cos*xb))" in fm


def test_air_classifier_tells_the_forms_apart():
    base = """define void @k() {
  %a = fpext half %h0 to float
  %b = fpext half %h1 to float
  %cc = tail call float @air.fast_cos.f32(float %t)
  %ss = tail call float @air.fast_sin.f32(float %t)
%BODY%
}
"""
    f2 = "  %n = fneg float %b\n  %m1 = fmul float %a, %cc\n  %r1 = tail call float @llvm.fmuladd.f32(float %n, float %ss, float %m1)\n" \
         "  %m2 = fmul float %a, %ss\n  %r2 = tail call float @llvm.fmuladd.f32(float %b, float %cc, float %m2)\n"
    f1 = "  %n = fneg float %ss\n  %m1 = fmul float %n, %b\n  %r1 = tail call float @llvm.fmuladd.f32(float %a, float %cc, float %m1)\n" \
         "  %m2 = fmul float %cc, %b\n  %r2 = tail call float @llvm.fmuladd.f32(float %a, float %ss, float %m2)\n"
    plain = "  %m1 = fmul float %a, %cc\n  %m2 = fmul float %b, %ss\n  %r1 = fsub float %m1, %m2\n"
    assert AIR.form_of(AIR.roles(base.replace("%BODY%", f1))) == "form1"
    assert AIR.form_of(AIR.roles(base.replace("%BODY%", f2))) == "form2"
    assert AIR.form_of(AIR.roles(base.replace("%BODY%", plain))) == "other"                     # no fmuladd at all: contract off


def test_bitcode_extent_walks_top_level_blocks():
    """magic + one empty block (id 13, abbrev width 2) + an empty STRTAB block (id 23) -> the extent ends after the STRTAB block, trailing junk ignored."""
    def block(bid):                                      # ENTER_SUBBLOCK: abbrev id 1 (2 bits), block id vbr8, new abbrev width vbr4, align 32, length word
        bits = 1 | (bid << 2) | (3 << 10)                # 2 + 8 + 4 = 14 bits, padded to 32
        return bits.to_bytes(4, "little") + (0).to_bytes(4, "little")
    raw = b"BC\xc0\xde" + block(13) + block(23)
    assert AIR.bitcode_extent(raw + b"\x99" * 40) == len(raw)


def test_rope_fma_flag_values():
    assert FL.FLAGS["rope_fma"].default == "3" and C.ROPE_FMA in (0, 1, 2)
    assert C.resolve_rope_fma(0) == 0 and C.resolve_rope_fma(1) == 1 and C.resolve_rope_fma(2) == 2
    assert C.resolve_rope_fma(3) in (0, 1, 2)
    old = C.ROPE_FMA
    try:
        FL.set("rope_fma", "1")
        assert C.ROPE_FMA == 1 and C._fk() == "_fma1"
        FL.set("rope_fma", "3")
        assert C.ROPE_FMA in (0, 1, 2)
    finally:
        FL.set("rope_fma", str(old))
    assert C.ROPE_FMA == old


def test_detect_picks_the_form_the_op_uses_on_this_stream():
    form = C.detect_rope_form(stream=mx.cpu)
    bad = C.ROPE_DETECT["bad"]
    assert form in (0, 1, 2), bad
    assert bad[form] == 0 and all(b > 1000 for f, b in bad.items() if f != form), bad          # the forms are far apart in fp32: ~20 % of the elements


def test_compare_reports_the_first_difference():
    a = np.arange(24, dtype=np.float16).reshape(2, 3, 4)
    assert KB.compare("op", a, a.copy())["bad"] == 0
    b = a.copy()
    b[1, 0, 2] = np.nextafter(b[1, 0, 2], np.float16(np.inf))
    b[1, 2, 3] = np.float16(3.0)
    r = KB.compare("op", a, b)
    assert r["bad"] == 2 and r["first"]["index"] == (1, 0, 2) and r["first"]["ulp"] == 1 and r["first"]["flat"] == 14
    assert "(1, 0, 2)" in KB.describe(r) and "1 ulp" in KB.describe(r)
    z = np.array([0.0, -0.0], np.float16)
    assert KB.compare("z", z, z[::-1].copy())["bad"] == 2 and KB.compare("z", z, z[::-1].copy())["first"]["ulp"] == 0
    n1, n2 = np.array([np.nan], np.float16), np.array([0x7e01], np.uint16).view(np.float16)
    assert KB.compare("nan", n1, n2)["bad"] == 0


@pytest.mark.parametrize("form", [0, 1, 2])
def test_stage_kernels_cut_the_production_source_and_nothing_else(form):
    st = KB.stage_sources(form)
    assert set(st) == {"kv_norm", "kv_rope", "kv_fp8"}
    prod = C.rope_src(C._KV_CHAIN, form)
    assert "out[o + lid * 4 + i] = nb[lid * 4 + i];" in st["kv_norm"] and "rsqrt" in st["kv_norm"] and "amax" in st["kv_norm"]
    assert "rsqrt" in st["kv_rope"] and "nb[lid * 4 + i] = xr[lid * 4 + i];" in st["kv_rope"] and "static_cast<T>(v[i]);\n    return;" in st["kv_rope"]
    assert "if (false) {" in st["kv_fp8"] and "rint" in st["kv_fp8"]
    for k, v in st.items():
        assert v.count("fma(") == prod.count("fma(") == (0 if form == 0 else 2), k                       # the rope tail text is the production one, form included
    assert KB.q_rope_source(form).count("fma(") == (0 if form == 0 else 2) and KB.inv_rope_source(form).count("fma(") == (0 if form == 0 else 2)
    assert C._Q_ROPE in KB.q_rope_source(0)
    assert "float v[DPT];" in KB.inv_rope_source(0)
