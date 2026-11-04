"""Flag vendor_bind (default 0): the per-call vendor lookups of the served hot paths (the fp8 round trip, its DISABLE switch, the draft head's block
collapse, the cache-name test) are either resolved through the generic tables at every call (0) or pinned once by flags.set (1). Both forms must answer
the same object and the same values: the flag measures the indirection, it changes no arithmetic. No model, no Metal needed for the fp8 / cache parts;
the collapse test needs the DS4.1 model code and is skipped without it."""
import mlx.core as mx
import pytest

from kyojin_mlx import archs as A, flags as FL


@pytest.fixture(autouse=True)
def _flag_off():
    FL.set("vendor_bind", 0)
    yield
    FL.set("vendor_bind", 0)


def test_default_is_0_and_runtime_switchable():
    d = FL.get_all()["vendor_bind"]
    assert d["env"] == "KYOJIN_VENDOR_BIND" and d["default"] == "0" and d["runtime"] is True
    assert FL.value("vendor_bind") == "0"
    assert A._FQ is None and A._FQ_DISABLE is None and A._HC_PRE is None and A._CACHE_NAMES is None
    FL.set("vendor_bind", 1)
    assert FL.value("vendor_bind") == "1"


def test_on_pins_the_same_objects():
    fq = A.sub("fakequant").fake_quant_fp8_ue8m0
    dis = A.sub("fakequant").DISABLE
    names = A.cache_class_names()
    FL.set("vendor_bind", 1)
    assert A._FQ is fq and A._FQ_DISABLE is dis and A._CACHE_NAMES == names
    assert A._CACHE_NAMES is not None and isinstance(A._CACHE_NAMES, frozenset)
    FL.set("vendor_bind", 0)
    assert A._FQ is None and A._FQ_DISABLE is None and A._CACHE_NAMES is None


def test_fq_is_bit_identical_in_both_states(monkeypatch):
    mx.random.seed(0)
    x = (mx.random.normal((8, 128)) * 0.5).astype(mx.float16)
    ref = A.sub("fakequant").fake_quant_fp8_ue8m0(x)
    out0 = A.fq(x)
    FL.set("vendor_bind", 1)
    out1 = A.fq(x)
    assert mx.array_equal(ref, out0) and mx.array_equal(ref, out1)
    assert A.fq_on() is True and A._FQ_DISABLE is False
    FL.set("vendor_bind", 0)
    monkeypatch.setattr(A.sub("fakequant"), "DISABLE", True)                 # the accuracy gate's --no-fq
    assert A.fq_on() is False                                                 # resolved per call: the flip is seen
    FL.set("vendor_bind", 1)                                                  # pinned: set() reads the flipped constant
    assert A.fq_on() is False and A._FQ_DISABLE is True


def test_cache_name_test_answers_the_same():
    class DeepseekV41Cache:                                                   # the name the active DS4.1 spec holds
        pass

    class Packed(DeepseekV41Cache):                                           # a subclass: fp4pool keeps the name in the MRO
        pass

    class Other:
        pass

    for on in (0, 1):
        FL.set("vendor_bind", on)
        assert A.is_model_cache(DeepseekV41Cache()) and A.is_model_cache(Packed()) and not A.is_model_cache(Other())


def test_hc_pre_is_the_same_function():
    pytest.importorskip("mlx_vlm.models.deepseek_v41")
    h = mx.zeros((2, 1, 4, 8), dtype=mx.float16)
    pre = mx.zeros((2, 1, 4), dtype=mx.float32)
    ref = A.cls("block").hc_pre(h, pre)
    mx.eval(ref)
    for on in (0, 1):
        FL.set("vendor_bind", on)
        got = A.hc_pre(h, pre)
        mx.eval(got)
        assert got.shape == ref.shape and bool(mx.array_equal(got, ref))
