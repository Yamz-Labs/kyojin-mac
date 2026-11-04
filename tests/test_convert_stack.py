"""tools/convert_stack.py: per-expert pack -> ragged stacked pack. The numpy part runs anywhere; the Metal part (skipped without
Metal) checks that an expert cut out of the stacked words runs through the existing GEMV and matches the per-expert tensor."""
import importlib.util, json, os, sys
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("convert_stack", os.path.join(os.path.dirname(__file__), "..", "tools", "convert_stack.py"))
CS = importlib.util.module_from_spec(spec); sys.modules["convert_stack"] = CS; spec.loader.exec_module(CS)

KT = NT = 8                                             # 128 x 128 weights: the smallest shape the GEMV accepts
KS = [2, 3, 3, 2, 2, 3]                                 # per-expert bits, mixed like DS4.1 hot/cold experts


def _write_pack(d, rng):
    ents, files = {}, {"layers.0.safetensors": [], "global.safetensors": []}
    for e, K in enumerate(KS):
        for proj in ("w1", "w2", "w3"):
            p = f"layers.0.ffn.experts.{e}.{proj}"
            tr = rng.integers(-32768, 32768, (KT, NT, 16 * K), dtype=np.int16)
            files["layers.0.safetensors"] += [
                CS.bytes_entry(p + ".trellis", "I16", list(tr.shape), tr),
                CS.bytes_entry(p + ".suh", "F16", [KT * 16], rng.standard_normal(KT * 16).astype("<f2")),
                CS.bytes_entry(p + ".svh", "F16", [NT * 16], rng.standard_normal(NT * 16).astype("<f2")),
                CS.bytes_entry(p + ".mul1", "I32", [], np.array(0x83DCD12D - (1 << 32), "<i4"))]
    for s in range(4):
        p = f"layers.0.attn.wo_a.slice.{s}"
        tr = rng.integers(-32768, 32768, (KT, NT, 80), dtype=np.int16)
        files["layers.0.safetensors"] += [
            CS.bytes_entry(p + ".trellis", "I16", list(tr.shape), tr),
            CS.bytes_entry(p + ".suh", "F16", [KT * 16], rng.standard_normal(KT * 16).astype("<f2")),
            CS.bytes_entry(p + ".svh", "F16", [NT * 16], rng.standard_normal(NT * 16).astype("<f2")),
            CS.bytes_entry(p + ".mul1", "I32", [], np.array(1, "<i4"))]
    files["global.safetensors"].append(CS.bytes_entry("embed.weight", "BF16", [4, 8], np.arange(32, dtype="<u2")))
    for f, es in files.items(): CS.write_st(os.path.join(d, f), es, {"format": "pt"})
    json.dump({"quantization_config": {"quant_method": "exl3"}}, open(os.path.join(d, "config.json"), "w"))


@pytest.fixture(scope="module")
def packs(tmp_path_factory):
    src_d, out_d = tmp_path_factory.mktemp("src"), tmp_path_factory.mktemp("out")
    _write_pack(str(src_d), np.random.default_rng(3))
    src, wmap = CS.convert(str(src_d), str(out_d))
    return src, wmap, str(out_d)


def test_layout_and_bytes(packs):
    src, wmap, out = packs
    CS.verify(src, out, wmap, 99)                       # every expert and slice byte-identical to the source
    assert wmap["embed.weight"] == "global.safetensors" and "layers.0.ffn.experts.0.w1.trellis" not in wmap
    assert json.load(open(os.path.join(out, "config.json")))["quantization_config"]["expert_layout"] == "ragged-u32-v1"
    for proj in ("w1", "w2", "w3"):
        st = CS.load_stack(out, wmap, "layers.0.ffn", proj)
        assert list(st["k"]) == KS and list(st["shape"]) == [KT, NT]
        assert list(np.diff(st["offset"])) == [KT * NT * 8 * k for k in KS] and st["offset"][-1] == len(st["words"])
        assert st["suh"].shape == (len(KS), KT * 16) and st["svh"].shape == (len(KS), NT * 16)


def test_rejects_gaps_and_mixed_flags(tmp_path):
    d, o = tmp_path / "s", tmp_path / "o"; d.mkdir()
    rng = np.random.default_rng(0)
    def one(e, flag):
        p = f"layers.0.ffn.experts.{e}.w1"
        return [CS.bytes_entry(p + ".trellis", "I16", [KT, NT, 32], rng.integers(-9, 9, (KT, NT, 32), dtype=np.int16)),
                CS.bytes_entry(p + ".suh", "F16", [128], np.zeros(128, "<f2")), CS.bytes_entry(p + ".svh", "F16", [128], np.zeros(128, "<f2")),
                CS.bytes_entry(p + ".mul1", "I32", [], np.array(flag, "<i4"))]
    CS.write_st(str(d / "a.safetensors"), one(0, 5) + one(2, 5), None)          # expert 1 missing
    with pytest.raises(AssertionError, match="not contiguous"): CS.convert(str(d), str(o))
    CS.write_st(str(d / "a.safetensors"), one(0, 5) + one(1, 6), None)          # two different codebook flags
    with pytest.raises(AssertionError, match="codebook flag differs"): CS.convert(str(d), str(o))


def test_metal_gemv_on_stacked_expert(packs):
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available(): pytest.skip("Metal not available")
    from kyojin_mlx import kernels as KN
    src, wmap, out = packs
    st = CS.load_stack(out, wmap, "layers.0.ffn", "w2")
    x = mx.array(np.random.default_rng(1).standard_normal((1, KT * 16)).astype(np.float16))
    for e, K in enumerate(KS):
        ref = KN.Mul1Gemv(np.frombuffer(src.read(f"layers.0.ffn.experts.{e}.w2.trellis"), "<i2").reshape(KT, NT, 16 * K).copy(),
                          np.array(st["suh"][e]), np.array(st["svh"][e]))
        got = KN.Mul1Gemv(CS.expert_trellis(st, e).copy(), np.array(st["suh"][e]), np.array(st["svh"][e]))
        assert np.array_equal(np.array(ref(x)).view(np.uint32), np.array(got(x)).view(np.uint32))


def test_mx_load_reads_stacked_file(packs):
    mx = pytest.importorskip("mlx.core")
    src, wmap, out = packs
    st = CS.load_stack(out, wmap, "layers.0.ffn", "w1")
    got = mx.load(os.path.join(out, wmap["layers.0.ffn.experts.w1.trellis"]))
    for name, ref in (("trellis", st["words"]), ("offset", st["offset"]), ("k", st["k"]), ("suh", st["suh"]), ("svh", st["svh"])):
        a = got[f"layers.0.ffn.experts.w1.{name}"]
        assert a.dtype == {"trellis": mx.uint32, "offset": mx.uint32, "k": mx.uint8, "suh": mx.float16, "svh": mx.float16}[name]
        assert np.array_equal(np.array(a), np.asarray(ref))
    assert got["layers.0.attn.wo_a.slices.trellis"].shape == (4, KT, NT, 80)


def test_load_time_restack_equals_convert_stack(packs):
    """kyojin_mlx.restack on the raw per-expert pack gives the same names, dtypes, shapes and bytes as the convert_stack output."""
    mx = pytest.importorskip("mlx.core")
    from kyojin_mlx import patch as P
    src, wmap, out = packs
    ref = P.load_pack(out)
    got = P.load_pack(src.dir)
    assert sorted(got) == sorted(ref)
    for k in ref:
        r, g = ref[k], got[k]
        if r.dtype == mx.bfloat16:
            r, g = r.view(mx.uint16), g.view(mx.uint16)
        a, b = np.array(r), np.array(g)
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), k
