"""Vendor-quantized converter inputs on the Mac: the byte-identical numpy reader (vendor_input) and its MLX glue
(vendor_src: lazy per-linear float16 dequant, companions dropped, prefix scans). CPU device only, no GPU needed."""
import json, os, struct, sys
from pathlib import Path

import numpy as np
import pytest
import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kyojin_mlx import vendor_input as vi
from kyojin_mlx import vendor_src as vs

rng = np.random.default_rng(7)


@pytest.fixture(autouse=True)
def cpu_device():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


def same(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        return False
    n = ~np.isnan(a)
    return np.array_equal(a[n].view(np.uint32), b[n].view(np.uint32))


def write_st(path, tensors):
    hdr, off, blobs = {}, 0, []
    for k, (tag, arr) in tensors.items():
        raw = np.ascontiguousarray(arr).tobytes()
        hdr[k] = {"dtype": tag, "shape": list(arr.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw); off += len(raw)
    h = json.dumps(hdr).encode(); h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h)
        for b in blobs: f.write(b)


def e4m3(shape): return (rng.integers(0, 0x7C, size=shape) | (rng.integers(0, 2, size=shape) << 7)).astype(np.uint8)
def pack(n): return (n[..., 0::2] | (n[..., 1::2] << 4)).astype(np.uint8)


def test_codebooks_against_mlx_and_hand_values():
    b = np.arange(256, dtype=np.uint8)
    ok = ~np.isnan(vi.E4M3FN_LUT)
    got = np.array(mx.from_fp8(mx.array(b), mx.float32))
    assert np.array_equal(got[ok].view(np.uint32), vi.E4M3FN_LUT[ok].view(np.uint32)) and np.isnan(got[~ok]).all()
    assert vi.E8M0_LUT[127] == 1.0 and vi.E8M0_LUT[130] == 8.0 and np.isnan(vi.E8M0_LUT[255])
    assert vi.E2M1_LUT.tolist()[:8] == [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    assert vi.unpack_e2m1(np.array([[0x21, 0xF7]], np.uint8)).tolist() == [[0.5, 1.0, 6.0, -6.0]]


@pytest.fixture
def src(tmp_path):
    q = "model.language_model.layers.0.self_attn.q_proj"
    wb, s = e4m3((256, 384)), (2.0 ** rng.integers(-4, 2, size=(2, 3))).astype(np.float32)
    nib = rng.integers(0, 16, size=(64, 128)).astype(np.uint8)
    sc = (rng.integers(0, 0x70, size=(64, 8)) & 0x7F).astype(np.uint8)
    mnib = rng.integers(0, 16, size=(64, 128)).astype(np.uint8)
    eb = rng.integers(120, 132, size=(64, 4)).astype(np.uint8)
    bf = rng.integers(0, 0x7F00, size=(8, 16)).astype(np.uint16)
    T = {q + ".weight": ("F8_E4M3", wb), q + ".weight_scale_inv": ("F32", s),
         "m.l.mlp.up_proj.weight": ("U8", pack(nib)), "m.l.mlp.up_proj.weight_scale": ("F8_E4M3", sc),
         "m.l.mlp.up_proj.weight_scale_2": ("F32", np.array(2.0 ** -9, np.float32)), "m.l.mlp.up_proj.input_scale": ("F32", np.array(1.0, np.float32)),
         "m.l.mlp.down_proj.weight": ("I8", pack(mnib).view(np.int8)), "m.l.mlp.down_proj.scale": ("F8_E8M0", eb),
         "model.embed_tokens.weight": ("BF16", bf)}
    write_st(tmp_path / "model.safetensors", T)
    json.dump({"quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}}, open(tmp_path / "config.json", "w"))
    return str(tmp_path), dict(q=q, wb=wb, s=s, nib=nib, sc=sc, mnib=mnib, eb=eb, bf=bf)


def test_lazy_source_dequantizes_each_format_exactly(src, monkeypatch):
    d, t = src
    log = []
    W = vs.load_source(d, ["model.safetensors"], log.append)
    assert isinstance(W, vs.SrcDict) and "source format: mixed" in log[0]
    assert not any(k.endswith(("weight_scale_inv", "weight_scale", "weight_scale_2", "input_scale", ".scale")) for k in W)   # companions dropped
    f16 = lambda x: np.asarray(x, np.float32).astype(np.float16)
    fp8 = vi.dequant_fp8(t["wb"], "F8_E4M3", t["s"], (128, 128))
    assert np.array_equal(np.array(W[t["q"] + ".weight"]), f16(fp8)) and W[t["q"] + ".weight"].dtype == mx.float16
    nv = vi.dequant_nvfp4(pack(t["nib"]), t["sc"], np.float32(2.0 ** -9))
    assert np.array_equal(np.array(W["m.l.mlp.up_proj.weight"]), f16(nv))
    mxw = vi.dequant_mxfp4_linear(pack(t["mnib"]), vi.e8m0_to_f32(t["eb"]))
    assert np.array_equal(np.array(W["m.l.mlp.down_proj.weight"]), f16(mxw))
    assert np.array_equal(np.array(W["model.embed_tokens.weight"].astype(mx.float32)), vi.bf16_to_f32(t["bf"]))   # BF16 untouched


def test_prefix_scan_only_dequantizes_matches(src, monkeypatch):
    d, t = src
    W = vs.load_source(d, ["model.safetensors"], lambda *a: None)
    calls = []
    real = vi.load_linear
    monkeypatch.setattr(vi, "load_linear", lambda ck, p, block=None: (calls.append(p), real(ck, p, block))[1])
    got = dict(vs.src_items(W, "m.l.mlp.up_proj"))
    assert list(got) == ["m.l.mlp.up_proj.weight"] and calls == ["m.l.mlp.up_proj"]
    assert dict(vs.src_items({"a.x": 1, "b.x": 2}, "a.")) == {"a.x": 1}      # plain dict (BF16 source) unchanged


def test_bf16_source_stays_a_plain_dict(tmp_path):
    write_st(tmp_path / "model.safetensors", {"w.weight": ("BF16", rng.integers(0, 0x7F00, size=(4, 8)).astype(np.uint16))})
    W = vs.load_source(str(tmp_path), ["model.safetensors"], lambda *a: None)
    assert type(W) is dict and W["w.weight"].dtype == mx.bfloat16


def test_e5m2_shard_falls_back_per_tensor(tmp_path):
    wb = e4m3((64, 128))
    write_st(tmp_path / "model.safetensors", {"l.weight": ("F8_E5M2", wb), "l.weight_scale": ("F32", np.array(0.5, np.float32)),
                                              "n.weight": ("F32", np.ones(4, np.float32))})
    W = vs.load_source(str(tmp_path), ["model.safetensors"], lambda *a: None)      # mx.load rejects F8_E5M2: per-tensor fallback
    ref = vi.E5M2_LUT[wb] * np.float32(0.5)
    assert np.array_equal(np.array(W["l.weight"]), ref.astype(np.float16), equal_nan=True) and "n.weight" in W


def test_gptq_style_source_is_refused(tmp_path):
    write_st(tmp_path / "model.safetensors", {"l.qweight": ("I32", np.zeros((8, 8), np.int32)), "l.weight": ("U8", np.zeros((8, 8), np.uint8))})
    with pytest.raises(SystemExit):
        vs.load_source(str(tmp_path), ["model.safetensors"], lambda *a: None)


def test_mxfp8_u8_scale_inv(tmp_path):
    wb, sb = e4m3((48, 128)), rng.integers(118, 134, size=(48, 4)).astype(np.uint8)
    write_st(tmp_path / "model.safetensors", {"l.weight": ("F8_E4M3", wb), "l.weight_scale_inv": ("U8", sb)})
    json.dump({"quantization_config": {"quant_method": "mxfp8", "weight_block_size": [1, 32]}}, open(tmp_path / "config.json", "w"))
    W = vs.load_source(str(tmp_path), ["model.safetensors"], lambda *a: None)
    ref = vi.dequant_fp8(wb, "F8_E4M3", vi.e8m0_to_f32(sb), (1, 32))
    assert np.array_equal(np.array(W["l.weight"]), ref.astype(np.float16)) and list(W) == ["l.weight"]
