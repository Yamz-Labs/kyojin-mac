"""End to end: a tiny random Qwen3.5-MoE converted from a vendor-quantized source (FP8 block / MXFP4 / NVFP4) must give the SAME pack,
tensor for tensor, as the same model converted from its own float16 dequant (the reader + lazy glue change nothing but the source format).
Needs the Metal GPU (run under the GPU lock)."""
import importlib, json, struct, sys
from pathlib import Path

import numpy as np
import pytest
import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "tools"), str(ROOT)]
from kyojin_mlx import vendor_input as vi

conv = importlib.import_module("conv_mlx")
mk = importlib.import_module("make_tiny_moe")
QUANT = ("linear_attn.in_proj_qkv.", "linear_attn.in_proj_z.", "linear_attn.out_proj.", "self_attn.q_proj.", "self_attn.k_proj.", "self_attn.v_proj.", "self_attn.o_proj.",
         "mlp.gate_proj.", "mlp.up_proj.", "mlp.down_proj.", "mlp.shared_expert.gate_proj.", "mlp.shared_expert.up_proj.", "mlp.shared_expert.down_proj.", "mlp.experts.")
rng = np.random.default_rng(11)


def write_st(path, tensors):
    hdr, off, blobs = {}, 0, []
    for k, (tag, arr) in tensors.items():
        raw = np.ascontiguousarray(arr).tobytes()
        hdr[k] = {"dtype": tag, "shape": list(arr.shape), "data_offsets": [off, off + len(raw)]}; blobs.append(raw); off += len(raw)
    h = json.dumps(hdr).encode(); h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h)
        for b in blobs: f.write(b)


def nearest(x, lut_pos):
    """index of the nearest magnitude in the ascending list lut_pos (ties to the lower index)."""
    i = np.clip(np.searchsorted(lut_pos, np.abs(x)), 1, len(lut_pos) - 1)
    lo, hi = lut_pos[i - 1], lut_pos[i]
    return np.where(np.abs(x) - lo <= hi - np.abs(x), i - 1, i)


E4_POS = np.sort(vi.E4M3FN_LUT[:127])                       # 0 .. 448, finite magnitudes, ascending bytes == ascending value
FP4_POS = vi.E2M1_LUT[:8]


def enc_e4m3(x):
    return (nearest(x, E4_POS).astype(np.uint8) | ((x < 0) << 7).astype(np.uint8))


def enc_e2m1(x):
    return (nearest(x, FP4_POS).astype(np.uint8) | ((x < 0) << 3).astype(np.uint8))


def pack(n):
    return (n[..., 0::2] | (n[..., 1::2] << 4)).astype(np.uint8)


def quantize(name, w, fmt, out):
    """adds the vendor tensors of linear `name` (".weight" stripped) for float32 w [o, i] to out."""
    o, i = w.shape
    if fmt == "fp8":
        g = w.reshape(o // 128, 128, i // 128, 128)
        s = 2.0 ** np.ceil(np.log2(np.maximum(np.abs(g).max(axis=(1, 3)), 1e-30) / 448.0))
        q = enc_e4m3((g / s[:, None, :, None]).reshape(o, i))
        out[name + ".weight"] = ("F8_E4M3", q); out[name + ".weight_scale_inv"] = ("F32", s.astype(np.float32))
    elif fmt == "mxfp4":
        g = w.reshape(o, i // 32, 32)
        e = np.ceil(np.log2(np.maximum(np.abs(g).max(-1), 1e-30) / 6.0)).astype(np.int32)
        q = enc_e2m1((g / (2.0 ** e)[..., None]).reshape(o, i))
        out[name + ".weight"] = ("U8", pack(q)); out[name + ".weight_scale"] = ("U8", (e + 127).astype(np.uint8))
    else:                                                    # nvfp4 ModelOpt
        g = w.reshape(o, i // 16, 16)
        gs = np.float32(2.0 ** np.ceil(np.log2(np.abs(w).max() / (6.0 * 448.0))))
        sc = enc_e4m3((np.abs(g).max(-1) / 6.0 / gs).astype(np.float32))
        sv = vi.e4m3_to_f32(sc)
        q = enc_e2m1((g / np.where(sv == 0, 1, sv * gs)[..., None]).reshape(o, i))
        out[name + ".weight"] = ("U8", pack(q)); out[name + ".weight_scale"] = ("F8_E4M3", sc)
        out[name + ".weight_scale_2"] = ("F32", np.array(gs, np.float32)); out[name + ".input_scale"] = ("F32", np.array(1.0, np.float32))


def is_quant(k):
    return k.endswith(".weight") and ".layers." in k and any(k.split(".layers.")[1].split(".", 1)[1].startswith(q) for q in QUANT)


@pytest.mark.parametrize("fmt", ["fp8", "mxfp4", "nvfp4"])
def test_vendor_source_equals_its_dequant_twin(tmp_path, monkeypatch, fmt):
    base, cal = tmp_path / "base", tmp_path / "cal.safetensors"
    monkeypatch.setattr(sys, "argv", ["mk", str(base), "--layout", "perexpert", "--layers", "2", "--cal", str(cal), "--rows", "8", "--cols", "128"])
    mk.main()
    ck = vi.Checkpoint(str(base))
    vendor, twin, nq, rel = {}, {}, 0, []
    for k, (tag, shape) in ck.info().items():
        if tag == "BF16" and len(shape) == 2 and is_quant(k) and shape[0] % 128 == 0 and shape[1] % 128 == 0:
            w = ck.f32(k)
            quantize(k[:-7], w, fmt, vendor)
            twin[k] = None                                   # filled below from the vendor tensors
            nq += 1
        else:
            vendor[k] = (tag, ck.raw(k)[2]); twin[k] = (tag, ck.raw(k)[2])
    assert nq >= 30
    for d in ("vendor", "twin"):
        (tmp_path / d).mkdir()
        json.dump(json.load(open(base / "config.json")) | ({"quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}} if fmt == "fp8" and d == "vendor" else {}),
                  open(tmp_path / d / "config.json", "w"))
    write_st(tmp_path / "vendor" / "model.safetensors", {k: (t, a) for k, (t, a) in vendor.items()})
    vck = vi.Checkpoint(str(tmp_path / "vendor"))
    for p in vi.linear_prefixes(vck.info()):
        c = vi.classify_linear(vck.info(), p)
        if c and not c["fmt"].startswith("dense_"):
            wf = vi.load_linear(vck, p)
            twin[p + ".weight"] = ("F16", wf.astype(np.float16))
            orig = ck.f32(p + ".weight")
            rel.append(np.linalg.norm(wf - orig) / np.linalg.norm(orig))
    assert len([1 for v in twin.values() if v and v[0] == "F16"]) == nq and max(rel) < {"fp8": 0.05, "mxfp4": 0.3, "nvfp4": 0.2}[fmt], max(rel)   # the vendor quantizer is sane
    twin = {k: v for k, v in twin.items() if v is not None}
    write_st(tmp_path / "twin" / "model.safetensors", twin)
    outs = {}
    for d in ("vendor", "twin"):
        out = tmp_path / ("out_" + d)
        monkeypatch.setattr(sys, "argv", ["conv", str(tmp_path / d), str(out), "--cal", str(cal), "--k-attn", "4", "--k-mlp", "4", "--k-expert", "3", "--head-bits", "4",
                                          "--expert-mode", "all", "--expert-batch", "4"])
        conv.main()
        outs[d] = mx.load(str(out / "model.safetensors"))
    a, b = outs["vendor"], outs["twin"]
    assert set(a) == set(b) and len(a) > 100
    bad = [k for k in a if not np.array_equal(np.array(a[k].astype(mx.float32)) if a[k].dtype == mx.bfloat16 else np.array(a[k]),
                                               np.array(b[k].astype(mx.float32)) if b[k].dtype == mx.bfloat16 else np.array(b[k]))]
    assert not bad, bad[:5]
    assert not any(k.endswith(("weight_scale_inv", "weight_scale", "weight_scale_2", "input_scale")) for k in a)     # no scale companion leaked into the pack
