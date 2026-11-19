"""MXFP8 linears as stored by modelopt (GLM-5.3-Flash FP4-Local attention / dense / indexer / lm_head): F8_E4M3 weight [n, k] + U8 (UE8M0) weight_scale
[n, k / 32]. They classify as fp8_block and pack to "fp8b" (one byte per weight, block scales per 32-weight unit), exact against the fp32 dequant."""
import json
import os
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kyojin_mlx import vendor_input as VI  # noqa: E402
from kyojin_mlx import foreign_linear as FL  # noqa: E402
from kyojin_mlx.csf_scale import CsfCheckpoint  # noqa: E402


def _e4m3(b):
    e = b.astype(np.int32)
    s = np.where(e & 0x80, -1.0, 1.0)
    ex, m = (e >> 3) & 0xF, e & 7
    v = np.where(ex == 0, m / 8 * 2.0 ** -6, (1 + m / 8) * 2.0 ** (ex - 7))
    return np.where((e & 0x7F) == 0x7F, 0.0, v * s).astype(np.float32)   # 0x7F / 0xFF = NaN in E4M3: not drawn below


def _write(path, tensors):
    hdr, blobs, off = {}, [], 0
    for k, (tag, arr) in tensors.items():
        b = np.ascontiguousarray(arr).tobytes()
        hdr[k] = {"dtype": tag, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


@pytest.mark.parametrize("n,k", [(64, 128), (48, 4096), (1536, 96)])
def test_mxfp8_linear_packs_as_stored_and_exact(tmp_path, n, k):
    rng = np.random.default_rng(n + k)
    w8 = rng.integers(0, 256, size=(n, k), dtype=np.uint8)
    w8[(w8 & 0x7F) == 0x7F] = 0x30                                    # no NaN codes
    sc = rng.integers(127 - 20, 127 + 3, size=(n, k // 32), dtype=np.uint8)   # 2^-20 .. 2^2 per 32 weights (wide row range)
    _write(os.path.join(tmp_path, "model.safetensors"),
           {"model.layers.0.self_attn.q_a_proj.weight": ("F8_E4M3", w8), "model.layers.0.self_attn.q_a_proj.weight_scale": ("U8", sc)})
    json.dump({"model_type": "glm5_next", "quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION"}},
              open(os.path.join(tmp_path, "config.json"), "w"))
    ck = CsfCheckpoint(str(tmp_path))
    c = VI.classify_linear(ck.info(), "model.layers.0.self_attn.q_a_proj")
    assert c["fmt"] == "fp8_block"
    ref = _e4m3(w8) * np.repeat(2.0 ** (sc.astype(np.float32) - 127), 32, axis=1)
    FL._FP8B_CACHE.clear()                                            # memoised per weight name: the cases reuse one name
    pk = FL._fp8b_pack(ck, c, VI._config_block(ck.config))
    assert pk is not None and pk.fmt == "fp8b" and pk.qs.dtype == np.uint8 and pk.qs.shape == (n, k)   # one byte per weight
    got = _e4m3(pk.qs) * np.repeat(pk.meta.astype(np.float32), 32, axis=1) * np.float32(pk.gscale)
    np.testing.assert_array_equal(got, ref)                           # power-of-two scales: exact


def test_concat_rows_exact_merges_fp8b_members_bit_for_bit():
    """sanitize's gate+up / q,k,v concatenation of fp8b linears: one Pack whose dequant equals the members' rows exactly (common power-of-two gscale)."""
    from kyojin_mlx import foreign_pack as FP
    rng = np.random.default_rng(7)
    packs, refs = [], []
    for n, p2 in ((64, 3), (32, 9), (96, 5)):
        qs = rng.integers(0, 256, size=(n, 128), dtype=np.uint8)
        qs[(qs & 0x7F) == 0x7F] = 0x30
        meta = (2.0 ** rng.integers(-4, 7, size=(n, 4))).astype(np.float16)
        packs.append(FP.Pack("fp8b", n, 128, qs, meta, np.float32(2.0 ** -p2)))
        refs.append(_e4m3(qs) * np.repeat(meta.astype(np.float32), 32, axis=1) * np.float32(2.0 ** -p2))
    pk = FL._concat_rows_exact(packs)
    got = _e4m3(pk.qs) * np.repeat(pk.meta.astype(np.float32), 32, axis=1) * np.float32(pk.gscale)
    assert pk.n == 192 and pk.fmt == "fp8b"
    np.testing.assert_array_equal(got, np.concatenate(refs, 0))
