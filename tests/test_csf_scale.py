"""csf / csf_32x2_v1 NVFP4 scale decoder (kyojin_mlx.csf_scale): bit-exact round trip against a test-only encoder written from the
format comment, the vLLM decoder as a cross-check when torch + the reference file are present, and a tiny modelopt checkpoint
read through CsfCheckpoint. numpy only, tiny tensors."""
import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kyojin_mlx import csf_scale as cs
from kyojin_mlx import vendor_input as vi

VLLM_REF = Path("/path/to/vllm_nvfp4_utils.py")


def make_scales(rows, cols, seed, outliers=0.03):
    """Rows with a narrow spread (codes 0..15 over a per-row base) plus scattered outliers (forced exceptions)."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 110, size=(rows, 1))
    s = base + rng.integers(0, 16, size=(rows, cols))
    hit = rng.random((rows, cols)) < outliers
    s[hit] = rng.integers(0, 127, size=int(hit.sum()))
    s[0, 0], s[-1, -1] = 0, 126
    return s.astype(np.uint8)


def row_split(s):
    base = np.minimum(s.min(axis=1), 240).astype(np.uint8)
    d = s.astype(np.int32) - base[:, None].astype(np.int32)
    ok = (d >= 0) & (d <= 15)
    return base, np.where(ok, d, 0).astype(np.uint8), ~ok


def pack_nibbles(codes):
    c = codes.reshape(-1)
    assert c.size % 2 == 0
    return (c[0::2] | (c[1::2] << 4)).astype(np.uint8)


def encode_csf(s):
    rows, cols = s.shape
    base, codes, bad = row_split(s)
    fixed = np.zeros((rows // 16, 16 * (1 + cols // 2)), np.uint8)
    for k in range(rows // 16):
        fixed[k, :16] = base[16 * k:16 * k + 16]
        fixed[k, 16:] = pack_nibbles(codes[16 * k:16 * k + 16])
    pos = np.flatnonzero(bad.reshape(-1))
    words = (pos.astype(np.uint32) | (s.reshape(-1)[pos].astype(np.uint32) << 24)).astype(np.uint32)
    return {"nvfp4_csf_fixed": fixed, "nvfp4_csf_exceptions": words}


def encode_csf32(s):
    rows, cols = s.shape
    base, codes, bad = row_split(s)
    ntr, ntc = -(-rows // 128), -(-cols // 4)
    cell_codes, words, page_counts = [], [], {}
    index = {}
    for tr in range(ntr):
        for tc in range(ntc):
            page = (tr * ntc + tc) // 32
            for pair in range(2):
                for blk in range(4):
                    r0, c0 = 128 * tr + 32 * blk, 4 * tc + 2 * pair
                    if r0 >= rows or c0 >= cols:
                        continue
                    w = min(2, cols - c0)
                    n = 0
                    for i in range(32):
                        for j in range(w):
                            cell_codes.append(codes[r0 + i, c0 + j])
                            if bad[r0 + i, c0 + j]:
                                words.append((2 * i + j) | (int(s[r0 + i, c0 + j]) << 8))
                                n += 1
                    page_counts[page] = page_counts.get(page, 0) + n
                    index.setdefault(page, [0]).append(page_counts[page])
    pages = sorted(index)
    per_page = [page_counts[p] for p in pages]
    origins = 2 * np.concatenate(([0], np.cumsum(per_page)[:-1])).astype(np.uint32)
    return {
        "csf32_bases": base,
        "csf32_codes": pack_nibbles(np.array(cell_codes, np.uint8)),
        "csf32_page_bases": origins.astype(np.uint32),
        "csf32_offsets": np.array([v for p in pages for v in index[p]], np.uint16),
        "csf32_exceptions": np.array(words, np.uint16),
    }


# (rows, cols): multiples of the tile and not; one page = 32 tiles of 128 x 4, so 128 x 160 spans two pages
SHAPES_32 = [(32, 2), (32, 6), (64, 7), (96, 18), (160, 13), (128, 160), (256, 130), (288, 133), (384, 515)]


@pytest.mark.parametrize("rows,cols", SHAPES_32)
def test_csf32_roundtrip_bit_exact(rows, cols):
    s = make_scales(rows, cols, rows * 1000 + cols)
    streams = encode_csf32(s)
    assert streams["csf32_exceptions"].size > 0
    out = cs.decode_csf_scale("csf_32x2_v1", streams)
    assert out.dtype == np.uint8 and out.shape == s.shape
    assert np.array_equal(out, s)


def test_csf32_no_exceptions_and_high_base():
    s = np.full((64, 8), 240, np.uint8) + np.arange(8, dtype=np.uint8) % 3
    s[:, 0] = 255
    s = np.minimum(s, 255).astype(np.uint8)
    streams = encode_csf32(s)
    assert np.array_equal(cs.decode_csf_scale("csf_32x2_v1", streams), s)


@pytest.mark.parametrize("rows,cols", [(16, 2), (32, 8), (48, 18), (64, 130)])
def test_csf_fixed_roundtrip_bit_exact(rows, cols):
    s = make_scales(rows, cols, rows + cols)
    out = cs.decode_csf_scale("csf", encode_csf(s))
    assert np.array_equal(out, s)


def test_corrupt_streams_rejected():
    s = make_scales(64, 8, 3)
    st = encode_csf32(s)
    bad = dict(st, csf32_offsets=st["csf32_offsets"][:-1])
    with pytest.raises(ValueError):
        cs.decode_csf_scale("csf_32x2_v1", bad)
    bad = dict(st, csf32_bases=np.full(64, 250, np.uint8))
    with pytest.raises(ValueError):
        cs.decode_csf_scale("csf_32x2_v1", bad)


def test_matches_vllm_decoder():
    torch = pytest.importorskip("torch")
    if not VLLM_REF.exists():
        pytest.skip("vLLM reference file not present")
    src = VLLM_REF.read_text()
    ns = {}
    try:
        # only the pure decoder part (from the first `_decode_csf` to `decode_csf_scale`), with its numpy / torch imports
        a, b = src.index("def _decode_csf("), src.index("def csf_weights_transform")
        exec("import functools\nimport numpy as np\nimport torch\nfrom collections.abc import Callable\n" + src[a:b], ns)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"vLLM reference not importable: {e}")
    for rows, cols in [(64, 7), (128, 160), (288, 133)]:
        s = make_scales(rows, cols, rows * 7 + cols)
        st = encode_csf32(s)
        ref = ns["decode_csf_scale"]("csf_32x2_v1", {k: torch.from_numpy(v.astype(np.int64)).to(
            {"csf32_page_bases": torch.uint32, "csf32_offsets": torch.uint16, "csf32_exceptions": torch.uint16}.get(k, torch.uint8))
            for k, v in st.items()})
        assert np.array_equal(ref.view(torch.uint8).numpy(), cs.decode_csf_scale("csf_32x2_v1", st))


def write_safetensors(path, tensors):
    tag = {np.dtype(np.uint8): "U8", np.dtype(np.uint16): "U16", np.dtype(np.uint32): "U32", np.dtype(np.float32): "F32"}
    hdr, off, blobs = {}, 0, []
    for k, (t, dt) in tensors.items():
        b = np.ascontiguousarray(t).tobytes()
        hdr[k] = {"dtype": dt or tag[t.dtype], "shape": list(t.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    Path(path).write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def test_checkpoint_view_feeds_nvfp4_path(tmp_path):
    rows, cols = 64, 10                                    # in_features = 160
    rng = np.random.default_rng(5)
    s = make_scales(rows, cols, 11)
    # keep the scale bytes finite E4M3 (0x7F / 0xFF are NaN)
    s = np.minimum(s, 120)
    w = rng.integers(0, 256, size=(rows, cols * 8), dtype=np.uint8)
    g = np.array(0.5, np.float32)
    st = encode_csf32(s)
    plain = {"m.weight": (w, "U8"), "m.weight_scale": (s, "F8_E4M3"), "m.weight_scale_2": (g, None)}
    comp = {"m.weight": (w, "U8"), "m.weight_scale_2": (g, None), "m.input_scale": (np.array(1.0, np.float32), None)}
    comp.update({f"m.weight_scale.{k}": (v, None) for k, v in st.items()})
    for name, t in (("plain", plain), ("csf", comp)):
        d = tmp_path / name
        d.mkdir()
        write_safetensors(d / "model.safetensors", t)
    ref = vi.load_linear(vi.Checkpoint(str(tmp_path / "plain")), "m")
    ck = cs.CsfCheckpoint(str(tmp_path / "csf"))
    assert ck.info()["m.weight_scale"] == ("F8_E4M3", (rows, cols))
    got = vi.load_linear(ck, "m")
    assert got.shape == (rows, cols * 16) and np.array_equal(got, ref)
