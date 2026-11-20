"""GGUF Q4_1 / Q5_0 / Q5_1 read as stored (flag KYOJIN_LEGACY5_STORED): CPU proof. The Metal output is tests/test_legacy5_metal.py (gpu_archkit/q5_types_metal.sh)."""
from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np
import pytest

from kyojin_mlx import foreign_gguf as GF, foreign_pack as FP

mx = pytest.importorskip("mlx.core")
FM = pytest.importorskip("kyojin_mlx.foreign_metal")
TYPES5 = ("Q4_1", "Q5_0", "Q5_1")
GEMMA3 = os.environ.get("KYOJIN_TEST_GGUF_Q5_0", "")      # any GGUF holding real Q5_0 tensors (for example gemma-3-1b Q4_K_M); unset = skipped


def ref_block(tname: str, b: np.ndarray) -> np.ndarray:
    """One 32-weight block decoded in plain python from the ggml-quants.c loops (dequantize_row_q4_1 / q5_0 / q5_1), float32 arithmetic."""
    f16 = lambda o: np.frombuffer(b[o:o + 2].tobytes(), "<f2")[0].astype(np.float32)
    d = f16(0)
    m = f16(2) if tname in ("Q4_1", "Q5_1") else np.float32(0)
    o = {"Q4_1": 4, "Q5_0": 2, "Q5_1": 4}[tname]
    qh = int(np.frombuffer(b[o:o + 4].tobytes(), "<u4")[0]) if tname != "Q4_1" else 0
    qs = b[o + (0 if tname == "Q4_1" else 4):][:16]
    y = np.zeros(32, np.float32)
    for j in range(16):
        xh0 = ((qh >> j) << 4) & 0x10
        xh1 = (qh >> (j + 12)) & 0x10
        x0, x1 = (int(qs[j]) & 15) | xh0, (int(qs[j]) >> 4) | xh1
        if tname == "Q5_0":
            y[j], y[j + 16] = np.float32(x0 - 16) * d, np.float32(x1 - 16) * d
        else:
            y[j], y[j + 16] = np.float32(x0) * d + m, np.float32(x1) * d + m
    return y


def rand_rows(tname, n, k, seed=0):
    rng = np.random.default_rng(seed)
    _, be, bb = GF.TYPES_BY_NAME[tname]
    b = rng.integers(0, 256, (n, k // be, bb), dtype=np.uint8)
    for o in ((0, 2) if tname != "Q5_0" else (0,)):               # fp16 d (and m): finite, normal sizes
        b[..., o:o + 2] = rng.uniform(-0.5, 0.5, (n, k // be, 1)).astype("<f2").view(np.uint8)
    return np.ascontiguousarray(b.reshape(n, -1))


def test_block_sizes_match_ggml():
    assert [GF.TYPES_BY_NAME[t][1:] for t in TYPES5] == [(32, 20), (32, 22), (32, 24)]
    for t, bb in zip(TYPES5, (20, 22, 24)):
        assert GF.TYPES_BY_NAME[t][2] == bb and t in GF.DEQUANT and t in GF.KERNEL_TYPES5 and t not in GF.KERNEL_TYPES + GF.KERNEL_TYPESK


@pytest.mark.parametrize("tname", TYPES5)
@pytest.mark.parametrize("k", [96, 256, 640, 704])             # 96 = odd block count (Q5_0 meta padding); 640 / 704 = the real down-expert widths of the census
def test_engine_dequant_pack_and_reference_agree(tname, k):
    raw = rand_rows(tname, 5, k, seed=k)
    _, be, bb = GF.TYPES_BY_NAME[tname]
    ref = np.stack([np.concatenate([ref_block(tname, raw[r, i * bb:(i + 1) * bb]) for i in range(k // be)]) for r in range(5)])
    eng = GF.dequant(tname, raw, k)
    np.testing.assert_array_equal(eng, ref)                                           # the engine's exact dequant = the ggml loop
    p = GF.to_pack(tname, raw, k)
    assert p.fmt == tname.lower() and p.qs.shape == (5, k // 2)
    np.testing.assert_array_equal(FP.dequant(p), ref)                                 # the device layout holds the same bits
    assert p.meta.dtype == np.uint32 and (("hi" in p.extra) == (tname != "Q4_1"))


@pytest.mark.parametrize("tname", TYPES5)
def test_row_slabs_and_stacked_experts_pack_alike(tname, monkeypatch):
    raw = rand_rows(tname, 7, 128, seed=3)
    whole = GF.to_pack(tname, raw, 128)
    monkeypatch.setattr(GF, "PACK_SLAB_ELEMS", 128 * 3)                              # 3 rows per slab
    slab = GF.to_pack(tname, raw, 128)
    for a, b in ((whole.qs, slab.qs), (whole.meta, slab.meta)) + tuple((whole.extra[x], slab.extra[x]) for x in whole.extra):
        np.testing.assert_array_equal(a, b)


@pytest.mark.skipif(not os.path.exists(GEMMA3), reason="KYOJIN_TEST_GGUF_Q5_0 not set")
def test_real_q5_0_tensor():
    g = GF.GgufFile(GEMMA3)
    names = [n for n, t in g.tensors.items() if t.tname == "Q5_0" and len(t.ne) == 2]
    assert names
    n = max(names, key=lambda x: g.tensors[x].ne[0])
    ti = g.tensors[n]
    raw = np.asarray(g.raw(n))[:64]
    p = GF.to_pack("Q5_0", raw, ti.ne[0])
    np.testing.assert_array_equal(FP.dequant(p), GF.dequant("Q5_0", raw, ti.ne[0]))


def test_pack_refuses_bad_shapes():
    raw = rand_rows("Q5_1", 2, 64)
    p = GF.to_pack("Q5_1", raw, 64)
    with pytest.raises(ValueError):
        FP.Pack("q5_1", 2, 64, p.qs, p.meta[:, :1], extra=p.extra)
    with pytest.raises(ValueError):
        FP.Pack("q5_1", 2, 64, p.qs, p.meta)                                         # hi missing
    with pytest.raises(ValueError):
        FP.Pack("q4_1", 2, 48, p.qs[:, :24], p.meta[:, :1])                          # meta width != k / 32


@pytest.mark.parametrize("fmt", ("q4_1", "q5_0", "q5_1"))
def test_kernel_sources_generate(fmt):
    for src in (FM.gemvk_source(fmt), FM.gemvk_source(fmt, gather=True), FM.gemvk_source(fmt, gather=True, glu=True), FM.dequantk_source(fmt), FM.fmma_source(fmt)):
        assert src.count("{") == src.count("}") and src.count("(") == src.count(")")
    assert FM._KDEC[fmt] in FM.hdrk(fmt) and FM._KLD[fmt].split("(")[0] in FM.hdrk(fmt)
    assert FM.fmma_inputs(fmt) == (["xt", "qs", "mt"] + ([] if fmt == "q4_1" else ["hi"]))
    assert FM.gemvk_inputs(fmt) == (["x", "qs", "mt"] + ([] if fmt == "q4_1" else ["hi"]))
    assert FM._FMMA_PAIR[fmt] == "kdec" and fmt not in FM.FORMATSK_MMA
    assert len({FM.gemvk_source(f) for f in ("q4_1", "q5_0", "q5_1")}) == 3


def test_existing_headers_unchanged():
    for f in ("q5_k", "q6_k", "iq4_xs", "q3_k"):
        assert "dec_q41" not in FM.hdrk(f) and "ld_q50" not in FM.hdrk(f)
    assert FM.hdrk("q6_k") == FM.HDRK and FM.hdrk("iq4_xs") == FM.HDRK


@pytest.mark.skipif(shutil.which("xcrun") is None or subprocess.run(["xcrun", "-sdk", "macosx", "-f", "metal"], capture_output=True).returncode != 0, reason="no Metal compiler")
@pytest.mark.parametrize("fmt", ("q4_1", "q5_0", "q5_1"))
def test_metal_header_compiles_offline(fmt, tmp_path):
    """The Metal compiler runs on the CPU: the loader and decoder of the new type compile with the shared header, as the Q4_0 / Q5_K ones do (no GPU, no run)."""
    ld = {"q4_1": "ld_q41(q4, mt, 0u, 0u, 8u)", "q5_0": "ld_q50(q4, mt, hi, 0u, 0u, 8u)", "q5_1": "ld_q51(q4, mt, hi, 0u, 0u, 8u)"}[fmt]
    src = ("#include <metal_stdlib>\nusing namespace metal;\n" + FM.hdrk(fmt) + "kernel void t(device float* o [[buffer(0)]], device const uint4* q4 [[buffer(1)]], "
           f"device const uint* mt [[buffer(2)]], device const uint* hi [[buffer(3)]]) {{ KU c = {ld}; thread float w[32]; {FM._KDEC[fmt]}(c, 0u, w); o[0] = w[0] + w[31]; }}\n")
    f = tmp_path / "t.metal"
    f.write_text(src)
    r = subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-c", str(f), "-o", str(tmp_path / "t.air")], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-600:]


def test_flag_default_off_and_tool_table(monkeypatch):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
    import day0_inspect as D
    ti = GF.TensorInfo("blk.0.ffn_down_exps.weight", (640, 64, 4), 7, 0)
    monkeypatch.delenv("KYOJIN_LEGACY5_STORED", raising=False)
    assert not GF.legacy5_stored() and D.mode_of(ti.name, ti) == "resident16"
    monkeypatch.setenv("KYOJIN_LEGACY5_STORED", "1")
    assert GF.legacy5_stored() and D.mode_of(ti.name, ti) == "stored"
    assert D.mode_of("blk.0.attn_norm.weight", GF.TensorInfo("blk.0.attn_norm.weight", (640,), 0, 0)) == "resident16"
