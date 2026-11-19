"""GGML type id 42 (Q2_0) as a stored weight type (flag KYOJIN_Q2_0_STORED, default off): CPU proof. The Metal output is tests/test_types2_metal.py (gpu_archkit/types2_metal.sh).
Layout (ggml-common.h block_q2_0, ggml-quants.c dequantize_row_q2_0): 64 weights = fp16 d + 16 bytes; code j = (qs[j / 4] >> 2 (j % 4)) & 3; w = (code - 1) * d."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from kyojin_mlx import foreign_gguf as GF, foreign_pack as FP, foreign_synth as SY

mx = pytest.importorskip("mlx.core")
FM = pytest.importorskip("kyojin_mlx.foreign_metal")
GGUF_PY = os.environ.get("KYOJIN_TEST_GGUF_PY", "")          # a llama.cpp gguf-py directory (second, independent decoder); unset = skipped


def ref_block(b: np.ndarray) -> np.ndarray:
    """One block decoded in plain python from the ggml-quants.c loop, float32 arithmetic."""
    d = np.frombuffer(b[0:2].tobytes(), "<f2")[0].astype(np.float32)
    y = np.zeros(64, np.float32)
    for j in range(64):
        q = (int(b[2 + j // 4]) >> ((j % 4) * 2)) & 3
        y[j] = np.float32(q - 1) * d
    return y


def rand_rows(n, k, seed=0):
    return SY.random_blocks("Q2_0", n, k, np.random.default_rng(seed))


def test_table_and_flag_default_off(monkeypatch):
    assert GF.TYPES_OPT == {42: ("Q2_0", 64, 18)} and GF.UPSTREAM_ONLY[42] == ("Q2_0", 64, 18) and 42 not in GF.TYPES
    assert "Q2_0" in GF.DEQUANT and GF.KERNEL_TYPES2 == ("Q2_0",) and "Q2_0" not in GF.KERNEL_TYPES + GF.KERNEL_TYPESK + GF.KERNEL_TYPES5
    monkeypatch.delenv("KYOJIN_Q2_0_STORED", raising=False)
    assert not GF.q2_0_stored()
    assert not GF.TensorInfo("w", (64, 2), 42, 0).known                         # off: an unknown type id, as before 
    monkeypatch.setenv("KYOJIN_Q2_0_STORED", "1")
    ti = GF.TensorInfo("w", (640, 3), 42, 0)
    assert ti.known and ti.tname == "Q2_0" and ti.nbytes == 3 * 10 * 18


@pytest.mark.parametrize("k", [64, 128, 640, 704])             # 704 = 11 blocks (odd: meta padding); 640 / 704 = the real expert widths
def test_engine_dequant_pack_and_reference_agree(k):
    n = 5
    raw = rand_rows(n, k, seed=k)
    ref = np.stack([np.concatenate([ref_block(raw[r, i * 18:(i + 1) * 18]) for i in range(k // 64)]) for r in range(n)])
    np.testing.assert_array_equal(GF.dequant("Q2_0", raw, k), ref)                    # the engine's exact dequant = the ggml loop
    p = GF.to_pack("Q2_0", raw, k)
    assert p.fmt == "q2_0" and p.qs.shape == (n, k // 4) and p.meta.shape == (n, (k // 64 + 1) // 2) and p.meta.dtype == np.uint32 and not p.extra
    np.testing.assert_array_equal(FP.dequant(p), ref)                                 # the device layout holds the same bits
    assert (FP.dequant(p) == 0).any() and (np.sign(ref) != 0).any()


def test_codes_cover_all_four_values_and_negative_zero():
    b = np.zeros((1, 18), np.uint8)
    b[0, 0:2] = np.array([-0.5], "<f2").view(np.uint8)
    b[0, 2:6] = [0b11100100, 0b00011011, 0, 0xFF]            # codes 0 1 2 3 | 3 2 1 0 | 1 1 1 1 | 3 3 3 3
    y = GF.dequant("Q2_0", b, 64)[0]
    assert y[:8].tolist() == [0.5, 0.0, -0.5, -1.0, -1.0, -0.5, 0.0, 0.5] and y[8:12].tolist() == [0.5] * 4 and y[12:16].tolist() == [-1.0] * 4
    assert np.signbit(y[1]) and np.signbit(y[6]) and not np.signbit(y[0])               # (1 - 1) * (-0.5) = -0.0, as the C loop


@pytest.mark.skipif(not GGUF_PY, reason="KYOJIN_TEST_GGUF_PY not set")
def test_gguf_py_decoder_agrees():
    sys.path.insert(0, GGUF_PY)
    from gguf import quants as Q, GGMLQuantizationType as T
    raw = rand_rows(4, 640, seed=9)
    ours = GF.dequant("Q2_0", raw, 640)
    theirs = Q.dequantize(raw.reshape(4, -1), T.Q2_0)
    np.testing.assert_array_equal(ours, theirs.reshape(4, 640))


def test_row_slabs_pack_alike(monkeypatch):
    raw = rand_rows(7, 128, seed=3)
    whole = GF.to_pack("Q2_0", raw, 128)
    monkeypatch.setattr(GF, "PACK_SLAB_ELEMS", 128 * 3)
    slab = GF.to_pack("Q2_0", raw, 128)
    np.testing.assert_array_equal(whole.qs, slab.qs)
    np.testing.assert_array_equal(whole.meta, slab.meta)


def test_pack_refuses_bad_shapes():
    p = GF.to_pack("Q2_0", rand_rows(2, 128), 128)
    with pytest.raises(ValueError):
        FP.Pack("q2_0", 2, 128, p.qs, p.meta[:, :0])
    with pytest.raises(ValueError):
        FP.Pack("q2_0", 2, 96, p.qs[:, :24], p.meta)                                  # k % 64 != 0


def test_reader_roundtrip_through_a_gguf_file(tmp_path, monkeypatch):
    raw = rand_rows(3, 128, seed=5)
    p = tmp_path / "q20.gguf"
    SY.write_gguf(p, {"general.architecture": "x"}, [("blk.0.ffn_down.weight", (128, 3), "Q2_0", raw)])
    monkeypatch.delenv("KYOJIN_Q2_0_STORED", raising=False)
    g = GF.GgufFile(p)
    assert not g.tensors["blk.0.ffn_down.weight"].known
    with pytest.raises(GF.GgufTypeError):
        g.raw("blk.0.ffn_down.weight")
    g.close()
    monkeypatch.setenv("KYOJIN_Q2_0_STORED", "1")
    g = GF.GgufFile(p)
    assert g.tensors["blk.0.ffn_down.weight"].known
    np.testing.assert_array_equal(np.asarray(g.raw("blk.0.ffn_down.weight")), raw)
    g.close()


def _emulate_unit(p: FP.Pack, n: int, u: int) -> np.ndarray:
    """ld_q20 + dec_q20 in numpy: the index expressions of the Metal text (uint2 per unit, ushort d at n * 2 ceil(nb / 2) + u / 2)."""
    UPR = p.k // 32
    q = np.ascontiguousarray(p.qs).view("<u4").reshape(p.n, UPR, 2)[n, u]
    mt16 = np.ascontiguousarray(p.meta).view("<u2").reshape(-1)
    d = mt16[n * (2 * ((UPR // 2 + 1) // 2)) + (u >> 1)].view(np.float16).astype(np.float32)
    w = np.zeros(32, np.float32)
    for e in range(16):
        w[e] = (np.float32((int(q[0]) >> (2 * e)) & 3) - np.float32(1)) * d
        w[16 + e] = (np.float32((int(q[1]) >> (2 * e)) & 3) - np.float32(1)) * d
    return w


@pytest.mark.parametrize("k", [64, 640, 704])
def test_metal_loader_index_math(k):
    raw = rand_rows(4, k, seed=k + 1)
    p = GF.to_pack("Q2_0", raw, k)
    ref = FP.dequant(p)
    for n in (0, 3):
        for u in range(k // 32):
            np.testing.assert_array_equal(_emulate_unit(p, n, u), ref[n, u * 32:(u + 1) * 32])


def test_kernel_sources_generate():
    fmt = "q2_0"
    for src in (FM.gemvk_source(fmt), FM.gemvk_source(fmt, gather=True), FM.gemvk_source(fmt, gather=True, glu=True), FM.dequantk_source(fmt), FM.fmma_source(fmt)):
        assert src.count("{") == src.count("}") and src.count("(") == src.count(")")
    assert FM._KDEC[fmt] in FM.hdrk(fmt) and FM._KLD[fmt].split("(")[0] in FM.hdrk(fmt)
    assert FM.gemvk_inputs(fmt) == ["x", "qs", "mt"] and FM.fmma_inputs(fmt) == ["xt", "qs", "mt"] and FM._FMMA_PAIR[fmt] == "kdec" and fmt not in FM.FORMATSK_MMA
    assert FM.gemvk_source(fmt) != FM.gemvk_source("q4_1")


def test_existing_headers_unchanged():
    for f in ("q5_k", "q6_k", "iq4_xs", "q3_k", "q4_1", "q5_0", "q5_1"):
        assert "dec_q20" not in FM.hdrk(f)
    assert FM.hdrk("q6_k") == FM.HDRK


def _metal_ok():
    return shutil.which("xcrun") is not None and subprocess.run(["xcrun", "-sdk", "macosx", "-f", "metal"], capture_output=True).returncode == 0


@pytest.mark.skipif(not _metal_ok(), reason="no Metal compiler")
def test_metal_header_compiles_offline(tmp_path):
    """The Metal compiler runs on the CPU: the loader and decoder compile with the shared header (no GPU, no run)."""
    src = ("#include <metal_stdlib>\nusing namespace metal;\n" + FM.hdrk("q2_0") + "kernel void t(device float* o [[buffer(0)]], device const uint* qs [[buffer(1)]], "
           "device const uint* mt [[buffer(2)]]) { KU c = ld_q20(qs, mt, 0u, 0u, 22u); thread float w[32]; dec_q20(c, 0u, w); o[0] = w[0] + w[31]; }\n")
    f = tmp_path / "t.metal"
    f.write_text(src)
    r = subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-c", str(f), "-o", str(tmp_path / "t.air")], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-600:]


def test_day0_inspect_verdict_off_and_on(tmp_path, monkeypatch):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
    import day0_inspect as D
    monkeypatch.delenv("KYOJIN_Q2_0_STORED", raising=False)
    ti = GF.TensorInfo("blk.0.ffn_down_exps.weight", (640, 64, 4), 42, 0)
    assert not ti.known and D.mode_of(ti.name, ti) == "unsupported"                    # off: TYPE NEEDED (the verdict path of tests/test_unknown_type_id.py)
    monkeypatch.setenv("KYOJIN_Q2_0_STORED", "1")
    ti = GF.TensorInfo("blk.0.ffn_down_exps.weight", (640, 64, 4), 42, 0)
    assert ti.known and D.mode_of(ti.name, ti) == "stored"
    assert D.mode_of("blk.0.attn_norm.weight", GF.TensorInfo("blk.0.attn_norm.weight", (640,), 0, 0)) == "resident16"


def test_day0_verdict_with_the_switch_off_and_on(tmp_path, monkeypatch):
    """A file holding a real Q2_0 tensor: switch off = TYPE NEEDED (unchanged), switch on = no type verdict, the tensor is read as stored."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
    import day0_inspect as D
    exps = "blk.0.ffn_down_exps.weight"
    kv = {"general.architecture": "qwen4exp", "qwen4exp.block_count": 1, "qwen4exp.expert_count": 4, "qwen4exp.expert_used_count": 2}
    p = tmp_path / "q20.gguf"
    SY.write_gguf(p, kv, [(exps, (128, 2), "Q2_0", rand_rows(2, 128)), ("token_embd.weight", (32, 2), "F32", np.zeros((2, 32), np.float32))])
    monkeypatch.delenv("KYOJIN_Q2_0_STORED", raising=False)
    r = D.inspect(str(p), use_census=False)
    assert r["verdict"] == "TYPE NEEDED" and "unknown weight type id 42" in r["why"]
    monkeypatch.setenv("KYOJIN_Q2_0_STORED", "1")
    r = D.inspect(str(p), use_census=False)
    print("verdict with the switch on:", r["verdict"], "|", r.get("why", "")[:200])
    assert r["verdict"] != "TYPE NEEDED" and not r.get("unknown_type_ids")
