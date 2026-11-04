"""Foreign 4-bit formats (NVFP4 / MXFP4 safetensors, GGUF): parse, tensor map, matrix-vector vs float64, Metal vs reference (GPU only). Small synthetic files, no downloads.

Ground truth for GGUF blocks is a literal python transcription of the reference C loops of ggml-quants.c (dequantize_row_*), written independently of the
vectorised decoders in foreign_gguf.py; for safetensors it is vendor_input.load_linear (the converter's dequantiser)."""
import os
import struct

import numpy as np
import pytest

from kyojin_mlx import foreign_gguf as GF, foreign_grids as G, foreign_metal as FM, foreign_pack as FP, foreign_synth as SY

RNG = lambda s=0: np.random.default_rng(s)
f16 = lambda b: float(np.frombuffer(bytes(b), "<f2")[0])
TYPES_Q = ("Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q4_K", "Q5_K", "Q6_K", "IQ3_XXS", "IQ3_S", "IQ4_NL", "IQ4_XS", "MXFP4", "NVFP4")
KV4 = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]
KVFP4 = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12]


def scale_min(j, q):
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)


def ue4m3(x):
    if x == 0 or x == 0x7F:
        return 0.0
    e, m = (x >> 3) & 15, x & 7
    raw = m * 2.0 ** -9 if e == 0 else (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return raw * 0.5


def ref_block(t, b):
    """literal transcription of dequantize_row_<t> for ONE block of bytes b (list of ints) -> list of floats"""
    y = []
    if t == "Q4_0":
        d = f16(b[0:2]); qs = b[2:18]
        y = [0.0] * 32
        for j in range(16):
            y[j] = ((qs[j] & 15) - 8) * d; y[j + 16] = ((qs[j] >> 4) - 8) * d
    elif t == "Q4_1":
        d, m = f16(b[0:2]), f16(b[2:4]); qs = b[4:20]; y = [0.0] * 32
        for j in range(16):
            y[j] = (qs[j] & 15) * d + m; y[j + 16] = (qs[j] >> 4) * d + m
    elif t in ("Q5_0", "Q5_1"):
        o = 2 if t == "Q5_0" else 4
        d = f16(b[0:2]); m = f16(b[2:4]) if t == "Q5_1" else 0.0
        qh = struct.unpack("<I", bytes(b[o:o + 4]))[0]; qs = b[o + 4:o + 20]; y = [0.0] * 32
        for j in range(16):
            xh0 = ((qh >> (j + 0)) << 4) & 0x10; xh1 = ((qh >> (j + 12))) & 0x10
            x0, x1 = (qs[j] & 15) | xh0, (qs[j] >> 4) | xh1
            if t == "Q5_0":
                x0, x1 = x0 - 16, x1 - 16
            y[j] = x0 * d + m; y[j + 16] = x1 * d + m
    elif t == "Q8_0":
        d = f16(b[0:2]); y = [(v - 256 if v > 127 else v) * d for v in b[2:34]]
    elif t == "Q4_K":
        d, mn = f16(b[0:2]), f16(b[2:4]); sc12 = b[4:16]; q = b[16:144]; qo = 0; is_ = 0
        for j in range(0, 256, 64):
            s, m = scale_min(is_, sc12); d1, m1 = d * s, mn * m
            s, m = scale_min(is_ + 1, sc12); d2, m2 = d * s, mn * m
            y += [d1 * (q[qo + l] & 15) - m1 for l in range(32)]
            y += [d2 * (q[qo + l] >> 4) - m2 for l in range(32)]
            qo += 32; is_ += 2
    elif t == "Q5_K":
        d, mn = f16(b[0:2]), f16(b[2:4]); sc12 = b[4:16]; qh = b[16:48]; ql = b[48:176]; qo = 0; is_ = 0; u1, u2 = 1, 2
        for j in range(0, 256, 64):
            s, m = scale_min(is_, sc12); d1, m1 = d * s, mn * m
            s, m = scale_min(is_ + 1, sc12); d2, m2 = d * s, mn * m
            y += [d1 * ((ql[qo + l] & 15) + (16 if qh[l] & u1 else 0)) - m1 for l in range(32)]
            y += [d2 * ((ql[qo + l] >> 4) + (16 if qh[l] & u2 else 0)) - m2 for l in range(32)]
            qo += 32; is_ += 2; u1 <<= 2; u2 <<= 2
    elif t == "Q6_K":
        ql, qh, sc = b[0:128], b[128:192], [v - 256 if v > 127 else v for v in b[192:208]]; d = f16(b[208:210]); y = [0.0] * 256
        qlo = qho = sco = yo = 0
        for n in range(0, 256, 128):
            for l in range(32):
                is_ = l // 16
                q1 = ((ql[qlo + l] & 15) | (((qh[qho + l] >> 0) & 3) << 4)) - 32
                q2 = ((ql[qlo + l + 32] & 15) | (((qh[qho + l] >> 2) & 3) << 4)) - 32
                q3 = ((ql[qlo + l] >> 4) | (((qh[qho + l] >> 4) & 3) << 4)) - 32
                q4 = ((ql[qlo + l + 32] >> 4) | (((qh[qho + l] >> 6) & 3) << 4)) - 32
                y[yo + l] = d * sc[sco + is_] * q1; y[yo + l + 32] = d * sc[sco + is_ + 2] * q2
                y[yo + l + 64] = d * sc[sco + is_ + 4] * q3; y[yo + l + 96] = d * sc[sco + is_ + 6] * q4
            yo += 128; qlo += 64; qho += 32; sco += 8
    elif t == "IQ4_NL":
        d = f16(b[0:2]); qs = b[2:18]; y = [0.0] * 32
        for j in range(16):
            y[j] = d * KV4[qs[j] & 15]; y[j + 16] = d * KV4[qs[j] >> 4]
    elif t == "IQ4_XS":
        d = f16(b[0:2]); sh = b[2] | (b[3] << 8); sl = b[4:8]; qs = b[8:136]; qo = 0
        for ib in range(8):
            ls = ((sl[ib // 2] >> (4 * (ib % 2))) & 15) | (((sh >> (2 * ib)) & 3) << 4); dl = d * (ls - 32)
            blk = [0.0] * 32
            for j in range(16):
                blk[j] = dl * KV4[qs[qo + j] & 15]; blk[j + 16] = dl * KV4[qs[qo + j] >> 4]
            y += blk; qo += 16
    elif t == "MXFP4":
        e = b[0]; d = (2.0 ** (e - 127)) * 0.5; qs = b[1:17]; y = [0.0] * 32
        for j in range(16):
            y[j] = KVFP4[qs[j] & 15] * d; y[j + 16] = KVFP4[qs[j] >> 4] * d
    elif t == "NVFP4":
        y = [0.0] * 64
        for s in range(4):
            d = ue4m3(b[s])
            for j in range(8):
                by = b[4 + s * 8 + j]
                y[s * 16 + j] = KVFP4[by & 15] * d; y[s * 16 + j + 8] = KVFP4[by >> 4] * d
    elif t == "IQ3_XXS":
        d = f16(b[0:2]); qs = b[2:66]; sas = b[66:98]; qo = 0
        grid = lambda i: [(G.IQ3XXS_GRID[i] >> (8 * j)) & 255 for j in range(4)]
        for ib32 in range(8):
            aux = struct.unpack("<I", bytes(sas[4 * ib32:4 * ib32 + 4]))[0]; db = d * (0.5 + (aux >> 28)) * 0.5
            for l in range(4):
                signs = G.KSIGNS_IQ2XS[(aux >> (7 * l)) & 127]
                g1, g2 = grid(qs[qo + 2 * l]), grid(qs[qo + 2 * l + 1])
                row = [0.0] * 8
                for j in range(4):
                    row[j] = db * g1[j] * (-1.0 if signs & (1 << j) else 1.0); row[j + 4] = db * g2[j] * (-1.0 if signs & (1 << (j + 4)) else 1.0)
                y += row
            qo += 8
    elif t == "IQ3_S":
        d = f16(b[0:2]); qs = b[2:66]; qh = b[66:74]; sg = b[74:106]; sc = b[106:110]; qo = so = qho = 0
        grid = lambda i: [(G.IQ3S_GRID[i] >> (8 * j)) & 255 for j in range(4)]
        for ib32 in range(0, 8, 2):
            db1 = d * (1 + 2 * (sc[ib32 // 2] & 15)); db2 = d * (1 + 2 * (sc[ib32 // 2] >> 4))
            for half, db in ((0, db1), (1, db2)):
                for l in range(4):
                    g1 = grid(qs[qo + 2 * l] | ((qh[qho + half] << (8 - 2 * l)) & 256)); g2 = grid(qs[qo + 2 * l + 1] | ((qh[qho + half] << (7 - 2 * l)) & 256))
                    row = [0.0] * 8
                    for j in range(4):
                        row[j] = db * g1[j] * (-1.0 if sg[so + l] & (1 << j) else 1.0); row[j + 4] = db * g2[j] * (-1.0 if sg[so + l] & (1 << (j + 4)) else 1.0)
                    y += row
                qo += 8; so += 4
            qho += 2
    else:
        raise AssertionError(t)
    return y


def ref_matrix(t, raw, k):
    _, be, bb = GF.TYPES_BY_NAME[t]
    out = []
    for row in raw:
        ys = []
        for i in range(k // be):
            ys += ref_block(t, [int(v) for v in row[i * bb:(i + 1) * bb]])
        out.append(ys)
    return np.array(out, np.float64)


def test_header_flags_and_types():
    assert GF.TYPES_BY_NAME["NVFP4"][2] == 36 and GF.TYPES_BY_NAME["MXFP4"][2] == 17 and GF.TYPES_BY_NAME["Q4_K"][2] == 144
    assert len(G.IQ3XXS_GRID) == 256 and len(G.IQ3S_GRID) == 512 and len(G.KSIGNS_IQ2XS) == 128
    assert G.IQ3XXS_GRID[0] == 0x04040404 and G.IQ3XXS_GRID[255] == 0x3e341c04


@pytest.mark.parametrize("t", TYPES_Q)
def test_dequant_matches_reference_loops(t):
    k = 512
    raw = SY.random_blocks(t, 3, k, RNG(1))
    got = GF.dequant(t, raw, k).astype(np.float64)
    want = ref_matrix(t, raw, k)
    np.testing.assert_allclose(got, want, rtol=2e-6, atol=1e-6 * (1 + np.abs(want).max()))


def test_gguf_parse_and_alignment(tmp_path):
    k, n = 256, 5
    raws = {"blk.0.attn_q.weight": ("Q4_K", SY.random_blocks("Q4_K", n, k, RNG(2))), "blk.0.attn_norm.weight": ("F32", RNG(3).standard_normal((1, n)).astype(np.float32).view(np.uint8))}
    ts = [("blk.0.attn_q.weight", (k, n), "Q4_K", raws["blk.0.attn_q.weight"][1]), ("blk.0.attn_norm.weight", (n,), "F32", raws["blk.0.attn_norm.weight"][1])]
    for al in (32, 64):
        p = tmp_path / f"m{al}.gguf"
        SY.write_gguf(p, {"general.architecture": "qwen3", "qwen3.block_count": 1, "tokenizer.ggml.tokens": ["a", "b"], "x.f": 0.5, "x.b": True}, ts, align=al)
        g = GF.GgufFile(p)
        assert g.version == 3 and g.kv["general.architecture"] == "qwen3" and g.kv["tokenizer.ggml.tokens"] == ["a", "b"] and g.kv["x.b"] is True
        ti = g.tensors["blk.0.attn_q.weight"]
        assert ti.ne == (k, n) and ti.shape == (n, k) and ti.tname == "Q4_K" and ti.nbytes == n * 144
        assert g.data_start % al == 0 and ti.offset % al == 0
        assert np.array_equal(g.raw("blk.0.attn_q.weight"), raws["blk.0.attn_q.weight"][1])
        np.testing.assert_allclose(g.tensor_f32("blk.0.attn_q.weight"), ref_matrix("Q4_K", raws["blk.0.attn_q.weight"][1], k), rtol=2e-6, atol=1e-5)
        assert g.tensor_f32("blk.0.attn_norm.weight").shape == (n,)
        g.close()


def test_gguf_rejects(tmp_path):
    p = tmp_path / "bad.gguf"
    p.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 0))
    with pytest.raises(GF.GgufError):
        GF.GgufFile(p)                                                      # truncated tensor info
    p.write_bytes(b"NOPE" + b"\0" * 40)
    with pytest.raises(GF.GgufError):
        GF.GgufFile(p)
    # TQ1_0 / TQ2_0 tensors open and decode (see test_gguf_types.py); they have no Metal kernel, so to_pack refuses them by name
    for tname in ("TQ1_0", "TQ2_0"):
        _, be, bb = GF.TYPES_BY_NAME[tname]
        SY.write_gguf(p, {"general.architecture": "qwen3"}, [("t.weight", (256, 2), tname, np.zeros(2 * bb, np.uint8))])
        g = GF.GgufFile(p)
        assert g.tensor_f32("t.weight").shape == (2, 256)
        with pytest.raises(GF.GgufTypeError, match=tname):
            GF.to_pack(tname, g.raw("t.weight"), 256)
    with open(p, "r+b") as f:                                               # unknown type id 9999
        data = bytearray(p.read_bytes())
    SY.write_gguf(p, {}, [("t", (32, 1), "Q4_0", np.zeros(18, np.uint8))])
    data = bytearray(p.read_bytes()); i = data.index(struct.pack("<I", 2) + struct.pack("<Q", 0))
    data[i:i + 4] = struct.pack("<I", 9999); p.write_bytes(bytes(data))
    with pytest.raises(GF.GgufTypeError, match="9999"):
        GF.GgufFile(p)


@pytest.mark.parametrize("t", GF.KERNEL_TYPES)
def test_to_pack_decodes_like_the_source(t):
    k = 512
    raw = SY.random_blocks(t, 8, k, RNG(5))
    p = GF.to_pack(t, raw, k)
    assert p.fmt == t.lower() and p.qs.dtype == np.uint8 and p.qs.shape == (8, k // 2)
    np.testing.assert_allclose(FP.dequant(p), GF.dequant(t, raw, k), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("t", GF.KERNEL_TYPES)
@pytest.mark.parametrize("R", (1, 5, 16))
def test_mma_kernel_logic_matches_float64(t, R):
    """the A/B fragment algebra of the MMA kernel (lane decode, k permutation, scale lanes) against the float64 matvec of the decoded source weights"""
    k, n = 512, 16
    raw = SY.random_blocks(t, n, k, RNG(6))
    p = GF.to_pack(t, raw, k)
    x = RNG(7).standard_normal((R, k)).astype(np.float16)
    want = x.astype(np.float64) @ GF.dequant(t, raw, k).astype(np.float64).T
    got = FP.emulate_mma(p, x)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4 * np.abs(want).max())


@pytest.mark.parametrize("fmt", FP.FORMATS)
def test_xperm_is_a_permutation_per_unit(fmt):
    idx = FP.xperm_index(fmt, 512)
    assert sorted(idx.tolist()) == list(range(512))
    u = idx.reshape(16, 32)
    for i in range(16):
        assert sorted((u[i] // 32).tolist()) in ([i] * 32, ) or fmt == "q4_k"       # a unit reads one 32-block (q4_k: the two 16-runs of a 64 group)


def _safetensors_dir(tmp_path, kind, n=32, k=512, modelopt=True):
    rng = RNG(11)
    d = tmp_path / f"{kind}{int(modelopt)}"
    d.mkdir()
    lin = SY.nvfp4_linear(rng, n, k, modelopt) if kind == "nvfp4" else SY.mxfp4_linear(rng, n, k)
    ts = {f"model.layers.0.self_attn.q_proj.{kk}": v for kk, v in lin.items()}
    ts["model.norm.weight"] = ("F32", (n,), rng.standard_normal(n).astype(np.float32).view(np.uint8))
    SY.write_safetensors(d / "model.safetensors", ts)
    (d / "config.json").write_text('{"model_type":"qwen3","quantization_config":{"quant_method":"modelopt","quant_algo":"NVFP4"}}')
    return d


@pytest.mark.parametrize("kind,modelopt", (("nvfp4", True), ("nvfp4", False), ("mxfp4", True)))
def test_safetensors_nvfp4_mxfp4(tmp_path, kind, modelopt):
    from kyojin_mlx import vendor_input as VI
    d = _safetensors_dir(tmp_path, kind, modelopt=modelopt)
    ck = VI.Checkpoint(str(d))
    rep = VI.detect_checkpoint(ck.info(), ck.config)
    assert rep["primary"] == kind
    pref = "model.layers.0.self_attn.q_proj"
    c = VI.classify_linear(ck.info(), pref)
    p = FP.pack_from_safetensors(ck, c)
    assert p.fmt == kind and p.qs.shape == (32, 256)
    want = VI.load_linear(ck, pref).astype(np.float64)
    np.testing.assert_allclose(FP.dequant(p).astype(np.float64), want, rtol=1e-6, atol=1e-30)
    x = RNG(3).standard_normal((5, 512)).astype(np.float16)
    ref = x.astype(np.float64) @ want.T
    got = FP.emulate_mma(p, x)
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4 * np.abs(ref).max())
    ck.close()


def test_mxfp4_group_other_than_32_is_refused(tmp_path):
    from kyojin_mlx import vendor_input as VI
    d = tmp_path / "mx16"; d.mkdir()
    rng = RNG(1)
    SY.write_safetensors(d / "model.safetensors", {"a.weight": ("U8", (8, 256), rng.integers(0, 256, (8, 256), dtype=np.uint8)),
                                                  "a.weight_scale": ("U8", (8, 32), rng.integers(118, 130, (8, 32), dtype=np.uint8))})
    ck = VI.Checkpoint(str(d))
    c = VI.classify_linear(ck.info(), "a")
    with pytest.raises(ValueError, match="per 32"):
        FP.pack_from_safetensors(ck, c)


def test_tensor_map_and_config():
    from kyojin_mlx import foreign_names as N
    assert N.gguf_name("blk.3.attn_q.weight", "qwen3") == "model.layers.3.self_attn.q_proj.weight"
    assert N.gguf_name("blk.3.ffn_down_exps.weight", "qwen3moe") == "model.layers.3.mlp.experts.down_proj.weight"
    assert N.gguf_name("blk.0.ffn_gate_inp.weight", "qwen3moe") == "model.layers.0.mlp.gate.weight"
    assert N.gguf_name("output.weight", "qwen3") == "lm_head.weight" and N.gguf_name("token_embd.weight", "qwen3") == "model.embed_tokens.weight"
    with pytest.raises(GF.GgufError, match="blk.0.ssm_a"):
        N.gguf_name("blk.0.ssm_a", "qwen3")
    with pytest.raises(GF.GgufError, match="mamba2"):
        N.gguf_name("blk.0.attn_q.weight", "mamba2")


def test_metal_sources_are_complete():
    for f in FM.FORMATS:
        for src in (FM.gemv_source(f, 2), FM.gemv_source(f, 1, True), FM.mma_source(f, 1), FM.mma_source(f, 2), FM.dequant_source(f)):
            assert src.count("{") == src.count("}") and src.count("(") == src.count(")") and "{mt}" not in src and "{dec}" not in src
    assert "gs[ex]" in FM.gemv_source("q4_k", 2, True)


# ------------------------------------------------------------------------------------------------------------ Metal (GPU only)
def _gpu():
    try:
        import mlx.core as mx
        return mx.metal.is_available()
    except Exception:
        return False


needs_gpu = pytest.mark.skipif(not _gpu(), reason="no Metal GPU")


@needs_gpu
@pytest.mark.parametrize("t", GF.KERNEL_TYPES)
@pytest.mark.parametrize("R", (1, 3, 4, 5, 9, 16, 24, 70))
def test_metal_linear_matches_reference(t, R):
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL
    k, n = 1024, 64
    raw = SY.random_blocks(t, n, k, RNG(8))
    p = GF.to_pack(t, raw, k)
    lin = FL.ForeignLinear(p)
    x = RNG(9).standard_normal((R, k)).astype(np.float16)
    want = x.astype(np.float64) @ GF.dequant(t, raw, k).astype(np.float64).T
    got = np.array(lin(mx.array(x))).astype(np.float64)
    assert np.abs(got - want).max() <= 4e-3 * np.abs(want).max()
    for fn in ("_gemv", "_mma"):                                              # both kernels, whatever the row count picked
        if fn == "_gemv" and (R > 16 or not lin.dev.gemv_ok()):
            continue
        y = np.array(getattr(lin.dev, fn)(mx.array(x))).astype(np.float64)
        assert np.abs(y - want).max() <= 4e-3 * np.abs(want).max(), fn
    w = np.array(lin.dev.dequant()).astype(np.float64)
    np.testing.assert_allclose(w, GF.dequant(t, raw, k) * (1.0), rtol=2e-3, atol=2e-3 * np.abs(w).max())


def _fmma_pack(fmt, n, k):
    rg = RNG(31)
    if fmt in ("q4_0", "q4_k", "q5_k", "q6_k", "iq4_xs"):
        t = fmt.upper()
        return GF.to_pack(t, SY.random_blocks(t, n, k, rg), k)
    qs = rg.integers(0, 256, (n, k // 2), dtype=np.uint8)
    if fmt == "nvfp4":
        return FP.Pack("nvfp4", n, k, qs, rg.integers(0x30, 0x3a, (n, k // 32, 2), dtype=np.uint8).view("<u2").reshape(n, k // 32), 1.5)
    return FP.Pack("mxfp4", n, k, qs, rg.integers(118, 124, (n, k // 32), dtype=np.uint8), 1.0)


@needs_gpu
@pytest.mark.parametrize("fmt", ("q4_0", "q4_k", "nvfp4", "mxfp4", "q5_k", "q6_k", "iq4_xs"))
@pytest.mark.parametrize("R", (2, 4, 5, 6, 8, 9, 12, 16))
def test_metal_fmma_matches_reference(fmt, R):
    """The few-row MMA (flag foreign_fmma) against the float64 dequantized weights; n = 72 is not a multiple of the 16 rows of a simdgroup (tail rows)."""
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL
    n, k = 72, 1024
    p = _fmma_pack(fmt, n, k)
    dp = FL.DevPack(p)
    x = RNG(9).standard_normal((R, k)).astype(np.float16)
    want = x.astype(np.float64) @ FP.dequant(p).astype(np.float64).T
    assert dp.fmma_ok(R) or R < 5                                             # below the per-format floor the GEMV serves, the kernel still runs
    for ws in (1, 4):
        y = np.array(dp._fmma(mx.array(x), ws=ws)).astype(np.float64)
        assert np.abs(y - want).max() <= 4e-3 * np.abs(want).max(), (fmt, R, ws)
    assert np.abs(np.array(dp.apply(mx.array(x))).astype(np.float64) - want).max() <= 4e-3 * np.abs(want).max()


@needs_gpu
def test_metal_expert_gather_matches_reference():
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL
    k, n, E = 512, 64, 4
    packs = [GF.to_pack("Q4_K", SY.random_blocks("Q4_K", n, k, RNG(20 + e)), k) for e in range(E)]
    dp = FL.DevPack(packs)
    x = RNG(2).standard_normal((3, k)).astype(np.float16)
    xrows, eids = np.array([0, 1, 2, 2, 0], np.uint32), np.array([3, 0, 1, 2, 2], np.uint32)
    y = np.array(dp.gather(mx.array(x), mx.array(xrows), mx.array(eids))).astype(np.float64)
    for s in range(5):
        want = x[xrows[s]].astype(np.float64) @ FP.dequant(packs[eids[s]]).astype(np.float64).T
        assert np.abs(y[s] - want).max() <= 4e-3 * np.abs(want).max()


def _tiny_kv():
    toks = list("abc ") + ["ab", "<|im_end|>", "<think>"]
    return {"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "qwen2", "tokenizer.ggml.tokens": toks, "tokenizer.ggml.merges": ["a b"],
            "tokenizer.ggml.token_type": [1, 1, 1, 1, 1, 3, 4], "tokenizer.ggml.eos_token_id": 5, "tokenizer.chat_template": "{{ messages }}"}


def test_gguf_tokenizer_from_metadata(tmp_path):
    from kyojin_mlx.foreign_names import write_gguf_tokenizer
    from transformers import AutoTokenizer
    write_gguf_tokenizer(_tiny_kv(), str(tmp_path))
    tok = AutoTokenizer.from_pretrained(str(tmp_path))
    assert tok.encode("abc<|im_end|><think>", add_special_tokens=False) == [4, 2, 5, 6]
    assert tok.eos_token_id == 5 and tok.chat_template == "{{ messages }}"
    assert tok.decode([4, 2]) == "abc"


def test_gguf_tokenizer_refuses_unmapped(tmp_path):
    from kyojin_mlx.foreign_names import write_gguf_tokenizer
    with pytest.raises(GF.GgufError):
        write_gguf_tokenizer(dict(_tiny_kv(), **{"tokenizer.ggml.pre": "no-such-pre"}), str(tmp_path))


def test_gguf_serve_dir_leaves_other_paths_alone(tmp_path):
    from kyojin_mlx.foreign_names import gguf_serve_dir
    assert gguf_serve_dir(str(tmp_path)) == str(tmp_path)
    assert gguf_serve_dir("/no/such/model") == "/no/such/model"


def test_find_mtp_dir_sees_a_draft_head_next_to_the_weights(tmp_path, monkeypatch):
    from kyojin_mlx.serve.q35mlx import find_mtp_dir
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert find_mtp_dir(str(tmp_path)) is None
    (tmp_path / "model_mtp.safetensors").write_bytes(b"x")
    d = find_mtp_dir(str(tmp_path))
    assert find_mtp_dir(str(tmp_path)) == d and os.path.islink(os.path.join(d, "model_mtp.safetensors"))


def test_spec_lazy_rows_serve_lookup_drafts():
    import mlx.core as mx
    from kyojin_mlx.serve.backends import LookupDrafts
    from kyojin_mlx.serve.q35mlx import _Lazy
    base = _Lazy({1: mx.array([7, 8], dtype=mx.int32)}, 2)
    ld = LookupDrafts({0: [3, 4]}, [2, 2], base=base)
    assert ld.toks.tolist() == [[3, 4], [7, 8]] and ld.tolist() == [[3, 4], [7, 8]]


# ---------------------------------------------------------------------------------------------------------------- load_foreign on CPU (dense-decoded F32 GGUF, no Metal)
def _tiny_qwen3_gguf(path, drop=(), extra_kv=None, tokens=None, embed_rows=32):
    H, V, I, NH, NK, HD = 64, embed_rows, 128, 2, 1, 32
    r = np.random.default_rng(0)
    ts = {"token_embd.weight": (H, V), "output_norm.weight": (H,), "blk.0.attn_norm.weight": (H,), "blk.0.ffn_norm.weight": (H,),
          "blk.0.attn_q.weight": (H, NH * HD), "blk.0.attn_k.weight": (H, NK * HD), "blk.0.attn_v.weight": (H, NK * HD), "blk.0.attn_output.weight": (NH * HD, H),
          "blk.0.attn_q_norm.weight": (HD,), "blk.0.attn_k_norm.weight": (HD,), "blk.0.ffn_gate.weight": (H, I), "blk.0.ffn_up.weight": (H, I), "blk.0.ffn_down.weight": (I, H)}
    tl = [(n, ne, "F32", (r.standard_normal(tuple(reversed(ne))) * 0.1).astype(np.float32)) for n, ne in ts.items() if n not in drop]
    kv = {"general.architecture": "qwen3", "qwen3.block_count": 1, "qwen3.embedding_length": H, "qwen3.feed_forward_length": I, "qwen3.attention.head_count": NH,
          "qwen3.attention.head_count_kv": NK, "qwen3.attention.key_length": HD, "qwen3.context_length": 128,
          "tokenizer.ggml.tokens": tokens if tokens is not None else [f"t{i}" for i in range(V)]}
    kv.update(extra_kv or {})
    SY.write_gguf(path, kv, tl)


def test_load_foreign_refuses_a_file_missing_tensors(tmp_path):
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from kyojin_mlx import foreign_linear as FL
    ok = tmp_path / "ok.gguf"
    _tiny_qwen3_gguf(ok)
    model, _ = FL.load_foreign(str(ok), log=lambda *a: None)
    assert model(__import__("mlx.core", fromlist=["array"]).array([[1, 2, 3]])).shape == (1, 3, 32)
    for gone in ("blk.0.attn_q_norm.weight", "blk.0.attn_q.weight", "output_norm.weight"):
        bad = tmp_path / "bad.gguf"
        _tiny_qwen3_gguf(bad, drop=(gone,))
        with pytest.raises(ValueError, match="no tensor for"):
            FL.load_foreign(str(bad), log=lambda *a: None)


def test_nvfp4_global_scale_must_be_one_scalar(tmp_path):
    from kyojin_mlx import vendor_input as VI
    rng = RNG(4)
    for name, g in (("many", np.array([0.01, 0.5], np.float32)), ("zero", np.array([0.0], np.float32))):
        d = tmp_path / name
        d.mkdir()
        lin = SY.nvfp4_linear(rng, 8, 64, True)
        lin["weight_scale_2"] = ("F32", g.shape, g.view(np.uint8))
        SY.write_safetensors(d / "model.safetensors", {f"a.{k}": v for k, v in lin.items()})
        ck = VI.Checkpoint(str(d))
        with pytest.raises(ValueError, match="global scale"):
            FP.pack_from_safetensors(ck, VI.classify_linear(ck.info(), "a"))
        ck.close()


def test_gguf_config_refuses_what_it_cannot_express(tmp_path):
    from kyojin_mlx import foreign_names as N
    base = tmp_path / "b.gguf"
    _tiny_qwen3_gguf(base, tokens=["a", "b"])                              # fewer metadata tokens than embedding rows
    assert N.gguf_config(GF.GgufFile(base))["vocab_size"] == 32
    for i, extra in enumerate(({"qwen3.rope.scaling.type": "yarn", "qwen3.rope.scaling.factor": 4.0}, {"qwen3.rope.dimension_count": 16})):
        p = tmp_path / f"r{i}.gguf"
        _tiny_qwen3_gguf(p, extra_kv=extra)
        with pytest.raises(GF.GgufError, match="rope"):
            N.gguf_config(GF.GgufFile(p))
    ok = tmp_path / "ok.gguf"
    _tiny_qwen3_gguf(ok, extra_kv={"qwen3.rope.scaling.type": "none", "qwen3.rope.dimension_count": 32})
    assert N.gguf_config(GF.GgufFile(ok))["head_dim"] == 32


def test_gguf_serve_dir_follows_the_files_it_copies(tmp_path):
    from kyojin_mlx.foreign_names import gguf_serve_dir
    m = tmp_path / "m"
    m.mkdir()
    _tiny_qwen3_gguf(m / "x.gguf")
    (m / "config.json").write_text('{"model_type":"qwen3","a":1}')
    (m / "tokenizer.json").write_text("{}")
    cache = str(tmp_path / "cache")
    d1 = gguf_serve_dir(str(m / "x.gguf"), cache)
    assert '"a":1' in open(os.path.join(d1, "config.json")).read()
    (m / "config.json").write_text('{"model_type":"qwen3","a":22}')
    d2 = gguf_serve_dir(str(m / "x.gguf"), cache)
    assert d2 != d1 and '"a":22' in open(os.path.join(d2, "config.json")).read()
    assert gguf_serve_dir(str(m / "x.gguf"), cache) == d2
