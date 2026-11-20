"""G4: dense MXFP8 -> NVFP4 (kyojin_mlx.glm_dense4): the stored bytes are what the loader's own dequant reads, the scale search never loses to the amax scale, a fused group
shares one global scale and concatenates as stored, the safetensors writer's header is the checkpoint reader's, and the MTP head helper keeps only complete fused modules."""
import json
import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from kyojin_mlx import foreign_linear as FL, foreign_pack as FP, glm_dense4 as G4, vendor_input as VI    # noqa: E402


def _w(n=64, k=256, seed=0, scale=0.05):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n, k)).astype(np.float32) * scale
    w[3, 5] = 4 * scale * 8                       # an outlier block
    return w


def _pack(q, n, k):
    qs, s = np.array(q["qs"]), np.array(q["scale"])
    return FP.Pack("nvfp4", n, k, qs, np.ascontiguousarray(s).view("<u2").reshape(n, k // 32), float(q["scale2"]))


def test_nvfp4_layout_is_the_loaders():
    w = _w()
    q = G4.quantize_matrix(mx.array(w), "nvfp4")
    ref = np.array(G4.dequant_nvfp4(np.array(q["qs"]), np.array(q["scale"]), float(q["scale2"])))
    pk = _pack(q, *w.shape)
    assert np.abs(FP.dequant(pk) - ref).max() < 1e-6
    mlx = np.array(mx.dequantize(mx.array(np.array(q["qs"])).view(mx.uint32), mx.array(np.array(q["scale"])), None, group_size=16, bits=4, mode="nvfp4", dtype=mx.float32)) * float(q["scale2"])
    assert np.abs(mlx - ref).max() < 1e-6
    rel = np.linalg.norm(ref - w) / np.linalg.norm(w)
    assert abs(rel - q["rel_rmse"]) < 1e-4 and rel < 0.12


def test_scale_search_never_loses():
    w = mx.array(_w(seed=1))
    g = G4.global_scale(float(mx.max(mx.abs(w))))
    e_full = float(mx.sum((w - G4.quant_nvfp4_rows(w, g)[2]) ** 2))
    e_amax = float(mx.sum((w - G4.quant_nvfp4_rows(w, g, search=(0,))[2]) ** 2))
    assert e_full <= e_amax


def test_mxfp4_roundtrip():
    w = _w(seed=2)
    q = G4.quantize_matrix(mx.array(w), "mxfp4")
    d = np.array(mx.dequantize(mx.array(np.array(q["qs"])).view(mx.uint32), mx.array(np.array(q["scale"])), None, group_size=32, bits=4, mode="mxfp4", dtype=mx.float32))
    assert np.linalg.norm(d - w) / np.linalg.norm(w) < 0.16
    assert abs(np.linalg.norm(d - w) / np.linalg.norm(w) - q["rel_rmse"]) < 1e-4


def test_fused_group_concat_is_exact():
    a, b = _w(seed=3, scale=0.02), _w(seed=4, scale=0.07)
    g = max(G4.global_scale(float(np.abs(a).max())), G4.global_scale(float(np.abs(b).max())))
    qa, qb = (G4.quantize_matrix(mx.array(x), "nvfp4", g=g) for x in (a, b))
    pa, pb = _pack(qa, *a.shape), _pack(qb, *b.shape)
    m = FL._concat_rows_exact([pa, pb])
    assert m.fmt == "nvfp4" and m.n == a.shape[0] + b.shape[0]
    assert np.array_equal(FP.dequant(m), np.concatenate([FP.dequant(pa), FP.dequant(pb)]))
    qc = G4.quantize_matrix(mx.array(a), "nvfp4")                                  # a global scale of its own: refused, with the way out
    with pytest.raises(ValueError, match="glm_dense4_convert"):
        FL._concat_rows_exact([pb, _pack(qc, *a.shape)])


def test_writer_header_and_classification(tmp_path):
    w = _w(n=32, k=256)
    q = G4.quantize_matrix(mx.array(w), "nvfp4")
    p = "model.language_model.layers.0.self_attn.q_proj"
    plan = [(p + ".weight", "U8", (32, 128)), (p + ".weight_scale", "F8_E4M3", (32, 16)), (p + ".weight_scale_2", "F32", ())]
    wr = G4.StWriter(str(tmp_path / "x.safetensors"), plan)
    wr.put(p + ".weight", np.array(q["qs"]))
    wr.put(p + ".weight_scale", np.array(q["scale"]))
    wr.put(p + ".weight_scale_2", np.array(q["scale2"], np.float32).reshape(()))
    wr.close()
    json.dump({}, open(tmp_path / "config.json", "w"))
    ck = VI.Checkpoint(str(tmp_path))
    c = VI.classify_linear(ck.info(), p)
    assert c["fmt"] == "nvfp4" and c["scale2"] == p + ".weight_scale_2"
    pk = FP.pack_from_safetensors(ck, c)
    assert np.array_equal(pk.qs, np.array(q["qs"])) and abs(pk.gscale - float(q["scale2"])) < 1e-12
    assert np.abs(FP.dequant(pk) - np.array(G4.dequant_nvfp4(np.array(q["qs"]), np.array(q["scale"]), float(q["scale2"])))).max() < 1e-6


def test_mtp_finish_keeps_complete_groups_only(monkeypatch):
    class S:
        pass
    s = S()
    q = G4.quantize_matrix(mx.array(_w(n=32, k=256)), "nvfp4")
    pk = _pack(q, 32, 256)
    L = "model.language_model.layers.45."
    s.mtp_packs = {L + "self_attn.q_a_proj": pk, L + "self_attn.o_proj": pk, L + "mlp.shared_experts.gate_proj": pk, L + "mlp.shared_experts.up_proj": pk}
    s.mtp = {}
    FL._g4_mtp_finish(s)
    assert set(s.mtp_packs) == {L + "self_attn.o_proj", L + "mlp.shared_experts.gate_proj", L + "mlp.shared_experts.up_proj"}      # q_a without kv_a: decoded to fp16
    assert (L + "self_attn.q_a_proj.weight") in s.mtp and s.mtp[L + "self_attn.q_a_proj.weight"].dtype == mx.float16


# ---------------------------------------------------------------------------------------------------------------------------------- G4b
class _FakeCk:
    """Just enough of vendor_input.Checkpoint for load_f32 / load_f32_rows: raw(name) -> (tag, shape, array), scale_f32(name)."""

    def __init__(self, d):
        self.d = d

    def raw(self, name):
        return self.d[name]

    def scale_f32(self, name):
        return self.d[name][2]


def _fake_fp8(n=256, k=512, scale="grid", seed=0):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 120, (n, k)).astype(np.uint8)
    if scale == "e8m0":
        sc = rng.integers(120, 130, (n, k // 32)).astype(np.uint8)
        ent = ("U8", sc.shape, sc)
    elif scale == "grid":
        sc = (rng.random((n // 128, 1, k // 128, 1)) + 0.5).astype(np.float32)
        ent = ("F32", sc.shape, sc)
    else:
        sc = np.float32(0.37)
        ent = ("F32", (), sc)
    return _FakeCk({"p.weight": ("F8_E4M3", (n, k), codes), "p.weight_scale": ent})


@pytest.mark.parametrize("scale", ["e8m0", "grid", "scalar"])
def test_load_rows_equals_load(scale):
    ck = _fake_fp8(scale=scale)
    full = G4.load_f32(ck, "p")
    for r0, r1 in ((0, 128), (128, 256), (37, 200)):
        assert np.array_equal(G4.load_f32_rows(ck, "p", r0, r1), full[r0:r1])


def test_stream_quantizer_is_the_matrix_quantizer():
    ck = _fake_fp8(scale="grid", seed=5)
    w = mx.array(G4.load_f32(ck, "p"))
    for search in (G4.NV_SEARCH, G4.NV_SEARCH_WIDE):
        g = 16 * G4.global_scale(float(mx.max(mx.abs(w))))
        ref = G4.quantize_matrix(w, "nvfp4", g=g)
        got = G4.quantize_stream(ck, "p", g=g, search=search, chunk_elems=40 * 512)       # chunks of 40 rows (not a multiple of 128 on purpose)
        if search == G4.NV_SEARCH:
            assert np.array_equal(np.array(ref["qs"]), got["qs"]) and np.array_equal(np.array(ref["scale"]), got["scale"]) and abs(ref["rel_rmse"] - got["rel_rmse"]) < 1e-5
        chunks = []
        class _S:
            def __call__(self, a):
                chunks.append(a)
        got2 = G4.quantize_stream(ck, "p", g=g, search=search, chunk_elems=40 * 512, sink=_S())
        assert got2["qs"] is None and np.array_equal(np.concatenate(chunks, 0), got["qs"])
    assert abs(G4.amax_stream(ck, "p", 40 * 512) - float(mx.max(mx.abs(w)))) < 1e-6


def test_wide_search_never_loses():
    w = mx.array(_w(n=128, k=512, seed=7))
    g = 16 * G4.global_scale(float(mx.max(mx.abs(w))))
    e = lambda s: float(mx.sum((w - G4.quant_nvfp4_rows(w, g, s)[2]) ** 2))
    assert e(G4.NV_SEARCH_WIDE) <= e(G4.NV_SEARCH) <= e((0,))


def test_row_sink_checks_bytes(tmp_path):
    plan = [("a", "U8", (4, 8)), ("b", "U8", (2, 2))]
    w = G4.StWriter(str(tmp_path / "s.safetensors"), plan)
    s = w.sink("a")
    s(np.zeros((3, 8), np.uint8))
    with pytest.raises(AssertionError):
        s.done()
    s(np.zeros((1, 8), np.uint8))
    s.done()
    w.put("b", np.zeros((2, 2), np.uint8))
    w.close()


def test_compose_selects_groups_and_layers(tmp_path):
    import importlib.util
    sp = importlib.util.spec_from_file_location("g4b_compose", os.path.join(os.path.dirname(__file__), "..", "tools", "g4b_compose.py"))
    C = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(C)
    groups = {"kda_qkv": 1, "kda_o": 1, "shared_gu": 1, "shared_down": 1, "lm_head": 1, "mtp_mla_o": 1}
    assert C.expand("shared@3-5", groups) == (["shared_gu", "shared_down"], 3, 5)
    assert C.expand("trunk", groups)[0] == ["kda_qkv", "kda_o", "shared_gu", "shared_down", "lm_head"] and C.expand("mtp", groups)[0] == ["mtp_mla_o"]
    with pytest.raises(SystemExit):
        C.expand("nope", groups)
    src, pool, dst = tmp_path / "src", tmp_path / "pool", tmp_path / "dst"
    src.mkdir(); pool.mkdir()
    json.dump({"weight_map": {"x.weight": "a.safetensors"}}, open(src / "model.safetensors.index.json", "w"))
    json.dump({"quantized_layers": {}}, open(src / "hf_quant_config.json", "w"))
    (src / "config.json").write_text("{}")
    T = {}
    for li in (2, 3, 4):
        p = f"model.language_model.layers.{li}.self_attn.o_proj"
        T[p] = dict(n=4096, k=8192, src="F8_E4M3", rel_rmse=0.08, layer=li, group="kda_o", shard=f"model-g4d4-kda_o-L{li:02d}.safetensors")
        (pool / T[p]["shard"]).write_text("")
    wm = {p + s: m["shard"] for p, m in T.items() for s in (".weight", ".weight_scale", ".weight_scale_2")}
    json.dump(dict(fmt="nvfp4", search=8, src=str(src), tensors=T, weight_map=wm), open(pool / "g4_pool.json", "w"))
    import sys
    argv = sys.argv
    sys.argv = ["x", "--pool", str(pool), "--dst", str(dst), "--select", "kda_o@3-4"]
    try:
        C.main()
    finally:
        sys.argv = argv
    idx = json.load(open(dst / "model.safetensors.index.json"))["weight_map"]
    assert idx["model.language_model.layers.3.self_attn.o_proj.weight"] == "model-g4d4-kda_o-L03.safetensors" and not any("layers.2." in k for k in idx)
    assert (dst / "model-g4d4-kda_o-L04.safetensors").is_symlink() and not (dst / "model-g4d4-kda_o-L02.safetensors").exists() and (dst / "config.json").is_symlink()
    man = json.load(open(dst / "g4_manifest.json"))
    assert man["n_tensors"] == 2 and abs(man["saved_gb_per_token"] - 2 * 4096 * 8192 * 0.5 / 1e9) < 1e-6


def test_e4m3_nearest_byte_matches_the_table():
    t = mx.array(np.concatenate([np.exp(np.random.default_rng(9).uniform(np.log(1e-4), np.log(500), 20000)), G4.E4M3_POS[1:], G4.E4M3_POS[1:] * 1.0001]).astype(np.float32))
    b = np.array(G4.e4m3_nearest_byte(t))
    ref = np.argmin(np.abs(np.array(t)[:, None] - G4.E4M3_POS[None, :]), axis=1)
    near = np.abs(G4.E4M3_POS[np.clip(b, 0, 126)] - np.array(t))
    assert (np.abs(b - ref) <= 1).all() and (near <= np.abs(G4.E4M3_POS[ref] - np.array(t)) * 1.0000001 + 1e-12).mean() > 0.999


# ---------------------------------------------------------------------------------------------------------------------------------- G4b activation-aware
def test_weighted_scale_search_lowers_the_weighted_error():
    w = mx.array(_w(n=128, k=512, seed=11, scale=0.05))
    rng = np.random.default_rng(3)
    h = np.exp(rng.normal(0, 1.5, 512)).astype(np.float32)
    h /= h.mean()
    g = 16 * G4.global_scale(float(mx.max(mx.abs(w))))
    ew = lambda d: float(mx.sum(mx.array(h) * (w - d) ** 2))
    flat = G4.quant_nvfp4_rows(w, g, G4.NV_SEARCH_WIDE)[2]
    wtd = G4.quant_nvfp4_rows(w, g, G4.NV_SEARCH_WIDE, h=h)[2]
    assert ew(wtd) < 0.92 * ew(flat)
    assert np.array_equal(np.array(G4.quant_nvfp4_rows(w, g, G4.NV_SEARCH_WIDE, h=np.ones(512, np.float32))[2]), np.array(flat))


def test_act_keys_and_weights():
    import importlib.util
    sp = importlib.util.spec_from_file_location("g4b_groups", os.path.join(os.path.dirname(__file__), "..", "tools", "g4b_groups.py"))
    M = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(M)
    L = "model.language_model.layers.7."
    assert M.act_key(L + "self_attn.k_proj") == "layers.7.input_layernorm" == M.act_key(L + "self_attn.kv_a_proj_with_mqa")
    assert M.act_key(L + "mlp.shared_experts.up_proj") == "layers.7.post_attention_layernorm" == M.act_key(L + "mlp.gate_proj")
    assert M.act_key(L + "self_attn.o_proj") == "layers.7.self_attn.o_proj" and M.act_key("lm_head") == "norm"
    st = {"layers.7.input_layernorm": np.array([1.0, 0.0, 3.0, 4.0])}
    h = M.act_weights(st, L + "self_attn.q_proj", floor=0.01, alpha=1.0, k=4)
    assert abs(h.mean() - 1) < 1e-6 and h[1] > 0 and h[3] > h[2] > h[0]
    assert M.act_weights(st, L + "self_attn.q_proj", k=5) is None and M.act_weights(st, L + "self_attn.o_proj", k=4) is None


def test_collector_hooks_a_tiny_model():
    import importlib.util, sys
    import mlx.nn as nn
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    sp = importlib.util.spec_from_file_location("g4b_actstat", os.path.join(os.path.dirname(__file__), "..", "tools", "g4b_actstat.py"))
    A = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(A)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = nn.RMSNorm(8)
            self.o_proj = nn.Linear(8, 8, bias=False)
            self.q_proj = nn.Linear(8, 8, bias=False)

        def __call__(self, x):
            return x + self.o_proj(self.q_proj(self.input_layernorm(x)))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [Layer(), Layer()]
            self.norm = nn.RMSNorm(8)

        def __call__(self, x):
            for l in self.layers:
                x = l(x)
            return self.norm(x)

    net = Net()
    col = A.Collector(mx, net, [nn.Linear], want_linear=lambda k: k.endswith("o_proj"))
    x = mx.array(np.random.default_rng(0).standard_normal((1, 5, 8)).astype(np.float32))
    y = net(x)
    mx.eval(y)
    col.close()
    r = col.result()
    assert set(k for k in r if "#" not in k) == {"layers.0.input_layernorm", "layers.1.input_layernorm", "layers.0.o_proj", "layers.1.o_proj", "norm"}
    ref = np.array(mx.mean(net.norm(net.layers[1](net.layers[0](x))).astype(mx.float32) ** 2, axis=(0, 1)))
    assert np.allclose(r["norm"], ref, rtol=1e-4) and int(r["norm#n"]) == 5
    assert np.allclose(r["layers.0.input_layernorm"], np.array(mx.mean(net.layers[0].input_layernorm(x) ** 2, axis=(0, 1))), rtol=1e-4)
    assert type(net.layers[0].o_proj).__call__ is nn.Linear.__call__                    # hooks removed
