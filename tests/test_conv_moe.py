"""MoE support of the Metal converter: expert-wise Hessians against a numpy brute force, and a tiny random Qwen3.5-MoE converted end to end
(fused and per-expert checkpoint layouts, calibration on all tokens / on routed tokens), packs decoded with the independent numpy decoder."""
import importlib, json, sys
from pathlib import Path

import numpy as np
import pytest
import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from kyojin_mlx import ref_numpy as R

conv = importlib.import_module("conv_mlx")
mk = importlib.import_module("make_tiny_moe")


def test_expert_hessians_match_numpy():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((200, 32)).astype(np.float16)
    wg = (rng.standard_normal((16, 32)) * 0.3).astype(np.float16); wu = (rng.standard_normal((16, 32)) * 0.3).astype(np.float16)
    rows = np.sort(rng.choice(200, 70, replace=False))
    H1, H2, n = conv.expert_hessians(mx.array(x), mx.array(wg), mx.array(wu), rows)
    xe = x[rows].astype(np.float32)
    g, u = xe @ wg.astype(np.float32).T, xe @ wu.astype(np.float32).T
    a = g / (1 + np.exp(-g)) * u
    assert n == 70
    assert np.allclose(np.array(H1), xe.T @ xe, rtol=2e-3, atol=2e-3)
    assert np.allclose(np.array(H2), a.T @ a, rtol=3e-2, atol=3e-2)
    H1a, _, na = conv.expert_hessians(mx.array(x), mx.array(wg), mx.array(wu), None)
    assert na == 200 and np.allclose(np.array(H1a), x.astype(np.float32).T @ x.astype(np.float32), rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("layout,mode", [("fused", "all"), ("perexpert", "routed")])
def test_tiny_moe_end_to_end(tmp_path, monkeypatch, layout, mode):
    src, out, cal = tmp_path / "src", tmp_path / "out", tmp_path / "cal.safetensors"
    monkeypatch.setattr(sys, "argv", ["mk", str(src), "--layout", layout, "--layers", "4", "--cal", str(cal), "--rows", "8", "--cols", "128"])
    mk.main()
    monkeypatch.setattr(sys, "argv", ["conv", str(src), str(out), "--cal", str(cal), "--k-attn", "4", "--k-mlp", "4", "--k-expert", "3", "--head-bits", "4",
                                      "--expert-mode", mode, "--min-expert-tokens", "16", "--expert-batch", "4"])
    conv.main()
    t = mx.load(str(out / "model.safetensors"))
    E = 8
    for li in range(4):
        lp = f"model.language_model.layers.{li}.mlp."
        for e in range(E):
            for n in ("gate_proj", "up_proj", "down_proj"):
                for s in ("trellis", "suh", "svh", "mul1"):
                    assert f"{lp}experts.{e}.{n}.{s}" in t
        for n in ("gate_proj", "up_proj", "down_proj"):
            assert f"{lp}shared_expert.{n}.trellis" in t
        assert lp + "gate.weight" in t and lp + "shared_expert_gate.weight" in t
        assert not any(k.startswith(lp + "experts.") and (k.endswith("gate_up_proj") or k.endswith(".weight")) for k in t)
    # expert tensors decode to the source weights up to the K = 3 quantisation noise (relative error ~ 2^-K), shapes [tk, tn, 16K]
    s_t = mx.load(str(src / "model.safetensors"))
    lp = "model.language_model.layers.1.mlp."
    if layout == "fused":
        gu = np.array(s_t[lp + "experts.gate_up_proj"].astype(mx.float32)); dn = np.array(s_t[lp + "experts.down_proj"].astype(mx.float32))
        ref = {"gate_proj": gu[:, :128], "up_proj": gu[:, 128:], "down_proj": dn}
    else:
        ref = {n: np.stack([np.array(s_t[f"{lp}experts.{e}.{n}.weight"].astype(mx.float32)) for e in range(E)]) for n in ("gate_proj", "up_proj", "down_proj")}
    for e in (0, 5):
        for n in ("gate_proj", "up_proj", "down_proj"):
            tr = np.array(t[f"{lp}experts.{e}.{n}.trellis"])
            w = R.reconstruct(tr, np.array(t[f"{lp}experts.{e}.{n}.suh"].astype(mx.float32)), np.array(t[f"{lp}experts.{e}.{n}.svh"].astype(mx.float32)))
            w = np.asarray(w, np.float32).T                                            # [n_out, k_in] as the source
            assert w.shape == ref[n][e].shape
            rel = np.linalg.norm(w - ref[n][e]) / np.linalg.norm(ref[n][e])
            assert rel < 0.25, (e, n, rel)
    rep = json.load(open(out / "conv_report.json"))
    assert sum(1 for r in rep if ".experts." in str(r["name"])) == 4 * E * 3
    qc = json.load(open(out / "quantization_config.json"))["tensor_storage"]
    assert f"{lp}experts.3.down_proj" in qc and qc[f"{lp}experts.3.down_proj"]["quant_format"] == "exl3"
