"""A converted MoE pack served through load_exl3: the routed experts run on the fused gather (qwen_moe.py) and match the experts decoded from the trellis
by the independent numpy decoder, with the model's own routing and shared expert."""
import importlib, sys
from pathlib import Path

import numpy as np
import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from kyojin_mlx import ref_numpy as R
from kyojin_mlx.exl3_linear import load_exl3

conv = importlib.import_module("conv_mlx")
mk = importlib.import_module("make_tiny_moe")


def test_pack_served_on_fused_gather_matches_decoded_experts(tmp_path, monkeypatch):
    src, out, cal = tmp_path / "src", tmp_path / "out", tmp_path / "cal.safetensors"
    monkeypatch.setattr(sys, "argv", ["mk", str(src), "--layout", "fused", "--layers", "2", "--cal", str(cal), "--rows", "8", "--cols", "128"])
    mk.main()
    monkeypatch.setattr(sys, "argv", ["conv", str(src), str(out), "--cal", str(cal), "--k-attn", "4", "--k-mlp", "4", "--k-expert", "4", "--head-bits", "4", "--expert-batch", "4"])
    conv.main()
    model, _ = load_exl3(str(out))
    blk = model.language_model.model.layers[1].mlp
    assert type(blk).__name__ == "KyojinQwenMoe" and blk.kmoe.topk == blk.top_k
    t = mx.load(str(out / "model.safetensors"))
    lp = "model.language_model.layers.1.mlp."
    E = blk.num_experts
    W = {n: [np.asarray(R.reconstruct(np.array(t[f"{lp}experts.{e}.{n}.trellis"]), np.array(t[f"{lp}experts.{e}.{n}.suh"].astype(mx.float32)),
                                      np.array(t[f"{lp}experts.{e}.{n}.svh"].astype(mx.float32))), np.float32).T for e in range(E)] for n in ("gate_proj", "up_proj", "down_proj")}
    x = mx.array(np.random.default_rng(3).standard_normal((1, 6, blk.gate.weight.shape[1])).astype(np.float16))
    y = np.array(blk(x).astype(mx.float32))[0]
    xn = np.array(x.astype(mx.float32))[0]
    gates = np.array(mx.softmax(blk.gate(x), axis=-1, precise=True).astype(mx.float32))[0]
    ref = np.zeros_like(xn)
    for r in range(xn.shape[0]):
        top = np.argsort(-gates[r])[:blk.top_k]
        s = gates[r][top] / gates[r][top].sum() if blk.norm_topk_prob else gates[r][top]
        for e, w in zip(top, s):
            g, u = W["gate_proj"][e] @ xn[r], W["up_proj"][e] @ xn[r]
            ref[r] += w * (W["down_proj"][e] @ (g / (1 + np.exp(-g)) * u))
    sh = np.array((mx.sigmoid(blk.shared_expert_gate(x)) * blk.shared_expert(x)).astype(mx.float32))[0]
    ref += sh
    assert np.linalg.norm(y - ref) / np.linalg.norm(ref) < 0.02, np.linalg.norm(y - ref) / np.linalg.norm(ref)


def test_cached_ladder_on_moe_blocks(tmp_path, monkeypatch):
    """--cache on a MoE model: a second run with more bits on one block's experts re-quantises only that block; the sensitivity file has the expert cell."""
    import json
    src, cal = tmp_path / "src", tmp_path / "cal.safetensors"
    monkeypatch.setattr(sys, "argv", ["mk", str(src), "--layout", "fused", "--layers", "2", "--cal", str(cal), "--rows", "8", "--cols", "128"])
    mk.main()
    base = ["conv", str(src), None, "--cal", str(cal), "--k-attn", "4", "--k-mlp", "4", "--k-expert", "2", "--head-bits", "4", "--expert-batch", "4", "--cache", str(tmp_path / "cache")]
    run = lambda out, strat: (monkeypatch.setattr(sys, "argv", [*base[:2], str(out), *base[3:], "--strategy", str(strat)]), conv.main())
    (tmp_path / "s0.json").write_text("{}"); (tmp_path / "s1.json").write_text(json.dumps({"L1.exp": 3}))
    run(tmp_path / "o0", tmp_path / "s0.json")
    s0 = json.load(open(tmp_path / "o0/conv_sens.json"))
    assert [r["K_exp"] for r in s0] == [2, 2] and all(r["params_exp"] > 0 and r["err_exp"] > 0 for r in s0)
    run(tmp_path / "o1", tmp_path / "s1.json")
    s1 = json.load(open(tmp_path / "o1/conv_sens.json"))
    assert [r["K_exp"] for r in s1] == [2, 3]
    assert s1[0]["err_exp"] == s0[0]["err_exp"] and s1[1]["err_exp"] < s0[1]["err_exp"]          # block 0 came from the cache, block 1 has one more bit
    t0, t1 = mx.load(str(tmp_path / "o0/model.safetensors")), mx.load(str(tmp_path / "o1/model.safetensors"))
    k = "model.language_model.layers.1.mlp.experts.0.down_proj.trellis"
    assert t0[k].shape[-1] == 16 * 2 and t1[k].shape[-1] == 16 * 3
    assert not any(".experts.gate_up_proj" in n or n.endswith("experts.down_proj") for n in t1)


def test_expert_hessian_without_input_matrix_equals_the_full_one():
    rng = np.random.default_rng(5)
    x = mx.array(rng.standard_normal((90, 32)).astype(np.float16))
    wg, wu = (mx.array((rng.standard_normal((16, 32)) * 0.3).astype(np.float16)) for _ in range(2))
    H1, H2, n = conv.expert_hessians(x, wg, wu, None)
    G1, G2, m = conv.expert_hessians(x, wg, wu, None, need_in=False)
    assert G1 is None and m == n and np.array_equal(np.array(H2), np.array(G2)) and H1.shape == (32, 32)


def test_pack_trellis_chunked_equals_whole():
    rng = np.random.default_rng(9)
    for K in range(1, 7):
        enc = rng.integers(0, 1 << 16, (3, 11, 256), dtype=np.uint16)
        whole = conv.pack_trellis(enc, K, chunk=100)
        assert whole.shape == (3, 11, 16 * K) and np.array_equal(whole, conv.pack_trellis(enc, K, chunk=4)) and np.array_equal(whole, conv.pack_trellis(enc, K, chunk=1))
