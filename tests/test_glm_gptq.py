"""G4c: GPTQ on the NVFP4 grid (CPU, tiny). Run: python -m pytest -q -p no:cacheprovider tests/test_glm_gptq.py"""
import os, sys
import numpy as np
import pytest
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kyojin_mlx import glm_dense4 as G4, glm_gptq as GQ

mx.set_default_device(mx.cpu)


def synth(n=96, k=256, T=4096, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((k // 4, k)).astype(np.float32) / 2                     # correlated inputs: low-rank mixing + a few massive channels
    Z = rng.standard_normal((T, k // 4)).astype(np.float32)
    X = Z @ A + 0.3 * rng.standard_normal((T, k)).astype(np.float32)
    X[:, rng.choice(k, 4, replace=False)] *= 12
    W = (rng.standard_normal((n, k)) * 0.05).astype(np.float32)
    return W, X[: T // 2], X[T // 2:]


def rel(W, Q, X):
    return float(np.linalg.norm(X @ (W - Q).T) / np.linalg.norm(X @ W.T))


@pytest.mark.parametrize("act_order", [False, True])
def test_gptq_beats_rtn_and_layout_roundtrips(act_order):
    W, Xt, Xe = synth()
    H = Xt.T @ Xt / len(Xt)
    g = 16 * G4.global_scale(float(np.abs(W).max()))
    r = GQ.gptq_matrix(W, H, g=g, act_order=act_order, return_deq=True)
    Q = np.array(r["deq"])
    Wr = np.array(G4.quant_nvfp4_rows(mx.array(W), g)[2])
    e_g, e_r = rel(W, Q, Xe), rel(W, Wr, Xe)
    assert e_g < 0.9 * e_r, (e_g, e_r)                                              # held-out output error clearly below RTN
    assert r["out_err"] < r["out_err_rtn"]
    dq = np.array(G4.dequant_nvfp4(np.array(r["qs"]), np.array(r["scale"]), g))     # the stored bytes decode to exactly what the sweep produced
    assert np.allclose(dq, Q, rtol=1e-5, atol=1e-7)


def test_codes_are_on_the_grid_and_dead_channels_are_safe():
    W, Xt, Xe = synth(n=32, k=128)
    Xt[:, 5] = 0                                                                       # a channel never active in calibration
    H = Xt.T @ Xt / len(Xt)
    g = 16 * G4.global_scale(float(np.abs(W).max()))
    r = GQ.gptq_matrix(W, H, g=g, return_deq=True)
    assert np.isfinite(np.array(r["deq"])).all()
    assert r["scale"].dtype == mx.uint8 and int(mx.max(r["scale"])) <= 126 and int(mx.min(r["scale"])) >= 1


def test_identity_hessian_is_close_to_rtn():
    W, _, _ = synth(n=32, k=128)
    g = 16 * G4.global_scale(float(np.abs(W).max()))
    r = GQ.gptq_matrix(W, np.eye(128, dtype=np.float32), g=g, return_deq=True)
    Wr = np.array(G4.quant_nvfp4_rows(mx.array(W), g)[2])
    # H = I: no cross-column compensation to speak of beyond damping; error stays at RTN level
    assert np.linalg.norm(np.array(r["deq"]) - W) <= 1.15 * np.linalg.norm(Wr - W)


def test_collector_passes_and_keys():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
    import g4c_collect as GC
    P = "model.language_model."
    assert GC.hkey(P + "layers.4.self_attn.q_proj") == GC.hkey(P + "layers.4.self_attn.k_proj") == GC.hkey(P + "layers.4.self_attn.v_proj") == "layers.4.qkv"
    assert GC.hkey(P + "layers.4.mlp.gate_proj") == GC.hkey(P + "layers.4.mlp.up_proj") == "layers.4.mlp_gu"
    assert GC.hkey(P + "layers.4.mlp.shared_experts.up_proj") == GC.hkey(P + "layers.4.mlp.shared_experts.gate_proj") == "layers.4.mlp.sh_gu"
    assert GC.hkey(P + "layers.4.self_attn.q_a_proj") == GC.hkey(P + "layers.4.self_attn.kv_a_proj_with_mqa") == "layers.4.qkva"
    assert GC.hkey(P + "layers.4.self_attn.o_proj") == "layers.4.self_attn.o_proj"
    assert GC.hkey("lm_head") == "lm_head"
    pl = {f"layers.{L}.x{i}": dict(k=4096, layer=L, members=[]) for L in range(10) for i in range(2)}
    pl["lm_head"] = dict(k=4096, layer=GC.LM_HEAD_LAYER, members=[])
    ps = GC.make_passes(pl, 0.35)                                                      # 64 MB per Hessian: 5 per pass
    assert sum(len(k) for k, _ in ps) == len(pl) and ps[-1][1] is None and all(last is not None for _, last in ps[:-1])
    for ks, last in ps[:-1]:
        assert max(pl[k]["layer"] for k in ks) == last                                 # a pass is cut right after its own last layer
