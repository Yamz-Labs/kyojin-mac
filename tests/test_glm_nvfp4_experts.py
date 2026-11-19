"""GLM-5.3 (glm5_next) routed experts served as stored NVFP4 (modelopt per-expert tensors, csf_32x2_v1 scale streams on some of them), through
foreign_linear.load_foreign: tiny synthetic pack (synth glm_shared with hidden / expert width 512 so the gather GEMV applies), against an fp32 reference
model whose experts are the exactly dequantized weights. Metal only."""
import dataclasses
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_vlm.models.glm5_next.language")
from kyojin_mlx import foreign_linear as FL, foreign_synth as FS, synth as S, vendor_input as VI  # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="no Metal GPU")

_spec = importlib.util.spec_from_file_location("t_csf", Path(__file__).with_name("test_csf_scale.py"))
T_CSF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T_CSF)

H = I = 512
SH0 = S.SHAPES["glm_shared"]
SH = dataclasses.replace(SH0, config={**SH0.config, "text_config": {**S.GLM_CFG, "hidden_size": H, "moe_intermediate_size": I}})
PROJ = {"gate_proj": (I, H), "up_proj": (I, H), "down_proj": (H, I)}


def _expert_tensors(rng, prefix, csf):
    """{name: (array, tag)} of one modelopt NVFP4 expert projection (+ the reference fp32 matrix)."""
    out = {}
    ref = {}
    for pj, (n, k) in PROJ.items():
        t = FS.nvfp4_linear(rng, n, k)
        w = np.frombuffer(t["weight"][2], np.uint8).reshape(n, k // 2)
        sc = np.frombuffer(t["weight_scale"][2], np.uint8).reshape(n, k // 16)
        g = np.frombuffer(t["weight_scale_2"][2], np.float32)
        ref[pj] = VI.dequant_nvfp4(w, sc, g)
        p = f"{prefix}.{pj}"
        out[p + ".weight"] = (w, "U8")
        out[p + ".weight_scale_2"] = (g.reshape(()), None)
        out[p + ".input_scale"] = (np.float32(1.0).reshape(()), None)
        if csf:
            out.update({f"{p}.weight_scale.{k_}": (v, None) for k_, v in T_CSF.encode_csf32(sc).items()})
        else:
            out[p + ".weight_scale"] = (sc, "F8_E4M3")
    return out, ref


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("glm_fp4"))
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        ref_model, _, _ = S.write_pack(SH, d)
    finally:
        mx.set_default_device(prev)
    from safetensors.numpy import load_file
    flat = load_file(f"{d}/model.safetensors")
    tag = {np.dtype(np.float32): "F32"}
    new = {}
    rng = np.random.default_rng(7)
    E = SH.text["n_routed_experts"]
    layers = ref_model.language_model.model.layers
    refw = {}
    for k, v in flat.items():
        if ".mlp.switch_mlp." in k:
            continue
        new[k] = (v, tag[v.dtype])
    for li, lay in enumerate(layers):
        if not hasattr(lay.mlp, "switch_mlp"):
            continue
        mp = f"model.language_model.layers.{li}.mlp"
        for e in range(E):
            t, r = _expert_tensors(rng, f"{mp}.experts.{e}", csf=(e % 2 == 1))
            new.update(t)
            for pj, w in r.items():
                refw.setdefault((li, pj), []).append(w)
    T_CSF.write_safetensors(f"{d}/model.safetensors", new)
    for (li, pj), ws in refw.items():                                 # the reference experts: the dequantized matrices, fp32
        getattr(layers[li].mlp.switch_mlp, pj).weight = mx.array(np.stack(ws))
    mx.eval(ref_model.language_model.parameters())
    cfg = json.load(open(f"{d}/config.json"))
    return d, ref_model, cfg


@pytest.fixture(scope="module")
def loaded(built):
    d, _, _ = built
    model, cfg = FL.load_foreign(d, log=lambda *_: None)
    return model


def test_is_foreign_and_experts_stay_nvfp4(built, loaded):
    d, ref_model, _ = built
    assert FL.is_foreign(d) == "safetensors"
    n_params = 0
    n_bytes = 0
    for li, lay in enumerate(loaded.language_model.model.layers):
        sw = getattr(lay.mlp, "switch_mlp", None)
        if sw is None:
            continue
        assert isinstance(sw, FL.ForeignSwitchGLU)
        assert sw.limit == ref_model.language_model.model.layers[li].mlp.switch_mlp.activation.limit
        assert all(dp.gemv_ok() for dp in (sw.g, sw.u, sw.d))      # the gather GEMV (decode) applies at this shape
        for dp in (sw.g, sw.u, sw.d):
            assert dp.fmt == "nvfp4" and dp.E == SH.text["n_routed_experts"]
            assert dp.gs.shape == (dp.E,)
            n_params += dp.E * dp.n * dp.k
            n_bytes += dp.nbytes()
    ratio = n_bytes / n_params
    print(f"\nexperts: {n_params} params, {n_bytes} resident bytes, {ratio:.4f} B/param")
    assert 0.5 < ratio < 0.5 + 1 / 16 + 0.01              # 4-bit codes + one E4M3 scale per 16 weights (+ one float per expert)


@pytest.mark.parametrize("T", [1, 3, 40])
def test_expert_block_matches_dequantized_fp32_reference(built, loaded, T):
    _, ref_model, _ = built
    rng = np.random.default_rng(T)
    x = mx.array(rng.standard_normal((1, T, H)).astype(np.float32) * 0.5)
    worst = 0.0
    for li, lay in enumerate(loaded.language_model.model.layers):
        if not hasattr(lay.mlp, "switch_mlp"):
            continue
        ref_blk = ref_model.language_model.model.layers[li].mlp
        want = ref_blk(x)
        got = lay.mlp(x)
        mx.eval(want, got)
        w, g = np.asarray(want, np.float32), np.asarray(got, np.float32)
        err = float(np.abs(g - w).max())
        worst = max(worst, err / float(np.abs(w).max()))
        np.testing.assert_allclose(g, w, rtol=2e-2, atol=2e-2 * np.abs(w).max())
    print(f"\nT={T}: max |err| / max |ref| = {worst:.2e}")


def test_expert_switch_alone_matches(built, loaded):
    """The routed part alone (no shared expert, no combine): ForeignSwitchGLU vs the stock SwitchGLU on the dequantized stacks."""
    _, ref_model, _ = built
    li = next(i for i, l in enumerate(loaded.language_model.model.layers) if hasattr(l.mlp, "switch_mlp"))
    sw, ref = loaded.language_model.model.layers[li].mlp.switch_mlp, ref_model.language_model.model.layers[li].mlp.switch_mlp
    rng = np.random.default_rng(3)
    x = mx.array(rng.standard_normal((1, 2, H)).astype(np.float32) * 0.5)
    inds = mx.array([[[0, 5], [7, 2]]], dtype=mx.uint32)
    want, got = ref(x, inds), sw(x, inds)
    mx.eval(want, got)
    w, g = np.asarray(want, np.float32), np.asarray(got, np.float32)
    err = float(np.abs(g - w).max())
    print(f"\nswitch alone: max abs err {err:.3e} (max |ref| {np.abs(w).max():.3e})")
    np.testing.assert_allclose(g, w, rtol=2e-2, atol=2e-2 * np.abs(w).max())


def test_clamped_swiglu_is_the_models_activation(loaded):
    from mlx_vlm.models.glm5_next.language import _limited_swiglu
    sw = next(l.mlp.switch_mlp for l in loaded.language_model.model.layers if hasattr(l.mlp, "switch_mlp"))
    a = mx.array(np.linspace(-30, 30, 257, dtype=np.float32))
    b = mx.array(np.linspace(40, -40, 257, dtype=np.float32))
    assert sw.limit is not None and float(mx.abs(a).max()) > sw.limit
    want = _limited_swiglu(a, b, sw.limit)
    np.testing.assert_allclose(np.asarray(sw._glu(a, b), np.float32), np.asarray(want), rtol=2e-3, atol=1e-3)


def test_server_backend_serves_the_pack_as_stored(built):
    """kyojin_mlx.serve.backends.load_backend takes a glm5_next modelopt pack (VlmBackend over the foreign_linear model), no refusal."""
    from kyojin_mlx.serve import backends as B
    d, _, _ = built
    be, cfg = B.load_backend(d, log=lambda *_: None)
    assert isinstance(be, B.VlmBackend) and be.pack == d and cfg["model_type"] == "glm5_next"
    sw = [m for _, m in be.model.named_modules() if type(m).__name__ == "ForeignSwitchGLU"]
    assert sw, "experts are not the NVFP4 ForeignSwitchGLU"
