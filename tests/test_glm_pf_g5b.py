"""G5b GLM-5.x prompt hooks: the mHC mix kernel against an f64 reference, flag defaults, install on a non-GLM model is a no-op. Needs a Mac (Metal)."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from kyojin_mlx import flags, glm_dsa_pf, glm_pf_glue as G, glm_pf_idx, patch as P   # noqa: E402


def test_flags_registered_and_off_by_default():
    for name in ("glm_dsa_pf", "glm_pf_hc", "glm_pf_idx", "glm_pf_untile"):
        assert name in flags.FLAGS
    assert flags.FLAGS["glm_pf_hc"].default == "0" and flags.FLAGS["glm_dsa_pf"].default == "0" and flags.FLAGS["glm_pf_idx"].default == "0"


def test_install_is_noop_on_other_models():
    class M:
        language_model = None

    assert glm_dsa_pf.install(M()) == 0 and glm_pf_idx.install(M()) == 0 and G.install(M()) == 0
    assert P.install_glm_pf.__doc__


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
@pytest.mark.parametrize("S", [16, 100, 1031])
def test_mix_vs_f64(S):
    rng = np.random.default_rng(S)
    K = 4096
    x = mx.array((rng.standard_normal((S, K)) * 3).astype(np.float16))
    fn = mx.array((rng.standard_normal((24, K)) * 0.02).astype(np.float32))
    xn, fnn = np.array(x).astype(np.float64), np.array(fn).astype(np.float64)
    ref = (xn / np.sqrt((xn ** 2).mean(1, keepdims=True) + 1e-6)) @ fnn.T
    got = np.array(G.mix(x, fn, 1e-6)).astype(np.float64)
    assert np.sqrt(((got - ref) ** 2).mean() / (ref ** 2).mean()) < 1e-5


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
@pytest.mark.parametrize("T,N,W", [(64, 64, 100), (96, 700, 300)])
def test_fused_dsa_vs_f64(T, N, W):
    from kyojin_mlx import glm_dsa_fused as GF
    rng = np.random.default_rng(T + N)
    H, D = 32, 512
    q = (rng.standard_normal((1, H, T, D)) * 0.4).astype(np.float16)
    lat = rng.standard_normal((1, 1, N, D)).astype(np.float16)
    ids = np.full((1, T, W), -1, dtype=np.int32)
    for i in range(T):
        vis = N - T + i + 1
        k = min(W, vis)
        ids[0, i, :k] = np.sort(rng.choice(vis, size=k, replace=False))
    qm, lm, im = mx.array(q), mx.array(lat), mx.array(ids)
    if not GF.supported(qm, lm, im):
        pytest.skip("fused kernel disabled")
    scale = 0.0625
    got = np.array(GF.latent_attention(qm, lm, im, scale)).astype(np.float64)
    ref = np.zeros_like(got)
    for i in range(T):
        sel = ids[0, i][ids[0, i] >= 0]
        kv = lat[0, 0, sel].astype(np.float64)
        s = (q[0, :, i].astype(np.float64) @ kv.T) * scale
        p = np.exp(s - s.max(-1, keepdims=True))
        ref[0, :, i] = (p / p.sum(-1, keepdims=True)) @ kv
    assert np.sqrt(((got - ref) ** 2).mean() / (ref ** 2).mean()) < 2e-3


def test_head_last_glm_sees_input_ids_keyword():
    """G5g: the served backend calls the language model as lm(input_ids=ids, cache=...) (inputs is None); the head-last wrapper must still ask for one logit row."""
    from kyojin_mlx import headlast as HL, patch as Pm

    seen = {}

    class LM:
        def __call__(self, inputs=None, cache=None, inputs_embeds=None, **kwargs):
            seen.update(kwargs)
            return None

    LM.__module__ = "mlx_vlm.models.glm5_next.language"

    class W:
        language_model = LM()

    w = W()
    Pm._head_last_row_glm(w.language_model)
    ids = mx.zeros((1, 300), dtype=mx.int32)
    w.language_model(input_ids=ids)                                  # outside last_row(): every row
    assert "num_logits_to_keep" not in seen
    seen.clear()
    with HL.last_row():
        w.language_model(input_ids=ids)
        assert seen.get("num_logits_to_keep") == 1
        seen.clear()
        w.language_model(ids)                                        # positional form (the G5b gate)
        assert seen.get("num_logits_to_keep") == 1
        seen.clear()
        w.language_model(input_ids=ids[:, :10])                      # decode / verify rows keep every row
        assert "num_logits_to_keep" not in seen
