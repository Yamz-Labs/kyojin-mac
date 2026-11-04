"""CPU tests for tools/fmt_gate_mac.py: every encoder / reference decoder round-trips, the kernels' unit layout (python emulation) matches the decoders,
and the Metal sources are generated for every variant. No GPU and no mlx needed."""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import fmt_gate_mac as G
import fmt_gate_metal as M


def test_selftest_all_roundtrips():
    assert G.selftest(verbose=False) == []


@pytest.mark.parametrize("fmt", sorted(G.ENC))
def test_bytes_per_weight_and_shapes(fmt):
    enc, dec = G.ENC[fmt]
    w = G.synth_weights(8, 1024, 5)
    qs, meta = enc(w)
    assert qs.dtype == np.uint8 and qs.shape == (8, 512)
    assert (qs.nbytes + meta.nbytes) / w.size == pytest.approx(0.5625)
    wd = dec(qs, meta, 1024)
    assert wd.shape == w.shape and np.isfinite(wd).all()
    assert np.abs(wd - w).mean() < 0.2            # 4-bit quantisation of N(0,1)


@pytest.mark.parametrize("fmt", sorted(G.ENC))
def test_kernel_layout_emulation_matches_decoder(fmt):
    enc, dec = G.ENC[fmt]
    rng = np.random.default_rng(11)
    K = 1024
    q, mt = enc(rng.standard_normal((4, K)).astype(np.float32))
    x = rng.standard_normal((3, K)).astype(np.float16)
    ye = G.emulate_gemv(fmt, q, mt, x, K)
    yr = G.reference_y(x, dec(q, mt, K))
    assert np.abs(ye - yr).max() <= 1e-4 * np.abs(yr).max()


def test_int8_activation_reference_close_to_float():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4, 512)).astype(np.float16)
    w = rng.standard_normal((16, 512)).astype(np.float32)
    a, b = G.reference_y(x, w, True), G.reference_y(x, w, False)
    assert np.abs(a - b).max() < 0.02 * np.abs(b).max()


def test_metal_sources_generate():
    for fmt, vs in M.VARIANTS.items():
        for v in vs:
            for nbr in (1, 2):
                s = M.gemv_source(fmt, v, nbr)
                assert "simd_shuffle_xor" in s and "y[" in s and s.count("{") == s.count("}")
                assert len(M.gemv_inputs(fmt, v)) == (4 if v == "i" else 3)


def _fake_mlx():
    """numpy stand-in for mlx.core: metal_kernel callables compute the reference result from the decoded weights (exercises main(): checks, timing, report)."""
    import types
    mc = types.ModuleType("mlx.core")
    mc.array = lambda x: np.array(x)
    mc.eval = lambda *a: None
    mc.float16, mc.float32, mc.int8, mc.uint32 = np.float16, np.float32, np.int8, np.uint32
    mc.set_cache_limit = lambda n: None
    mc.clear_cache = lambda: None
    mc.get_peak_memory = lambda: 0
    mc.get_active_memory = lambda: 0

    def device_info():
        raise RuntimeError("fake")
    mc.metal = types.SimpleNamespace(device_info=device_info)

    def metal_kernel(name, input_names, output_names, source, header=""):
        def call(inputs, template=None, grid=None, threadgroup=None, output_shapes=None, output_dtypes=None):
            t = dict(template) if template else {}
            if name == "fg_actq":
                q, s = G.act_quant_ref(inputs[0])
                return [q.astype(np.int8), s]
            fmt, var = name[3:].rsplit("_", 2)[0], name.rsplit("_", 2)[1]
            if var == "i":
                xq, xsa, qs, mt = inputs
                x = (xq.reshape(xq.shape[0], -1, 32) * xsa[..., None]).reshape(xq.shape)
            else:
                x, qs, mt = inputs
            N, K = t["N"], t["KIN"]
            w = G.ENC[fmt][1](np.ascontiguousarray(qs).view(np.uint8).reshape(N, K // 2), mt, K)
            return [(x.astype(np.float64) @ w.astype(np.float64).T).astype(np.float16)]
        return call
    mc.fast = types.SimpleNamespace(metal_kernel=metal_kernel)
    pkg = types.ModuleType("mlx")
    pkg.core = mc
    return pkg, mc


def test_main_with_fake_mlx(monkeypatch, capsys):
    pkg, mc = _fake_mlx()
    monkeypatch.setitem(sys.modules, "mlx", pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mc)
    monkeypatch.setattr(sys, "argv", ["fmt_gate_mac.py", "--in", "1024", "--out", "512", "--rows", "1,4", "--no-exl3", "--pool", "2", "--reps", "2", "--chain", "4"])
    assert G.main() == 0
    out = capsys.readouterr().out
    for lab in ("NVFP4", "Q4_0", "Q4_K", "IQ4_NL"):
        assert lab in out
    assert "MISMATCH" not in out and "(no result)" not in out.split("--- all checked")[0].split("=== 4")[0]
    assert "[i,nbr2]" in out and "[lut,nbr1]" in out
