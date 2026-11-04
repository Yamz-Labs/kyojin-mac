"""Numpy reference vs the engine's CPU codec (exllamav3 modules/quant/exl3_cpu.py). Linux, needs torch.
   KYOJIN_ENGINE_ROOT=<engine checkout> python -m pytest tests/test_ref_engine.py -s"""
import importlib.util, os, sys, types

import numpy as np
import pytest

from kyojin_mlx import ref_numpy as R

ROOT = os.environ.get("KYOJIN_ENGINE_ROOT")
torch = pytest.importorskip("torch")
if not ROOT:
    pytest.skip("KYOJIN_ENGINE_ROOT not set", allow_module_level=True)


def _load_engine_cpu(root):
    """exl3_cpu.py without the CUDA extension: stub packages, real files."""
    base = os.path.join(root, "exllamav3")
    for name, sub in (("exllamav3", ""), ("exllamav3.util", "util"), ("exllamav3.modules", "modules"),
                      ("exllamav3.modules.quant", "modules/quant")):
        m = types.ModuleType(name); m.__path__ = [os.path.join(base, sub)]; sys.modules[name] = m
    ext = types.ModuleType("exllamav3.ext"); ext.exllamav3_ext = None; sys.modules["exllamav3.ext"] = ext
    spec = importlib.util.spec_from_file_location("exllamav3.modules.quant.exl3_cpu", os.path.join(base, "modules/quant/exl3_cpu.py"))
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def eng():
    return _load_engine_cpu(ROOT)


def test_states_and_codebook(case, eng):
    _, K, t, _, _ = case
    got = R.decode_states(R.trellis_words(t), K)
    want = eng.unpack_states(t.reshape(-1, 16 * K), K)
    assert np.array_equal(got, want)
    cb = R.mul1_values(np.arange(65536, dtype=np.uint16))
    assert np.array_equal(cb.view(np.uint16), eng.mul1_lut().astype(np.float16).view(np.uint16))


def test_tile_perm(eng):
    assert np.array_equal(R.tile_perm(), eng.tile_perm())


def test_wr_bit_exact(case, eng):
    _, K, t, _, _ = case
    got = R.decode_tiles(t)
    want = eng.decode_tiles(torch.from_numpy(t)).numpy().astype(np.float16)
    assert np.array_equal(got.view(np.uint16), want.view(np.uint16))


def test_full_layer(case, eng):
    """Dense W: fp32 values may differ by a few ulp (butterfly vs the engine's fp32 matmul order). The engine
    stores W in fp16, so also report the fp16 mismatch count."""
    name, K, t, su, sv = case
    got = R.reconstruct(t, su, sv)
    want = eng.reconstruct(torch.from_numpy(t), torch.from_numpy(su), torch.from_numpy(sv)).numpy()
    ulp = np.abs(got.view(np.int32).astype(np.int64) - want.view(np.int32).astype(np.int64))
    h = (got.astype(np.float16) != want.astype(np.float16)).sum()
    rel = np.abs(got - want).max() / np.abs(want).max()
    print(f"\n[{name}] fp32: bit-equal {np.array_equal(ulp, 0 * ulp)} max_ulp {ulp.max()} mism {(ulp > 0).sum()}/{ulp.size}"
          f" | fp16 W mismatches {h}/{ulp.size} | max rel err {rel:.2e}")
    assert rel < 1e-5


def test_gemv_vs_dense(case):
    """x @ W (dense, fp32) equals the factored GEMV in float64."""
    _, K, t, su, sv = case
    x = np.random.default_rng(1).standard_normal((3, t.shape[0] * 16)).astype(np.float64)
    y = R.gemv(x, t, su, sv)
    d = x @ R.reconstruct(t, su, sv).astype(np.float64)
    assert np.abs(y - d).max() / np.abs(d).max() < 1e-5
