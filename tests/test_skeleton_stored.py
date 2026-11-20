"""foreign_skeleton_stored.load_stored on synthetic Hy4-shaped GGUFs (the 2-layer model of test_skeleton_hy4_e2e).
  CPU tier: RowExperts (experts decoded from the mapped file on use) and the dense rest = load_cpu, logits equal (f32, same values).
  layers=1: the second layer is mapped and checked, not read.
  Metal tier (tiny, < 50 MB): the same model at width 256 with Q3_K gate / up experts, Q4_K down experts and Q8_0 linears, as on the Hy4-flash Q3_K
  file; experts on ForeignSwitchGLU, linears on ForeignLinear vs load_cpu on the CPU (every tensor decoded to f32): equal within the fp16 activation
  rounding of the stored kernels, prefill and cached one-token steps. Skipped without a Metal device or with KYOJIN_NO_METAL_TESTS=1."""
import importlib.util
import os

import numpy as np
import pytest

import mlx.core as mx

from kyojin_mlx import foreign_gguf as GF  # noqa: E402
from kyojin_mlx import foreign_skeleton as SK  # noqa: E402
from kyojin_mlx import foreign_skeleton_rows as R  # noqa: E402
from kyojin_mlx import foreign_skeleton_stored as ST  # noqa: E402
from kyojin_mlx import foreign_synth as SY  # noqa: E402
from kyojin_mlx import engram_gguf as EG  # noqa: E402

A = "hy_v4_flash"
HERE = os.path.dirname(os.path.abspath(__file__))


def _e2e(tag, **dims):
    """A fresh copy of the e2e module (its own globals), with some sizes changed."""
    spec = importlib.util.spec_from_file_location(f"_hy4_e2e_{tag}", os.path.join(HERE, "test_skeleton_hy4_e2e.py"))
    m = importlib.util.module_from_spec(spec)
    dev = mx.default_device()
    spec.loader.exec_module(m)
    mx.set_default_device(dev)                                            # the e2e module pins the CPU at import
    for k, v in dims.items():
        setattr(m, k, v)
    return m


def _table(m, rng):
    tab = rng.integers(0, 256, (sum(m.PRIMES), m.HD)).astype(np.uint8)
    tab[(tab & 127) == 127] = 0x38
    return tab


def _q(tname, a, rng, scale):
    """random valid blocks of `tname` for a tensor of numpy shape a.shape (k = last axis), fp16 scales ~ scale."""
    k = a.shape[-1]
    n = int(np.prod(a.shape[:-1]))
    b = SY.random_blocks(tname, n, k, rng)
    _, be, bb = GF.TYPES_BY_NAME[tname]
    b3 = b.reshape(n, k // be, bb)
    SY.set_scales(tname, b3, rng.uniform(0.5, 1.0, (n, k // be)) * scale)
    return b3.reshape(n, -1)


QSCALE = {"Q8_0": 6e-4, "Q3_K": 1.2e-3, "Q4_K": 2e-4}


def _write_quant(m, path, rng):
    t = m._model_tensors(rng)
    tab = _table(m, rng)
    out = [(EG.TABLE, (m.HD, tab.shape[0]), "F8_E4M3", tab)]
    types = {}
    for n, a in t.items():
        tn = "F32"
        if n.endswith(("ffn_gate_exps.weight", "ffn_up_exps.weight")):
            tn = "Q3_K"
        elif n.endswith("ffn_down_exps.weight"):
            tn = "Q4_K"
        elif a.ndim == 2 and a.shape[-1] % 256 == 0 and not n.startswith(("token_embd", "output")) and "hc_" not in n and "gate_inp" not in n:
            tn = "Q8_0"
        types[n] = tn
        out.append((n, a.shape[::-1], tn, a.astype(np.float32) if tn == "F32" else _q(tn, a, rng, QSCALE[tn])))
    SY.write_gguf(path, m._kv(), out)
    return types


def _logits(model, ids):
    return np.array(model(mx.array([ids]))[0].astype(mx.float32)).astype(np.float64)


def _cached(model, ids, cut):
    from mlx_lm.models.cache import KVCache
    cache = [KVCache() for _ in model.layers]
    a = _logits_c(model, ids[:cut], cache)
    rest = [_logits_c(model, [i], cache) for i in ids[cut:]]                 # one-token calls: the decode path (gather kernels)
    return np.concatenate([a] + rest)


def _logits_c(model, ids, cache):
    return np.array(model(mx.array([ids]), cache=cache)[0].astype(mx.float32)).astype(np.float64)


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    m = _e2e("small")
    mx.set_default_device(mx.cpu)
    rng = np.random.default_rng(7)
    t = m._model_tensors(rng)
    tab = _table(m, rng)
    p = str(tmp_path_factory.mktemp("hy4s") / "hy4_2l.gguf")
    m._write(p, t, tab)
    return m, p


def test_cpu_tier_equals_load_cpu(small):
    m, p = small
    mx.set_default_device(mx.cpu)
    ref, _ = SK.load_cpu(p, row=R.ROWS[A])
    got, rep = ST.load_stored(p, metal=False, log=lambda *a: None)
    assert rep["expert_blocks"] == 1 and rep["layers_loaded"] == 2 and rep["mapped"] == len(GF.GgufFile(p).tensors) - 1
    ids = m._ids()
    a, b = _logits(ref, ids), _logits(got, ids)
    assert np.abs(a - b).max() / np.abs(a).max() < 1e-6
    c = _cached(got, ids, 9)
    assert np.abs(c - a).max() / np.abs(a).max() < 1e-5
    lru = got.__dict__["_expert_lru"]
    assert lru.miss <= m.E and lru.hit > 0                                 # each expert decoded once, then served from the cache


def test_layers_cut_maps_all_reads_some(small):
    m, p = small
    mx.set_default_device(mx.cpu)
    got, rep = ST.load_stored(p, metal=False, layers=1, log=lambda *a: None)
    assert rep["layers_loaded"] == 1 and len(got.layers) == 1 and rep["mapped_not_loaded"] > 0 and rep["expert_blocks"] == 0
    assert np.isfinite(_logits(got, m._ids())).all()


def _metal_ok():
    return os.environ.get("KYOJIN_NO_METAL_TESTS", "0") in ("0", "") and mx.metal.is_available()


@pytest.mark.skipif(not _metal_ok(), reason="no Metal device (or KYOJIN_NO_METAL_TESTS=1)")
def test_metal_tier_stored_q3k_q4k_q8_0(tmp_path):
    m = _e2e("w256", D=256, FF=256, LAT=256, EFF=256)
    rng = np.random.default_rng(11)
    p = str(tmp_path / "hy4_2l_q.gguf")
    types = _write_quant(m, p, rng)
    assert {"Q3_K", "Q4_K", "Q8_0"} <= set(types.values())
    mx.set_default_device(mx.cpu)
    ref, _ = SK.load_cpu(p, row=R.ROWS[A])
    ids = m._ids()
    want = _logits(ref, ids)
    want_c = _cached(ref, ids, 9)
    dev = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        got, rep = ST.load_stored(p, metal=True, log=lambda *a: None)
        fm = rep["formats"]
        assert fm.get("experts_q3_k") == 2 and fm.get("experts_q4_k") == 1 and rep["stored_linears"] >= 8, fm
        assert type(got.layers[1].mlp.block().switch_mlp).__name__ == "ForeignSwitchGLU"
        g = _logits(got, ids)
        gc = _cached(got, ids, 9)
    finally:
        mx.set_default_device(dev)
    for x, w in ((g, want), (gc, want_c)):
        rel = np.linalg.norm(x - w) / np.linalg.norm(w)
        cos = float((x * w).sum() / (np.linalg.norm(x) * np.linalg.norm(w)))
        print(f"metal vs load_cpu logits: cos {cos:.7f} rel {rel:.2e}")
        assert np.isfinite(x).all() and cos > 0.9999 and rel < 1e-2, (cos, rel)
    assert (g.argmax(-1) == want.argmax(-1)).mean() >= 0.9
