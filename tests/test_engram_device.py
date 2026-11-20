"""KYOJIN_ENGRAM_DEVICE: the Hy4 Engram projections stay on the device; logits BIT-EQUAL to the numpy round-trip path (tiny synthetic Hy4, CPU)."""
import importlib.util
import os

import numpy as np
import pytest

import mlx.core as mx

HERE = os.path.dirname(__file__)
METAL = os.environ.get("KYOJIN_METAL_TESTS") == "1"
if not METAL:
    mx.set_default_device(mx.cpu)


def _helpers():
    spec = importlib.util.spec_from_file_location("hy4_backend_helpers_eng", os.path.join(HERE, "test_hy4_serve_backend.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def backend(tmp_path_factory):
    keys = {"KYOJIN_SKELETON": "1", "KYOJIN_KIT_UNPROVEN_ROWS": "1", "KYOJIN_TQ": "0", "KYOJIN_WIRED_MB": "0"}
    old = {k: os.environ.get(k) for k in keys}
    os.environ.update(keys)
    from kyojin_mlx.foreign_names import gguf_serve_dir
    from kyojin_mlx.serve.backends import load_backend
    d = tmp_path_factory.mktemp("eng")
    g = str(d / "model" / "hy4_2l.gguf")
    os.makedirs(os.path.dirname(g))
    _helpers()._write(g)
    be, _ = load_backend(gguf_serve_dir(os.path.dirname(g), cache=str(d / "cache")), log=lambda *a: None)
    yield be
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _run(be, on, steps=24):
    from kyojin_mlx import engram_gguf as EG
    EG.DEVICE = on
    c = be.make_cache()
    lg = be.prefill_chunk(c, [5, 9, 13, 7, 22, 31, 4, 8])
    out = [np.array(lg)]
    t = int(mx.argmax(lg[-1] if lg.ndim > 1 else lg))
    for _ in range(steps):
        lg = be.step(c, mx.array([t]))
        out.append(np.array(lg))
        t = int(mx.argmax(lg.reshape(-1)))
    return out


def test_flag_registered():
    from kyojin_mlx import flags
    assert flags.FLAGS["engram_device"].env == "KYOJIN_ENGRAM_DEVICE" if hasattr(flags.FLAGS["engram_device"], "env") else "engram_device" in flags.FLAGS


def test_bit_equal_and_no_projection_roundtrip(backend, monkeypatch):
    from kyojin_mlx import engram_gguf as EG
    try:
        off = _run(backend, False)
        n_proj = {"n": 0}
        real = np.array
        shapes = set()
        for m in backend.model.modules() if hasattr(backend, "model") else []:
            for k in ("kv", "gate"):
                a = getattr(m, k, None) if m.__class__.__name__ == "EngramParams" else None
                if a is not None:
                    shapes.add(tuple(a.shape))
        assert shapes, "no EngramParams found"

        def counting(a, *args, **kw):
            if isinstance(a, mx.array) and tuple(a.shape) in shapes:
                n_proj["n"] += 1
            return real(a, *args, **kw)
        _run(backend, True, steps=1)                      # warm: the one-time device copies
        monkeypatch.setattr(np, "array", counting)
        on = _run(backend, True)
        monkeypatch.undo()
        assert n_proj["n"] == 0, f"{n_proj['n']} numpy conversions of Engram projections with the flag on"
        assert len(on) == len(off) == 25
        assert max(float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) for a, b in zip(on, off)) == 0.0
        # the OFF path does convert (the counter sees it)
        n_proj["n"] = 0
        monkeypatch.setattr(np, "array", counting)
        _run(backend, False, steps=2)
        monkeypatch.undo()
        assert n_proj["n"] > 0
    finally:
        EG.DEVICE = False


@pytest.mark.skipif(not METAL, reason="KYOJIN_METAL_TESTS=1 (behind the can_build guard)")
def test_metal_bit_equal(backend):
    from kyojin_mlx import engram_gguf as EG
    try:
        off, on = _run(backend, False, 6), _run(backend, True, 6)
    finally:
        EG.DEVICE = False
    assert all(np.array_equal(a, b) for a, b in zip(on, off))
