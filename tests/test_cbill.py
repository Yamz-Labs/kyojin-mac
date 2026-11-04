"""KYOJIN_DENSE_VIEWS (cbill): policy, view contents, caching, flag table. Runs on the CPU build of MLX (no kernel is launched)."""
import os
import subprocess
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from kyojin_mlx import cbill as CB          # noqa: E402
from kyojin_mlx import flags as FL          # noqa: E402


def test_policy():
    assert not CB.want_view(8 << 20, 1 << 21, enabled=False)
    assert CB.want_view(8 << 20, 1 << 21, enabled=True)
    assert not CB.want_view(1 << 19, 1 << 17, enabled=True)                    # under the byte threshold
    assert not CB.want_view(8 << 20, 1 << 16, enabled=True)                    # not larger than the view itself
    assert CB.want_view(1 << 10, 1 << 8, enabled=True, min_bytes=0, view_words=16)


def test_flag_off_binds_the_array(monkeypatch):
    monkeypatch.setattr(CB, "DENSE_VIEWS", False)
    w = mx.arange(1 << 21, dtype=mx.uint32)
    assert CB.bind(w, CB.make_view(w)) is w


def test_view_is_a_prefix_and_bind_never_evaluates(monkeypatch):
    monkeypatch.setattr(CB, "MIN_BYTES", 0)
    monkeypatch.setattr(CB, "VIEW_WORDS", 128)
    w = mx.arange(1 << 12, dtype=mx.uint32)
    v = CB.make_view(w)                                                        # built whatever the flag says
    assert v.size == 128 and np.array_equal(np.array(v), np.arange(128, dtype=np.uint32))
    monkeypatch.setattr(CB, "DENSE_VIEWS", True)
    assert CB.bind(w, v) is v and CB.bind(w, None) is w
    monkeypatch.setattr(mx, "eval", lambda *a: (_ for _ in ()).throw(AssertionError("bind evaluated")))
    assert CB.bind(w, v) is v


def test_small_array_has_no_view(monkeypatch):
    w = mx.zeros((1 << 10,), dtype=mx.uint32)
    assert CB.make_view(w) is None
    monkeypatch.setattr(CB, "DENSE_VIEWS", True)
    assert CB.bind(w, None) is w


def test_runtime_flip_inside_a_compiled_function(monkeypatch):
    monkeypatch.setattr(CB, "MIN_BYTES", 0)
    monkeypatch.setattr(CB, "VIEW_WORDS", 16)
    w = mx.arange(1024, dtype=mx.uint32)
    v = CB.make_view(w)
    f = mx.compile(lambda x: x + CB.bind(w, v)[:16].astype(mx.float32))
    for on in (False, True, False):
        monkeypatch.setattr(CB, "DENSE_VIEWS", on)
        assert np.array_equal(np.array(f(mx.zeros(16))), np.arange(16, dtype=np.float32))


def test_flag_row_and_module_literal():
    f = FL.FLAGS["dense_views"]
    assert f.env == "KYOJIN_DENSE_VIEWS" and f.default == "0" and f.runtime
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {k: v for k, v in os.environ.items() if not k.startswith("KYOJIN_")}
    env["PYTHONPATH"] = root
    r = subprocess.run([sys.executable, "-c", "import kyojin_mlx\nfrom kyojin_mlx import cbill\nprint(cbill.DENSE_VIEWS)"], capture_output=True, text=True, env=env, cwd=root)
    assert r.stdout.split() == [str(f.default == "1")], r.stderr[-300:]


def test_every_decode_bind_goes_through_bind():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for fn in ("kernels", "grouped", "fused_group", "mma_gemv", "metal_nax_sh", "metal_nax_shw"):
        assert "CB.bind(" in open(os.path.join(root, "kyojin_mlx", fn + ".py")).read(), fn
