# SPDX-License-Identifier: Apache-2.0
"""Probe gpu_kern: the counters must name every compute dispatch of an eval (tools/cbcount records the Metal function name of the pipeline
at creation and the dispatch at encode). Needs a Mac with Metal; no model, no pack."""
import ctypes
import os
import sys
import time

import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from kyojin_mlx import gpuspan as GS


@pytest.fixture(scope="module")
def lib():
    if not GS.install():
        pytest.skip("libcbcount unavailable")
    L = GS._lib
    L.cb_commits.restype = ctypes.c_long
    L.cb_ndisp.argtypes = [ctypes.c_long]
    L.cb_ndisp.restype = ctypes.c_long
    L.cb_disp.argtypes = [ctypes.c_long, ctypes.c_int]
    L.cb_disp.restype = ctypes.c_int
    L.cb_rec.argtypes = [ctypes.c_int]
    yield L
    L.cb_rec(0)


def test_every_dispatch_of_an_eval_is_named(lib):
    """A matmul + a softmax: every committed buffer carries at least one dispatch with a non-empty Metal function name."""
    mx.eval(mx.ones((8, 8)) * 2.0)                       # warm: compile the kernels first
    lib.cb_rec(1)
    s0 = lib.cb_commits()
    x = mx.random.normal((8, 64)).astype(mx.float16)
    y = mx.softmax(x @ mx.random.normal((64, 64)).astype(mx.float16), axis=-1)
    mx.eval(y)
    lib.cb_rec(0)
    s1 = lib.cb_commits()
    assert s1 > s0, "no command buffer committed"
    names, ndisp = [], 0
    for s in range(s0, s1):
        n = lib.cb_ndisp(s)
        ndisp += n
        names += [GS.kname(lib.cb_disp(s, i)) for i in range(n)]
    assert ndisp >= 1
    assert all(n and n != "?" for n in names), f"unnamed dispatch: {names}"


def test_recording_off_records_nothing(lib):
    """Default off: no dispatch is recorded, so the probe costs nothing on the served path."""
    lib.cb_rec(0)
    s0 = lib.cb_commits()
    mx.eval(mx.ones((4, 4)) + 1.0)
    s1 = lib.cb_commits()
    assert s1 > s0
    assert all(lib.cb_ndisp(s) == 0 for s in range(s0, s1))


def test_prefill_chunk_bracket_keys_by_its_rows(lib):
    """The engine brackets a prefill chunk like a tick (`_prefill_step`: begin, the chunk's eval, end with `end - pos`), so the
    chunk is priced per kernel under its own row count (an 8192-row chunk writes gpu_*_r8192), and the kernel rows add up to
    gpu_kspan. Needs no model: an eval stands in for the chunk."""
    mx.eval(mx.ones((8, 8)) * 1.5)                        # compile outside the bracket
    on, kern = GS.ON, GS.KERN
    GS.ON, GS.KERN = True, True
    st = {}
    try:
        GS.begin()
        mx.eval(mx.sum(mx.ones((64, 64)) * 3.0))           # the chunk's own eval: blocks, so the buffer stamps have landed
        time.sleep(0.05)                                   # the driver's completion handler runs off the eval's fence
        GS.end(st, 8192)
    finally:
        GS.ON, GS.KERN = on, kern
        lib.cb_rec(0)
    assert st.get("gpu_n_r8192") == 1
    assert st["gpu_busy_r8192"] > 0.0
    assert not [k for k in st if k.startswith("gpu_busy_r") and not k.endswith("_r8192")]
    kms = sum(v for k, v in st.items() if k.startswith("gpu_k_"))
    kc = sum(v for k, v in st.items() if k.startswith("gpu_kc_"))
    assert kc >= 1 and kms > 0.0
    assert abs(kms - st["gpu_kspan_r8192"]) < 1e-6
