"""kyojin_mlx.fired: the lever-fired counters are off by default, count when switched on, and the sites that need no GPU bump them."""
import gc

import pytest

mx = pytest.importorskip("mlx.core")

from kyojin_mlx import cbill, fired as FD, flags as FL          # noqa: E402
from kyojin_mlx.serve import hostloop as HL                       # noqa: E402


@pytest.fixture(autouse=True)
def clean():
    on, dv = FD.ON, cbill.DENSE_VIEWS
    FD.reset()
    yield
    FD.ON, cbill.DENSE_VIEWS = on, dv
    FD.reset()


def test_default_is_off_and_registered():
    assert FL.FLAGS["fired"].default == "0" and FL.FLAGS["fired"].env == "KYOJIN_FIRED"
    FD.ON = False
    FD.hit("x")
    assert FD.HITS == {}


def test_flag_switches_the_counter_at_runtime():
    FL.set("fired", 1)
    try:
        FD.hit("x")
        FD.hit("x", 3)
        assert FD.HITS == {"x": 4}
    finally:
        FL.set("fired", 0)
    FD.hit("x")
    assert FD.HITS == {"x": 4}


def test_dense_views_bind_counts_view_and_whole():
    FD.ON, cbill.DENSE_VIEWS = True, True
    w = mx.zeros((1 << 18,), dtype=mx.uint32)
    v = cbill.make_view(w)
    assert v is not None and cbill.bind(w, v) is v and cbill.bind(w, None) is w
    assert FD.HITS == {"dense_views": 1, "dense_views_whole": 1}
    cbill.DENSE_VIEWS = False
    cbill.bind(w, v)
    assert FD.HITS == {"dense_views": 1, "dense_views_whole": 1}


def test_serve_gc_and_switch_count_when_applied():
    FD.ON = True
    saved = (HL.GC, HL._gc_on, HL._gc_saved, gc.get_threshold(), HL.SWITCH_US, HL._sw_state, HL._sw_orig)
    try:
        HL.GC, HL._gc_on = True, False
        HL.gc_sync()
        HL.gc_sync()                                              # already applied: no second count
        HL.idle()
        HL.SWITCH_US, HL._sw_state = 2000, 0
        HL.switch_sync()
        assert FD.HITS == {"serve_gc": 1, "serve_gc_idle": 1, "serve_switch": 1}
    finally:
        HL.GC = False
        HL.gc_sync()
        HL.SWITCH_US = 0
        HL.switch_sync()
        HL.GC, HL._gc_on, HL._gc_saved, th, HL.SWITCH_US, HL._sw_state, HL._sw_orig = saved
        gc.set_threshold(*th)
