"""memfit sizing (pure integer arithmetic) and the flag rows. No Metal needed."""
import os
import subprocess
import sys

import pytest

from kyojin_mlx import memfit as M

GIB = 1 << 30


def test_fit_wired_is_live_plus_headroom_rounded_to_mib():
    assert M.fit_wired(100 * GIB + 1, 192 * GIB, 1024) == 101 * GIB + (1 << 20)
    assert M.fit_wired(100 * GIB, 192 * GIB, 0) == 100 * GIB                       # the rival's zero headroom


def test_fit_wired_never_above_recommended():
    assert M.fit_wired(191 * GIB, 192 * GIB, 4096) == 192 * GIB
    assert M.fit_wired(5, 0, 1) == (1 << 20) + (1 << 20)                           # no cap known


def test_fit_cache_lowers_to_the_room_under_the_gc_line():
    rec = 192 * GIB
    room = int(rec * 0.95) - 172 * GIB - 2048 * (1 << 20)
    assert M.fit_cache(172 * GIB, rec, 16 * GIB, 2048) == room                     # 16 GiB asked, ~6.4 GiB fits
    assert room < 16 * GIB


def test_fit_cache_never_raised_and_has_a_floor():
    rec = 192 * GIB
    assert M.fit_cache(10 * GIB, rec, 256 << 20, 2048) == 256 << 20                # small limit stays
    assert M.fit_cache(190 * GIB, rec, 16 * GIB, 2048) == 256 << 20                # no room: the floor
    assert M.fit_cache(1, 0, 123, 1) == 123                                        # unknown device


def test_flag_rows_and_module_literals():
    from kyojin_mlx import flags as FL
    for n, attr in (("wired_fit", "WIRED_FIT"), ("cache_fit", "CACHE_FIT")):
        f = FL.FLAGS[n]
        assert f.default == "0" and not f.runtime and f.reason
    assert FL.FLAGS["wired_fit_mb"].default == str(M.WIRED_FIT_MB) and FL.FLAGS["cache_fit_mb"].default == str(M.CACHE_FIT_MB)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {k: v for k, v in os.environ.items() if not k.startswith("KYOJIN_")}
    env["PYTHONPATH"] = root
    r = subprocess.run([sys.executable, "-c", "from kyojin_mlx import memfit as M\nprint(M.WIRED_FIT, M.CACHE_FIT)"], capture_output=True, text=True, env=env, cwd=root)
    assert r.stdout.split() == ["False", "False"], r.stderr[-300:]
