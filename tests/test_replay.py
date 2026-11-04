"""kyojin_mlx.replay: static-region replay. The Metal hooks must exist before MLX builds its pipelines, so every replay check runs in a fresh process
(tools/replay_toy.py, KYOJIN_REPLAY=1) next to the unit tests that need no hooks."""
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
darwin = pytest.mark.skipif(sys.platform != "darwin", reason="Metal replay needs macOS")


def _toy(mode: str) -> str:
    env = dict(os.environ, KYOJIN_REPLAY="1", PYTHONPATH=os.pathsep.join([ROOT, os.environ.get("PYTHONPATH", "")]))
    p = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "replay_toy.py"), "--mode", mode], env=env, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    return p.stdout


def test_disabled_region_is_the_function():
    from kyojin_mlx import replay
    if replay.ENABLED:
        pytest.skip("KYOJIN_REPLAY is on in this process")
    f = lambda x: x
    assert replay.region(f, "x") is f


@darwin
def test_toy_region_is_bitwise_and_replays():
    out = _toy("toy")
    assert "mismatching calls 0" in out and "'replay': 5" in out and "'dead': 0" in out


@darwin
def test_a_new_shape_is_a_new_entry():
    assert "mismatching calls 0" in _toy("shapes")


@darwin
def test_a_region_that_is_not_static_is_caught_by_the_shadow_check():
    out = _toy("stateful")
    assert "mismatching calls 0" in out and "'dead': 1" in out


@darwin
def test_replays_queued_before_the_first_evaluation_get_their_own_icb():
    assert "mismatching calls 0" in _toy("inflight")


@darwin
def test_inputs_that_share_memory_are_not_recorded():
    assert "mismatching calls 0" in _toy("alias")
