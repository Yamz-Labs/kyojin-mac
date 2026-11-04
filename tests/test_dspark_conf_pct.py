"""dspark_conf_pct as a start value: KYOJIN_DSPARK_CONF_PCT in the launch environment sets the gate threshold (it was only reachable through POST /admin/flags)."""
import subprocess
import sys

import pytest

pytest.importorskip("mlx.core")

CODE = "import kyojin_mlx.dspark as D; print(D.CONF_OVERRIDE)"


def run(pct):
    env = {"PATH": "/usr/bin:/bin", "HOME": "/tmp"}
    if pct is not None:
        env["KYOJIN_DSPARK_CONF_PCT"] = pct
    return subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True, env=env, timeout=120).stdout.strip()


@pytest.mark.parametrize("pct,want", [(None, "None"), ("0", "None"), ("", "None"), ("35", "0.35")])
def test_start_value(pct, want):
    assert run(pct) == want


def test_refine_first_pass_is_the_stock_chain(monkeypatch):
    """KYOJIN_DSPARK_REFINE in the rows launch: pass 1 runs the stock (not teacher-forced) region, pass 2 the refine region on pass 1's tokens."""
    import mlx.core as mx
    from kyojin_mlx import dspark as D
    monkeypatch.setattr(D, "REFINE", True)
    dr = object.__new__(D.Drafter)
    dr.block, dr.noise = 3, 7
    calls = []

    def region(name, toks):
        def f(ids, pend, pos, *rings):
            calls.append((name, ids.reshape(-1).tolist()))
            return mx.array([toks]), mx.zeros((1, 3))
        return f
    dr._rows_region, dr._rows_region_r = region("stock", [4, 5, 6]), region("refine", [1, 2, 3])
    dr._rows_region_q = dr._rows_region_rq = None
    toks, conf, lg = dr._rows_launch([types_ctx()], [9], [0])
    assert calls == [("stock", [9, 7, 7]), ("refine", [9, 4, 5])]            # noise ids for pass 1, pass 1's proposals (shifted by the pending token) for pass 2
    assert toks.tolist() == [[1, 2, 3]] and lg is None


def types_ctx():
    import types
    return types.SimpleNamespace(rows=[])
