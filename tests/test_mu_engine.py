"""Flag mu_tuner: the round plan of several greedy sessions on a backend whose verify pass is shared (engine._multi_tuner)."""
import pytest

from kyojin_mlx.serve.engine import Engine, mu_auto_on
from test_serve_engine import collect
from test_serve_spec import FakeSpec


class Auto(FakeSpec):
    """A backend that declares the shared verify pass itself (the Q35 path): the flag must not change its answer."""
    multi_auto = True


class Fixed(Auto):
    multi_auto = False


def test_mu_auto_on(monkeypatch):
    monkeypatch.delenv("KYOJIN_MU_TUNER", raising=False)
    assert mu_auto_on(Auto()) is True                       # declared by the backend
    assert mu_auto_on(Fixed()) is False                     # declared off: the flag does not override it
    assert mu_auto_on(FakeSpec()) is False                  # undeclared: the flag decides, off by default
    monkeypatch.setenv("KYOJIN_MU_TUNER", "1")
    assert mu_auto_on(FakeSpec()) is True
    assert mu_auto_on(FakeSpec(), greedy=False) is False    # the tuner measures greedy arms only
    assert mu_auto_on(Fixed()) is False


def _run(backend, n=2, mtp_depth=2, max_new=14):
    eng = Engine(backend, sessions=n, ctx=4096, chunk=4, block=8, prefix_bytes=0, mtp_depth=mtp_depth)
    eng.start()
    try:
        hs = [eng.submit([7 + 11 * i, 3, 5, 9 + i], max_new=max_new) for i in range(n)]
        out = [collect(h, timeout=60) for h in hs]
    finally:
        eng.stop()
    return eng, out


def test_flag_wires_the_tuner_on_an_undeclared_backend(monkeypatch):
    monkeypatch.setenv("KYOJIN_MU_TUNER", "1")
    eng, out = _run(FakeSpec())
    assert all(len(t) == 14 for t, _, _ in out)
    assert eng._tuner is not None
    assert eng.status()["tuner_ms"]                      # costs of the arms of N = 2 were measured
    assert set(eng._tuner.cost) == {2}
    assert all(0 <= a <= 2 for a in eng._tuner.cost[2])  # dmax = max_spec_rows // N - 1, mtp_depth = 2


def test_flag_off_keeps_the_solo_plan(monkeypatch):
    monkeypatch.delenv("KYOJIN_MU_TUNER", raising=False)
    eng, out = _run(FakeSpec())
    assert all(len(t) == 14 for t, _, _ in out)
    assert eng._tuner is None


def test_declared_backend_is_unchanged_by_the_flag(monkeypatch):
    monkeypatch.delenv("KYOJIN_MU_TUNER", raising=False)
    eng, _ = _run(Auto())
    assert eng._tuner is not None


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
