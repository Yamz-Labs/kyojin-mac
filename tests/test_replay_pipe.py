"""Flag replay_pipe (KYOJIN_REPLAY_PIPE): the pipelined cache replay of VlmBackend gives the bits of the synchronous loop (fakes, CPU-cheap)."""
import pytest

mx = pytest.importorskip("mlx.core")

from kyojin_mlx import flags
from kyojin_mlx.serve import backends as B


class Row:
    offset = 0


class Cache:
    def __init__(self):
        self.rows = [Row()]
        self.acc = mx.zeros((4,))

    def stacked_arrays(self):
        return [self.acc]


class PS:
    def __init__(self, bc, grp):
        self.bc, self.grp = bc, grp
        self.logits = mx.array(grp, dtype=mx.float32) * 2.0
        self.taps = {}

    def commit(self, acc):
        c = self.bc[0]
        c.acc = c.acc * 0.9 + mx.sum(self.logits) * mx.arange(4)          # order-dependent state
        c.rows[0].offset += len(self.grp)


class R:
    @staticmethod
    def verify(model, bc, lists):
        return PS(bc, lists[0])


class Fake(B.VlmBackend):
    supports_spec = True

    def __init__(self):
        self.R, self.model = R, None
        self.groups = []

    def batch(self, caches):
        return [caches[0]]

    def unbatch(self, bc, n):
        return [bc[0]]

    def _ring(self, d):
        return []


def run(monkeypatch, val, toks):
    monkeypatch.setenv("KYOJIN_REPLAY_PIPE", val)
    return Fake().replay(Cache(), toks)


@pytest.mark.parametrize("n", [1, 8, 9, 37, 130])
def test_pipelined_replay_matches_synchronous(monkeypatch, n):
    toks = [(7 * i + 3) % 101 for i in range(n)]
    a = run(monkeypatch, "0", toks)
    b = run(monkeypatch, "1", toks)
    assert a.rows[0].offset == b.rows[0].offset == n
    assert mx.array_equal(a.acc, b.acc).item()


def test_empty_replay(monkeypatch):
    assert run(monkeypatch, "1", []).rows[0].offset == 0


def test_flag_registered_default_on():
    f = flags.FLAGS["replay_pipe"]
    assert f.env == "KYOJIN_REPLAY_PIPE" and f.default == "1"
