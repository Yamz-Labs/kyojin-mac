"""Loader speed-ups (G1): the threaded stacking / pack builds are the serial result, bit for bit."""
import numpy as np

from kyojin_mlx import foreign_linear as FL
from kyojin_mlx.csf_scale import _csf32_blocks, _csf32_rows


def test_stack_par_equals_np_stack(monkeypatch):
    rng = np.random.default_rng(0)
    arrs = [rng.integers(0, 2 ** 32, (7, 5), dtype=np.uint32) for _ in range(37)]
    for n in ("1", "8"):
        monkeypatch.setenv("KYOJIN_LOAD_THREADS", n)
        assert np.array_equal(FL._stack_par(arrs), np.stack(arrs))


def test_pmap_keeps_order(monkeypatch):
    for n in ("1", "6"):
        monkeypatch.setenv("KYOJIN_LOAD_THREADS", n)
        assert FL.pmap(lambda x: x * x, range(100)) == [x * x for x in range(100)]


def test_csf32_rows_is_the_division():
    cells = _csf32_blocks(256, 16)[4]
    assert np.array_equal(_csf32_rows(256, 16), cells // 16)


def test_owned_index_matches_scan():
    info = {"a.b.weight": 1, "a.b.weight_scale": 1, "a.b.c.weight": 1, "a.b.csf32_bases": 1, "a.bb.weight": 1, "x.weight": 1}
    own = {}
    for k in info:
        if k.rsplit(".", 1)[-1] in FL._LIN_SFX:
            a = -1
            while (a := k.find(".", a + 1)) >= 0:
                own.setdefault(k[:a], []).append(k)
    for p in ("a", "a.b", "a.b.c", "a.bb", "x", "y"):
        want = sorted(k for k in info if k.startswith(p + ".") and k.rsplit(".", 1)[-1] in FL._LIN_SFX)
        assert sorted(own.get(p, ())) == want
