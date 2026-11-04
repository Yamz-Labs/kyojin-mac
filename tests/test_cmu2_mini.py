"""cmu2: N sessions on the batched core, flag stack_dedupe 0 against 1. CPU, tiny random DS4.1 (mini_ds41).
(1) N = 1: the flag changes nothing (no layer released, stacked cache bit and byte identical): the solo path is untouched by construction.
(2) N = 2..4: bytes held by the stacked cache per session do not grow with the flag and shrink for the layers with rows; layers come back bit for bit.
The batched decode core is Metal only: this file stacks and writes back (the part the flag touches); logits of real steps are checked by tools/cmu2_mac.py."""
import os
import sys

import mlx.core as mx
import pytest

sys.path.insert(0, os.path.dirname(__file__))
pytest.importorskip("mlx_vlm.models.deepseek_v41")
import mini_ds41 as M  # noqa: E402
from kyojin_mlx import dsv4_core as DC, fp4pool as FP, rows as R  # noqa: E402
from kyojin_mlx.serve.backends import arrays_of  # noqa: E402


def nb(v):
    return sum(a.nbytes for a in arrays_of(v))


def stacked(flag, n, n0=320):
    os.environ["KYOJIN_STACK_DEDUPE"] = flag
    try:
        lm, _ = M.make(sliding_window=16)
        users = []
        for i in range(n):
            c = FP.wrap_cache(lm.make_cache())
            mx.eval(lm(mx.array([M.tokens(n0 + 16 * i, seed=1 + i)]), cache=c).logits, *c[0].state)
            users.append(c)
        bc = R.batch(users)
        st = DC.stack_rows(bc[0])
        mx.eval(*st.stacked_arrays())
        released = sum(1 for r in st.rows for x in r.compress if x is None)
        counted = nb(bc)
        dirty = st._dirty
        st.sync_rows()
        back = [[None if x is None else mx.array(x) for k in ("compress", "keys", "window") for x in getattr(r, k)] for r in st.rows]
        mx.eval(*[x for b in back for x in b if x is not None])
        return counted, released, dirty, back
    finally:
        os.environ.pop("KYOJIN_STACK_DEDUPE", None)


def test_solo_is_untouched_by_the_flag():
    c0, rel0, d0, b0 = stacked("0", 1)
    c1, rel1, d1, b1 = stacked("1", 1)
    assert c0 == c1 and rel0 == rel1 and d0 is False and d1 is False
    for x, y in zip(b0[0], b1[0]):
        assert (x is None) == (y is None) and (x is None or bool(mx.array_equal(x, y).item()))


@pytest.mark.parametrize("n", [2, 3, 4])
def test_bytes_per_session_and_roundtrip(n):
    c0, _, _, b0 = stacked("0", n)
    c1, rel1, dirty1, b1 = stacked("1", n)
    print(f"\nN={n}: counted {c0 / n:.0f} -> {c1 / n:.0f} B per session ({c1 / c0:.3f}x)")
    assert dirty1 and c1 < c0
    for u in range(n):
        for x, y in zip(b0[u], b1[u]):
            assert (x is None) == (y is None) and (x is None or bool(mx.array_equal(x, y).item()))
