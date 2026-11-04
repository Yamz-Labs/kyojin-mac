"""Codebook cross-check against PonyExl3's independent CPU codec. KYOJIN_PONY_ROOT=<PonyExl3 checkout>."""
import importlib.util, os

import numpy as np
import pytest

from kyojin_mlx import ref_numpy as R

root = os.environ.get("KYOJIN_PONY_ROOT")
if not root:
    pytest.skip("KYOJIN_PONY_ROOT not set", allow_module_level=True)


@pytest.mark.parametrize("cb,mode", [("mul1", 2), ("mcg", 1), ("3inst", 0)])
def test_codebooks_vs_pony(cb, mode):
    spec = importlib.util.spec_from_file_location("pony_codebook", os.path.join(root, "ponyexl3/ref/codebook.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    s = np.arange(0, 65536, 3, dtype=np.uint16)
    want = np.array([m.decode_3inst(int(v), mode) for v in s], np.float16)
    assert np.array_equal(R.codebook_values(s, cb).view(np.uint16), want.view(np.uint16))
