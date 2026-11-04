import os
os.environ.setdefault("KYOJIN_PF_IDX_HEADS", "0")  # exactness tests pin the exact first pass; the served default (8 heads) is gated by quality, test_pfheads covers it
import os
import numpy as np
import pytest

from kyojin_mlx import ref_numpy as R



@pytest.fixture(autouse=True)
def _fp32_rows(monkeypatch):
    """Bitwise tests check the fp32 row FMAs; the packed-fp16 rows (served default HALF_ROWS=4) have their own tests that set it."""
    from kyojin_mlx import metal_src as S
    monkeypatch.setattr(S, "HALF_ROWS", 0)


DATA = os.path.join(os.path.dirname(__file__), "..", "data", "real_slices.npz")


def cases():
    """(name, K, trellis, suh, svh): real mul1 slices from a GLM pack (K 2..5) plus random linears (K 1..4)."""
    out = []
    if os.path.exists(DATA):
        d = np.load(DATA)
        for K in (2, 3, 4, 5):
            out.append((f"real_K{K}", K, d[f"K{K}_trellis"], d[f"K{K}_suh"].view(np.float16), d[f"K{K}_svh"].view(np.float16)))
    for K in (1, 2, 3, 4):
        t, su, sv = R.random_linear(384, 256, K, seed=100 + K)
        out.append((f"random_K{K}", K, t, su, sv))
    return out


def pytest_generate_tests(metafunc):
    if "case" in metafunc.fixturenames:
        cs = cases()
        metafunc.parametrize("case", cs, ids=[c[0] for c in cs])
