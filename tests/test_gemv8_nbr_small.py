# SPDX-License-Identifier: Apache-2.0
"""Flag foreign_nbr_small: the 8-bit stored GEMV with one output row per simdgroup (more threadgroups) gives the same bits as NBR 2 (the kernel part is behind KYOJIN_METAL_TESTS=1)."""
import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from kyojin_mlx import flags as F
from kyojin_mlx import foreign_linear as FL, foreign_pack as FP

METAL = os.environ.get("KYOJIN_METAL_TESTS") == "1"


def test_flag_registered_and_selects():
    assert F.FLAGS["foreign_nbr_small"].runtime and F.FLAGS["foreign_nbr_small"].kind == "int"
    old = FL.NBR_SMALL

    class Fake:
        n, k = 1536, 3072
    try:
        FL.NBR_SMALL = 0
        assert FL.DevPack._nbr8(Fake) == FL.NBR
        FL.NBR_SMALL = 1
        assert FL.DevPack._nbr8(Fake) == 1
        Fake.n, Fake.k = 3072, 16384                                # 50M weights: the global NBR
        assert FL.DevPack._nbr8(Fake) == FL.NBR
        assert FL.DevPack._nbr8(Fake, 4) == 4                       # an explicit nbr wins
    finally:
        FL.NBR_SMALL = old


@pytest.mark.skipif(not METAL, reason="Metal run: KYOJIN_METAL_TESTS=1")
@pytest.mark.parametrize("n,k", [(1536, 3072), (3072, 1536), (1024, 3072)])
def test_nbr1_equals_nbr2(n, k):
    rng = np.random.default_rng(n + k)
    q = rng.integers(-127, 128, (n, k), dtype=np.int8)
    d = FP.Pack("q8_0", n, k, q.view(np.uint8), rng.uniform(1e-3, 3e-3, (n, k // 32)).astype(np.float16), np.ones(n, np.float32))
    dev = FL.DevPack(d)
    for R in (1, 4, 6):
        x = mx.array(rng.normal(0, 1, (R, k)).astype(np.float16))
        a, b = dev._gemv8(x, 2), dev._gemv8(x, 1)
        mx.eval(a, b)
        assert bool((a == b).all())
