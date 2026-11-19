# SPDX-License-Identifier: Apache-2.0
"""moe_union: the union expert kernels (each distinct expert read once for all the rows that chose it) equal the per-slot gather path, bit for bit, for K-family
and legacy-family stored experts, T = 2..8 rows with overlapping routing; the device-side tables match a host reference."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
from kyojin_mlx import foreign_gguf as GF, foreign_linear as FL, moe_union as MU  # noqa: E402

BS = {"Q3_K": (110, 108), "Q4_K": (144, 0), "Q6_K": (210, 208)}
H = F = 512
E = 24
K = 6


def stack(tname, n, k, rng):
    bs, nb = BS[tname]
    row = k // 256 * bs
    packs = []
    for _ in range(E):
        raw = rng.integers(0, 256, (n, row), dtype=np.uint8)
        for b in range(k // 256):
            o = b * bs + nb
            if tname == "Q3_K":
                raw[:, o:o + 2] = np.array([0x00, 0x1c], np.uint8)
            elif tname == "Q4_K":
                raw[:, o:o + 2] = np.array([0x00, 0x1c], np.uint8)
                raw[:, o + 2:o + 4] = np.array([0x00, 0x18], np.uint8)
            else:
                raw[:, o:o + 2] = np.array([0x00, 0x1c], np.uint8)
        packs.append(GF.to_pack(tname, raw, k))
    return FL.DevPack(packs)


def host_tables(inds, R):
    T, Kk = inds.shape
    S = T * Kk
    order = []
    for e in inds.reshape(-1):
        if int(e) not in order:
            order.append(int(e))
    groups = []
    for e in order:
        rows = [r * Kk + k for r in range(T) for k in range(Kk) if inds[r, k] == e]
        for c in range(0, len(rows), R):
            groups.append((e, rows[c:c + R]))
    eids = np.zeros(S, np.uint32)
    osl = np.full((S, R), 0xFFFFFFFF, np.uint32)
    for g, (e, rows) in enumerate(groups):
        eids[g] = e
        osl[g, :len(rows)] = rows
    return eids, osl.reshape(-1), len(groups)


@pytest.mark.parametrize("R", [1, 2, 3, 8])
@pytest.mark.parametrize("T", [2, 5, 6, 8])
def test_tables(T, R):
    rng = np.random.default_rng(T)
    inds = np.stack([rng.choice(E, K, replace=False) for _ in range(T)]).astype(np.uint32)
    eu, osl = MU.tables(mx.array(inds), T, K, R)
    he, ho, n = host_tables(inds, R)
    assert np.array_equal(np.array(eu)[:n], he[:n]) and np.array_equal(np.array(osl), ho)


@pytest.mark.parametrize("types", [("Q3_K", "Q3_K", "Q4_K"), ("Q3_K", "Q3_K", "Q6_K"), ("Q4_K", "Q4_K", "Q4_K")])
@pytest.mark.parametrize("R", [1, 2, 3, 8])
@pytest.mark.parametrize("T", [2, 6, 8])
def test_union_equals_gather(types, T, R, monkeypatch):
    monkeypatch.setenv("KYOJIN_MOE_UNION_R", str(R))
    rng = np.random.default_rng(7)
    sw = FL.ForeignSwitchGLU(stack(types[0], F, H, rng), stack(types[1], F, H, rng), stack(types[2], H, F, rng))
    x = mx.array((rng.standard_normal((T, 1, H)) * 0.5).astype(np.float32)).astype(mx.float16)
    inds = np.stack([rng.choice(E, K, replace=False) for _ in range(T)])[:, None, :].astype(np.uint32)
    old = (FL.GATEUP_PAIR, MU.UNION)
    try:
        FL.GATEUP_PAIR = True
        MU.UNION = False
        ref = sw(x, mx.array(inds))
        MU.UNION = True
        n0 = MU.PATHS["union"]
        got = sw(x, mx.array(inds))
        mx.eval(ref, got)
        assert MU.PATHS["union"] == n0 + 1, "the union path did not run"
    finally:
        FL.GATEUP_PAIR, MU.UNION = old
    a, b = np.array(ref).astype(np.float32), np.array(got).astype(np.float32)
    assert np.array_equal(a, b), f"max |diff| {np.abs(a - b).max()}"
