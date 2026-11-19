"""G2c rows path: group table + pair_glu + down against the G2a launches (pair gather, glu_rows, down gather) on a tiny synthetic NVFP4 expert block. Bit-identical."""
import types

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
from kyojin_mlx import foreign_linear as FL  # noqa: E402
from kyojin_mlx import glm_moe_rows as R  # noqa: E402

E, KIN, N, NEXP_PICK, LIMIT = 16, 1024, 1024, 4, 10.0


def mk(k, n, seed):
    rng = np.random.default_rng(seed)
    qs = mx.array(rng.integers(0, 256, size=(E, n, k // 2), dtype=np.uint8))
    mt = mx.array(rng.integers(0x20, 0x48, size=(E, n, k // 16), dtype=np.uint8))
    gs = mx.array(rng.uniform(0.001, 0.01, size=(E,)).astype(np.float32))
    return types.SimpleNamespace(qs=qs, meta=mt, gs=gs, k=k, n=n)


def ref(g, u, d, x2, inds, S, K):
    tok, ar = FL._slot_rows(S, K)
    kern = FL._g2a_kern()

    def one(dev, x, xr):
        return kern(inputs=[x, xr, inds, dev.qs, dev.meta, dev.gs], template=[("KIN", dev.k), ("N", dev.n), ("NBR", 1), ("SGN", FL.G2A_SGN)],
                    grid=(dev.n // FL.G2A_SGN * 32 * FL.G2A_SGN, S, 1), threadgroup=(32 * FL.G2A_SGN, 1, 1), output_shapes=[(S, dev.n)], output_dtypes=[mx.float16])[0]
    h = FL.glu_rows(mx.concatenate([one(g, x2, tok), one(u, x2, tok)], axis=0), S, g.n, LIMIT)
    return h, one(d, h, ar)


@pytest.mark.parametrize("T", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("overlap", [False, True])
def test_rows_match_g2a(T, overlap):
    K = NEXP_PICK
    S = T * K
    g, u, d = mk(KIN, N, 1), mk(KIN, N, 2), mk(N, KIN, 3)
    rng = np.random.default_rng(T)
    x2 = mx.array((rng.normal(size=(T, KIN)) * 0.5).astype(np.float16))
    pool = 6 if overlap else E
    inds = np.concatenate([rng.choice(pool, size=K, replace=False) for _ in range(T)]).astype(np.uint32)
    inds = mx.array(inds)
    h0, y0 = ref(g, u, d, x2, inds, S, K)
    tab = R.group(inds, T, K) if T > 1 else None
    h1 = R.pair_glu(g, u, x2, inds, tab, T, K, LIMIT)
    y1 = R.down(d, h1, inds, tab, T, K)
    mx.eval(h0, y0, h1, y1)
    assert mx.array_equal(h0, h1) and mx.array_equal(y0, y1)


def test_group_table():
    T, K = 3, 4
    inds = np.array([5, 1, 7, 2, 1, 5, 9, 3, 5, 2, 11, 0], np.uint32)
    tab = np.asarray(R.group(mx.array(inds), T, K)).reshape(T * K, 2 + T)
    seen = {}
    for p in range(T * K):
        ex, c = tab[p, 0], tab[p, 1]
        if c == 0:
            continue
        slots = list(tab[p, 2:2 + c])
        assert all(inds[s] == ex for s in slots) and slots == sorted(slots) and ex not in seen
        seen[ex] = slots
    assert sorted(seen) == sorted(set(inds.tolist()))
    assert sum(len(v) for v in seen.values()) == T * K
