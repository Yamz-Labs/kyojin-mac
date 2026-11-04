"""q4mma (few-row affine 4-bit GEMV on simdgroup_matrix) against MLX's quantized_matmul: any (N, K) the Q4Linear accepts, any row count 1..8, fp16 and bf16.
Gate: the output of a row does not depend on how many rows ride along, and stays within rounding of MLX qmv / qmm."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")

# (N, K) of dense projections of several models: Qwen3.5-4B (2560 x 9216 / 9216 x 2560), 7B class (4096 x 4096, 4096 x 14336), small head (1536 x 2048)
SHAPES = [(2560, 9216), (9216, 2560), (4096, 4096), (14336, 4096), (2048, 1536 * 2)]


def make(N, K, dtype, seed=0):
    from kyojin_mlx import q4mma as QM
    r = np.random.default_rng(seed)
    lin = nn.Linear(K, N, bias=False)
    lin.weight = mx.array((r.standard_normal((N, K)) * 0.05).astype(np.float32)).astype(dtype)
    q = nn.QuantizedLinear.from_linear(lin, group_size=64, bits=4)
    mx.eval(q.parameters())
    return q, QM.Q4Linear(q)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("N,K", SHAPES)
def test_rows_match_mlx_and_are_row_invariant(N, K, dtype):
    from kyojin_mlx import q4mma as QM
    q, f = make(N, K, dtype)
    r = np.random.default_rng(1)
    x = mx.array(r.standard_normal((1, 8, K)).astype(np.float32)).astype(dtype)
    ref = q(x).astype(mx.float32)
    old = QM.MIN_ROWS, list(QM._NAX_ON)
    QM.MIN_ROWS = 1
    QM._NAX_ON[:] = [False]
    try:
        full = f(x).astype(mx.float32)
        mx.eval(full, ref)
        scale = float(mx.abs(ref).max())
        assert float(mx.abs(full - ref).max()) <= 2e-2 * scale
        for R in (1, 2, 3, 5, 8):
            part = f(x[:, :R]).astype(mx.float32)
            mx.eval(part)
            assert np.array_equal(np.array(part), np.array(full[:, :R])), f"row result depends on the row count (R={R})"
    finally:
        QM.MIN_ROWS, QM._NAX_ON[:] = old


def test_tuning_table_is_keyed_by_chip(tmp_path):
    import json
    from kyojin_mlx import q4mma as Q
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"devices": {Q.device_name(): {"min_rows": {"100x512": 3}}, "Some Other Chip": {"min_rows": {"100x512": 8}}}}))
    try:
        assert Q.load_tuning(str(p)) == 1
        assert Q.min_rows_for(100, 512) == 3 and Q.min_rows_for(7, 512) == Q.MIN_ROWS
        p.write_text(json.dumps({"devices": {"Some Other Chip": {"min_rows": {"100x512": 8}}}}))
        assert Q.load_tuning(str(p)) == 0 and Q.min_rows_for(100, 512) == Q.MIN_ROWS
    finally:
        Q.load_tuning()


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("N,K", SHAPES)
def test_sixteen_rows_match_mlx_and_are_row_invariant(N, K, dtype):
    """9..16 rows ride two B tiles of the same decoded weights: every row equals its value in a 1-row or 8-row call (bit for bit)."""
    from kyojin_mlx import q4mma as QM
    q, f = make(N, K, dtype)
    r = np.random.default_rng(2)
    x = mx.array(r.standard_normal((16, 1, K)).astype(np.float32)).astype(dtype)
    ref = q(x).astype(mx.float32)
    old = QM.MIN_ROWS, QM._ENV_MAX, list(QM._NAX_ON)
    QM.load_tuning("/nonexistent")
    QM.MIN_ROWS, QM._ENV_MAX = 1, "16"
    QM._NAX_ON[:] = [False]
    try:
        full = f(x).astype(mx.float32)
        mx.eval(full, ref)
        assert float(mx.abs(full - ref).max()) <= 2e-2 * float(mx.abs(ref).max())
        for R in (1, 5, 8, 9, 12, 16):
            part = f(x[:R]).astype(mx.float32)
            mx.eval(part)
            assert np.array_equal(np.array(part), np.array(full[:R])), f"row result depends on the row count (R={R})"
    finally:
        QM.MIN_ROWS, QM._ENV_MAX, QM._NAX_ON[:] = old
        QM.load_tuning()


def test_tuning_row_window(tmp_path):
    import json
    from kyojin_mlx import q4mma as Q
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"devices": {Q.device_name(): {"min_rows": {"100x512": [4, 12], "32x512": [17, 16], "64x512": 5}}}}))
    old = Q._ENV_MAX
    try:
        Q._ENV_MAX = None
        assert Q.load_tuning(str(p)) == 3
        assert [Q.serves(100, 512, R) for R in (3, 4, 12, 13)] == [False, True, True, False]
        assert not any(Q.serves(32, 512, R) for R in range(1, 17))
        assert Q.rows_for(64, 512) == (5, 8) and Q.rows_for(7, 512) == (Q.MIN_ROWS, Q.DEFAULT_MAX)   # legacy int = 8-row cap; untuned = defaults
        Q._ENV_MAX = "8"
        assert Q.rows_for(100, 512) == (4, 8) and Q.rows_for(7, 512) == (Q.MIN_ROWS, 8)
    finally:
        Q._ENV_MAX = old
        Q.load_tuning()


@pytest.mark.parametrize("cfg", [(2, 2, 1), (2, 4, 1), (1, 2, 2), (2, 2, 2)])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("N,K", SHAPES)
def test_nax_rows_match_mlx_and_are_row_invariant(N, K, dtype, cfg):
    """q4nax (weights dequantized once into the NAX matmul right input) for 1..32 rows: within rounding of MLX qmm, and inside one M tile
    (1..16 rows, 17..32 rows) a row does not depend on how many rows ride along."""
    from kyojin_mlx import q4mma as QM
    if not QM.nax_on():
        pytest.skip("no neural accelerators")
    q, _ = make(N, K, dtype)
    r = np.random.default_rng(3)
    x = mx.array(r.standard_normal((32, K)).astype(np.float32)).astype(dtype)
    ref = q(x).astype(mx.float32)
    run = lambda R: QM.q4nax(QM.prep_xn(x[:R], cfg[0]), q.weight, q.scales, q.biases, R, *cfg).astype(mx.float32)
    full, half = run(32), run(16)
    mx.eval(full, half, ref)
    scale = float(mx.abs(ref).max())
    assert float(mx.abs(full - ref).max()) <= 2e-2 * scale and float(mx.abs(half - ref[:16]).max()) <= 2e-2 * scale
    for R, base in ((1, half), (9, half), (17, full), (24, full)):
        part = run(R)
        mx.eval(part)
        assert np.array_equal(np.array(part), np.array(base[:R])), f"row result depends on the row count (R={R})"


def test_nax_tuning_window(tmp_path):
    import json
    from kyojin_mlx import q4mma as Q
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"devices": {Q.device_name(): {"min_rows": {"256x512": [4, 8]}, "nax": {"256x512": [6, 20, 2, 4, 1], "64x512": [40, 32, 2, 2, 1]}}}}))
    old = list(Q._NAX_ON)
    try:
        Q.load_tuning(str(p))
        Q._NAX_ON[:] = [True]
        assert [Q.nax_cfg(256, 512, R) for R in (5, 6, 20, 21)] == [None, (2, 4, 1), (2, 4, 1), None]
        assert not any(Q.nax_cfg(64, 512, R) for R in range(1, 33))
        assert Q.nax_cfg(96, 512, Q.NAX_DEFAULT[0]) == Q.NAX_DEFAULT[2:] and Q.nax_cfg(96, 512, 33) is None and Q.nax_cfg(100, 512, 12) is None
        Q._NAX_ON[:] = [False]
        assert Q.nax_cfg(256, 512, 8) is None
    finally:
        Q._NAX_ON[:] = old
        Q.load_tuning()
