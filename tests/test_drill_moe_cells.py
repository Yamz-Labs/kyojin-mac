import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import drill_mlx as D


def _moe(errs):
    return [dict(layer=i, err_attn=ea, err_mlp=em, err_exp=ee, params_attn=10, params_mlp=5, params_exp=100, K_attn=5, K_mlp=4, K_exp=3) for i, (ea, em, ee) in enumerate(errs)]


def test_moe_blocks_get_a_third_cell_and_its_bits_count():
    s = _moe([(0.01, 0.01, 0.2), (0.02, 0.01, 0.9)])
    assert D.cells_of(s[0]) == ("attn", "mlp", "exp") and D.body_bpw(s) == (5 * 10 + 4 * 5 + 3 * 100) / 115
    strat, raised = D.next_cells(s, k_max=6, step_frac=0.3, max_bpw=5.0)
    assert strat["L1.exp"] == 4 and strat["L0.exp"] == 3 and (1, "exp", 4) in raised         # the worst expert cell per parameter is raised; every cell is in the strategy
    assert set(strat) == {f"L{i}.{g}" for i in (0, 1) for g in ("attn", "mlp", "exp")}


def test_dense_blocks_keep_two_cells():
    s = [dict(layer=0, err_attn=0.1, err_mlp=0.2, params_attn=10, params_mlp=30, K_attn=5, K_mlp=4)]
    assert D.cells_of(s[0]) == ("attn", "mlp") and D.body_bpw(s) == (50 + 120) / 40
