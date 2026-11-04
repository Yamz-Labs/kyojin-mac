"""Per-(device, shape) tuning table (kyojin_mlx/tuning.py): file handling, and that an absent / empty entry leaves the Exl3Linear bits untouched while a tuned
entry changes only the launch geometry (k-splits round at 1e-6 relative, the NAX tile at fp16 matmul level)."""
import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

from kyojin_mlx import exl3_linear as EL
from kyojin_mlx import tuning as TN


@pytest.fixture(autouse=True)
def clean():
    TN.TABLE.clear()
    yield
    TN.TABLE.clear()


def lin(kin=512, nout=1024, K=3, tune=None, seed=0):
    rng = np.random.default_rng(seed)
    tr = mx.array(rng.integers(-32768, 32768, (kin // 16, nout // 16, 16 * K), dtype=np.int16))
    suh = mx.array(rng.choice([-1.0, 1.0], kin).astype(np.float16))
    svh = mx.array((rng.standard_normal(nout) * 0.05).astype(np.float16))
    return EL.Exl3Linear(tr, suh, svh, cb="mcg", tune=tune)


def x_of(rows, kin=512):
    return mx.array(np.random.default_rng(1).standard_normal((1, rows, kin)).astype(np.float16))


def test_install_status_and_stale(tmp_path):
    assert TN.install(tmp_path) == "none"
    TN.write(tmp_path, {"512x1024x3": {"tgk": 2}}, dict(pack="p"))
    assert TN.install(tmp_path) == "loaded" and TN.lookup(512, 1024, 3) == {"tgk": 2} and TN.lookup(512, 1024, 2) == {}
    f = TN.path_for(tmp_path)
    d = json.loads(f.read_text())
    d["kernel_version"] = "0" * 12
    f.write_text(json.dumps(d))
    assert TN.install(tmp_path) == "stale" and TN.lookup(512, 1024, 3) == {}
    d["kernel_version"] = TN.kernel_version()
    f.write_text(json.dumps(d))
    assert TN.install(tmp_path) == "loaded"


def test_env_switch_off(tmp_path, monkeypatch):
    TN.write(tmp_path, {"512x1024x3": {"tgk": 2}}, dict(pack="p"))
    monkeypatch.setenv("KYOJIN_TUNING", "0")
    assert TN.install(tmp_path) == "off" and TN.lookup(512, 1024, 3) == {}


def test_cfg_for_takes_the_next_tuned_row_count():
    e = {"nax_cfg": {"32": [16, 1, 32, 64], "512": [32, 2, 32, 64]}}
    assert TN.cfg_for(e, 8) == (16, 1, 32, 64) and TN.cfg_for(e, 32) == (16, 1, 32, 64)
    assert TN.cfg_for(e, 33) == (32, 2, 32, 64) and TN.cfg_for(e, 4096) == (32, 2, 32, 64) and TN.cfg_for({}, 8) is None


def test_table_entry_is_found_by_the_module(tmp_path):
    TN.write(tmp_path, {"512x1024x3": {"tgk": 2, "fused_max": 64}}, dict(pack="p"))
    TN.install(tmp_path)
    m = lin()
    assert m.tune == {"tgk": 2, "fused_max": 64} and m.fused_max == 64 and m._g[1].tgk == 2
    assert lin(kin=512, nout=2048).tune == {} and lin(kin=512, nout=2048).fused_max == TN.chip().get("dense_fused_max", EL.DENSE_FUSED_MAX)


def test_empty_entry_is_bit_identical():
    for rows in (1, 3, 16, 300):
        a, b = lin(), lin(tune={})
        assert (a(x_of(rows)) == b(x_of(rows))).all().item()


def test_tuned_tgk_changes_only_the_rounding_of_the_k_sum():
    base = lin()
    other = 4 if base._g[1].tgk != 4 else 2
    t = lin(tune={"tgk": other})
    assert t._g[1].tgk == other
    for rows in (1, 4):
        y0, y1 = base(x_of(rows)).astype(mx.float32), t(x_of(rows)).astype(mx.float32)
        assert float(mx.max(mx.abs(y0 - y1)) / mx.max(mx.abs(y0))) < 2e-3


def test_tuned_fused_max_and_tile_route_the_rows():
    if not EL.NX.nax_available():
        pytest.skip("no neural accelerators")
    base = lin()
    dense = lin(tune={"fused_max": 0})
    tile = lin(tune={"nax_cfg": {"64": [16, 1, 32, 64]}})
    x = x_of(64)
    y0 = base(x).astype(mx.float32)
    assert not dense._fused_ok(64) and base._fused_ok(64) and tile._nax_rows == {}
    for m in (dense, tile):
        y = m(x).astype(mx.float32)
        assert float(mx.max(mx.abs(y0 - y)) / mx.max(mx.abs(y0))) < 5e-3
    assert tile._nax_rows[64][1] == (16, 1, 32, 64)


def test_autotune_entry_from_measurements():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "qualify"))
    import autotune_mlx as AT

    def row(pick, dflt="fused64"):
        return dict(default=dflt, pick=pick)
    dec = dict(default=4, pick=8)
    pf = {"32": row("fused16", "fused32"), "128": row("fused64"), "512": row("fused64"), "1024": row("dense", "fused64"), "2048": row("dense", "dense")}
    e = AT.entry_from(dec, pf)
    assert e["tgk"] == 8 and e["fused_max"] == 512 and e["nax_cfg"] == {"32": [16, 1, 32, 64], "128": [32, 2, 32, 64], "512": [32, 2, 32, 64]}
    assert AT.changes_of(e) == ["tgk", "fused_max=512", "cfg@32"]
    # no crossover on a grid that stops below the constant (the wide head): the constant stands, no tgk change = no key
    e2 = AT.entry_from(dict(default=1, pick=1), {"8": row("fused16", "fused16"), "256": row("fused64")})
    assert "fused_max" not in e2 and "tgk" not in e2
    # fused wins up to the constant: the table states it
    assert AT.entry_from(dec, {"1024": row("fused64")})["fused_max"] == 1024


def test_mma_max_per_shape_and_chip_defaults(monkeypatch):
    c = TN.chip()
    assert c["mma_max"] == 8 and c["nax_ks_weights"] > 0 and c["nax_ks_tgs"] > c["nax_ks_few"]
    base = lin(K=4)
    assert base._mma_on(8) and not base._mma_on(9) and not base._mma_on(4)       # chip default: rows 5..8
    wide = lin(K=4, tune={"mma_max": 12})
    off = lin(K=4, tune={"mma_max": 4})
    assert wide._mma_on(12) and not wide._mma_on(13) and not off._mma_on(5)
    monkeypatch.setattr(EL, "MMA_MAX", 6)                                         # the module override (env KYOJIN_MMA_MAX) beats the entry
    assert wide._mma_on(6) and not wide._mma_on(7)


def test_chip_defaults_scale_with_cores(monkeypatch):
    monkeypatch.setattr(TN, "_chip", None)
    monkeypatch.setenv("KYOJIN_GPU_CORES", "80")
    c = TN.chip()
    monkeypatch.setattr(TN, "_chip", None)
    assert c["nax_ks_tgs"] == 1024 and c["nax_ks_few"] == 256 and c["nax_ks_weights"] == 128_000_000 and c["cores"] == 80


def test_autotune_mma_max_from_measurements():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "qualify"))
    import autotune_mlx as AT
    mm = lambda pick, dflt: dict(default="fused16", pick="fused16", mma=dict(default=dflt, pick=pick))
    # chip default (mma up to 8): measured as the default everywhere -> no key; MMA also wins at 12 -> 12; loses at 5 -> off
    assert "mma_max" not in AT.entry_from(dict(default=1, pick=1), {"5": mm("mma", "mma"), "8": mm("mma", "mma"), "12": mm("nax", "nax")})
    assert AT.entry_from(dict(default=1, pick=1), {"5": mm("mma", "mma"), "8": mm("mma", "mma"), "12": mm("mma", "nax")})["mma_max"] == 12
    assert AT.entry_from(dict(default=1, pick=1), {"5": mm("nax", "mma"), "8": mm("nax", "mma")})["mma_max"] == 4


def test_prefill_entry_and_seam_thresholds(monkeypatch):
    from kyojin_mlx import layer_pf as LP, mlp_pf as MP
    assert TN.pf("layer_max_rows", 1024) == 1024 and LP.max_rows() == 1024 and LP.enabled() and MP.enabled() and MP.min_rows() == 64
    TN.TABLE["prefill"] = {"layer_max_rows": 2048, "layer_split_max_rows": 4096, "mlp_pf": 0, "mlp_min_rows": 128, "layer_pf": 0, "layer_min_rows": 32}
    assert LP.max_rows() == 2048 and LP.split_max_rows() == 4096 and not MP.enabled() and MP.min_rows() == 128 and not LP.enabled() and LP.min_rows() == 32
    monkeypatch.setattr(LP, "MAX_ROWS", 512)                                     # a module attribute (tools) or the env var beats the table
    assert LP.max_rows() == 512
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "qualify"))
    import autotune_mlx as AT
    T = lambda off, mlp, layer, split: dict(off=off, mlp=mlp, layer=layer, split=split)
    # layer seams win up to 1024 and lose at 2048, split matches there: the Air's shape
    air = {256: T(1.0, .98, .97, .97), 512: T(2.0, 1.96, 1.94, 1.94), 1024: T(4.1, 4.0, 3.95, 3.97), 2048: T(8.0, 7.9, 8.6, 7.9)}
    assert AT.seam_thresholds(air) == {"layer_max_rows": 1024}
    split_wins = dict(air); split_wins[2048] = T(8.0, 7.9, 8.6, 7.5)
    assert AT.seam_thresholds(split_wins) == {"layer_max_rows": 1024, "layer_split_max_rows": 2048}
    assert AT.seam_thresholds({r: T(1, 1.1, 1.1, 1.1) for r in (256, 512)}) == {"mlp_pf": 0, "layer_pf": 0}      # the MLP middle loses everywhere
    assert AT.seam_thresholds({256: T(1, 1.2, 1.2, 1.2), 512: T(2, 1.9, 1.9, 1.9), 1024: T(4, 3.8, 3.8, 3.8)}) == {"mlp_min_rows": 512, "layer_max_rows": 1024}


def test_serve_entry_merge_keeps_the_shapes(tmp_path):
    """set_entry adds the server priors next to the autotuned shapes; serve() reads them after install, constants without a table."""
    TN.TABLE.clear()
    assert TN.serve("row_cost", 0.4) == 0.4
    TN.write(tmp_path, {"2048x512x3": {"mma_max": 8}}, {"note": "x"})
    TN.set_entry(tmp_path, "serve", {"row_cost": 0.33, "draft_cost": 0.2})
    TN.set_entry(tmp_path, "serve", {"draft_cost": 0.18})
    assert TN.install(tmp_path) == "loaded"
    assert TN.lookup(2048, 512, 3) == {"mma_max": 8}
    assert (TN.serve("row_cost", 0.4), TN.serve("draft_cost", 0.15)) == (0.33, 0.18)
    assert json.loads(TN.path_for(tmp_path).read_text())["note"] == "x"
    TN.TABLE.clear()


def test_serve_entry_keeps_the_spec_table(tmp_path, monkeypatch):
    """set_entry next to the MTP round cost table (write_spec): the spec table survives, also when a stale kernel version resets the shapes."""
    monkeypatch.delenv("KYOJIN_SPEC_COSTS", raising=False)
    TN.write(tmp_path, {"2048x512x3": {"mma_max": 8}}, {})
    TN.write_spec(tmp_path, "fp16", 4096, {"round": [10.0, 12.0]})
    TN.set_entry(tmp_path, "serve", {"row_cost": 0.3})
    f = TN.path_for(tmp_path)
    d = json.loads(f.read_text())
    assert d["spec"]["fp16"]["4096"]["round"] == [10.0, 12.0] and d["shapes"]["2048x512x3"] == {"mma_max": 8}
    d["kernel_version"] = "0" * 12
    f.write_text(json.dumps(d))
    TN.set_entry(tmp_path, "serve", {"draft_cost": 0.2})
    d = json.loads(f.read_text())
    assert d["shapes"] == {"serve": {"draft_cost": 0.2}} and d["spec"]["fp16"]["4096"]["round"] == [10.0, 12.0]
    monkeypatch.setenv("KYOJIN_SPEC_COSTS", "1")
    assert TN.spec_costs(tmp_path) == {4096: [10.0, 12.0]}
    TN.TABLE.clear()
