import sys, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import drill_mlx as D


def test_open_license_list_has_apache_and_not_noncommercial():
    assert "apache-2.0" in D.OPEN_LICENSES and "cc-by-nc-4.0" not in D.OPEN_LICENSES


def test_fetch_refuses_closed_license_before_download(monkeypatch, tmp_path):
    import huggingface_hub as H
    info = types.SimpleNamespace(tags=["license:cc-by-nc-4.0"], siblings=[], sha="x")
    monkeypatch.setattr(H.HfApi, "model_info", lambda self, repo, files_metadata=False: info)
    monkeypatch.setattr(H, "snapshot_download", lambda *a, **k: (_ for _ in ()).throw(AssertionError("downloaded")))
    try:
        D.fetch("x/y", tmp_path / "src", 10)
        assert False
    except SystemExit as e:
        assert "license" in str(e)


def test_fetch_refuses_when_disk_floor_would_be_crossed(monkeypatch, tmp_path):
    import huggingface_hub as H
    sib = types.SimpleNamespace(rfilename="m.safetensors", size=int(1e12))
    info = types.SimpleNamespace(tags=["license:apache-2.0"], siblings=[sib], sha="x")
    monkeypatch.setattr(H.HfApi, "model_info", lambda self, repo, files_metadata=False: info)
    monkeypatch.setattr(H, "snapshot_download", lambda *a, **k: (_ for _ in ()).throw(AssertionError("downloaded")))
    try:
        D.fetch("x/y", tmp_path / "src", 40)
        assert False
    except SystemExit as e:
        assert "floor" in str(e)


def _sens(errs, ka=5, km=4):
    return [dict(layer=i, err_attn=ea, err_mlp=em, params_attn=10, params_mlp=30, K_attn=ka, K_mlp=km) for i, (ea, em) in enumerate(errs)]


def test_ladder_raises_attention_of_worst_blocks_first_and_respects_step():
    s = _sens([(0.01, 0.02), (0.08, 0.05), (0.02, 0.03), (0.01, 0.01)])
    strat, raised = D.next_cells(s, k_max=6, step_frac=0.25, max_bpw=5.0)
    assert raised[0] == (1, "attn", 6) and strat["L1.attn"] == 6        # highest error per parameter first
    added = sum(10 if g == "attn" else 30 for _, g, _ in raised)
    assert added >= 0.25 * 160 and added < 0.25 * 160 + 30              # stops once +0.25 bpw is covered
    assert strat["L3.attn"] == 5 and strat["L3.mlp"] == 4               # untouched cells keep their bits


def test_ladder_size_cap_and_kmax():
    s = _sens([(0.05, 0.05)] * 2, ka=6, km=6)
    assert D.next_cells(s, 6, 0.25, 8.0) == (None, [])                  # everything at k_max
    s = _sens([(0.05, 0.05)] * 2, ka=5, km=4)
    assert D.body_bpw(s) == (5 * 10 + 4 * 30) / 40
    assert D.next_cells(s, 6, 0.25, 4.25)[0] is None                    # 4.25 bpw already at the cap: nothing fits


def test_parity_gate_matches_qualify_rules():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools/qualify"))
    import rules
    par = dict(kl_mean=0.2, top1_mean=0.7, by_domain={"a": dict(top1=0.9, kl=0.1), "b": dict(top1=0.6, kl=0.4)})
    assert D.parity_gate(par) == [r for r in rules.verdict({"parity": par})["reasons"]]
    par["kl_mean"] = 0.1; par["by_domain"]["b"]["top1"] = 0.8
    assert D.parity_gate(par) == [] == rules.verdict({"parity": par})["reasons"]
