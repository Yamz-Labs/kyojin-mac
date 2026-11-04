"""Loader checks (kyojin_mlx.packcheck): a pack that lacks what its descriptor promises is named, field by field. Header-only, CPU, tiny synthetic packs."""
import json
import os
import re
import shutil

import mlx.core as mx
import pytest

pytest.importorskip("mlx_vlm.models.deepseek_v41")
pytest.importorskip("mlx_lm")
from kyojin_mlx import archs as A, flags as FL, packcheck as PC, synth as S  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


@pytest.fixture(scope="module")
def packs(tmp_path_factory):
    out = {}
    for name in ("ds_moe_ced", "ds_moe_plain", "q35_dense"):
        p = tmp_path_factory.mktemp(name)
        S.write_pack(S.SHAPES[name], str(p))
        out[name] = str(p)
    return out


def fields(findings):
    return sorted({f.field for f in findings})


def variant(src, tmp_path, drop=None, config=None, tokenizer=None):
    """Copy of the pack with tensors removed (`drop(name)`), the config edited (`config(dict)`) or tokenizer.json edited."""
    d = str(tmp_path / "v")
    shutil.copytree(src, d)
    if drop:
        t = {k: v for k, v in mx.load(os.path.join(d, "model.safetensors")).items() if not drop(k)}
        mx.save_safetensors(os.path.join(d, "model.safetensors"), t)
    if config:
        c = json.load(open(os.path.join(d, "config.json")))
        config(c)
        json.dump(c, open(os.path.join(d, "config.json"), "w"))
    if tokenizer:
        tj = json.load(open(os.path.join(d, "tokenizer.json")))
        tokenizer(tj)
        json.dump(tj, open(os.path.join(d, "tokenizer.json"), "w"))
    return d


def test_clean_packs_have_no_findings(packs):
    assert PC.check(packs["ds_moe_ced"], ced=True) == []
    assert PC.check(packs["ds_moe_plain"]) == []
    assert PC.check(packs["q35_dense"]) == []


def test_missing_layer_is_named(packs, tmp_path):
    f = PC.check(variant(packs["ds_moe_ced"], tmp_path, drop=lambda k: ".layers.3." in k))
    assert fields(f) == ["layers"] and "[3]" in f[0].msg


def test_missing_experts_hc_indexer_are_named(packs, tmp_path):
    assert fields(PC.check(variant(packs["ds_moe_ced"], tmp_path, drop=lambda k: "switch_mlp" in k or "experts" in k))) == ["moe"]
    shutil.rmtree(tmp_path / "v")
    assert fields(PC.check(variant(packs["ds_moe_ced"], tmp_path, drop=lambda k: "hc_" in k))) == ["hc"]
    shutil.rmtree(tmp_path / "v")
    assert fields(PC.check(variant(packs["ds_moe_ced"], tmp_path, drop=lambda k: "indexer" in k))) == ["index"]


def test_config_promises_without_tensors(packs, tmp_path):
    f = PC.check(variant(packs["ds_moe_ced"], tmp_path, config=lambda c: c["text_config"].update(engram_layer_ids=[1])))
    assert fields(f) == ["engram"] and "[1]" in f[0].msg


def test_dense_row_with_experts_and_moe_row_without(packs, tmp_path):
    # a Qwen pack whose config grows an expert count: the dense promise of the pack is broken only for rows that say moe="no"; Qwen says "config"
    assert PC.check(variant(packs["q35_dense"], tmp_path, config=lambda c: c["text_config"].update(num_experts=4))) == []
    shutil.rmtree(tmp_path / "v")
    f = PC.check(variant(packs["ds_moe_ced"], tmp_path, config=lambda c: [c["text_config"].pop(k, None) for k in ("n_routed_experts",)]))
    assert "moe" in fields(f)


def test_draft_head_requested_without_tensors_or_keys(packs, tmp_path):
    f = PC.check(packs["ds_moe_ced"], mtp=True)
    assert fields(f) == ["mtp"] and any("mtp." in x.msg for x in f) and any("MtpSpec.keys" in x.msg for x in f)
    # tensors present, config keys absent
    t = dict(mx.load(os.path.join(packs["ds_moe_ced"], "model.safetensors")))
    t["mtp.0.main_proj.weight"] = mx.zeros((2, 2))
    d = str(tmp_path / "w")
    shutil.copytree(packs["ds_moe_ced"], d)
    mx.save_safetensors(os.path.join(d, "model.safetensors"), t)
    f = PC.check(d, mtp=True)
    assert fields(f) == ["mtp"] and not any("no tensors under" in x.msg for x in f)


def test_ced_promise_is_checked_against_the_layout(packs):
    f = PC.check(packs["ds_moe_plain"], ced=True)
    assert fields(f) == ["ced"] and "no even split" in f[0].msg
    assert PC.check(packs["ds_moe_plain"], ced=False) == []                      # not asked for: no promise to break


def test_ced_spec_mid_is_honoured_by_the_check(packs, monkeypatch):
    # an explicit midpoint in the CedSpec lifts the even-split rule; layer 3 of the 7 is a KV+index source only if the config says so
    monkeypatch.setattr(A.DS41, "ced", A.CedSpec(adapter="dsv4", mid=3))
    f = PC.check(packs["ds_moe_plain"], ced=True)
    assert f and "no even split" not in f[0].msg


def test_vendor_class_missing_is_named(packs, monkeypatch):
    monkeypatch.setattr(A.DS41, "vendor", A.Vendor(module=A.DS41.vendor.module, block="NoSuchBlock", cache="DeepseekV41Cache"))
    f = PC.check(packs["ds_moe_ced"])
    assert fields(f) == ["vendor"] and "NoSuchBlock" in f[0].msg
    monkeypatch.setattr(A.DS41, "vendor", A.Vendor(module="mlx_vlm.models.no_such_model.language", block="B"))
    assert "cannot import" in PC.check(packs["ds_moe_ced"])[0].msg


def test_tokenizer_stop_token_and_template(packs, tmp_path):
    f = PC.check(variant(packs["q35_dense"], tmp_path, tokenizer=lambda tj: tj.update(added_tokens=[])))
    assert fields(f) == ["stop_tokens"] and "<|im_end|>" in f[0].msg
    shutil.rmtree(tmp_path / "v")
    d = variant(packs["q35_dense"], tmp_path)
    os.remove(os.path.join(d, "tokenizer.json"))
    os.remove(os.path.join(d, "tokenizer_config.json"))
    assert fields(PC.check(d)) == ["template", "tokenizer"]
    # the DS4.1 row names a builtin template and its own tokenizer dir, so a pack without either is still complete for it
    d2 = variant(packs["ds_moe_ced"], tmp_path / "x")
    os.remove(os.path.join(d2, "tokenizer.json"))
    os.remove(os.path.join(d2, "tokenizer_config.json"))
    assert PC.check(d2) == []


def test_unknown_model_and_missing_geometry_keys(packs, tmp_path):
    f = PC.check(variant(packs["q35_dense"], tmp_path, config=lambda c: c.update(model_type="acme_v9", architectures=["AcmeForCausalLM"],
                                                                               text_config={**c["text_config"], "model_type": "acme_v9"})))
    assert fields(f) == ["row"] and "add a row" in f[0].msg
    shutil.rmtree(tmp_path / "v")
    f = PC.check(variant(packs["q35_dense"], tmp_path, config=lambda c: c["text_config"].pop("num_hidden_layers")))
    assert fields(f) == ["config"]


def test_enforce_strict_raises_with_every_finding_and_default_logs(packs, monkeypatch):
    findings = PC.check(packs["ds_moe_plain"], mtp=True, ced=True)
    assert len(fields(findings)) == 2
    lines = []
    monkeypatch.delenv("KYOJIN_DESC_STRICT", raising=False)
    assert PC.strict() is False and PC.enforce(findings, log=lines.append) == findings and len(lines) == len(findings)
    assert all("KYOJIN_DESC_STRICT=0" in l for l in lines)
    monkeypatch.setenv("KYOJIN_DESC_STRICT", "1")
    with pytest.raises(PC.PackError) as e:
        PC.enforce(findings, log=lines.append)
    assert "[ced]" in str(e.value) and "[mtp]" in str(e.value)
    assert PC.enforce([], log=lines.append) == []                                # nothing to raise on


def test_flag_default_literal_and_levers_row():
    f = FL.FLAGS["desc_strict"]
    assert (f.env, f.default, f.kind, f.runtime) == ("KYOJIN_DESC_STRICT", "0", "bool", False) and f.reason
    assert PC.STRICT_DEFAULT == f.default
    assert re.search(r'os\.environ\.get\("KYOJIN_DESC_STRICT", STRICT_DEFAULT\)', open(os.path.join(ROOT, "kyojin_mlx", "packcheck.py")).read())
    rows = [l for l in open(os.path.join(ROOT, "docs", "cloud", "LEVERS.md")).read().splitlines() if l.startswith("| `KYOJIN_DESC_STRICT`")]
    assert len(rows) == 1 and "| desc_strict |" in rows[0] and "| 0 |" in rows[0]


def test_loaders_call_the_check_before_building_anything():
    src = open(os.path.join(ROOT, "kyojin_mlx", "serve", "backends.py")).read()
    assert re.search(r"def load_vlm\(.*?packcheck\.check_pack\(path, mtp=mtp, ced=ced, log=log\).*?P\.build_model\(path\)", src, re.S)
    assert re.search(r"def load_q35\(.*?packcheck\.check_pack\(path, log=log\)", src, re.S)
