"""Descriptor fields that replaced DS4.1 literals: each one equals the frozen literal it replaced (so the DS4.1 code path is unchanged), and no
DS4.1 class / module literal is left in executable code outside archs.py / modeldesc.py. CPU only, no Metal."""
import ast
import glob
import os
import types

import pytest

from kyojin_mlx import archs as A, ced as C, modeldesc as MD

ROOT = os.path.join(os.path.dirname(__file__), "..", "kyojin_mlx")


def test_ds41_values_are_the_replaced_literals():
    d = MD.by_model_type("deepseek_v41")
    assert d.stop_tokens == ("<｜end▁of▁sentence｜>",) and d.mtp_auto is False
    v = A.DS41.vendor
    assert (v.module, v.block, v.attn, v.moe, v.cache, v.batch_cache) == (
        "mlx_vlm.models.deepseek_v41.language", "DeepseekV41Block", "DeepseekV41Attention", "DeepseekV41MoE", "DeepseekV41Cache", "BatchDeepseekV41Cache")
    assert A.DS41.trunk == "language_model" and A.DS41.mtp.keep == 128 and A.DS41.mtp.prefix == "mtp."
    assert (A.DS41.ced.mid, A.DS41.ced.tail) == (None, None)
    assert MD.KERNEL.PF_NORM_HIDDEN == 5120 and MD.KERNEL.HC == 4


def test_other_rows_keep_their_old_table():
    assert MD.stop_tokens_of("qwen") == ("<|im_end|>", "<|endoftext|>")
    assert MD.stop_tokens_of("glm") == ("<|user|>", "<|observation|>", "<|endoftext|>")
    assert MD.by_model_type("qwen3_5").mtp_auto is True and MD.by_model_type("glm5_next").mtp_auto is False
    assert A.cache_class_names() == frozenset({"DeepseekV41Cache"})


def test_active_arch_defaults_to_ds41_and_resolves_classes():
    assert A.active() is A.DS41
    pytest.importorskip("mlx_vlm.models.deepseek_v41")
    assert A.cls("cache").__name__ == "DeepseekV41Cache" and A.cls("batch_cache").__name__ == "BatchDeepseekV41Cache"
    assert A.cls("block").__name__ == "DeepseekV41Block" and A.lang().__name__ == "mlx_vlm.models.deepseek_v41.language"
    assert A.sub("fakequant").__name__.endswith("deepseek_v41.fakequant")


def test_vendor_role_errors_name_the_role():
    v = A.Vendor(module="mlx_vlm.models.deepseek_v41.language", block="NoSuchBlock")
    with pytest.raises(ValueError, match="NoSuchBlock"):
        v.cls("block")
    with pytest.raises(ValueError, match="no class named for role 'moe'"):
        v.cls("moe")


def test_is_model_cache_matches_subclasses_by_name():
    class DeepseekV41Cache: ...
    class Packed(DeepseekV41Cache): ...
    assert A.is_model_cache(DeepseekV41Cache()) and A.is_model_cache(Packed()) and not A.is_model_cache(object())


def ds41_shaped_cfg():
    """The real DS4.1 layer layout the lane runs: 40 layers, window 128, ratio-1 decoder half from layer 20 (synthetic numbers, no pack read)."""
    return types.SimpleNamespace(num_hidden_layers=40, sliding_window=128, compress_ratios=[0, 0] + [2, 2] * 9 + [1] * 20,
                                 kv_source_layer_ids=[2, 20], index_source_layer_ids=[2, 20], engram_layer_ids=[1], candidate_source_layer_id=20)


def test_ced_midpoint_and_tail_stay_20_and_128_for_a_ds41_shaped_config():
    cfg = ds41_shaped_cfg()
    assert C.explain(cfg) is None
    mid, tail = C.spec_defaults(A.DS41, None, None)
    lay = C.layout(cfg, mid)
    assert (lay.mid, lay.window) == (20, 128) and (128 if tail is None else tail) == 128
    # an explicit value still wins, and a CedSpec value is the default for a row that names one
    assert C.spec_defaults(A.DS41, 3, 256) == (3, 256)
    row = A.ArchSpec(name="x", ced=A.CedSpec(adapter="dsv4", mid=7, tail=192))
    assert C.spec_defaults(row, None, None) == (7, 192) and C.spec_defaults(row, 5, None) == (5, 192)


def test_explicit_mid_lifts_the_even_layer_rule_and_still_validates():
    cfg = types.SimpleNamespace(num_hidden_layers=7, sliding_window=16, compress_ratios=[0, 0, 2, 1, 1, 1, 1],
                                kv_source_layer_ids=[2, 3], index_source_layer_ids=[2, 3], engram_layer_ids=[], candidate_source_layer_id=3)
    assert "no even split" in C.explain(cfg)
    assert C.explain(cfg, 3) is None and C.layout(cfg, 3).mid == 3
    assert C.explain(cfg, 2) is not None


FORBIDDEN = ("DeepseekV41", "deepseek_v41", "glm5_next")
ALLOW_FILES = {"archs.py", "modeldesc.py", "synth.py"}          # the three tables: descriptor rows, ArchSpecs, synthetic configs


def _code_literals(path):
    """String constants of executable code (docstrings and comments excluded)."""
    tree = ast.parse(open(path).read())
    doc = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            b = node.body
            if b and isinstance(b[0], ast.Expr) and isinstance(getattr(b[0], "value", None), ast.Constant) and isinstance(b[0].value.value, str):
                doc.add(id(b[0].value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in doc:
            yield node.lineno, node.value


def test_no_vendor_literal_in_executable_kernel_or_serve_code():
    hits = []
    for f in sorted(glob.glob(os.path.join(ROOT, "*.py")) + glob.glob(os.path.join(ROOT, "serve", "*.py"))):
        if os.path.basename(f) in ALLOW_FILES:
            continue
        for ln, s in _code_literals(f):
            if len(s) < 200 and any(k in s for k in FORBIDDEN):          # long strings are Metal sources / prose
                hits.append(f"{os.path.relpath(f, ROOT)}:{ln}: {s[:70]!r}")
    assert not hits, "DS4.1 literal in executable code (move it behind the ArchSpec / ModelDesc):\n" + "\n".join(hits)


def test_new_model_doc_names_every_descriptor_field():
    import dataclasses
    doc = open(os.path.join(os.path.dirname(__file__), "..", "docs", "NEW_MODEL.md")).read()
    exempt = {"name", "kind", "taps_at_input", "name_rules", "dense_rules", "keys"}                 # keys / rules are explained by their own words in the doc
    missing = [f"{c.__name__}.{f.name}" for c in (MD.ModelDesc, A.CedSpec, A.MtpSpec, A.Vendor) for f in dataclasses.fields(c)
               if f.name not in exempt and f"`{f.name}`" not in doc and f"{f.name}" not in doc]
    missing += [f"ArchSpec.{n}" for n in ("vendor", "trunk", "ced", "mtp", "moe", "rows", "fuse") if f"`{n}`" not in doc]
    assert not missing, missing


@pytest.fixture
def lb(monkeypatch):
    pytest.importorskip("mlx.core")
    from kyojin_mlx.serve import backends as B
    seen = {}
    monkeypatch.setattr(B, "wire_memory", lambda *a, **k: 0)
    monkeypatch.setattr(B, "load_q35", lambda path, mtp=None, mtp_head=None, log=print: seen.update(kind="q35", mtp=mtp) or ("q", {}))
    monkeypatch.setattr(B, "load_vlm", lambda path, mtp=False, **kw: seen.update(kind="vlm", mtp=mtp) or ("v", {}))
    return B, seen


def _pack(tmp_path, model_type, with_mtp):
    import json
    from mlx.core import save_safetensors, zeros
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    save_safetensors(str(tmp_path / "model.safetensors"), {"mtp.0.x": zeros((1,))} if with_mtp else {"layers.0.x": zeros((1,))})
    return str(tmp_path)


@pytest.mark.parametrize("mt,with_mtp,kw,want", [
    ("qwen3_5", True, {}, ("q35", True)), ("qwen3_5", False, {}, ("q35", False)), ("qwen3_5", True, {"mtp": False}, ("q35", False)),
    ("deepseek_v41", True, {}, ("vlm", False)), ("deepseek_v41", True, {"mtp": True}, ("vlm", True)), ("deepseek_v41", False, {"mtp": False}, ("vlm", False)),
    ("glm5_next", True, {}, ("vlm", False))])
def test_load_backend_mtp_rule_is_the_one_the_code_had(lb, tmp_path, mt, with_mtp, kw, want):
    B, seen = lb
    B.load_backend(_pack(tmp_path, mt, with_mtp), **kw)
    assert (seen["kind"], seen["mtp"]) == want


def test_a_row_with_mtp_auto_gets_the_draft_head_by_itself(lb, tmp_path, monkeypatch):
    B, seen = lb
    row = MD.by_model_type("deepseek_v41")
    import dataclasses
    monkeypatch.setattr(MD, "DESCS", tuple(dataclasses.replace(d, mtp_auto=True) if d is row else d for d in MD.DESCS))
    monkeypatch.setattr(MD, "_BY_TYPE", {t: d for d in MD.DESCS for t in d.model_types})
    B.load_backend(_pack(tmp_path, "deepseek_v41", True))
    assert (seen["kind"], seen["mtp"]) == ("vlm", True)
