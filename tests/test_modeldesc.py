"""Model table + geometry derivation. No Metal, no mlx: everything here is config and strings."""
import dataclasses
import sys
from pathlib import Path

import pytest

from kyojin_mlx import archs, modeldesc as MD

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "qualify"))

# DS4.1-shaped config (the numbers the fused kernels are written for); synthetic, not a released checkpoint
DS_SHAPED = dict(model_type="deepseek_v41", architectures=["DeepseekV41ForCausalLM"], num_hidden_layers=40, hidden_size=5120,
                 num_attention_heads=64, num_key_value_heads=1, head_dim=512, qk_rope_head_dim=64, o_groups=8, hc_mult=4,
                 index_n_heads=32, index_head_dim=128, n_routed_experts=384, num_experts_per_tok=6, sliding_window=128,
                 compress_ratios=[0, 0] + [4, 128] * 19, num_nextn_predict_layers=1)

# an architecture the table has never seen: 2 hyper-connection streams, 8 experts, 24 layers, head 128
UNSEEN = dict(model_type="acme_moe_v2", architectures=["AcmeMoeForCausalLM"], num_hidden_layers=24, hidden_size=2048, num_attention_heads=16,
              num_key_value_heads=4, qk_rope_head_dim=32, hc_mult=2, num_experts=8, num_experts_per_tok=2, first_k_dense_replace=1,
              sliding_window=64)


def test_geometry_ds_shaped():
    g = MD.geometry(DS_SHAPED)
    assert (g.layers, g.experts, g.topk, g.hc_mult, g.n_mix) == (40, 384, 6, 4, 24)
    assert (g.head_dim, g.rope_dims, g.o_groups, g.index_heads, g.index_dim) == (512, 64, 8, 32, 128)
    assert g.moe_layers == 40 and g.window == 128 and g.mtp_layers == 1 and len(g.compress_ratios) == 40


def test_kernel_envelope_is_consistent():
    assert MD.KERNEL.MIXES == (2 + MD.KERNEL.HC) * MD.KERNEL.HC
    assert all(ok for ok, _ in MD.fast_paths(MD.geometry(DS_SHAPED)).values())


def test_geometry_unseen_model_derived_from_config_only():
    g = MD.geometry(UNSEEN)
    assert (g.layers, g.experts, g.topk, g.hc_mult, g.n_mix) == (24, 8, 2, 2, 8)
    assert g.head_dim == 2048 // 16 == 128 and g.moe_layers == 23 and g.kv_layers == 24 and g.o_groups == 0 and g.index_heads == 0


def test_unseen_model_reports_the_fallbacks_with_reasons():
    row, g = MD.describe(UNSEEN)
    assert row is None
    paths = MD.fast_paths(g)
    bad = {k: why for k, (ok, why) in paths.items() if not ok}
    assert any(k.startswith("hc_block_decode") for k in bad) and "hc_mult=2" in next(v for k, v in bad.items() if k.startswith("hc_block_decode"))
    assert any(k.startswith("indexer_score") for k in bad)
    assert paths["rows_router (decrows)"][0]                      # 8 experts, top-2: inside the router limits
    assert "NONE" in MD.report(UNSEEN)


def test_text_config_nesting():
    nested = {"architectures": ["X"], "text_config": {k: v for k, v in UNSEEN.items() if k != "architectures"}}
    assert MD.geometry(nested) == MD.geometry(UNSEEN)


def test_adding_a_model_is_one_row(monkeypatch):
    row = dataclasses.replace(MD.by_model_type("deepseek_v41"), name="acme", model_types=("acme_moe_v2",),
                              architectures={"AcmeMoeForCausalLM": False}, arch=None, one_buffer_gate=False, chunk=1024)
    monkeypatch.setattr(MD, "_BY_TYPE", {**MD._BY_TYPE, "acme_moe_v2": row})
    monkeypatch.setattr(MD, "_BY_ARCH", {**MD._BY_ARCH, "AcmeMoeForCausalLM": row})
    d, _ = MD.describe(UNSEEN)
    assert d is row and MD.backend_desc("acme_moe_v2").backend == "vlm" and MD.chat_desc("acme_moe_v2").server == "ds41"
    with pytest.raises(ValueError):                               # no ArchSpec: patch_model still refuses, as for Qwen3.5
        archs.arch_of({"model_type": "acme_moe_v2"})


# ---- the old name switches, pinned (behaviour for existing models identical, unlisted names unchanged)
@pytest.mark.parametrize("mt,want", [("qwen3_5", "qwen"), ("qwen3_5_moe", "qwen"), ("qwen2", "qwen"), ("qwen3", "qwen"), ("glm5_next", "glm"),
                                     ("glm4", "glm"), ("deepseek_v41", "ds41"), ("deepseek_v3", "ds41")])
def test_chat_family_unchanged(mt, want):
    assert MD.chat_desc(mt).server == want


@pytest.mark.parametrize("mt", ["llama", "mistral", "", "minimax_m3"])
def test_chat_family_unknown(mt):
    assert MD.chat_desc(mt) is None


@pytest.mark.parametrize("mt,want", [("qwen3_5", "q35"), ("qwen3_5_moe", "q35"), ("qwen3", "q35"), ("deepseek_v41", "vlm"), ("glm5_next", "vlm")])
def test_backend_unchanged(mt, want):
    assert MD.backend_desc(mt).backend == want


@pytest.mark.parametrize("mt", ["qwen2", "glm4", "deepseek_v3", "llama"])
def test_backend_unlisted_still_refused(mt):
    assert MD.backend_desc(mt) is None


def test_arch_of_unchanged():
    assert archs.arch_of({"model_type": "deepseek_v41"}) is archs.DS41
    assert archs.arch_of({"model_type": "glm5_next"}) is archs.GLM5
    for mt in ("qwen3_5", "qwen3_5_moe", "glm4", "deepseek_v3", ""):
        with pytest.raises(ValueError):
            archs.arch_of({"model_type": mt})
    assert {d.arch for d in MD.DESCS if d.arch} == set(archs.ARCHS)


def test_qualify_families_unchanged():
    import pack
    old = {"Qwen3_5ForConditionalGeneration": ("hybrid_gdn", "qwen", "qwen35", "kvwin", True),
           "Qwen3_5ForCausalLM": ("hybrid_gdn", "qwen", "qwen35", "kvwin", False),
           "Qwen3_5MoeForConditionalGeneration": ("hybrid_gdn", "qwen", "qwen35", "kvwin", True),
           "Qwen3_5MoeForCausalLM": ("hybrid_gdn", "qwen", "qwen35", "kvwin", False),
           "DeepseekV41ForCausalLM": ("ds41", "ds41", None, "native", False),
           "Glm5NextForConditionalGeneration": ("glm5_next", "glm", None, "native", True)}
    assert {a: (f["family"], f["server"], f["ref"], f["kv"], f["vision"]) for a, f in pack.FAMILIES.items()} == old


def test_qualify_chunk_and_mtp():
    import pack, stages
    meta = lambda arch, mtp: dict(arch=arch, mtp_layers=mtp)
    assert stages.chunk_for(meta("DeepseekV41ForCausalLM", 1)) == 2048 and stages.chunk_for(meta("Qwen3_5ForCausalLM", 0)) == 512
    assert stages.chunk_for(meta("Unknown", 0)) == 512
    assert pack.mtp_depth(meta("DeepseekV41ForCausalLM", 1)) == 3 and pack.mtp_depth(meta("DeepseekV41ForCausalLM", 0)) == 0
    assert pack.mtp_depth(meta("Qwen3_5MoeForCausalLM", 1)) == 0 and pack.mtp_depth(meta("Glm5NextForConditionalGeneration", 1)) == 0


def test_one_buffer_gate_only_ds41():
    assert [d.name for d in MD.DESCS if d.one_buffer_gate] == ["deepseek_v41"]


def test_ds41_argv_reads_the_table(tmp_path, monkeypatch):
    import json
    import kyojin_mlx
    for mt, want in (("deepseek_v41", True), ("glm5_next", False), ("qwen3_5", False), ("acme", False)):
        d = tmp_path / mt
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"model_type": mt}))
        monkeypatch.setattr(sys, "argv", ["x", str(d)])
        assert kyojin_mlx._ds41_argv() is want
