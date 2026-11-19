# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next FP4-Local pack: header mapping (synthetic headers, a stand-in class) and the NVFP4 n-gram table read by row id. CPU only, no weights."""
import numpy as np
import pytest
import mlx.core as mx

mx.set_default_device(mx.cpu)

from kyojin_mlx import foreign_qwen4exp_st as Q

LM = "model.language_model."
PLE = LM + "layers.1.ple.ple_embedding.ngram_embedding"


def _info(shards=3, rows=(5, 7, 4), k=32, experts=2):
    i = {}
    for s in range(shards):
        i[f"{PLE}.shard_{s}.weight"] = ("U8", (rows[s], k // 2))
        i[f"{PLE}.shard_{s}.weight_scale"] = ("F8_E4M3", (rows[s], k // 16))
    i[f"{PLE}.weight_scale_2"] = ("F32", (1,))
    for e in range(experts):                                    # CsfCheckpoint.info(): the decoded scale is listed, the raw streams too
        for pj, (n, kk) in dict(gate_proj=(8, 64), up_proj=(8, 64), down_proj=(64, 8)).items():
            b = f"{LM}layers.0.mlp.experts.{e}.{pj}"
            i[b + ".weight"] = ("U8", (n, kk // 2))
            i[b + ".weight_scale"] = ("F8_E4M3", (n, kk // 16))
            i[b + ".weight_scale_2"] = ("F32", ())
            i[b + ".weight_scale.nvfp4_csf_fixed"] = ("U8", (2, 9))
            i[b + ".weight_scale.nvfp4_csf_exceptions"] = ("U32", (3,))
    i[LM + "layers.0.linear_attn.in_proj_qkv.weight"] = ("F8_E4M3", (96, 64))
    i[LM + "layers.0.linear_attn.in_proj_qkv.weight_scale"] = ("U8", (96, 2))
    i[LM + "layers.0.linear_attn.conv1d.weight"] = ("BF16", (96, 1, 4))
    i[LM + "embed_tokens.weight"] = ("BF16", (50, 64))
    i["lm_head.weight"] = ("BF16", (50, 64))
    i["mtp.fc_hidden.weight"] = ("BF16", (64, 64))
    i["model.visual.pos_embed.weight"] = ("BF16", (4, 4))
    return i


class _Fake:
    """Stand-in for the class: sanitize renames like mlx-vlm's (language_model.model.*, shards, conv1d moveaxis, mtp dropped)."""
    def sanitize(self, w):
        out = {}
        for k, v in w.items():
            if k.startswith("mtp."):
                continue
            k = k.replace("model.language_model.", "language_model.model.", 1).replace(".shard_", ".shards.")
            if k.startswith("lm_head"):
                k = "language_model." + k
            if "conv1d.weight" in k and v.shape[-1] != 1:
                v = v.moveaxis(2, 1)
            out[k] = v
        return out


def _cls():
    P = "language_model.model."
    c = {"language_model.lm_head.weight": (50, 64), P + "embed_tokens.weight": (50, 64), P + "layers.0.linear_attn.conv1d.weight": (96, 4, 1),
         P + "layers.0.linear_attn.in_proj_qkv.weight": (96, 64), P + "layers.0.mlp.switch_mlp.gate_proj.weight": (2, 8, 64),
         P + "layers.0.mlp.switch_mlp.up_proj.weight": (2, 8, 64), P + "layers.0.mlp.switch_mlp.down_proj.weight": (2, 64, 8),
         "vision_tower.pos_embed.weight": (4, 4)}
    for s, r in enumerate((5, 7, 4)):
        c[P + f"layers.1.ple.ple_embedding.ngram_embedding.shards.{s}.weight"] = (r, 32)
    return c


def _csf(info):
    return [k for k in info if "csf_" in k]


def test_ple_tables_layout():
    t = Q.ple_tables(_info())
    assert list(t) == [PLE] and len(t[PLE]["shards"]) == 3 and t[PLE]["rows"] == [5, 7, 4] and t[PLE]["k"] == 32 and t[PLE]["g2"] == PLE + ".weight_scale_2"
    assert len(Q.table_keys(t)) == 7


def test_ple_tables_refuse_half_described():
    i = _info()
    del i[f"{PLE}.shard_1.weight_scale"]
    with pytest.raises(ValueError):
        Q.ple_tables(i)
    i = _info()
    del i[PLE + ".weight_scale_2"]
    with pytest.raises(ValueError):
        Q.ple_tables(i)
    i = _info()
    i[f"{PLE}.shard_0.weight_scale"] = ("F8_E4M3", (5, 3))                 # not one scale per 16 elements
    with pytest.raises(ValueError):
        Q.ple_tables(i)


def test_map_header_both_ways():
    info = _info()
    r = Q.map_header(info, {}, model=_Fake(), class_shapes=_cls(), csf=_csf(info))
    assert r["unmatched_header"] == [] and r["class_unmatched"] == [] and r["shape_mismatch"] == []
    c = r["by_category"]
    assert c["ple:table"] == 3 and c["aux:ple_scale"] == 3 and c["aux:ple_global_scale"] == 1
    assert c["expert:nvfp4"] == 6 and c["aux:csf_stream"] == 12 and c["pack:fp8_block"] == 1
    assert c["dropped:mtp"] == 1 and c["dropped:vision"] == 1 and r["vision_class_params"] == 1
    assert c["dense"] == 3                                                  # conv1d (moved axis), embed_tokens, lm_head
    assert r["class_params"] == r["class_params_matched"] == 10 and r["ple_tables"][PLE]["rows"] == 16


def test_map_header_reports_what_does_not_match():
    info = _info()
    cs = _cls()
    cs["language_model.model.layers.0.extra.weight"] = (3, 3)              # class parameter with no tensor
    cs["language_model.model.layers.0.mlp.switch_mlp.up_proj.weight"] = (2, 8, 66)
    info[LM + "layers.0.surprise.weight"] = ("BF16", (3,))                 # tensor the class does not know
    r = Q.map_header(info, {}, model=_Fake(), class_shapes=cs, csf=_csf(info))
    assert "language_model.model.layers.0.extra.weight" in r["class_unmatched"]
    assert any(m[0].endswith("up_proj.weight") for m in r["shape_mismatch"])
    assert r["class_params_matched"] == r["class_params"] - 1
    assert r["unmatched_header"] == [LM + "layers.0.surprise.weight"]


def test_switch_default_off(monkeypatch):
    monkeypatch.delenv(Q.FLAG, raising=False)
    assert Q.mapped_row_nvfp4() is False
    monkeypatch.setenv(Q.FLAG, "1")
    assert Q.mapped_row_nvfp4() is True
    assert Q.ROW["proven"] is False and Q.ROW["unproven"]
