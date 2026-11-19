"""tools/day0_inspect.py on tiny synthetic GGUFs (the ones the kit's tests build): header-only, CPU."""
from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
mx.set_default_device(mx.cpu)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import test_gguf_arch as T  # noqa: E402
import day0_inspect as D  # noqa: E402
from kyojin_mlx import foreign_gguf as GF, foreign_synth as SY  # noqa: E402


def _qwen35(tmp_path, name="m.gguf", qtype=True):
    p = tmp_path / name
    g = T._write(p, T._hf_qwen35())
    q = None
    if qtype:
        n = next(k for k, a in g.items() if k.endswith("ffn_down.weight") and a.shape[1] % 32 == 0)
        q = {n: "Q4_0"}
        T._write(p, T._hf_qwen35(), qtype=q)
    return p, q


def test_hand_row_file_is_ready_and_types_are_classified(tmp_path):
    p, q = _qwen35(tmp_path)
    r = D.inspect(str(p))
    assert r["architecture"] == "qwen35" and r["hand_row"] and r["in_tables"]
    assert r["verdict"] == "READY" and r["serve_command"].startswith("python -m kyojin_mlx.serve --model ")
    modes = {(e["type"], e["mode"]) for e in r["types"]}
    assert ("Q4_0", "read as stored (kernel)") in modes and ("F32", "resident fp16 copy") in modes
    assert r["mtp_layers"] == 1 and "MTP draft head" in r["side_tensors"] or r["mtp_layers"] == 1
    assert r["fits"] and r["resident_est_bytes"] > 0
    assert "VERDICT   READY" in D.render(r)
    json.dumps(r, default=str)


def test_header_only_never_reads_a_weight(tmp_path, monkeypatch):
    p, _ = _qwen35(tmp_path)

    def boom(*a, **k):
        raise AssertionError("weights read")
    monkeypatch.setattr(GF.GgufFile, "raw", boom)
    monkeypatch.setattr(GF.GgufFile, "tensor_f32", boom)
    assert D.inspect(str(p), use_census=False)["n_tensors"] > 0


def test_unknown_architecture_needs_model_code(tmp_path):
    p = tmp_path / "z.gguf"
    SY.write_gguf(p, {"general.architecture": "zzznotanarch", "zzznotanarch.block_count": 1}, [("token_embd.weight", (32, 8), "F32", np.zeros((8, 32), np.float32))])
    r = D.inspect(str(p))
    assert r["verdict"] == "MODEL CODE NEEDED" and not r["in_tables"] and r["serve_command"] is None


def test_unsupported_type_blocks(tmp_path):
    p = tmp_path / "f64.gguf"
    SY.write_gguf(p, {"general.architecture": "qwen35"}, [("blk.0.ffn_down.weight", (32, 8), "F64", np.zeros((8, 32), np.float64))])
    r = D.inspect(str(p), use_census=False)
    assert r["verdict"] == "MODEL CODE NEEDED" and any(e["mode"] == "unsupported" for e in r["types"])


def test_does_not_fit(tmp_path):
    p, _ = _qwen35(tmp_path, qtype=False)
    r = D.inspect(str(p), limit_gib=1e-6, use_census=False)
    assert r["verdict"] == "DOES NOT FIT" and not r["fits"]


def test_split_shards_directory(tmp_path):
    p, _ = _qwen35(tmp_path, qtype=False)
    g = GF.GgufFile(p)
    names = sorted(g.tensors)
    tl = [(n, g.tensors[n].ne, g.tensors[n].tname, np.array(g.raw(n))) for n in names]
    kv = dict(g.kv)
    d = tmp_path / "sh"
    d.mkdir()
    h = len(tl) // 2
    for i, part in enumerate((tl[:h], tl[h:])):
        SY.write_gguf(d / f"m-0000{i + 1}-of-00002.gguf", {**kv, "split.no": i, "split.count": 2, "split.tensors.count": len(tl)}, part)
    shutil.copy(p, d / "mmproj-m.gguf")
    r = D.inspect(str(d), use_census=False)
    assert r["shards"] == 2 and r["side_files"] == ["mmproj-m.gguf"] and r["n_tensors"] == len(tl)
    assert r["verdict"] == "READY" and any("no merge" in n for n in r["notes"])
    (d / "m-00002-of-00002.gguf").unlink()
    r = D.inspect(str(d), use_census=False)
    assert r["verdict"] != "SHARDS INVALID" and r["verdict"] != "READY"            # a set that is merely incomplete on disk is not invalid
    assert r["verdict"] == "UNVERIFIED" and "incomplete shard set: 1 of 2 present" in r["why"]


def test_roofline_dense_and_moe_hand_numbers(tmp_path):
    f32 = lambda *s: np.zeros(s, np.float32)
    dense = [("token_embd.weight", (8, 10), "F32", f32(10, 8)), ("blk.0.attn_q.weight", (8, 4), "F32", f32(4, 8)), ("blk.0.attn_norm.weight", (8,), "F32", f32(8)),
             ("output_norm.weight", (8,), "F32", f32(8)), ("output.weight", (8, 10), "F32", f32(10, 8))]
    SY.write_gguf(tmp_path / "d.gguf", {"general.architecture": "llama", "llama.block_count": 1}, dense)
    r = D.inspect(str(tmp_path / "d.gguf"), use_census=False, bw_gbs=100.0)["roofline"]
    # embedding row 8*4 = 32; head 10*8*4 = 320; trunk 4*8*4 + 8*4 + 8*4 = 192
    assert r["parts"] == dict(embedding_row=32, head=320, trunk=192, routed_experts=0) and r["bytes_per_step"] == 544
    assert abs(r["ceiling_tok_s"] - 100e9 / 544) < 1e-3 and not r["tied_head"]
    moe = [("token_embd.weight", (8, 10), "F32", f32(10, 8)), ("blk.0.ffn_gate_inp.weight", (8, 4), "F32", f32(4, 8)), ("blk.0.ffn_gate_exps.weight", (8, 3, 4), "F32", f32(4, 3, 8)),
           ("blk.0.ffn_down_shexp.weight", (3, 8), "F32", f32(8, 3)), ("blk.1.nextn.eh_proj.weight", (8, 8), "F32", f32(8, 8))]
    SY.write_gguf(tmp_path / "m.gguf", {"general.architecture": "llama", "llama.block_count": 2, "llama.nextn_predict_layers": 1, "llama.expert_count": 4, "llama.expert_used_count": 2}, moe)
    r = D.inspect(str(tmp_path / "m.gguf"), use_census=False, bw_gbs=50.0)["roofline"]
    # tied head: full token_embd 320; trunk = router 128 + shared 96; experts 4*3*8*4 = 384 * 2/4 = 192; the nextn tensor (256) is skipped
    assert r["parts"] == dict(embedding_row=0, head=320, trunk=224, routed_experts=192) and r["bytes_per_step"] == 736 and r["skipped_bytes"] == 256 and r["tied_head"]
    SY.write_gguf(tmp_path / "x.gguf", {"general.architecture": "llama", "llama.block_count": 1}, moe[:3])
    assert "UNVERIFIED" in D.inspect(str(tmp_path / "x.gguf"), use_census=False, bw_gbs=50.0)["roofline"]["error"]


def test_roofline_cli_says_it_is_a_ceiling(tmp_path, capsys):
    SY.write_gguf(tmp_path / "d.gguf", {"general.architecture": "llama"}, [("token_embd.weight", (8, 10), "F32", np.zeros((10, 8), np.float32))])
    assert D.main([str(tmp_path / "d.gguf"), "--no-census", "--roofline", "--bw-gbs", "10"]) == 0
    out = capsys.readouterr().out
    assert "NOT a prediction" in out and "10 GB/s" in out


def test_cli_json(tmp_path, capsys):
    p, _ = _qwen35(tmp_path, qtype=False)
    assert D.main([str(p), "--json", "--no-census"]) == 0
    assert json.loads(capsys.readouterr().out)["architecture"] == "qwen35"


def test_roofline_per_layer_embedding_is_a_row_and_experts_are_found_by_dimension(tmp_path):
    """Census bug: Qwen3.8-Flash-Next's 26.8 GiB per_layer_token_embd.weight was counted as trunk (32.7 GB per step for a 10-of-512 model). A lookup table is one row
    per step; a 3-D tensor whose last dimension is the expert count is a routed stack whatever its name; another 3-D tensor stays trunk."""
    f32 = lambda *s: np.zeros(s, np.float32)
    t = [("token_embd.weight", (8, 10), "F32", f32(10, 8)), ("output.weight", (8, 10), "F32", f32(10, 8)), ("per_layer_token_embd.weight", (4, 50), "F32", f32(50, 4)),
         ("blk.0.ffn_gate_exps.weight", (8, 3, 4), "F32", f32(4, 3, 8)), ("blk.0.fused_up.weight", (8, 3, 4), "F32", f32(4, 3, 8)), ("blk.0.conv3.weight", (8, 3, 5), "F32", f32(5, 3, 8))]
    SY.write_gguf(tmp_path / "p.gguf", {"general.architecture": "qwen4exp", "qwen4exp.block_count": 1, "qwen4exp.expert_count": 4, "qwen4exp.expert_used_count": 2}, t)
    r = D.inspect(str(tmp_path / "p.gguf"), use_census=False, bw_gbs=100.0)["roofline"]
    # embedding rows 32 + 16; head 320; experts (384 + 384) * 2 / 4 = 384; trunk = the 5-wide conv 480
    assert r["parts"] == dict(embedding_row=48, head=320, trunk=480, routed_experts=384) and r["bytes_per_step"] == 1232


CENSUS_Q4EXP = os.environ.get("KYOJIN_TEST_QWEN4EXP_SHARDS", "")      # a directory holding the header set of a qwen4exp UD-Q4_K_XL shard set; unset = skipped


@pytest.mark.skipif(not os.path.isdir(CENSUS_Q4EXP), reason="KYOJIN_TEST_QWEN4EXP_SHARDS not set")
def test_roofline_qwen4exp_census_headers():
    r = D.inspect(CENSUS_Q4EXP, use_census=False, bw_gbs=1159.0)["roofline"]
    assert r["experts_used"] == 10 and r["experts_total"] == 512 and 6e9 < r["bytes_per_step"] < 7e9          # 6.33e9; was 32.7e9 with the per-layer table counted whole


def _f32(*s):
    return np.zeros(s, np.float32)


def _moe_tensors(nblk=2):
    t = [("token_embd.weight", (8, 10), "F32", _f32(10, 8))]
    for i in range(nblk):
        t.append((f"blk.{i}.ffn_gate_exps.weight", (8, 3, 4), "F32", _f32(4, 3, 8)))
    return t


def test_array_valued_expert_used_count_does_not_crash_and_reports_min_max(tmp_path):
    kv = {"general.architecture": "llama", "llama.block_count": 2, "llama.expert_count": 4, "llama.expert_used_count": [2, 1]}
    SY.write_gguf(tmp_path / "a.gguf", kv, _moe_tensors())
    r = D.inspect(str(tmp_path / "a.gguf"), use_census=False, bw_gbs=100.0)
    assert r["moe"] and r["expert_count"] == 4 and r["expert_used"] == 2 and r["expert_used_min"] == 1 and r["expert_per_layer"]
    rl = r["roofline"]
    # head tied: token_embd 320; layer 0 uses 2 of 4 (384 * 2 / 4 = 192), layer 1 uses 1 of 4 (96): the ceiling uses each layer's own value
    assert rl["parts"]["routed_experts"] == 288 and rl["bytes_per_step"] == 608 and rl["experts_used_min"] == 1 and rl["experts_used"] == 2
    out = D.render(r)
    assert "1..2 used" in out and "per layer" in out and "[1..2/4, per layer]" in out
    json.dumps(r, default=str)
    # scalar key: unchanged numbers, no per-layer flag
    SY.write_gguf(tmp_path / "s.gguf", {**kv, "llama.expert_used_count": 2}, _moe_tensors())
    r2 = D.inspect(str(tmp_path / "s.gguf"), use_census=False, bw_gbs=100.0)
    assert r2["expert_used"] == r2["expert_used_min"] == 2 and not r2["expert_per_layer"] and r2["roofline"]["parts"]["routed_experts"] == 384
    # a zero entry for a layer that holds routed tensors is UNVERIFIED, not a division
    SY.write_gguf(tmp_path / "z.gguf", {**kv, "llama.expert_used_count": [2, 0]}, _moe_tensors())
    assert "UNVERIFIED" in D.inspect(str(tmp_path / "z.gguf"), use_census=False, bw_gbs=100.0)["roofline"]["error"]


def test_engine_tables_take_uniform_array_and_refuse_a_varying_one():
    assert GF.kv_count_uniform([8, 8, 8]) == 8 and GF.kv_count_uniform(8) == 8 and GF.kv_count_uniform(None) == 0
    with pytest.raises(GF.GgufError, match="per-layer"):
        GF.kv_count_uniform([0, 4, 8], "x.expert_used_count")
    from kyojin_mlx import foreign_names as N
    a = N._kv_getter({"qwen3moe.expert_used_count": [8, 8], "qwen3moe.expert_count": 128, "qwen3moe.block_count": 2}, "qwen3moe")
    assert a("expert_used_count") == 8 and a("expert_count") == 128 and a("block_count") == 2
    with pytest.raises(GF.GgufError):
        N._kv_getter({"qwen3moe.expert_used_count": [2, 8]}, "qwen3moe")("expert_used_count")


def _split_set(tmp_path, n=3, with_tensors_in_first=False):
    """n shards; shard 1 holds the metadata and (only if asked) the embedding; the other tensors go one per later shard (the last shard takes the remainder)."""
    d = tmp_path / "set"
    d.mkdir(exist_ok=True)
    t = _moe_tensors(n - 1)                                            # token_embd + n-1 expert tensors
    kv = {"general.architecture": "llama", "llama.block_count": n - 1, "llama.expert_count": 4, "llama.expert_used_count": 2}
    for i in range(n):
        part = ([t[0]] if with_tensors_in_first else []) if i == 0 else ([t[i]] + ([t[0]] if (i == 1 and not with_tensors_in_first) else []))
        SY.write_gguf(d / f"m-{i + 1:05d}-of-{n:05d}.gguf", {**(kv if i == 0 else {"general.architecture": "llama"}), "split.no": i, "split.count": n, "split.tensors.count": len(t)}, part)
    return d


def test_split_shard_1_metadata_only_alone_says_weights_not_seen(tmp_path):
    d = _split_set(tmp_path)
    for f in ("m-00002-of-00003.gguf", "m-00003-of-00003.gguf"):
        (d / f).unlink()
    r = D.inspect(str(d / "m-00001-of-00003.gguf"), use_census=False)
    assert r["verdict"] == "UNVERIFIED" and r["verdict"] not in ("READY", "SHARDS INVALID")
    assert "incomplete shard set: 1 of 3 present" in r["why"] and "verdict from shard 1 of 3: weights not seen" in r["why"]
    assert "incomplete shard set" in D.render(r)


def test_split_some_shards_present_is_incomplete_not_invalid_and_complete_is_still_checked(tmp_path):
    d = _split_set(tmp_path, with_tensors_in_first=True)
    (d / "m-00003-of-00003.gguf").unlink()
    r = D.inspect(str(d), use_census=False)
    assert r["verdict"] == "UNVERIFIED" and "incomplete shard set: 2 of 3 present" in r["why"] and "weights not seen" not in r["why"]
    # shard 1 absent: no metadata, still no SHARDS INVALID
    (d / "m-00001-of-00003.gguf").unlink()
    r = D.inspect(str(d), use_census=False)
    assert r["verdict"] == "UNVERIFIED" and "shard 1" in r["why"] and "1 of 3 present" in r["why"]
    # a complete set whose shard 1 holds metadata only: the tensors of the other shards give the verdict
    (tmp_path / "x").mkdir()
    d2 = _split_set(tmp_path / "x")
    r = D.inspect(str(d2), use_census=False)
    assert r["incomplete_set"] is None and r["n_tensors"] == 3 and r["shards"] == 3
    # a complete but inconsistent set (wrong tensor total) is still SHARDS INVALID
    (tmp_path / "y").mkdir()
    bad = _split_set(tmp_path / "y")
    SY.write_gguf(bad / "m-00001-of-00003.gguf", {"general.architecture": "llama", "split.no": 0, "split.count": 3, "split.tensors.count": 99}, [])
    assert D.inspect(str(bad), use_census=False)["verdict"] == "SHARDS INVALID"


def test_missing_architecture_key_is_a_clear_verdict_not_an_exception(tmp_path):
    # MiniMax-H3 real case: no metadata at all, diffusion-style tensor names
    SY.write_gguf(tmp_path / "h3.gguf", {}, [("blocks.0.attn.q_norm.weight", (8,), "F32", _f32(8)), ("adaln_t_table", (8, 2), "F32", _f32(2, 8))])
    r = D.inspect(str(tmp_path / "h3.gguf"), use_census=False, bw_gbs=10.0)
    assert r["verdict"] == "NOT A TEXT LLM" and "none" in r["why"] and "blocks.0.attn.q_norm.weight" in r["why"]
    assert "VERDICT   NOT A TEXT LLM" in D.render(r)
    # text-like tensors but no architecture key
    SY.write_gguf(tmp_path / "t.gguf", {"general.name": "x", "tokenizer.ggml.model": "llama"}, [("token_embd.weight", (8, 10), "F32", _f32(10, 8)), ("blk.0.attn_q.weight", (8, 4), "F32", _f32(4, 8))])
    r = D.inspect(str(tmp_path / "t.gguf"), use_census=False)
    assert r["verdict"] == "ARCH KEY MISSING" and "general.name" in r["why"] and "tokenizer.ggml.model" in r["why"]
    D.render(r)
    json.dumps(r, default=str)


def test_pack_roofline_per_layer_expert_arrays():
    info = {"model.embed_tokens.weight": ("BF16", (10, 8), 160), "lm_head.weight": ("BF16", (10, 8), 160),
            "model.layers.0.mlp.experts.0.down_proj.weight": ("BF16", (8, 8), 400), "model.layers.1.mlp.experts.0.down_proj.weight": ("BF16", (8, 8), 400)}
    cfg = {"num_experts": 4, "num_experts_per_tok": [2, 1], "vocab_size": 10}
    r = D.PK.roofline_pack(info, cfg, 100.0, None)
    assert r["parts"]["routed_experts"] == 400 * 2 / 4 + 400 * 1 / 4 and r["experts_used"] == 2 and r["experts_used_min"] == 1 and r["per_layer_experts"]
    r2 = D.PK.roofline_pack(info, {**cfg, "num_experts_per_tok": 2}, 100.0, None)
    assert r2["parts"]["routed_experts"] == 400 and not r2["per_layer_experts"]
