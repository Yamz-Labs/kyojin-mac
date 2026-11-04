# SPDX-License-Identifier: Apache-2.0
"""The pack-derived serve defaults (no model load, no MLX): a stub DS4.1 config must resolve like the lane script."""
import json
import os

from kyojin_mlx.serve import autoconfig as AC


def make_pack(root, *, max_ctx=1048576, draft_block=5, mtp=True):
    root.mkdir(parents=True, exist_ok=True)
    cfg = {"model_type": "deepseek_v41", "architectures": ["DeepseekV41ForCausalLM"],
           "text_config": {"model_type": "deepseek_v41_text", "max_position_embeddings": max_ctx,
                           "dspark_block_size": draft_block}}
    (root / "config.json").write_text(json.dumps(cfg))
    (root / "tokenizer_config.json").write_text("{}")
    if mtp:
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
            "mtp.0.token_embd.weight": "mtp.0.safetensors", "layers.0.self_attn.q_proj.weight": "layers.0.safetensors"}}))
        (root / "mtp.0.safetensors").write_bytes(b"x")
    (root / "layers.0.safetensors").write_bytes(b"x")
    return str(root)


def test_ds41_stub_resolves_like_the_lane_script(tmp_path):
    pack = AC.read_pack(make_pack(tmp_path / "ds41-stub"))
    opts = AC.resolve_options(pack)
    assert opts["sessions"] == 4
    assert opts["ctx"] == 1048576
    assert opts["prefill_chunk"] == 8192
    assert opts["prefix_block"] == 8192
    assert opts["prefix_cache_mb"] == 16384
    assert opts["kv_disk_gib"] == 100
    assert opts["kv_disk_block"] == 8192
    assert opts["mtp_depth"] == 5
    assert opts["thinking"] is True
    assert opts["reasoning_effort"] == "medium"


def test_kv_disk_dir_is_under_the_user_cache_and_keyed_by_the_pack(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(AC.sys, "platform", "linux")               # force the XDG branch on macOS
    a = AC.resolve_options(AC.read_pack(make_pack(tmp_path / "pack-a")))["kv_disk"]
    b = AC.resolve_options(AC.read_pack(make_pack(tmp_path / "pack-b")))["kv_disk"]
    assert a.startswith(str(tmp_path / "cache" / "kyojin_mlx" / "kvdisk"))
    assert os.path.basename(a).startswith("pack-a-") and a != b
    assert AC.resolve_options(AC.read_pack(make_pack(tmp_path / "pack-a")), no_kv_disk=True)["kv_disk"] is None


def test_explicit_values_win(tmp_path):
    pack = AC.read_pack(make_pack(tmp_path / "ds41-stub"))
    opts = AC.resolve_options(pack, ctx=4096, prefill_chunk=2048, prefix_block=2048, prefix_cache_mb=0,
                              kv_disk=str(tmp_path / "kv"), kv_disk_gib=7, kv_disk_block=2048, mtp_depth=0,
                              thinking=False, reasoning_effort="high")
    assert (opts["ctx"], opts["prefill_chunk"], opts["prefix_block"], opts["prefix_cache_mb"]) == (4096, 2048, 2048, 0)
    assert (opts["kv_disk"], opts["kv_disk_gib"], opts["kv_disk_block"]) == (str(tmp_path / "kv"), 7.0, 2048)
    assert opts["mtp_depth"] == 0 and opts["thinking"] is False and opts["reasoning_effort"] == "high"


def test_no_draft_head_means_no_mtp_and_short_ctx_keeps_auto_chunk(tmp_path):
    pack = AC.read_pack(make_pack(tmp_path / "small", max_ctx=32768, draft_block=None, mtp=False))
    opts = AC.resolve_options(pack)
    assert opts["mtp_depth"] == 0
    assert opts["prefill_chunk"] is None                          # the memory-driven auto chunk in server.resolve_chunk
    assert opts["prefix_cache_mb"] == 1024


def test_print_config_cli_uses_model_flag_without_loading(tmp_path, capsys):
    from kyojin_mlx.serve import server
    pack = make_pack(tmp_path / "ds41-stub")
    server.main(["--model", pack, "--print-config"])
    cfg = json.loads(capsys.readouterr().out)
    assert cfg["ctx"] == 1048576 and cfg["prefill_chunk"] == 8192 and cfg["prefix_block"] == 8192
    assert cfg["prefix_cache_mb"] == 16384 and cfg["mtp_depth"] == 5
    assert cfg["thinking"] is True and cfg["reasoning_effort"] == "medium"


def test_positional_path_keeps_fixed_defaults(tmp_path):
    from kyojin_mlx.serve import server
    args = server.build_parser().parse_args([str(tmp_path)])
    AC.resolve_serve_args(args)
    assert (args.ctx, args.prefix_cache_mb, args.kv_disk, args.kv_disk_gib) == (32768, 1024, None, 0.0)
    assert args.mtp_depth is None and args.thinking is None and args.reasoning_effort is None
