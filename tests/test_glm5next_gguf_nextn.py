# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash GGUF nextn (MTP) head: block 45 names -> the names glm_nextn.NextN.build renames from (CPU; real Unsloth header when present, else mocked)."""
import glob
import re
import types

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from kyojin_mlx import foreign_glm_gguf as GG
from kyojin_mlx import foreign_gguf as GF
from kyojin_mlx import foreign_linear as FL
from kyojin_mlx import foreign_names as FN
from tests.test_glm_nextn_variants import tiny_cfg

HDR = glob.glob("/path/to/data")


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    monkeypatch.setenv("KYOJIN_KIT_UNPROVEN_ROWS", "1")
    monkeypatch.setenv("KYOJIN_FOREIGN_MTP", "1")
    GG.register()
    yield
    monkeypatch.undo()
    GG.register()


def _tail(k):
    return re.sub(r"^.*?layers\.\d+\.", "", k)


def test_switch_off_skips():
    import os
    os.environ["KYOJIN_FOREIGN_MTP"] = "0"
    GG.register()
    assert FN.gguf_name("blk.45.nextn.eh_proj.weight", "glm5-next", 45) is None


@pytest.mark.skipif(not HDR, reason="Unsloth header not on this machine")
def test_all_nextn_tensors_map_and_cover_the_module():
    g = GF.GgufFile(HDR[0])
    nt = FN.n_trunk_of(g.kv, "glm5-next")
    assert nt == 45 and int(g.kv["glm5-next.nextn_predict_layers"]) == 1
    head = {n: FN.gguf_name(n, "glm5-next", nt) for n in g.tensors if n.startswith("blk.45.")}
    assert len(head) == 29 and all(v and v.startswith("mtp.layers.0.") for v in head.values())          # 0 unmapped, 0 skipped
    mine = {_tail(v) for v in head.values()}
    # module side: names Glm5NextMTPDraftModel (tiny config) takes, in HF form after build()'s rename + sanitize (q_a + kv_a, gate + up fuse; experts = switch)
    from mlx_vlm.speculative.drafters.glm5_next_mtp.config import Glm5NextMTPConfig
    from mlx_vlm.speculative.drafters.glm5_next_mtp.glm5_next_mtp import Glm5NextMTPDraftModel
    mx.set_default_device(mx.cpu)
    d = Glm5NextMTPDraftModel(Glm5NextMTPConfig.from_dict({"text_config": tiny_cfg()}))
    want = set()
    for k, v in tree_flatten(d.parameters()):
        k = k.replace("mtp_block.", "").replace("shared_head_norm.", "shared_head.norm.")
        if ".switch_mlp." in k:
            continue
        k = k.replace("qkv_a_proj", "q_a_proj").replace("gate_up_proj", "gate_proj")
        want.add(k)
        if "q_a_proj" in k:
            want.add(k.replace("q_a_proj", "kv_a_proj_with_mqa"))
        if "shared_experts.gate_proj" in k:
            want.add(k.replace("gate_proj", "up_proj"))
    assert want <= mine, sorted(want - mine)
    assert GG.MTP["nextn.eh_proj.weight"].endswith("eh_proj.weight") and g.tensors["blk.45.nextn.eh_proj.weight"].shape == (4096, 2 * 4096)


def test_read_gguf_head_mocked(monkeypatch):
    H = 8
    ten = {"token_embd.weight": (16, H), "output_norm.weight": (H,), "blk.4.nextn.eh_proj.weight": (H, 2 * H), "blk.4.nextn.enorm.weight": (H,), "blk.4.nextn.hnorm.weight": (H,),
           "blk.4.nextn.shared_head_norm.weight": (H,), "blk.4.attn_norm.weight": (H,), "blk.4.ffn_gate_inp.weight": (4, H), "blk.4.exp_probs_b.bias": (4,)}
    ti = {n: types.SimpleNamespace(ne=tuple(reversed(s)), tname="F32", shape=s) for n, s in ten.items()}
    g = types.SimpleNamespace(kv={"general.architecture": "glm5-next", "glm5-next.block_count": 5, "glm5-next.nextn_predict_layers": 1}, tensors=ti,
                              raw=lambda n: np.ones(ten[n], np.float32))
    monkeypatch.setattr(GF, "GgufFile", lambda p: g)
    monkeypatch.setattr(GF, "find_gguf", lambda p: p)
    src = FL._read_gguf("x", "glm5-next", mtp=True)
    assert set(src.mtp) == {"mtp.layers.0." + k for k in ("eh_proj.weight", "enorm.weight", "hnorm.weight", "shared_head.norm.weight", "input_layernorm.weight",
                                                         "mlp.gate.weight", "mlp.gate.e_score_correction_bias")}
    assert src.mtp["mtp.layers.0.eh_proj.weight"].shape == (H, 2 * H) and not src.mtp_packs
    assert not FL._read_gguf("x", "glm5-next", mtp=False).mtp          # the load_foreign(mtp=False) path never reads the head


def test_mixed_qkv_a_stays_stored_as_split():
    """GLM-5.3-Flash head: attn_q_a Q6_K + attn_kv_a_mqa Q8_0 stay stored (SplitLinear, as the trunk), no fp16 copy."""
    from kyojin_mlx import foreign_pack as FP
    from kyojin_mlx import glm_nextn as GN
    rng = np.random.default_rng(0)
    k = 256
    raw = rng.integers(0, 256, (24, 210), dtype=np.uint8)
    raw[:, 208:210] = np.array([0x00, 0x1C], np.uint8)
    qa = GF.to_pack("Q6_K", raw, k)
    kva = FP.Pack("q8_0", 8, k, rng.integers(0, 256, (8, k), dtype=np.uint8), np.full((8, k // 32), 1e-3, np.float16))
    src = FL._Src()
    src.mtp_packs = {"mtp.layers.0.self_attn.q_a_proj": qa, "mtp.layers.0.self_attn.kv_a_proj_with_mqa": kva}
    FL._g4_mtp_finish(src)
    assert set(src.mtp_packs) == {"mtp.layers.0.self_attn.q_a_proj", "mtp.layers.0.self_attn.kv_a_proj_with_mqa"} and not src.mtp
    lin = GN._stored_linears(src.mtp_packs)["mtp_block.self_attn.qkv_a_proj"]
    assert isinstance(lin, FL.SplitLinear) and lin.out_features == 32 and lin.in_features == k
    src = FL._Src()                                                    # a lone member (partner missing) is still decoded
    src.mtp_packs = {"mtp.layers.0.self_attn.q_a_proj": qa}
    FL._g4_mtp_finish(src)
    assert not src.mtp_packs and src.mtp["mtp.layers.0.self_attn.q_a_proj.weight"].shape == (24, k)
