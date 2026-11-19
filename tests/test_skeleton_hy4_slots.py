# SPDX-License-Identifier: Apache-2.0
"""Hy4-flash Q3_K GGUF header -> skeleton slots (stage 1, no weights): every tensor of both shards (1415 in shard 1 + the shard-2 Engram table
per_layer_token_embd = split.tensors.count 1416) claimed by exactly one block (block + role), every metadata key placed by exactly one block.
Fixture tests/data/hdr_hy4_flash_q3k.json = header range reads of AngelSlim/Hy4-flash-GGUF (names, shapes, types; no tokenizer)."""
import collections
import inspect
import json
import os

import pytest

from kyojin_mlx import foreign_skeleton as SK
from kyojin_mlx import foreign_skeleton_blocks as B
from kyojin_mlx import foreign_skeleton_kpool as KP
from kyojin_mlx import foreign_skeleton_moe as SM
from kyojin_mlx import foreign_skeleton_rows as R
from kyojin_mlx import foreign_unproven as U

DATA = os.path.join(os.path.dirname(__file__), "data")


def _hdr(name):
    return json.load(open(os.path.join(DATA, name)))


@pytest.fixture(scope="module")
def hy4():
    return _hdr("hdr_hy4_flash_q3k.json")


def test_counts(hy4):
    assert hy4["tensors_per_shard"] == [1415, 1]
    assert hy4["kv"]["split.tensors.count"] == 1416 == len(hy4["tensors"])
    assert hy4["tensors"]["per_layer_token_embd.weight"] == [[96, 660006416], 43]          # the Engram table, F8_E4M3 (N5: branch f8e4m3-rows)


def test_every_tensor_one_slot(hy4):
    m = B.slot_map(hy4["tensors"])
    assert len(m) == 1416 and set(m) == set(hy4["tensors"])
    roles = collections.Counter(r for _, r in m.values())
    assert max(roles.values()) == 1                                    # no two tensors on one role
    per_block = collections.Counter(b for b, _ in m.values())
    # = hy4_flash_tensors.tsv by pattern: skeleton 3 top + 7 x 43 + 3 dense FFN; moe 8 x 42 (router, bias, 3 experts, 3 shexp); N1 7 x 43; N2 6 x 42 + 3 head;
    # N3 8 x 13 (12 sparse layers + nextn); N4 6 x 4 + the table; N6 2 x 42; nextn 4
    assert per_block == {"skeleton": 307, "moe": 336, "N1_mla": 301, "N2_hc": 255, "N3_kpool": 104, "N4_engram": 25,
                         "N6_latent_alias": 84, "nextn": 4}, per_block
    shard1 = [n for n in hy4["tensors"] if n != "per_layer_token_embd.weight"]
    assert len(B.slot_map(shard1)) == 1415
    assert m["blk.5.indexer.k_norm.weight"][0] == "N3_kpool"              # kpool layer: N3 owns the shared kind
    assert m["blk.3.ffn_routed_down.weight"] == ("N6_latent_alias", "model.layers.3.mlp.routed_down.weight")


def test_leftovers_are_listed():
    with pytest.raises(B.SlotError, match="blk.0.not_a_kind.weight"):
        B.slot_map(["token_embd.weight", "blk.0.not_a_kind.weight"])


def test_every_key_one_block(hy4):
    km = B.key_map(dict.fromkeys(hy4["kv_keys"]) | {"general.architecture": hy4["kv"]["general.architecture"]})
    assert len(km) == 76
    c = collections.Counter(km.values())
    assert c["N1_mla"] == 4 and c["N2_hc"] == 3 and c["N3_kpool"] == 2 and c["N4_engram"] == 9 and c["N6_latent_alias"] == 1, c


def test_blocks_disjoint():
    """Each kind / key belongs to one block, except the generic kinds a firing block takes over (FIRES)."""
    seen = collections.defaultdict(list)
    for b, kinds in B.BLOCK_KINDS.items():
        for k in kinds:
            seen[k].append(b)
    shared = {k: v for k, v in seen.items() if len(v) > 1}
    assert shared == {"indexer.k_norm": ["skeleton", "N3_kpool"]}
    keys = collections.defaultdict(list)
    for b, ks in B.BLOCK_KEYS.items():
        for k in ks:
            keys[k].append(b)
    assert not {k: v for k, v in keys.items() if len(v) > 1}


def test_no_arch_name_in_blocks():
    for mod in (B, KP, SM, SK):
        assert "hy_v4" not in inspect.getsource(mod) and "hunyuan" not in inspect.getsource(mod).replace("hunyuan-dense", ""), mod.__name__


def test_row_and_gate(monkeypatch):
    assert R.ROWS["hy_v4_flash"] == dict(norm="rms", act="silu", rope="norm", norm_pos="sandwich", moe=dict(clamp_mode="pre_act"))
    assert "hy_v4_flash" in U.UNPROVEN_ARCHS
    monkeypatch.delenv(U.ENV, raising=False)
    assert not SK.has_rows("hy_v4_flash")
    monkeypatch.setenv(U.ENV, "1")
    assert SK.has_rows("hy_v4_flash")
