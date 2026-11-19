# SPDX-License-Identifier: Apache-2.0
"""Second models of the new skeleton blocks, proven on GGUF headers (names, shapes, types, keys; no weights): for each block, the tensors and keys
of its family in a second architecture's header are claimed by that block alone, or what is left is listed. A block with something left on its
second header is recorded "one model" (HY4_BUILD.md); nothing left = "two headers map". A form that only the second header has (a predicate key
or kind) stays "mapped, not run" until that model produces tokens through it. Fixtures: tests/data/hdr_*.json, header range reads of public GGUF files."""
import collections
import json
import os
import re

import pytest

from kyojin_mlx import foreign_skeleton_blocks as B

DATA = os.path.join(os.path.dirname(__file__), "data")
FAMILY = {                                   # what a header of the block's family looks like (tensor kinds and keys), to find what the block leaves
    "N1_mla": r"attn_(q_a|q_b|kv_a|k_b|v_b|kv_b)|lora_rank|_mla$",
    "N2_hc": r"(^|\.)hc_|output_hc|hyper_connection",
    "N3_kpool": r"indexer",
    "N4_engram": r"engram|per_layer_token|\.ple\.|per_layer_input",
    "N6_latent_alias": r"ffn_routed|ffn_latent|latent_(length|size)",
}


def _load(name):
    return json.load(open(os.path.join(DATA, name)))


def report(h, block):
    """-> (number of tensors claimed by `block`, sorted kinds / keys of the block's family that no block (or two blocks) take)."""
    T = h["tensors"]
    lt = B._layers(T)
    n, left = 0, set()
    fam = re.compile(FAMILY[block])
    for name in T:
        m = B._BLK.match(name)
        c = B.claims_of(name, lt.get(m.group(1)) if m else None)
        kind = re.sub(r"^blk\.\d+\.", "blk.N.", name)
        if len(c) == 1 and c[0][0] == block:
            n += 1
        elif len(c) != 1 and fam.search(kind):
            left.add(kind)
    arch = h["kv"]["general.architecture"]
    for k in h["kv_keys"]:
        sub = k[len(arch) + 1:] if k.startswith(arch + ".") else None
        if sub and fam.search(k) and sum(sub in ks for ks in B.BLOCK_KEYS.values()) != 1:
            left.add(sub)
    return n, sorted(left)


# (fixture, block) -> (tensors claimed, what is left): the recorded state of each block's second header
EXPECT = {
    ("hdr_hy4_preview_q4km.json", "N1_mla"): (7 * 78, []),
    ("hdr_kimi_k3_iq1m.json", "N1_mla"): (7 * 24, []),
    ("hdr_hy4_preview_q4km.json", "N2_hc"): (6 * 78 + 3, []),
    ("hdr_glm53_flash_iq3xxs.json", "N2_hc"): (270, []),
    ("hdr_ds41_flash_udq4kxl.json", "N2_hc"): (240, []),
    ("hdr_glm53_flash_iq3xxs.json", "N3_kpool"): (7 * 12, []),
    ("hdr_ds41_flash_udq4kxl.json", "N4_engram"): (5, []),
    ("hdr_kimi_k3_iq1m.json", "N6_latent_alias"): (3 * 92, []),
}


@pytest.mark.parametrize("fx,block", list(EXPECT))
def test_second_header(fx, block):
    h = _load(fx)
    assert sum(h["tensors_per_shard"]) == len(h["tensors"])
    if "split.tensors.count" in h["kv"]:
        assert h["kv"]["split.tensors.count"] == len(h["tensors"]), "every shard header read"
    assert report(h, block) == EXPECT[(fx, block)]


def test_hc_forms():
    """N2 picks its form by key: identity without sinkhorn_iterations (hy_v4 preview, as Hy4-flash), Sinkhorn with it (glm5-next, deepseek41)."""
    for fx, sink in (("hdr_hy4_flash_q3k.json", False), ("hdr_hy4_preview_q4km.json", False), ("hdr_glm53_flash_iq3xxs.json", True), ("hdr_ds41_flash_udq4kxl.json", True)):
        h = _load(fx)
        a = h["kv"]["general.architecture"]
        assert (f"{a}.hyper_connection.sinkhorn_iterations" in h["kv_keys"]) == sink, fx


def test_kpool_variants_by_presence():
    """N3 on glm5-next: same tensor family minus compressor_value / compressor_norm, LayerNorm key norm (bias); the gate reads the hidden state
    (in-dim 4096 = hidden) where Hy4-flash's reads the normed index key (in-dim 128). The variant is chosen by these presences / shapes."""
    g, f = _load("hdr_glm53_flash_iq3xxs.json")["tensors"], _load("hdr_hy4_flash_q3k.json")["tensors"]
    gk = {re.sub(r"^blk\.\d+\.", "", n) for n in g if "indexer" in n}
    fk = {re.sub(r"^blk\.\d+\.", "", n) for n in f if "indexer" in n}
    assert fk - gk == {"indexer_compressor_value.weight", "indexer_compressor_norm.weight"} and gk - fk == {"indexer.k_norm.bias"}
    first = lambda T: next(v for n, v in T.items() if n.endswith("indexer_compressor_gate.weight"))[0]      # noqa: E731
    assert first(g) == [4096, 128] and first(f) == [128, 128]


def test_engram_table_f8_on_ds41():
    """N4's table: one F8_E4M3 (type 43) per_layer_token_embd in both files (row width 96 in Hy4-flash, 256 in DS4.1); N5 = branch f8e4m3-rows."""
    assert _load("hdr_ds41_flash_udq4kxl.json")["tensors"]["per_layer_token_embd.weight"] == [[256, 768022850], 43]
    assert _load("hdr_hy4_flash_q3k.json")["tensors"]["per_layer_token_embd.weight"] == [[96, 660006416], 43]


def _kv(h):
    return {k: h["kv"].get(k) for k in h["kv_keys"]}


def test_predicate_keys_two_headers():
    """The three keys / kinds that left a block on one model are predicates: each is placed by exactly one block, fires on the header that
    carries it and not on Hy4-flash (whose form is the default). Header-mapped only: the forms they switch on are mapped, not run."""
    from kyojin_mlx import foreign_skeleton_kpool as KP
    from kyojin_mlx import foreign_skeleton_moe as SM
    hy4 = _load("hdr_hy4_flash_q3k.json")
    glm, ds, kimi = (_load(f) for f in ("hdr_glm53_flash_iq3xxs.json", "hdr_ds41_flash_udq4kxl.json", "hdr_kimi_k3_iq1m.json"))
    for key, block in ((KP.SHARE_MTP, "N3_kpool"), ("engram.vocab_size", "N4_engram"), ("engram.compressed_vocab_size", "N4_engram")):
        assert [b for b, ks in B.BLOCK_KEYS.items() if key in ks] == [block], key
    assert KP.shares_mtp(_kv(glm)) is True and KP.shares_mtp(_kv(hy4)) is False
    assert ds["kv"]["deepseek41.engram.vocab_size"] == 16000000 and ds["kv"]["deepseek41.engram.compressed_vocab_size"] == 99092
    assert not any(k.endswith(("engram.vocab_size", "engram.compressed_vocab_size")) for k in hy4["kv_keys"])
    lk, lh = B._layers(kimi["tensors"]), B._layers(hy4["tensors"])
    assert sum(SM.routed_norm_fires(t) for t in lk.values()) == 92 and not any(SM.routed_norm_fires(t) for t in lh.values())
    lat = kimi["kv"]["kimi-k3.expert_latent_length"]
    assert {(tuple(v[0]), v[1]) for n, v in kimi["tensors"].items() if SM.ROUTED_NORM in n} == {((lat,), 0)}      # [latent] F32
    for n in kimi["tensors"]:
        if SM.ROUTED_NORM in n:
            assert B.claims_of(n, lk[n.split(".")[1]]) == [("N6_latent_alias", f"model.layers.{n.split('.')[1]}.mlp.routed_norm.weight")]
    B.key_map(_kv(hy4))                                     # Hy4-flash: every key still placed by exactly one block
