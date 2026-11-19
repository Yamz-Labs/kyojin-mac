# SPDX-License-Identifier: Apache-2.0
"""Which block of the decoder skeleton claims each tensor and each metadata key of a GGUF: the header-level contract of the new shared blocks.

Each block declares the tensor kinds (after `blk.N.`, without .weight / .bias) and top-level tensors it claims, with a role (module path), and the
metadata keys (after `<arch>.`) it places. `slot_map` assigns every tensor of a header to exactly one (block, role) and refuses leftovers or double
claims; `key_map` does the same for the keys. Keyed on tensor names and key names only: the architecture string is read from the file as the key
prefix, no architecture name appears here.

Blocks (HY4_DESIGN.md §3): skeleton + moe exist; N1 mla, N2 hc, N4 engram and the nextn head are declared here (their modules come with stage 2 / 5),
N3 kpool lives in foreign_skeleton_kpool, N6 (latent alias) in foreign_skeleton_moe.ALIASES. N5 (the F8_E4M3 row type of the Engram table) is a
stored type, not a slot: branch f8e4m3-rows (commit a1dd9f1c) decodes it in foreign_gguf.
"""
from __future__ import annotations

import re

from . import foreign_skeleton as SK
from . import foreign_skeleton_kpool as KP
from . import foreign_skeleton_moe as SM


class SlotError(ValueError):
    pass


# N1, MLA projections (absorbed): q_a -> norm -> q_b; kv_a_mqa -> latent + rope; k_b folded into q, v_b after the attention
MLA_KINDS = {k: v for k, v in SK.N_NAMES.items() if k.startswith("attn_")}     # the module paths of foreign_skeleton (MlaAttention as self_attn)
MLA_KEYS = {"attention.q_lora_rank", "attention.kv_lora_rank", "attention.key_length_mla", "attention.value_length_mla"}

# N2, hyper-connection streams: pre / post mixes per sub-layer, head reduce at the top
HC_KINDS = {k: v for k, v in SK.N_NAMES.items() if k.startswith("hc_")}
HC_TOP = dict(SK.N_TOP)
HC_KEYS = {"hyper_connection.count", "hyper_connection.epsilon", "hyper_connection.magnitude", "hyper_connection.sinkhorn_iterations"}

# N4, Engram from a GGUF: per-layer projections / gate in the blocks, ONE table at the top (shard 2 of Hy4-flash, F8_E4M3 rows: N5)
ENGRAM_KINDS = {k: v for k, v in SK.N_NAMES.items() if k.startswith("engram_")}
ENGRAM_TOP = {"per_layer_token_embd.weight": "model.engram.table"}
ENGRAM_KEYS = {"ple.layers", "ple.ngram_size", "ple.heads_per_ngram", "ple.layer_multipliers", "ple.head_offsets", "ple.head_vocab_sizes",
               "embedding_length_per_layer_input", "engram.pad_token_id", "engram.token_map",
               # predicate keys (DS4.1 writes them, Hy4-flash not): present = engram_gguf checks the primes lie above vocab_size and token_map's range
               # equals compressed_vocab_size; absent = the compressed vocab is token_map's range, no prime floor
               "engram.vocab_size", "engram.compressed_vocab_size"}

# nextn draft head (stage 5: adapter onto glm_nextn.NextN)
NEXTN_KINDS = {k: k for k in ("nextn.eh_proj", "nextn.enorm", "nextn.hnorm", "nextn.shared_head_norm", "nextn.embed_tokens", "nextn.shared_head_head")}

BLOCK_KINDS = {
    "skeleton": dict(SK.BLK_NAMES),
    "moe": {k: "mlp." + v for k, v in SM.NAMES.items() if k not in SM.ALIASES},
    "N1_mla": MLA_KINDS,
    "N2_hc": HC_KINDS,
    "N3_kpool": KP.KINDS,
    "N4_engram": ENGRAM_KINDS,
    "N6_latent_alias": {**{a: "mlp." + SM.NAMES[a] for a in SM.ALIASES}, SM.ROUTED_NORM: "mlp.routed_norm"},
    "nextn": NEXTN_KINDS,
}
BLOCK_TOP = {"skeleton": dict(SK.TOP_NAMES), "N2_hc": HC_TOP, "N4_engram": ENGRAM_TOP}
BLOCK_KEYS = {
    "skeleton": set(SK._KNOWN), "moe": SM.MOE_KEYS - set(SM.KEY_ALIASES), "N1_mla": MLA_KEYS, "N2_hc": HC_KEYS, "N3_kpool": KP.KEYS,
    "N4_engram": ENGRAM_KEYS, "N6_latent_alias": set(SM.KEY_ALIASES),
}
FILE_KEY_PREFIXES = ("general.", "tokenizer.", "quantize.", "split.")      # not model structure: the reader and the tokenizer place them

# a block that fires on a layer (its own tensors are there) takes the kinds it shares with the skeleton's generic form, e.g. indexer.k_norm of
# the block-sparse indexer: the same weight-only rms norm, owned by N3 on a kpool layer
FIRES = {"N3_kpool": KP.fires}

_BLK = re.compile(r"blk\.(\d+)\.(.+)\.(weight|bias|scale)$")


def _layers(names):
    lt = {}
    for name in names:
        m = re.match(r"blk\.(\d+)\.(.+)$", name)
        if m:
            lt.setdefault(m.group(1), set()).add(m.group(2))
    return lt


def claims_of(name: str, layer: set | None = None):
    """-> list of (block, role) that claim this tensor name; `layer` = the tensor kinds (with suffix) of its block, for FIRES."""
    out = [(b, r) for b, top in BLOCK_TOP.items() for n, r in top.items() if n == name]
    m = _BLK.match(name)
    if m:
        il, kind, suf = m.groups()
        c = [(b, f"model.layers.{il}.{kinds[kind]}.{suf}") for b, kinds in BLOCK_KINDS.items() if kind in kinds]
        fired = [x for x in c if x[0] in FIRES and FIRES[x[0]](layer or set())]
        out += fired if fired and len(c) == 2 and any(b == "skeleton" for b, _ in c) else c
    return out


def slot_map(tensors) -> dict:
    """tensor names (iterable or dict) -> {name: (block, role)}; SlotError listing every unclaimed or doubly claimed name."""
    out, left, double = {}, [], []
    lt = _layers(tensors)
    for name in tensors:
        m = _BLK.match(name)
        c = claims_of(name, lt.get(m.group(1)) if m else None)
        if len(c) == 1:
            out[name] = c[0]
        else:
            (left if not c else double).append(name)
    if left or double:
        raise SlotError(f"{len(left)} tensors no block claims {sorted(left)[:12]}; {len(double)} claimed twice {sorted(double)[:12]}")
    return out


def key_map(kv: dict) -> dict:
    """metadata keys -> {key: block}; the model keys are `<general.architecture>.` + a key of exactly one block."""
    arch = kv.get("general.architecture", "")
    p = arch + "."
    out, left, double = {}, [], []
    for k in kv:
        if k.startswith(FILE_KEY_PREFIXES):
            out[k] = "file"
            continue
        sub = k[len(p):] if arch and k.startswith(p) else None
        bs = [b for b, keys in BLOCK_KEYS.items() if sub in keys]
        if len(bs) == 1:
            out[k] = bs[0]
        else:
            (left if not bs else double).append(k)
    if left or double:
        raise SlotError(f"{len(left)} keys no block places {left[:12]}; {len(double)} placed twice {double[:12]}")
    return out
