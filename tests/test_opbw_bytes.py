"""opbw_bytes: the per-kind accounting of tools/opbw_bytes.py on a fake pack (no model, no GPU)."""
import json
import os
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))


def _write(path, tensors):
    """Minimal safetensors writer: tensors = [(name, dtype, shape, bytes)]."""
    head, blobs, off = {"__metadata__": {"format": "pt"}}, [], 0
    for name, dt, shape, raw in tensors:
        head[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(head).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def _pack(tmp_path):
    """One layer with a 128x64 attention trellis, two routed experts (3 projections each) and a 3x smaller shared expert,
    plus a head, an embedding table and a vision tensor that a text step never reads."""
    i16 = lambda n: np.zeros(n, np.int16).tobytes()
    f16 = lambda n: np.zeros(n, np.float16).tobytes()
    bf16 = lambda n: np.zeros(n, np.uint16).tobytes()
    lay = [
        ("layers.0.attn.wq_a.trellis", "I16", (1, 1, 16), i16(16)),
        ("layers.0.attn.wq_a.suh", "F16", (16,), f16(16)),
        ("layers.0.attn.wq_a.svh", "F16", (16,), f16(16)),
        ("layers.0.attn_norm.weight", "BF16", (16,), bf16(16)),
        ("layers.0.hc_attn_fn", "F32", (2, 16), np.zeros(32, np.float32).tobytes()),
        ("layers.0.ffn.shared_experts.w1.trellis", "I16", (1, 1, 16), i16(16)),
        ("layers.0.ffn.shared_experts.w2.trellis", "I16", (1, 1, 16), i16(16)),
        ("layers.0.ffn.shared_experts.w3.trellis", "I16", (1, 1, 16), i16(16)),
    ]
    for e in (0, 1):
        for proj in ("w1", "w2", "w3"):
            lay.append((f"layers.0.ffn.experts.{e}.{proj}.trellis", "I16", (1, 1, 32), i16(32)))
    _write(tmp_path / "layers.0.safetensors", lay)
    glb = [
        ("head.trellis", "I16", (1, 1, 64), i16(64)),
        ("head.suh", "F16", (16,), f16(16)),
        ("head.svh", "F16", (16,), f16(16)),
        ("embed.weight", "BF16", (16, 16), bf16(256)),
        ("vision.blocks.0.attn.wo.weight", "BF16", (16, 16), bf16(256)),
    ]
    _write(tmp_path / "global.safetensors", glb)
    return tmp_path


def test_kinds_and_totals(tmp_path):
    from opbw_bytes import account, load_index, kind_of

    idx = load_index(str(_pack(tmp_path)))
    tot, routed_layers, per_layer, skipped = account(idx)
    assert kind_of("layers.3.ffn.experts.7.w2.trellis") == "routed"
    assert kind_of("layers.3.ffn.shared_experts.w2.trellis") == "shared"
    assert kind_of("layers.3.attn.wq_b.trellis") == "attn"
    assert kind_of("layers.3.hc_ffn_fn") == "glue"
    assert kind_of("head.trellis") == "head" and kind_of("embed.weight") == "embed"
    assert kind_of("vision.blocks.0.attn.wo.weight") is None
    assert skipped == 512                                        # the vision tensor only
    assert tot["routed"] == 2 * 3 * 64                           # 2 experts x 3 proj x 64 B
    assert tot["shared"] == 3 * 32
    assert tot["attn"] == 3 * 32                                 # trellis 32 B + suh + svh
    assert tot["glue"] == 128 + 32                               # hc_attn_fn 2 x 16 f32 + attn_norm
    assert tot["head"] == 128 + 32 + 32                          # trellis 64 i16 + suh + svh
    assert tot["embed"] == 512
    assert routed_layers == {"0": 384}


def test_step_and_chunk_follow_the_geometry(tmp_path):
    """The step counts topk experts per layer out of all of them; a chunk counts every expert once per MoE call."""
    from opbw_bytes import account, load_index

    idx = load_index(str(_pack(tmp_path)))
    tot, routed_layers, per_layer, _ = account(idx)
    one_expert = tot["routed"] / 384.0                           # the per-layer mean expert
    step = tot["attn"] + tot["shared"] + tot["glue"] + tot["head"] + one_expert * 6
    assert step == 96 + 96 + 160 + 192 + 6 * 384 / 384.0
    # one 8192-row chunk with 4096-row MoE calls sweeps the bank twice; with 8192-row calls once
    for cap, sweeps in ((4096, 2), (8192, 1)):
        assert (8192 + cap - 1) // cap == sweeps
        chunk = tot["attn"] + tot["shared"] + tot["glue"] + tot["routed"] * sweeps
        assert chunk == 96 + 96 + 160 + 384 * sweeps


@pytest.mark.parametrize("cap,sweeps", [(4096, 2), (8192, 1), (2048, 4)])
def test_moe_sweeps(cap, sweeps):
    assert (8192 + cap - 1) // cap == sweeps
