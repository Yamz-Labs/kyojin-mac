"""Prefill lever fixes that need no Metal GPU: kernel source structure (padded-row staging), the hot-expert limit of the device-plan MoE path at 384
experts, the idxblk row gate and the flag table defaults. Timing and bit-identity of the kernels are checked on the Mac lane (test_pf_moe_copies.py)."""
import types

import numpy as np
import pytest

pytest.importorskip("mlx.core")

from kyojin_mlx import flags as FL, idxblk, metal_nax as NX, metal_nax_gu as GU, moe_prefill as MP


@pytest.mark.parametrize("cfg", [(32, 2, 32, 64), (16, 1, 32, 64), (32, 1, 32, 64)])
@pytest.mark.parametrize("nproj", [1, 2])
def test_rowmap_gemm_skips_padded_rows(cfg, nproj):
    bmu, nmb, bns, bks = cfg
    src = NX.coop_rm_source((2, 3), "mul1", bmu, nmb, bns, bks, True, nproj, True, 8)
    rows = bmu * nmb
    assert f"i < {rows} * 128u" in src and "atg[i] = half(0.0f)" in src      # A starts at zero, so a skipped row equals a staged zero row
    assert "if (r >= rblk) continue;" in src and "if (r >= %d) continue;" % rows not in src
    assert src.index("atg[i] = half(0.0f)") < src.index("for (uint kb")      # zeroed before the first barrier


def test_epi_rowmap_skips_padded_rows():
    src = GU.gu_source(3, 3, "mul1", 32, 1, 16, 64, 7.0, True, True, 1.0, 0.0, True, True)
    assert "atg[i] = half(0.0f)" in src and "if (r >= rblk) continue;" in src
    assert src.index("atg[i] = half(0.0f)") < src.index("for (uint kb")


def synth_counts(tokens=8192, experts=384, topk=6, hot=None, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.stack([rng.choice(experts, topk, replace=False) for _ in range(tokens)])
    if hot is not None:                                  # a token set that routes alike: every token takes expert `hot`
        ids[:, 0] = hot
        for r in range(tokens):
            while len(set(ids[r])) < topk:
                ids[r, 1:] = rng.choice(experts, topk - 1, replace=False)
    return np.bincount(ids.reshape(-1), minlength=experts)


def mk(chunk):
    return MP.MoePrefill(types.SimpleNamespace(), chunk=chunk, fused=False)


def test_hot_expert_limit_384_experts(monkeypatch):
    counts = synth_counts(hot=17)
    assert len(counts) == 384 and counts.max() == 8192
    ok = synth_counts()
    assert ok.max() < MP.FUSED_MAX                                  # uniform routing never reaches the limit
    pf8, pf4 = mk(8192), mk(4096)
    monkeypatch.setattr(MP, "HOT_FUSED", False)
    assert counts.max() > pf8.glu_limit()                          # default: this layer bails to the host plan at 8192 rows ...
    assert synth_counts(tokens=4096, hot=17).max() <= pf4.glu_limit()   # ... while a 4096-row pass can never exceed the limit
    monkeypatch.setattr(MP, "HOT_FUSED", True)
    assert counts.max() <= pf8.glu_limit() == 8192                 # flag: stays on the device plan
    assert pf4.glu_limit() == MP.FUSED_MAX                         # nothing changes at 4096


def test_idxblk_row_gate(monkeypatch):
    monkeypatch.setattr(idxblk, "MAX_ROWS", 0)
    assert idxblk.applies(512, 32768, 512) and idxblk.applies(8, 1 << 20, 512)
    monkeypatch.setattr(idxblk, "MAX_ROWS", 16)
    assert not idxblk.applies(512, 32768, 512) and idxblk.applies(8, 1 << 20, 512)


def test_flag_defaults_off_and_wired():
    for name in ("pf_moe_hot_fused", "dec_topk_blocks_rows", "pfroof2", "pf_moe_rowmap", "pf_moe_epi", "dec_topk_blocks"):
        assert FL.FLAGS[name].default == "0"
    FL.set_many({"pf_moe_hot_fused": "1", "dec_topk_blocks_rows": "16"})
    try:
        assert MP.HOT_FUSED is True and idxblk.MAX_ROWS == 16
    finally:
        FL.set_many({"pf_moe_hot_fused": "0", "dec_topk_blocks_rows": "0"})
    assert MP.HOT_FUSED is False and idxblk.MAX_ROWS == 0
